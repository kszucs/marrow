# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Statistics for a plan's sources, gathered because a caller asked.

A plan is a description, and building one touches no data. `analyze` is the
step that looks: every `InMemoryTable` is summarised from its batch and every
`ParquetScan` from its file's footer, and each keeps the result as the
statistics its `estimate` answers with — row and null counts, bounds, distinct
counts, and a width for columns whose type does not fix one.

It is its own step and its own module so that nothing else pays for it: an AOT
binary that never calls `analyze` links neither the kernels it runs nor the
Parquet reader, and a plan built without it estimates what it always did.
"""

from std.sys import bit_width_of

from ..arrays import DynArray
from ..dtypes import BinaryLikeType, PrimitiveType
from ..execution import ExecContext
from ..io import DynSource
from ..kernels.distinct import approx_count_distinct
from ..parquet.reader import LeafSet, ParquetFile
from ..scalars import DynScalar, NullScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .bindings import Bindings
from .estimates import Approx, ColumnEstimate, Estimate
from .index import ColumnZones, Index
from .logical import DynRelation, InMemoryTable, ParquetScan


def analyze(
    plan: DynRelation,
    ctx: ExecContext = ExecContext.auto(),
    bindings: Bindings = Bindings(),
) raises -> DynRelation:
    """`plan` with statistics attached to every source that has none.

    A source that already carries statistics keeps them, so analysing twice
    reads nothing twice. A scan whose path is a parameter this execution has
    not bound is left without: there is no file to read yet. One whose path
    `bindings` does bind is summarised from *that* file, and the statistics
    stay with the plan: analyse it again, from the unanalysed plan, for a
    different binding. A scan of several files is summarised from all of
    them, each file's row groups chunks of one index.
    """
    if plan.isa[InMemoryTable]():
        ref source = plan.get[InMemoryTable]()
        if source.statistics:
            return plan.copy()
        var out: DynRelation = source.with_statistics(
            summarize(source.batch, ctx)
        )
        return out^

    if plan.isa[ParquetScan]():
        ref scan = plan.get[ParquetScan]()
        if scan.statistics:
            return plan.copy()
        var paths: List[String]
        try:
            paths = scan.path.resolve(bindings)
        except:
            return plan.copy()
        var schema = scan.schema()
        var index = Index()
        var bytes = List[Int](length=len(schema.fields), fill=0)
        for i in range(len(paths)):
            var file = ParquetFile[DynSource, LeafSet.all()](
                DynSource.open(paths[i])
            )
            var part = Index.from_parquet(file)
            _add_decoded_bytes(bytes, schema, part, file)
            if i == 0:
                index = part^
            else:
                index = _appended(index, part)
        var estimate = Estimate.from_index(index, schema)
        # The data's own average where every row group can say, over the
        # default a variable-width dtype is otherwise priced at.
        var rows = estimate.rows.known()
        if rows and rows.value() > 0:
            for i in range(len(schema.fields)):
                ref c = estimate.columns[i]
                if bytes[i] >= 0 and not c.width.is_exact():
                    c.width = Approx.estimated(
                        (bytes[i] + rows.value() - 1) // rows.value()
                    )
        var out: DynRelation = scan.with_statistics(estimate^)
        return out^

    def descend(child: DynRelation) raises {imm} -> DynRelation:
        return analyze(child, ctx, bindings)

    return plan.traverse(descend)


def summarize(
    batch: RecordBatch, ctx: ExecContext = ExecContext.auto()
) raises -> Estimate:
    """What a batch holds: exact row and null counts, each primitive column's
    bounds, an estimated distinct count for every column the hash kernel
    takes, and each column's width.

    The distinct count is a HyperLogLog sketch (`approx_count_distinct`),
    never more than the column's non-null values. A dtype the hash kernel rejects keeps an unknown count rather than
    failing the whole summary.
    """
    var rows = batch.num_rows()
    var cols = List[ColumnEstimate](capacity=batch.num_columns())
    for i in range(batch.num_columns()):
        ref f = batch.schema.fields[i]
        ref column = batch.column(i)
        var nulls = column.null_count()
        var bounds = _bounds(column)

        var ndv = Approx.unknown()
        try:
            var distinct = Int(approx_count_distinct(column, ctx).value())
            ndv = Approx.estimated(min(distinct, rows - nulls))
        except:
            pass

        # The data's own average where it can be measured, over the default a
        # variable-width dtype is otherwise priced at.
        var width = ColumnEstimate.width_of(f.dtype)
        if not width.is_exact():
            var measured = _average_width(column)
            if measured.is_known():
                width = measured

        cols.append(
            ColumnEstimate(
                f.name.copy(),
                bounds[0].copy(),
                bounds[1].copy(),
                nulls=Approx.exact(nulls),
                ndv=ndv,
                width=width,
            )
        )
    return Estimate(Approx.exact(rows), cols^)


def _bounds(column: DynArray) raises -> Tuple[DynScalar, DynScalar]:
    """A primitive column's smallest and largest value, NULL and NaN skipped;
    null scalars for any other column or one holding no such value.

    NaN is skipped for the reason a Parquet writer skips it: a bound is read
    as a proof, and a NaN maximum would prove `x > 5` false and prune rows
    that match it.
    """
    var dtype = column.dtype()
    if not dtype.is_primitive():
        return (NullScalar().to_dyn(), NullScalar().to_dyn())

    def arm[
        T: PrimitiveType
    ](witness: T) raises {imm} -> Tuple[DynScalar, DynScalar]:
        ref values = column.as_primitive[T]()
        return ColumnEstimate.bounds_of[skip_nan=T.native.is_floating_point()](
            values, values, witness
        )

    return dtype.dispatch_primitive(arm)


def _average_width(column: DynArray) raises -> Approx:
    """A binary-like column's bytes per value, rounded up, its offset
    included; unknown for any other column.

    The value bytes are `total_values_length`, not the length of the value
    buffer a slice shares with its parent.
    """
    var dtype = column.dtype()
    if not dtype.is_binary_like():
        return Approx.unknown()

    def arm[T: BinaryLikeType](witness: T) raises {imm} -> Approx:
        ref array = column.as_binary_like[T]()
        var offset_width = bit_width_of[T.offset]() // 8
        var n = len(array)
        if n == 0:
            return Approx.exact(offset_width)
        var bytes = array.total_values_length()
        return Approx.estimated((bytes + n - 1) // n + offset_width)

    return dtype.dispatch_binarylike(arm)


def _add_decoded_bytes(
    mut bytes: List[Int],
    schema: Schema,
    index: Index,
    file: ParquetFile[DynSource, LeafSet.all()],
) raises:
    """Add to `bytes`, per field of `schema`, what `file`'s row groups decode
    to (`ColumnMetaData.decoded_bytes`); `-1` once one cannot say.

    Only for the columns `index` holds, which are the file's top-level fields
    when each is one leaf.
    """
    var leaves = file.schema()
    ref meta = file.metadata()
    for i in range(len(schema.fields)):
        if bytes[i] < 0:
            continue
        var leaf = leaves.get_field_index(schema.fields[i].name)
        var zones = index.find(schema.fields[i].name)
        if leaf < 0 or zones < 0:
            bytes[i] = -1
            continue
        ref distinct = index.columns[zones].distinct_counts
        for rg in range(index.chunks()):
            var decoded = (
                meta.row_groups[rg]
                .columns[leaf]
                .meta_data.decoded_bytes(distinct[rg])
            )
            if decoded < 0:
                bytes[i] = -1
                break
            bytes[i] += decoded


def _appended(index: Index, more: Index) raises -> Index:
    """`index` followed by `more`'s chunks, keeping the columns both hold in
    one recorded dtype — the next file of a scan, its row groups as further
    chunks."""
    var rows = index.rows.copy()
    rows.extend(more.rows.copy())
    var columns = List[ColumnZones](capacity=len(index.columns))
    for ref c in index.columns:
        var j = more.find(c.name)
        if j < 0:
            continue
        ref d = more.columns[j]
        if not (c.dtype.is_null() or d.dtype.is_null() or c.dtype == d.dtype):
            continue
        var mins = c.mins.copy()
        mins.extend(d.mins.copy())
        var maxes = c.maxes.copy()
        maxes.extend(d.maxes.copy())
        var nulls = c.null_counts.copy()
        nulls.extend(d.null_counts.copy())
        var distinct = _per_chunk(c.distinct_counts, c.num_chunks())
        distinct.extend(_per_chunk(d.distinct_counts, d.num_chunks()))
        columns.append(
            ColumnZones(c.name.copy(), mins^, maxes^, nulls^, distinct^)
        )
    return Index(rows^, columns^)


def _per_chunk(counts: List[Int], chunks: Int) -> List[Int]:
    """`counts`, or `-1` per chunk when none was recorded."""
    if len(counts) == 0:
        return List[Int](length=chunks, fill=-1)
    return counts.copy()
