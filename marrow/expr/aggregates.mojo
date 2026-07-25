"""Runtime aggregate routing — the *only* place a function **name** or **tag**
resolves to a comptime ``AggKernel``.

The kernel layer (``marrow.kernels.aggregate`` / ``marrow.kernels.groupby``)
knows aggregates only as types: ``SumKernel``, ``AggState[K, V]``,
``GroupBy.aggregate[K]``. It has no vocabulary for ``"sum"`` and no ``UInt8``
tag. Everything that maps a *runtime* identity onto those types lives here, next
to ``DynValue``'s tag interpreter — the layer that already exists to turn
runtime descriptions into kernel calls:

- ``agg_tag_from_name`` / ``agg_name_from_tag`` — the name <-> tag vocabulary;
- ``for_agg_tag`` — tag -> comptime ``AggKernel`` (the fold aggregates);
- ``agg_out_dtype`` — the aggregate's output dtype, for planners that need the
  output schema before the data exists;
- ``aggregate_column`` — one aggregate column over precomputed group ids, the
  shared per-column routing (distinct kernels, string/temporal min/max,
  non-numeric ``count``, typed ``AggState`` folds);
- ``aggregate_grouped`` / ``aggregate_whole`` — the runtime, multi-aggregate
  drivers: group once (or not at all, for ``SELECT agg(x)`` with no ``GROUP
  BY``) and apply N runtime-selected aggregates in the same pass.

Consumers are the expression layer's ``AggregateProcessor``, the relational
planner's output-schema derivation, and the Python
``group_by(...).aggregate([...])`` binding (the F1 frontend) — every one of them
starts from a runtime string.
"""

from std.algorithm.functional import sync_parallelize

from ..arrays import (
    AnyArray,
    StructArray,
    Int32Array,
    Int64Array,
    UInt64Array,
)
from ..builders import Int32Builder
from ..dtypes import (
    AnyDataType,
    Field,
    NumericType,
    int32,
    int64,
    float64,
)
from ..schema import Schema
from ..scalars import AnyScalar
from ..tabular import RecordBatch
from ..kernels.aggregate import (
    AggKernel,
    AggState,
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
    min_max_string_grouped,
    count_valid_grouped,
    reinterpret_array,
    temporal_backing_dtype,
)
from ..kernels.concat import concat
from ..kernels.distinct import (
    count_distinct,
    approx_count_distinct,
    count_distinct_grouped,
    approx_count_distinct_grouped,
)
from ..kernels.execution import ExecutionContext
from ..kernels.filter import take
from ..kernels.groupby import (
    GroupBy,
    HashGrouper,
    slice_struct,
    GROUP_THREAD_LOCAL,
)


# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------

comptime AGG_SUM: UInt8 = 0
comptime AGG_MIN: UInt8 = 1
comptime AGG_MAX: UInt8 = 2
comptime AGG_COUNT: UInt8 = 3
comptime AGG_MEAN: UInt8 = 4
comptime AGG_PRODUCT: UInt8 = 5
# Distinct aggregates are not `AggKernel` folds (they carry a hash set / HLL
# sketch, not a scalar accumulator), so they have tags but no `for_agg_tag`
# case — the drivers below route them to the `distinct` kernels instead.
comptime AGG_COUNT_DISTINCT: UInt8 = 6
comptime AGG_APPROX_COUNT_DISTINCT: UInt8 = 7


def agg_is_distinct(tag: UInt8) -> Bool:
    """Whether ``tag`` is a distinct aggregate (routed to the distinct kernels
    rather than the `AggState` fold path)."""
    return tag == AGG_COUNT_DISTINCT or tag == AGG_APPROX_COUNT_DISTINCT


def agg_tag_from_name(name: String) raises -> UInt8:
    """Map an aggregate function name to its tag."""
    if name == "sum":
        return AGG_SUM
    elif name == "min":
        return AGG_MIN
    elif name == "max":
        return AGG_MAX
    elif name == "count":
        return AGG_COUNT
    elif name == "mean":
        return AGG_MEAN
    elif name == "product":
        return AGG_PRODUCT
    elif name == "count_distinct":
        return AGG_COUNT_DISTINCT
    elif name == "approx_count_distinct":
        return AGG_APPROX_COUNT_DISTINCT
    raise Error("unknown aggregate function: ", name)


def for_agg_tag[
    job: def[K: AggKernel]() raises capturing[_] -> None
](tag: UInt8) raises:
    """Resolve a runtime aggregate tag to its comptime kernel, run `job[K]`."""
    if tag == AGG_SUM:
        job[SumKernel]()
    elif tag == AGG_MIN:
        job[MinKernel]()
    elif tag == AGG_MAX:
        job[MaxKernel]()
    elif tag == AGG_COUNT:
        job[CountKernel]()
    elif tag == AGG_MEAN:
        job[MeanKernel]()
    elif tag == AGG_PRODUCT:
        job[ProductKernel]()
    else:
        raise Error("unknown aggregate tag ", Int(tag))


def agg_name_from_tag(tag: UInt8) raises -> String:
    """The kernel name for an aggregate tag (default output column name)."""
    if tag == AGG_COUNT_DISTINCT:
        return "count_distinct"
    elif tag == AGG_APPROX_COUNT_DISTINCT:
        return "approx_count_distinct"
    var box = List[String]()

    @parameter
    def name[K: AggKernel]() raises:
        box.append(String(K.name))

    for_agg_tag[name](tag)
    return box[0]


def agg_out_dtype(tag: UInt8, value_dtype: AnyDataType) raises -> AnyDataType:
    """The output dtype of aggregate `tag` over a `value_dtype` column.

    The single home of the rule — shared by the group-by drivers (which produce
    the column) and by any planner that needs the aggregate's output schema
    before the data exists:

    - `count` / `count_distinct` / `approx_count_distinct` → `int64`;
    - `mean` → `float64`;
    - `min` / `max` are order-preserving, so they keep the *input* dtype —
      numeric, string, or temporal (unit/tz included);
    - the remaining folds (`sum` / `product`) widen to their accumulator dtype
      (`AggKernel.AccType`), so narrow integers don't overflow."""
    if tag == AGG_COUNT or agg_is_distinct(tag):
        return AnyDataType(int64)
    elif tag == AGG_MEAN:
        return AnyDataType(float64)
    elif tag == AGG_MIN or tag == AGG_MAX:
        return value_dtype.copy()
    else:
        var box = List[AnyDataType]()

        @parameter
        def by_kind[K: AggKernel]() raises:
            @parameter
            def by_value[V: NumericType](d: V) raises:
                box.append(AnyDataType(K.AccType[V]()))

            value_dtype.dispatch_numeric[by_value]()

        for_agg_tag[by_kind](tag)
        return box[0].copy()


# ---------------------------------------------------------------------------
# Per-column routing over precomputed group ids
# ---------------------------------------------------------------------------


def aggregate_column(
    gids: Int32Array, value: AnyArray, num_groups: Int, tag: UInt8
) raises -> AnyArray:
    """Compute one aggregate column over precomputed ``gids``.

    The shared per-column entry point for *every* runtime-tagged aggregate —
    used by the multi-aggregate drivers below and by the expression layer's
    ``AggregateProcessor``, so a caller holding group ids never repeats the
    routing:

    - distinct aggregates → the ``distinct`` kernels;
    - string min/max → the dedicated bytewise scan;
    - temporal min/max → its (order-preserving) signed-integer backing,
      reinterpreted here and relabelled to the temporal dtype on the way out;
    - non-numeric ``count`` → the validity-only per-group scan;
    - every remaining fold → the typed ``AggState`` path.

    ``aggregate_grouped`` reinterprets temporal columns *before* partitioning
    (``take``/``concat`` don't handle temporal dtypes) and relabels the output
    columns itself, so the temporal branch here is inert on that path."""
    var vdt = value.dtype()
    if tag == AGG_COUNT_DISTINCT:
        return count_distinct_grouped(gids, value, num_groups)
    elif tag == AGG_APPROX_COUNT_DISTINCT:
        return approx_count_distinct_grouped(gids, value, num_groups)
    elif tag == AGG_COUNT and not vdt.is_numeric():
        return count_valid_grouped(gids, value, num_groups).to_any()
    elif (tag == AGG_MIN or tag == AGG_MAX) and (
        vdt.is_string() or vdt.is_large_string()
    ):
        return min_max_string_grouped(gids, value, num_groups, tag == AGG_MIN)
    elif (tag == AGG_MIN or tag == AGG_MAX) and vdt.is_temporal():
        var backing = temporal_backing_dtype(vdt)
        var out = _agg_over_gids(
            gids, reinterpret_array(value, backing), num_groups, tag
        )
        return reinterpret_array(out, vdt)
    return _agg_over_gids(gids, value, num_groups, tag)


def _agg_over_gids(
    gids: Int32Array, value: AnyArray, num_groups: Int, tag: UInt8
) raises -> AnyArray:
    """Aggregate ``value`` over precomputed group ids ``gids`` (one typed
    ``AggState`` resolved from the runtime tag + value dtype)."""
    var box = List[AnyArray]()

    @parameter
    def run[K: AggKernel]() raises:
        @parameter
        def by_value[V: NumericType](d: V) raises:
            var state = AggState[K, V]()
            state.update(gids, value.as_primitive[V](), num_groups)
            box.append(state.finish(num_groups).to_any())

        value.dtype().dispatch_numeric[by_value]()

    for_agg_tag[run](tag)
    return box[0].copy()


# ---------------------------------------------------------------------------
# Runtime multi-aggregate drivers — group ONCE, apply N runtime-selected
# aggregates in the same pass. The aggregate columns are named by kernel
# (callers rename as needed).
# ---------------------------------------------------------------------------


def aggregate_grouped(
    gb: GroupBy, values: List[AnyArray], tags: List[UInt8]
) raises -> RecordBatch:
    """Apply several aggregates over one grouping of ``gb``'s keys.

    ``values[j]`` is aggregated with the kernel for ``tags[j]``. Returns the
    unique key columns followed by one column per aggregate. This is the path
    the Python ``group_by(...).aggregate([...])`` binding uses, so a multi-agg
    query hashes/probes the keys once instead of once per aggregate."""
    # Temporal min/max are order-preserving on the signed-integer backing, so
    # reinterpret the value column to int up front. Then every downstream
    # path — serial / thread-local / radix, `take`, `concat`, `AggState` — is
    # fully numeric (`take` doesn't handle temporal dtypes), and the output
    # column is relabelled back to the temporal dtype at the end.
    var work = List[AnyArray]()
    var relabel = List[Optional[AnyDataType]]()  # per aggregate column
    var names = List[String]()
    for j in range(len(tags)):
        var vdt = values[j].dtype()
        if (tags[j] == AGG_MIN or tags[j] == AGG_MAX) and vdt.is_temporal():
            var native_dt = temporal_backing_dtype(vdt)
            work.append(reinterpret_array(values[j], native_dt))
            relabel.append(vdt.copy())
        else:
            work.append(values[j].copy())
            relabel.append(None)
        names.append(agg_name_from_tag(tags[j]))

    # Some aggregates can't be combined by the thread-local partial + merge
    # path, so they must take the radix (or serial) route instead:
    #   * distinct (count_distinct / approx) carry a hash set / HLL sketch,
    #     not a mergeable scalar;
    #   * string min/max and non-numeric `count` are dedicated per-group
    #     scans in `aggregate_column`, not the numeric `AggState` fold the
    #     thread-local merge understands.
    # The radix path partitions by key hash, so a group lands wholly in one
    # partition and its result is final without any cross-partition merge.
    var avoid_thread_local = False
    for j in range(len(tags)):
        var wdt = work[j].dtype()
        if agg_is_distinct(tags[j]):
            avoid_thread_local = True
            break
        elif tags[j] == AGG_COUNT and not wdt.is_numeric():
            avoid_thread_local = True
            break
        elif (tags[j] == AGG_MIN or tags[j] == AGG_MAX) and (
            wdt.is_string() or wdt.is_large_string()
        ):
            avoid_thread_local = True
            break

    var result: RecordBatch
    if gb.strategy() == GROUP_THREAD_LOCAL and not avoid_thread_local:
        result = _thread_local_multi(
            gb.keys(), work, tags, names, gb.num_threads()
        )
    else:
        # `aggregate_columns` groups once and picks serial vs. radix itself;
        # the tag routing is supplied as the comptime aggregator.
        @parameter
        def by_tag(
            j: Int, gids: Int32Array, value: AnyArray, ng: Int
        ) raises -> AnyArray:
            return aggregate_column(gids, value, ng, tags[j])

        result = gb.aggregate_columns[by_tag](work, names)

    return _relabel_temporal(result^, len(tags), relabel)


def _relabel_temporal(
    var result: RecordBatch,
    num_aggs: Int,
    relabel: List[Optional[AnyDataType]],
) raises -> RecordBatch:
    """Relabel the temporal min/max aggregate columns (computed over their
    integer backing) back to their temporal dtype. A no-op when no aggregate
    was reinterpreted."""
    var any_relabel = False
    for j in range(len(relabel)):
        if relabel[j]:
            any_relabel = True
            break
    if not any_relabel:
        return result^

    var num_keys = result.num_columns() - num_aggs
    var out_fields = List[Field]()
    var out_cols = List[AnyArray]()
    for c in range(result.num_columns()):
        var agg_idx = c - num_keys
        if agg_idx >= 0 and relabel[agg_idx]:
            var dt = relabel[agg_idx].value().copy()
            out_cols.append(reinterpret_array(result.column(c), dt))
            out_fields.append(Field(result.schema.fields[c].name, dt.copy()))
        else:
            out_fields.append(result.schema.fields[c].copy())
            out_cols.append(result.column(c).copy())
    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


def _thread_local_multi(
    keys: StructArray,
    values: List[AnyArray],
    tags: List[UInt8],
    names: List[String],
    num_threads: Int,
) raises -> RecordBatch:
    """Thread-local partial aggregation for N runtime-tagged *fold* aggregates.

    Every worker groups an equal contiguous chunk once and folds each aggregate
    into its own `AggState`; a serial merge then re-keys the chunks into a global
    grouper and folds the partials. Only valid when every aggregate is a
    mergeable `AggKernel` fold — the caller gates on that."""
    var n = len(keys)
    var na = len(tags)
    var chunk = (n + num_threads - 1) // num_threads

    var part_keys = List[Optional[StructArray]](length=num_threads, fill=None)
    # Per (thread, aggregate) partial state, flattened as [t * na + j].
    var part_acc = List[Optional[AnyArray]](length=num_threads * na, fill=None)
    var part_cnt = List[Optional[Int64Array]](
        length=num_threads * na, fill=None
    )

    @parameter
    def worker(t: Int) raises:
        var start = t * chunk
        if start >= n:
            return
        var length = min(chunk, n - start)
        var kchunk = slice_struct(keys, start, length)

        var grouper = HashGrouper()
        var gids = grouper.consume_keys(kchunk)  # group this chunk ONCE
        var ng = grouper.num_groups()

        var kfields = grouper.key_fields(kchunk)
        var kcols = grouper.key_columns(kfields)
        part_keys[t] = StructArray(
            dtype=keys.dtype.copy(),
            length=ng,
            nulls=0,
            offset=0,
            bitmap=None,
            children=kcols^,
        )

        for j in range(na):
            var vchunk = values[j].slice(start, length)
            # Compute the flat slot in the worker's direct scope; capturing
            # `t`/`na`/`j` through the doubly-nested `@parameter` closures
            # below corrupts the runtime index.
            var slot = t * na + j

            @parameter
            def run_local[K: AggKernel]() raises:
                @parameter
                def by_value[V: NumericType](d: V) raises:
                    var state = AggState[K, V]()
                    state.update(gids, vchunk.as_primitive[V](), ng)
                    var parts = state.into_partials()
                    part_acc[slot] = parts[0].copy().to_any()
                    part_cnt[slot] = parts[1].copy()

                vchunk.dtype().dispatch_numeric[by_value]()

            for_agg_tag[run_local](tags[j])

    sync_parallelize[worker](num_threads)

    # Merge — re-key every chunk into the global grouper ONCE (shared across
    # aggregates), then fold each aggregate's partials at the global ids.
    var gg = HashGrouper()
    var l2g = List[Int32Array]()
    for t in range(num_threads):
        if part_keys[t]:
            l2g.append(gg.consume_keys(part_keys[t].value()))
        else:
            var empty = Int32Builder(0)
            l2g.append(empty.finish())
    var ngg = gg.num_groups()

    var kfields = gg.key_fields(keys)
    var out_fields = List[Field]()
    for k in range(len(kfields)):
        out_fields.append(kfields[k].copy())
    var out_cols = gg.key_columns(kfields)

    for j in range(na):
        var box = List[AnyArray]()

        @parameter
        def run_merge[K: AggKernel]() raises:
            @parameter
            def by_value[V: NumericType](d: V) raises:
                var gstate = AggState[K, V]()
                for t in range(num_threads):
                    if not part_keys[t]:
                        continue
                    gstate.merge(
                        l2g[t],
                        part_acc[t * na + j]
                        .value()
                        .as_primitive[K.AccType[V]](),
                        part_cnt[t * na + j].value(),
                        ngg,
                    )
                box.append(gstate.finish(ngg).to_any())

            values[j].dtype().dispatch_numeric[by_value]()

        for_agg_tag[run_merge](tags[j])
        out_fields.append(Field(names[j], box[0].dtype().copy()))
        out_cols.append(box[0].copy())

    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


# ---------------------------------------------------------------------------
# Whole-table aggregation (no GROUP BY)
# ---------------------------------------------------------------------------


def aggregate_whole(
    values: List[AnyArray], tags: List[UInt8], num_threads: Int = 0
) raises -> RecordBatch:
    """Whole-table aggregation — ``SELECT agg(x), ...`` with no GROUP BY.

    A single implicit group, computed with the vectorized whole-array
    reductions (SIMD ``AggKernel.reduce``, ``O(1)`` count, direct scalar
    ``count_distinct``) rather than the grouped scatter. Returns a one-row
    batch of the aggregate columns (named by kernel; callers rename).

    Distinct aggregates get a parallel ctx (``count_distinct`` self-gates on
    size, going radix-partition-parallel at scale); fold reductions stay
    serial — the whole-array SIMD reduce only benefits from threads well above
    these sizes, and that gating belongs in the reduce primitive itself."""
    var par = ExecutionContext.parallel(num_threads)
    var ser = ExecutionContext.serial()
    var out_fields = List[Field]()
    var out_cols = List[AnyArray]()
    for j in range(len(tags)):
        var col: AnyArray
        if agg_is_distinct(tags[j]):
            col = _whole_col(values[j], tags[j], par)
        else:
            col = _whole_col(values[j], tags[j], ser)
        out_fields.append(Field(agg_name_from_tag(tags[j]), col.dtype().copy()))
        out_cols.append(col^)
    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


def _whole_col(
    value: AnyArray, tag: UInt8, ctx: ExecutionContext
) raises -> AnyArray:
    """One whole-table aggregate as a 1-row column — the SIMD/``O(1)`` scalar
    reduction broadcast to length 1 (``AnyScalar.repeat``)."""
    if tag == AGG_COUNT_DISTINCT:
        return count_distinct(value, ctx).to_any().repeat(1)
    elif tag == AGG_APPROX_COUNT_DISTINCT:
        return approx_count_distinct(value, ctx).to_any().repeat(1)
    elif tag == AGG_MIN or tag == AGG_MAX:
        var vdt = value.dtype()
        if vdt.is_string() or vdt.is_large_string():
            # String min/max doesn't broadcast through `AnyScalar.repeat`;
            # compute the single-group result by scanning every row into
            # group 0.
            var gb = Int32Builder(len(value))
            for _ in range(len(value)):
                gb.append(Scalar[int32.native](0))
            return min_max_string_grouped(gb.finish(), value, 1, tag == AGG_MIN)
        elif vdt.is_temporal():
            # Reduce over the integer backing, then relabel to the temporal
            # dtype (the whole-array reduce already handles all-null → null).
            var native_dt = temporal_backing_dtype(vdt)
            var iv = reinterpret_array(value, native_dt)
            var tbox = List[AnyScalar]()

            @parameter
            def trun[K: AggKernel]() raises:
                tbox.append(K.reduce(iv, ctx))

            for_agg_tag[trun](tag)
            return reinterpret_array(tbox[0].repeat(1), vdt)
    var box = List[AnyScalar]()

    @parameter
    def run[K: AggKernel]() raises:
        box.append(K.reduce(value, ctx))

    for_agg_tag[run](tag)
    return box[0].repeat(1)
