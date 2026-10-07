# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""An index: what a source knows about its data before reading it.

Given a filter's predicates and an index over a source, a plan can read less:
fewer row groups, later fewer pages. That is the whole purpose of this module,
and `Value.mask` is the one method that spends it.

A **chunk** is whatever unit the source can skip whole — a Parquet row group, a
page, a partition, a file. Nothing here says which; whoever built the index
knows, and every array below is one entry per chunk.

## Pruning is the same computation over a different domain

One statistic per chunk instead of one value per row, and then the *same*
comparison kernel. `a > 150` prunes by asking `max(a) > 150` over the chunk
maxima, which is `GtKernel` over a three-element array — so there is one
definition of `>` per dtype rather than two that can drift apart. The answer is
an ordinary `BoolArray`, one bit per chunk, and composing two predicates is
`AndKernel`.

**Which dtypes prune is a property of the lane, not of this module.** The
comparison kernels take every `PrimitiveType`, so the *runtime* lane prunes
whatever the index recorded — temporal and decimal included:
`RuntimeValue._statistics` recovers the dtype and dispatches, on the column
side and the literal side alike. The *comptime* lane prunes numerics only, and
that is a decision rather than an oversight. A fused node would need a dtype
*instance* to ask the index with, and a temporal type carries a unit, so
`Stat()` — free where `NumericType` extends `Defaultable` — does not exist. The
alternatives are a required `TemporalValue` member holding one, or a witness
threaded through every call; both are machinery for a single dtype family, and
decimal cannot be expressed in that lane at all for want of a value family. So
`TemporalCompare` inherits `Value.mask`'s default and keeps every chunk, which
is correct, and temporal predicates prune through the runtime lane.

## A chunk is a row group, and then it is a page

`page_selections` is the same decision one level down: a row group that
survives `read_plan` still decodes every row, and the Parquet page index says
which of its pages could match. It reuses everything above — the only thing it
cannot reuse is the *chunking*, because Parquet pages are per column and split
at different rows, so each column gets its own single-column `Index` and the
per-row selections are intersected.

## The three states are carried by the data, not by sentinels

| situation | representation | consequence |
|---|---|---|
| the node cannot prune | mask is **all true** (`keep_every`) | identity for `AND`; composition needs no special case |
| the statistic is absent | the array is **all null** | the comparison yields null, and a null bit means "must read" |
| the source has no index | `Index.columns` is **empty** | every lookup is all-null, so nothing prunes |

**The error is one-sided, and every line here keeps it that way.** A caller may
skip only what it has *proven* cannot match: a wrong "keep" costs time, a wrong
"skip" costs the answer. So the only question asked is *could this be TRUE
here?*, and every unknown resolves to "read it" — by construction, in the
values, rather than in a branch anyone can forget. Kleene semantics are exactly
the right algebra and the boolean kernels already implement them: `false AND
unknown` is `false`, sound because one conjunct proving a chunk empty proves
the conjunction empty.

## Adding an index kind is adding a field

A zone map is one kind — per-chunk `[min, max]` and null counts, so it answers
ordering questions conservatively and equality only as far as interval overlap.
A bloom index would answer exact membership far better and ordering not at all;
a token or bitmap index answers containment. Each is a separate field on
`Index`, carried *beside* the zone maps rather than inside them, because a node
reads the index whose question it can use and produces no answer otherwise.

The alternative — an index *trait*, so a node could take any index — cannot
work here: `DynValue._mask` is a `thin` function pointer, whose signature is
concrete, so a generic index would force parameterising `Filter`,
`ParquetScan` and `DynRelation` on it, or erasing it behind a second open
dispatcher, or one trampoline slot per kind. The recorded prior for an extra
slot on an erased box is +3.2 MB.

## Two members, and they are the module's whole point

`Index.from_parquet(file)` builds one from a footer, and
`Index.from_manifest(files, schema, spec)` one from an Iceberg manifest, a data
file per chunk; `index.read_plan(predicates, bindings)` spends it, answering
which chunks are left to read. A source knows how to describe itself, and
`read_plan` never learns which one described this.

They are members rather than free functions because that is what they are:
a constructor from a source, and the one verb an index has. They are not
inlined into `ParquetScanOperator.drain`, their only caller, because the
decision has to be reachable without executing a query — every case in
`test_scan_pruning.mojo` builds an index and asks for a read plan directly.
"""


from ..errors import InvalidError
from ..arrays import BoolArray, PrimitiveArray
from ..builders import BoolBuilder, PrimitiveBuilder
from ..dtypes import DynType, NullType, PrimitiveType
from ..kernels.boolean import AndKernel
from ..iceberg.bounds import decode_bound
from ..iceberg.manifest import DataFile
from ..iceberg.metadata import PartitionSpec
from ..parquet.reader import (
    LeafSet,
    PageBounds,
    ParquetFile,
    RowSelection,
)
from ..io import ByteSource
from ..scalars import BoolScalar, DynScalar, NullScalar, PrimitiveScalar
from ..schema import Schema
from .logical import Bindings
from .logical import DynValue


# ---------------------------------------------------------------------------
# The index -- what a source knows about its data before reading it
# ---------------------------------------------------------------------------
struct ColumnZones(Copyable, Movable):
    """One column's statistics across every chunk of a source.

    **Stored per column, read per chunk.** A reading node wants one statistic
    for all chunks at once -- that is the array a comparison kernel takes --
    so the layout is columnar even though the source produces it chunk by
    chunk.

    A statistic the source did not record is a **null** `DynScalar` rather than
    an absence: comparing against null yields null, and a null answer already
    means "cannot prove anything, read it". That is the same rule an absent
    column and an unrecognised predicate get, expressed once in the data.
    """

    var name: String
    var mins: List[DynScalar]
    var maxes: List[DynScalar]
    var null_counts: List[Int]
    """Nulls per chunk, or `-1` where the source recorded none. Never silently
    zero -- reading a missing count as zero is a soundness choice this type
    declines to make."""
    var distinct_counts: List[Int]
    """Distinct non-null values per chunk, `-1` where the source recorded
    none. Per chunk: the counts do not combine into the column's, which is
    why `Estimate.from_index` estimates it."""
    var dtype: DynType
    """The type the statistics were recorded in, `null` when none were — read
    once here, from the first recorded bound, rather than on every lookup."""

    def __init__(
        out self,
        var name: String,
        var mins: List[DynScalar],
        var maxes: List[DynScalar],
        var null_counts: List[Int],
        var distinct_counts: List[Int] = [],
    ) raises:
        """The bound and count lists must agree in length -- one entry per
        chunk each. `distinct_counts` may instead be empty, meaning no chunk
        recorded one.

        Checked rather than assumed because nothing else can check it: the
        chunk count lives on `Index`, so a column shorter than the index it
        goes into would be read past its end. A source that cannot record a
        statistic appends a *null* for that chunk, which is what keeps the
        lengths equal without pretending to know anything.
        """
        var distinct = len(distinct_counts)
        var distinct_ok = distinct == 0 or distinct == len(mins)
        if (
            len(mins) != len(maxes)
            or len(mins) != len(null_counts)
            or not distinct_ok
        ):
            raise InvalidError(
                t"ColumnZones '{name}': {len(mins)} mins, {len(maxes)} maxes, "
                t"{len(null_counts)} null counts and {len(distinct_counts)} "
                t"distinct counts -- one of each per chunk (distinct counts "
                t"may also be absent entirely)"
            )
        self.dtype = DynType(NullType())
        for c in range(len(mins)):
            if maxes[c].is_valid():
                self.dtype = maxes[c].type()
                break
            if mins[c].is_valid():
                self.dtype = mins[c].type()
                break
        self.name = name^
        self.mins = mins^
        self.maxes = maxes^
        self.null_counts = null_counts^
        self.distinct_counts = distinct_counts^

    def num_chunks(self) -> Int:
        """How many chunks this column describes."""
        return len(self.mins)

    def statistics[
        T: PrimitiveType
    ](self, dtype: T, upper: Bool) raises -> PrimitiveArray[T]:
        """One statistic for every chunk, in the type the reading node expects.

        **One instantiation per dtype, and no runtime ladder.** The node knows
        `T` at compile time, so the unwrap is a single `as_primitive[T]` arm --
        which is the whole reason this design costs a few kilobytes where
        materialising an erased statistics batch would cost ninety.

        `dtype` is the instance the caller already holds, not a reconstructed
        one: `PrimitiveBuilder` needs it for temporal and decimal, whose types
        carry a unit or a precision and so are not `Defaultable`. Requiring it
        is what lets those dtypes prune at all.

        **`T` is checked, not assumed.** A chunk whose statistic is absent,
        null, or recorded in some *other* type all answer null -- and a null
        answer means "read it". The third case is why the check exists rather
        than a comment: a plan declaring `col("a", int64)` against a file that
        wrote `a` as `int32` would otherwise unwrap an `Int32Scalar` as
        `Int64Scalar`, which is not a raise but a **process abort**, and one
        that `Index.read_plan`'s `except` cannot catch. Nothing verifies a
        supplied scan schema against the file it names, so that mismatch is a
        user error away.
        """
        var out = PrimitiveBuilder[T](dtype, capacity=self.num_chunks())
        var want = DynType(dtype)
        ref stats = self.maxes if upper else self.mins
        for c in range(self.num_chunks()):
            if stats[c].is_valid() and stats[c].type() == want:
                out.append(stats[c].as_primitive[T]().value())
            else:
                out.append_null()
        return out.finish()


def _or_null(bound: Optional[DynScalar]) -> DynScalar:
    """A recorded bound, or a null scalar when the source wrote none.

    `ColumnZones` states the rule -- an absent statistic is a *null*, so it
    propagates through the comparison and reads as "read it" -- and this is the
    one place that applies it. Both builders below take their bounds as
    `Optional[DynScalar]` (`ColumnStatistics` per row group, `PageBounds` per
    page), so without this the rule is written out four times.
    """
    if bound:
        return bound.value().copy()
    return NullScalar().to_dyn()


def _decoded(
    dtype: DynType, bounds: Dict[Int, List[UInt8]], id: Int
) -> DynScalar:
    """Column `id`'s Iceberg bound as `dtype`, or a null scalar when the
    manifest wrote none or it does not decode -- either way it proves
    nothing, the rule `_or_null` applies to Parquet statistics."""
    try:
        return _decode_bound(dtype, bounds, id)
    except:
        return NullScalar().to_dyn()


def _decode_bound(
    dtype: DynType, bounds: Dict[Int, List[UInt8]], id: Int
) raises -> DynScalar:
    if id not in bounds:
        return NullScalar().to_dyn()
    return decode_bound(dtype, Span(bounds[id]))


struct Index(Copyable, Movable):
    """Everything a source knows about its own data without reading it: rows
    per chunk, and per column its zone maps — per-chunk `[min, max]`, null and
    distinct counts.

    **One field per index kind, and adding a kind is adding a field.** A zone
    map answers ordering questions conservatively and equality only as far as
    interval overlap; a bloom index would answer membership far better and
    ordering not at all, and would be a field beside `columns`. The
    alternative -- a trait, so a node could take any index -- cannot work here:
    the erased predicate box holds a `thin` function pointer, whose signature
    is concrete, so a generic index would force parameterising `Filter`,
    `ParquetScan` and `DynRelation` on it, or erasing it behind a second open
    dispatcher, or one trampoline slot per kind. The recorded prior for one
    extra slot on an erased box is +3.2 MB.

    A node reads the fields it can use and produces no answer otherwise, so an
    index that cannot answer a question prunes nothing rather than erroring.
    Composing two kinds is an `AND` of their masks. A column with no zone maps
    answers all-null, so nothing prunes on it.

    A **chunk** is whatever unit the source can skip whole -- a row group, a
    page, a file for a partitioned source. This type never says which; whoever
    built it knows. **`rows` is what says how many there are**, one entry per
    chunk, so every column is checked against it once, here.
    """

    var rows: List[Int]
    """Rows per chunk, `-1` where the source did not say. Read to decide
    whether a column is *provably* all null, which needs a row count to compare
    its null count against, and to turn surviving pages back into rows."""
    var columns: List[ColumnZones]

    def __init__(
        out self,
        var rows: List[Int] = [],
        var columns: List[ColumnZones] = [],
    ) raises:
        for ref c in columns:
            if c.num_chunks() != len(rows):
                raise InvalidError(
                    t"Index: column '{c.name}' describes {c.num_chunks()}"
                    t" chunks of {len(rows)}"
                )
        self.rows = rows^
        self.columns = columns^

    def chunks(self) -> Int:
        """How many chunks the source has."""
        return len(self.rows)

    def find(self, name: String) -> Int:
        """`name`'s position in `columns`, `-1` when it has no zone maps."""
        for i in range(len(self.columns)):
            if self.columns[i].name == name:
                return i
        return -1

    def statistics[
        T: PrimitiveType
    ](self, name: String, dtype: T, upper: Bool) raises -> PrimitiveArray[T]:
        """Each chunk's largest non-null value of `name` when `upper`, its
        smallest otherwise, as `T` — null for every chunk when `name` has no
        zone maps."""
        var i = self.find(name)
        if i < 0:
            var out = PrimitiveBuilder[T](dtype, capacity=self.chunks())
            out.append_nulls(self.chunks())
            return out.finish()
        return self.columns[i].statistics[T](dtype, upper)

    def dtype_of(self, name: String) -> DynType:
        """The type `name`'s statistics were recorded in, `null` when there
        are none — what a caller that must *recover* the type reads.

        The interpreted lane needs it; a comptime node passes its own `T` to
        `statistics` and never asks.
        """
        var i = self.find(name)
        if i < 0:
            return DynType(NullType())
        return self.columns[i].dtype.copy()

    def defined(self, name: String) raises -> BoolArray:
        """Where `name` has at least one non-null value, as far as can be shown.

        **The one exactly-provable skip.** Every comparison with NULL is NULL
        and a filter keeps only rows whose mask bit is valid and true, so a
        chunk whose column is entirely null cannot produce a surviving row —
        whatever its bounds say, and even when it has none.

        Needs both a stored null count and a known row count; either missing
        makes it unprovable, and unprovable answers `True`.
        """
        var out = BoolBuilder(capacity=self.chunks())
        var i = self.find(name)
        for c in range(self.chunks()):
            var known = (
                i >= 0
                and self.rows[c] >= 0
                and self.columns[i].null_counts[c] >= 0
            )
            out.append(
                not (known and self.columns[i].null_counts[c] == self.rows[c])
            )
        return out.finish()

    @staticmethod
    def from_pages(
        var name: String, pages: List[PageBounds], rows: Int
    ) raises -> Index:
        """One column's data pages of one row group, as chunks.

        The finer granularity this type's docstring promises, and a **single
        column**: Parquet pages are per column and split at different rows, so
        pages are the one chunking that is not shared. `rows` is the group's
        row count, and a page index that does not account for all of it is one
        this cannot trust -- that and an absent page index both answer an index
        of **no chunks**, which the caller reads as "this column says nothing".

        `Index.rows` carries the per-page row counts, which is what turns the
        surviving chunks back into a `RowSelection`.
        """
        var mins = List[DynScalar](capacity=len(pages))
        var maxes = List[DynScalar](capacity=len(pages))
        var nulls = List[Int](capacity=len(pages))
        var page_rows = List[Int](capacity=len(pages))
        var covered = 0
        for ref page in pages:
            mins.append(_or_null(page.min))
            maxes.append(_or_null(page.max))
            # A page index records which pages are entirely null, which is
            # their whole null count; of any other it records none.
            nulls.append(page.num_rows if page.all_null else -1)
            page_rows.append(page.num_rows)
            covered += page.num_rows
        if len(pages) == 0 or covered != rows:
            return Index()

        # A page index records no distinct count, so the list is left absent
        # rather than filled with sentinels.
        var columns = List[ColumnZones](capacity=1)
        columns.append(ColumnZones(name^, mins^, maxes^, nulls^))
        return Index(page_rows^, columns^)

    @staticmethod
    def from_parquet[
        S: ByteSource, leaves: LeafSet
    ](file: ParquetFile[S, leaves]) raises -> Index:
        """A row group is a chunk: one min, max, null count and distinct count
        per column per row group, keyed by the file's top-level column names.

        The distinct count is almost always `-1` — pyarrow and parquet-cpp
        omit `Statistics.distinct_count` outright, and marrow's own writer
        emits it only for a `RLE_DICTIONARY` chunk. An absent one is unknown
        and stays unknown; see `ColumnZones.distinct_counts`.

        **The leaf-alignment guard is the load-bearing line.**
        `ParquetFile.statistics()` is indexed by *leaf* position, and an expression
        names a *top-level column*. Those two agree only when every top-level
        field contributes exactly one leaf — and since every field contributes
        at least one and leaves are emitted in field order, "as many leaves as
        fields" proves that one-to-one correspondence outright. Anything else
        (a struct with two members, a list) yields more leaves than fields, and
        this returns an index with no zone maps rather than handing field `i`
        the bounds of somebody else's leaf. That misattribution is a live
        defect class here: D14 in `2026-08-25-pruning-indexing-findings.md` was
        `struct<x>`'s bounds being handed to a top-level `x`.

        Two further protections are already in the reader and are relied on
        rather than repeated: `_trusted_leaf` returns no statistics for a leaf
        that is not a whole top-level column or whose ColumnOrder the footer
        never declared, and `ColumnStatistics.from_metadata` keeps min/max only
        when *both* decode.

        **This reads every leaf's statistics, not just the predicate's.**
        `ParquetFile` exposes only whole-file `statistics()`; the per-`(row group,
        leaf)` narrowing is private, so a three-column predicate on a
        105-column file decodes 210 chunk statistics to use 6. That is footer
        work, not column data, and it is bounded by the footer's size — but it
        is the first thing to fix if pruning ever shows up in a profile.
        """
        var arrow = file.schema()
        ref meta = file.metadata()
        var stats = file.statistics()
        var chunks = len(stats)

        var rows = List[Int](capacity=chunks)
        for rg in range(chunks):
            rows.append(meta.row_groups[rg].num_rows)

        var aligned = chunks > 0 and len(arrow.fields) == len(stats[0])
        if not aligned:
            # No usable correspondence between leaves and fields, so the file
            # says nothing this can trust. No zone maps prune nothing.
            return Index(rows^)

        var columns = List[ColumnZones](capacity=len(arrow.fields))
        for i in range(len(arrow.fields)):
            var mins = List[DynScalar](capacity=chunks)
            var maxes = List[DynScalar](capacity=chunks)
            var nulls = List[Int](capacity=chunks)
            var distinct = List[Int](capacity=chunks)
            for rg in range(chunks):
                mins.append(_or_null(stats[rg][i].min))
                maxes.append(_or_null(stats[rg][i].max))
                nulls.append(stats[rg][i].null_count)
                distinct.append(stats[rg][i].distinct_count)
            columns.append(
                ColumnZones(
                    arrow.fields[i].name.copy(),
                    mins^,
                    maxes^,
                    nulls^,
                    distinct^,
                )
            )
        return Index(rows^, columns^)

    @staticmethod
    def from_manifest(
        files: List[DataFile], schema: Schema, spec: PartitionSpec
    ) raises -> Index:
        """A data file is a chunk: one min, max and null count per top-level
        column per file of an Iceberg manifest, keyed by `schema`'s column
        names and looked up by each column's field id. `spec` is the
        partition spec every file in `files` was written with.

        Only flat columns with a field id are described — primitive, bool,
        string, binary, fixed-size binary; a nested column's id names no
        leaf statistic. A missing or undecodable bound is a null scalar and
        a missing null count is `-1`: absence proves nothing.

        An identity partition fills in what the bounds leave out: every row
        of the file holds the partition value, so it is both min and max and
        the column has no nulls — or, for a null value, every row is null.
        Other transforms prune nothing here.
        """
        var chunks = len(files)
        var rows = List[Int](capacity=chunks)
        var identities = List[Dict[Int, DynScalar]](capacity=chunks)
        for ref f in files:
            rows.append(f.record_count)
            identities.append(f.identity_values(spec))

        var columns = List[ColumnZones](capacity=len(schema.fields))
        for ref field in schema.fields:
            ref dtype = field.dtype
            var flat = (
                dtype.is_primitive()
                or dtype.is_bool()
                or dtype.is_binary_like()
                or dtype.is_fixed_size_binary()
            )
            var id = field.field_id()
            if not flat or not id:
                continue
            var fid = id.value()

            var mins = List[DynScalar](capacity=chunks)
            var maxes = List[DynScalar](capacity=chunks)
            var nulls = List[Int](capacity=chunks)
            for k in range(chunks):
                ref f = files[k]
                var lo = _decoded(dtype, f.lower_bounds, fid)
                var hi = _decoded(dtype, f.upper_bounds, fid)
                var null_count = f.null_value_counts.get(fid).or_else(-1)
                if fid in identities[k]:
                    ref value = identities[k][fid]
                    if value.is_null():
                        null_count = f.record_count
                    elif value.type() == dtype:
                        if lo.is_null():
                            lo = value.copy()
                        if hi.is_null():
                            hi = value.copy()
                        if null_count < 0:
                            null_count = 0
                mins.append(lo^)
                maxes.append(hi^)
                nulls.append(null_count)
            columns.append(
                ColumnZones(field.name.copy(), mins^, maxes^, nulls^)
            )
        return Index(rows^, columns^)

    def read_plan(
        self,
        predicates: List[DynValue],
        bindings: Bindings = Bindings(),
    ) -> List[Int]:
        """Which chunks a source still has to read, in order.

        **The conjunction is a kernel, not a loop.** The predicates came from
        nested `Filter`s, so a chunk survives only if every one keeps it —
        which is `AND`, evaluated once over the whole mask rather than per
        chunk. Kleene semantics are exactly right here and the boolean kernel
        already implements them: `false AND unknown` is `false`, sound because
        one conjunct proving a chunk empty proves the conjunction empty.

        A **null** bit means nothing could be proven, so it keeps the chunk.
        That is read straight off the mask rather than through `fill_null`,
        which would link a conditional kernel into every binary that filters to
        answer a question one bit test already answers.

        A method on the index rather than a free function taking one, and the
        index's only verb: everything above builds the description, this is
        what spends it. It reads no file, so the decision is testable with
        hand-written statistics and no Parquet at all. `bindings` are this
        execution's parameter values, which a predicate naming a `param` needs
        before it can say anything.
        """
        var keep = List[Int](capacity=self.chunks())
        var live = self.surviving(predicates, bindings)
        for i in range(self.chunks()):
            if live[i]:
                keep.append(i)
        return keep^

    def surviving(
        self,
        predicates: List[DynValue],
        bindings: Bindings = Bindings(),
    ) -> List[Bool]:
        """One flag per chunk: could any of its rows satisfy every predicate.

        What `read_plan` answers, before it is turned into a list of indices.
        Separate because the two consumers want different shapes -- a row group
        is skipped by *not being read*, where a page is skipped by a
        `RowSelection` that needs a flag for every page, kept or not.
        """
        var keep = List[Bool](capacity=self.chunks())
        try:
            var live = keep_every(self.chunks())
            for ref predicate in predicates:
                live = AndKernel.apply(live, predicate.mask(self, bindings))
            for i in range(self.chunks()):
                keep.append(live.is_null(i) or live[i].value())
        except:
            # A predicate that fails proves nothing; read everything rather
            # than turning one unreadable footer into a failed query.
            keep.clear()
            for _ in range(self.chunks()):
                keep.append(True)
        return keep^


def page_selections[
    S: ByteSource, leaves: LeafSet
](
    file: ParquetFile[S, leaves],
    index: Index,
    row_groups: List[Int],
    predicates: List[DynValue],
    bindings: Bindings = Bindings(),
) raises -> List[RowSelection]:
    """A row selection per row group, from the Parquet **page** index.

    The finer half of the same machinery: a row group that survives
    `read_plan` still decodes every one of its rows, and a page index says
    which of its pages could hold a match. One `RowSelection` per entry of
    `row_groups`, in that order, which is the shape `ParquetFile.read` takes —
    or an **empty list**, meaning this file offers no page pruning at all and
    every surviving group should be read whole.

    `index` is the one the caller already built to choose `row_groups`; it is
    read for its per-group row counts.

    **A page is not a chunk shared by every column, and that is the one place
    this differs from `Index.from_parquet`.** Parquet pages are per column, so
    column `a` and column `b` of the same row group split at different rows.
    Each column therefore gets its own single-column `Index` whose chunks are
    *its* pages, and the per-row selections are intersected. That is sound for
    a disjunction as well as a conjunction: a predicate naming a column this
    index does not describe reads null, so `a > 5 OR b < 3` evaluated against
    `a` alone keeps every row rather than pretending `b` said something.

    **Only the columns a predicate names are read.** Any other column answers
    all-null, so its selection is all-true and intersecting it is the identity
    — a page-index decode and a merge per column that cannot change the
    answer. Cheaper than it was, since a selection is runs rather than a row
    per byte, but still bought with nothing.

    **Nested columns are refused outright, and the test is positive.**
    `ColumnReader.decode` *ignores* a selection for a leveled leaf, so a
    repeated column would return misaligned columns rather than fewer rows: a
    wrong answer, not a slow one. Every field must therefore be flat —
    primitive, string or binary — which is also what makes the one-leaf-per-
    field rule `page_bounds` needs hold, since nesting is exactly what
    multiplies leaves. Checking the dtypes rather than counting leaves is the
    difference between a rule that admits what it has been shown and one that
    admits what nobody has thought about yet.

    A column with no page index, and one whose pages do not account for every
    row of the group, are skipped the same way — see `Index.from_pages`.
    """
    var out = List[RowSelection]()
    if len(row_groups) == 0:
        return out^

    var arrow = file.schema()
    for ref f in arrow.fields:
        # Byte strings in either layout: a view column is as flat as an
        # offsets one, and its pages are the same pages.
        if not (
            f.dtype.is_primitive()
            or f.dtype.is_string_like()
            or f.dtype.is_binary_like()
            or f.dtype.is_string_view()
            or f.dtype.is_binary_view()
        ):
            return out^

    var wanted = List[Int]()
    for i in range(len(arrow.fields)):
        for ref predicate in predicates:
            if arrow.fields[i].name in predicate.columns():
                wanted.append(i)
                break
    if len(wanted) == 0:
        return out^

    # Decoded once for the whole file, not once per group: `page_bounds` has no
    # per-(group, leaf) entry point -- the same narrowing `Index.from_parquet`
    # notes and does not have either.
    var pages = file.page_bounds()

    var read_any = False
    for rg in row_groups:
        var rows = index.rows[rg]
        if rows < 0:
            return List[RowSelection]()
        var sel = RowSelection.all(rows)
        for i in wanted:
            var column = Index.from_pages(
                arrow.fields[i].name.copy(), pages[rg][i], rows
            )
            if column.chunks() == 0:
                continue
            read_any = True
            sel = sel.intersect(
                RowSelection.from_pages(
                    column.surviving(predicates, bindings), column.rows
                )
            )
        out.append(sel^)

    if not read_any:
        # Not one column had a usable page index, so every selection built
        # above is `RowSelection.all`, which says exactly what no selection
        # says -- and an empty list additionally tells `read` not to decode an
        # `OffsetIndex` per (group, leaf). Answer the empty list instead,
        # which is the same "read them whole" the other early returns give.
        return List[RowSelection]()
    return out^


def keep_every(chunks: Int) raises -> BoolArray:
    """A mask that keeps every chunk — what a node unable to prune answers.

    All-true rather than an absent answer, so composition needs no special
    case: it is the identity for `AND`, and for `OR` it correctly swallows
    whatever the other side proved. Named for that argument; the array itself
    is one broadcast scalar.
    """
    return BoolScalar(True).repeat(chunks)
