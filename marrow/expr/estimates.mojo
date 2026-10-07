# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a plan node expects to produce, and what producing it should cost.

`index.mojo` describes what a source knows before it is read; this describes
what each node's output is expected to be, propagated up the plan, so that a
plan can be scored before it runs. The optimizer's join ordering spends the
score.

Counts are `Approx`: exact, estimated or unknown, with unknown absorbing, so
no figure is invented where nothing is known. Column bounds are `DynScalar`s
with null for unrecorded; a bound is only ever claimed to contain the data, so
it is sound wherever present, which lets a filter prove itself empty from an
estimated input. A filter's `Selectivity` comes from `Value.mask`, the
comparisons row-group pruning uses: over a one-chunk index built from its
input's estimate for what it proves, and over an equal-width histogram of the
column it reads for what it keeps. A `Cost` weighs rows moved, comparisons
made and bytes held over each input's `Size`.
"""

from std.bit import bit_width
from std.sys import bit_width_of

from ..arrays import PrimitiveArray
from ..dtypes import DynType, PrimitiveType
from ..kernels.join import (
    BUILD_LEFT,
    BUILD_RIGHT,
    JOIN_ALL,
    JOIN_INNER,
    JoinBuildSide,
    JoinKind,
)
from ..scalars import DynScalar, NullScalar, PrimitiveScalar
from ..schema import Schema
from ..parquet.statistics import valid_min_max
from .index import ColumnZones, Index


comptime DEFAULT_SELECTIVITY = 20
"""Percent of its input a filter is assumed to keep when nothing can be proven.

The figure is DataFusion's `default_filter_selectivity`, taken rather than
invented because a made-up constant with no provenance is exactly what this
module is supposed to make visible. It applies **only** when the predicate
could not be decided against the child's bounds; a decided predicate answers
exactly zero and never comes here.
"""

comptime DEFAULT_STRING_WIDTH = 20
"""Bytes per value assumed for a string column nobody measured. An estimate,
so a plan over an unanalysed table can still be priced, and `analyze()`
replaces it with the data's own average."""

comptime DEFAULT_BINARY_WIDTH = 100
"""Bytes per value assumed for a binary column nobody measured."""

comptime HISTOGRAM_BUCKETS = 64
"""The most buckets a filter's selectivity is read over — see
`Filter.verdict_of`. Enough that an equality on a key of up to this many
values keeps its one value's share."""


struct Selectivity(Copyable, ImplicitlyCopyable, Movable):
    """What a filter keeps of its input: proven nothing, proven everything,
    or an estimated share.

    Filters combine by product, which is independent of the order they run
    in, and only a proof keeps a count exact.
    """

    comptime WHOLE = 1_000_000
    """A share is in parts per million."""

    var empty: Bool
    var whole: Bool
    var kept: Int

    def __init__(out self, empty: Bool, whole: Bool, kept: Int):
        self.empty = empty
        self.whole = whole
        self.kept = kept

    @staticmethod
    def everything() -> Self:
        """Provably every row: the identity of the product."""
        return Self(False, True, Self.WHOLE)

    @staticmethod
    def nothing() -> Self:
        """Provably no row."""
        return Self(True, False, 0)

    @staticmethod
    def share(parts: Int, of: Int) -> Self:
        """About `parts` in every `of` rows — never provably all or none."""
        var kept = parts * Self.WHOLE // of if of > 0 else 0
        return Self(False, False, kept)

    @staticmethod
    def default() -> Self:
        """What a filter nothing can be read from keeps."""
        return Self.share(DEFAULT_SELECTIVITY, 100)

    def __mul__(self, other: Self) -> Self:
        if self.empty or other.empty:
            return Self.nothing()
        if self.whole:
            return other
        if other.whole:
            return self
        return Self(False, False, self.kept * other.kept // Self.WHOLE)

    def apply(self, rows: Approx) -> Approx:
        """`rows` once filtered: exactly none, the same count, or the share
        as an estimate — never zero, which only a proof answers."""
        if self.empty:
            return Approx.exact(0)
        if self.whole:
            return rows
        return rows.scaled(self.kept, Self.WHOLE)


comptime ROW_WEIGHT = 1
"""What moving one row through one operator costs, in the arbitrary unit
`Cost.total` sums. The unit is arbitrary; the *ratios* below are the model."""

comptime COMPARE_WEIGHT = 2
"""A key comparison, hash or probe against a row's mere movement. Twice,
because it reads a value and branches where movement copies."""

comptime BYTE_WEIGHT = 1
"""A byte materialised into a buffer a pipeline breaker has to hold.

Per byte, against `ROW_WEIGHT` per row — so an eight-byte column makes
buffering a row eight times the cost of passing it along, which is the ratio
that should make a plan prefer to filter before it sorts.
"""


# ---------------------------------------------------------------------------
# Approx -- a count that may be exact, estimated, or unknown
# ---------------------------------------------------------------------------
struct Approx(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """A count and how much it is worth believing: exact, estimated or unknown.

    Unknown absorbs every operator and is never read as zero, so a count nobody
    knows reaches the root as unknown. Two exact counts combine to an exact one;
    anything involving an estimate is an estimate, and nothing promotes it back.
    """

    comptime _UNKNOWN = UInt8(0)
    comptime _ESTIMATED = UInt8(1)
    comptime _EXACT = UInt8(2)

    var _value: Int
    var _state: UInt8

    def __init__(out self):
        """Unknown — the default, so a field nobody set claims nothing."""
        self._value = 0
        self._state = Self._UNKNOWN

    def __init__(out self, value: Int, state: UInt8):
        self._value = value if state != Self._UNKNOWN else 0
        self._state = state

    @staticmethod
    def unknown() -> Self:
        """Nothing is known."""
        return Self()

    @staticmethod
    def exact(value: Int) -> Self:
        """`value`, and it is the number. Negative counts clamp to zero: every
        count here is a cardinality, and a caller that computed a negative one
        subtracted too much rather than discovered a negative population."""
        return Self(value if value > 0 else 0, Self._EXACT)

    @staticmethod
    def estimated(value: Int) -> Self:
        """About `value`."""
        return Self(value if value > 0 else 0, Self._ESTIMATED)

    def known(self) -> Optional[Int]:
        """The number, or `None` when nothing is known.

        An `Optional` rather than a sentinel because every caller has to make
        the "or else" decision explicitly, which is the whole discipline this
        type exists to enforce."""
        if self._state == Self._UNKNOWN:
            return None
        return self._value

    def is_known(self) -> Bool:
        return self._state != Self._UNKNOWN

    def is_exact(self) -> Bool:
        return self._state == Self._EXACT

    def to_estimate(self) -> Self:
        """This count, believed less. Unknown stays unknown — it is a different
        state, not a less precise one."""
        if self._state == Self._EXACT:
            return Self.estimated(self._value)
        return self.copy()

    def _combine(self, other: Self, value: Int) -> Self:
        """`value` at the weaker of the two states — the one rule every binary
        operator below shares."""
        if not (self.is_known() and other.is_known()):
            return Self.unknown()
        if self.is_exact() and other.is_exact():
            return Self.exact(value)
        return Self.estimated(value)

    @staticmethod
    def _sum(a: Int, b: Int) -> Int:
        """`a + b` for two counts, saturating at `Int.MAX` rather than wrapping.

        A count here is never negative, so overflow can only go up, and a
        wrapped sum would come back negative and clamp to zero — pricing the
        largest intermediate a search can build as the cheapest.
        """
        if a > Int.MAX - b:
            return Int.MAX
        return a + b

    @staticmethod
    def _product(a: Int, b: Int) -> Int:
        """`a * b` for two counts, saturating at `Int.MAX` — see `_sum`."""
        if a == 0 or b == 0:
            return 0
        if a > Int.MAX // b:
            return Int.MAX
        return a * b

    def __add__(self, other: Self) -> Self:
        return self._combine(other, Self._sum(self._value, other._value))

    def __sub__(self, other: Self) -> Self:
        """Saturating at zero — see `exact`."""
        return self._combine(other, self._value - other._value)

    def __mul__(self, other: Self) -> Self:
        return self._combine(other, Self._product(self._value, other._value))

    def __floordiv__(self, other: Self) -> Self:
        """Integer division; unknown unless the divisor is a known positive."""
        if other.is_known() and other._value > 0:
            return self._combine(other, self._value // other._value)
        else:
            return Self.unknown()

    def at_most(self, other: Self) -> Self:
        """This count capped by `other`; unknown if either is.

        Deliberately **not** "the known one when the other is unknown". A cap
        by an unknown bound is not a cap, and answering the uncapped value
        would silently assert the cap did not bite.
        """
        var value = self._value if self._value < other._value else other._value
        return self._combine(other, value)

    def at_least(self, other: Self) -> Self:
        """This count floored by `other`; unknown if either is."""
        var value = self._value if self._value > other._value else other._value
        return self._combine(other, value)

    def times(self, factor: Int) -> Self:
        """This count multiplied by a known integer, keeping its state and
        saturating at `Int.MAX`."""
        return self._combine(self, Self._product(self._value, factor))

    def scaled(self, parts: Int, of: Int = 100) -> Self:
        """`parts` in every `of` of this count — a percentage by default —
        always an *estimate*.

        Rounded to nearest, then floored at one whenever the input held a row
        and the share is not zero: **zero is reserved for provably empty**. A
        filter that merely might not match must not report the same
        cardinality as one proven to match nothing, because the two are read
        identically downstream and only one of them is a proof.
        """
        if not self.is_known():
            return Self.unknown()
        # Whole multiples of `of` first, so a count near saturation still
        # scales below a smaller one rather than collapsing.
        var scaled = Self._sum(
            Self._product(self._value // of, parts),
            (self._value % of * parts + of // 2) // of,
        )
        if scaled == 0 and self._value > 0 and parts > 0:
            scaled = 1
        return Self.estimated(scaled)

    def __eq__(self, other: Self) -> Bool:
        return self._state == other._state and self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        if self._state == Self._UNKNOWN:
            writer.write("?")
        elif self._state == Self._ESTIMATED:
            writer.write("~", self._value)
        else:
            writer.write(self._value)


# ---------------------------------------------------------------------------
# ColumnEstimate -- one column's summary
# ---------------------------------------------------------------------------
struct ColumnEstimate(Copyable, Movable, Writable):
    """What is known about one output column.

    `min`/`max` are `DynScalar`s with null for unrecorded, as in `ColumnZones`,
    which makes `Estimate.to_index` a direct copy. The counts are `Approx`
    because each can be unknown: most writers record no distinct count, and a
    computed column records nothing.
    """

    var name: String
    var min: DynScalar
    """The smallest value this column takes, null when unrecorded. Always a
    sound lower *bound*: narrowed only by a proof, never by a heuristic."""
    var max: DynScalar
    """The largest value, on the same terms as `min`."""
    var nulls: Approx
    var ndv: Approx
    """Distinct values, never more than estimated. Most writers record none,
    so `max_distinct` supplies the fallback where one is needed."""
    var width: Approx
    """Average bytes per value.

    Exact for a fixed-width dtype, where it is a property of the type rather
    than of the data. For a variable-width one an estimate: the data's own
    average where something measured it, a default otherwise (`width_of`).
    """

    def __init__(
        out self,
        var name: String,
        var min: DynScalar = NullScalar().to_dyn(),
        var max: DynScalar = NullScalar().to_dyn(),
        nulls: Approx = Approx(),
        ndv: Approx = Approx(),
        width: Approx = Approx(),
    ):
        self.name = name^
        self.min = min^
        self.max = max^
        self.nulls = nulls
        self.ndv = ndv
        self.width = width

    @staticmethod
    def unknown(var name: String, dtype: DynType) -> Self:
        """A column nothing is known about, except how wide its values are.

        The width survives because it comes from the *type*, and a projection
        that computes a column still knows the type it computes.
        """
        return Self(name^, width=Self.width_of(dtype))

    @staticmethod
    def width_of(dtype: DynType) -> Approx:
        """Bytes per value: exact for a dtype whose values all have one size,
        a default otherwise — `DEFAULT_STRING_WIDTH` for a string,
        `DEFAULT_BINARY_WIDTH` for a binary, one element for a list or a map
        — as an estimate, so an unmeasured column prices rather than blinding
        every plan that reads it.

        `DynType.byte_width` answers 0 for anything but a primitive, which is
        not a width. A bool is charged a whole byte rather than a bit, erring
        toward the larger cost. The other fixed shapes are sized from the type
        alone: a fixed-size binary by its size, a dictionary by its index —
        its values are shared, not held per row — a null column by nothing,
        and a struct or fixed-size list by what it holds. Defined here rather
        than on `DynType`, which must not depend on the estimator.
        """
        var width = dtype.byte_width()
        if width > 0:
            return Approx.exact(width)
        if dtype.is_bool():
            return Approx.exact(1)
        if dtype.is_null():
            return Approx.exact(0)
        if dtype.is_fixed_size_binary():
            return Approx.exact(dtype.as_fixed_size_binary().byte_width)
        if dtype.is_dictionary():
            return Self.width_of(dtype.as_dictionary().index_type())
        if dtype.is_fixed_size_list():
            ref fixed = dtype.as_fixed_size_list()
            return Self.width_of(fixed.item[].dtype).times(fixed.size)
        if dtype.is_struct():
            var total = Approx.exact(0)
            for ref f in dtype.as_struct().fields:
                total = total + Self.width_of(f.dtype)
            return total
        if dtype.is_string_like() or dtype.is_string_view():
            return Approx.estimated(DEFAULT_STRING_WIDTH)
        if dtype.is_binary_like() or dtype.is_binary_view():
            return Approx.estimated(DEFAULT_BINARY_WIDTH)
        if dtype.is_list():
            return Self.width_of(dtype.as_list().value_type()).to_estimate()
        if dtype.is_large_list():
            return Self.width_of(
                dtype.as_large_list().value_type()
            ).to_estimate()
        if dtype.is_map():
            return Self.width_of(dtype.as_map().item_type()).to_estimate()
        return Approx.unknown()

    @staticmethod
    def bounds_of[
        T: PrimitiveType, //, skip_nan: Bool = False
    ](
        lows: PrimitiveArray[T], highs: PrimitiveArray[T], witness: T
    ) raises -> Tuple[DynScalar, DynScalar]:
        """The smallest valid value of `lows` and the largest of `highs`, null
        scalars where there is none; `skip_nan` passes over NaN."""
        var lo = valid_min_max[skip_nan=skip_nan](lows)
        var hi = valid_min_max[skip_nan=skip_nan](highs)
        var low = Optional[Scalar[T.native]](lo[0]) if lo[2] else None
        var high = Optional[Scalar[T.native]](hi[1]) if hi[2] else None
        return (
            PrimitiveScalar[T](low, witness).to_dyn(),
            PrimitiveScalar[T](high, witness).to_dyn(),
        )

    def span(self) -> Approx:
        """How many values fit between the bounds, `max - min + 1`.

        A column stored as integers cannot hold more distinct values than its
        range does, so this bounds `ndv` from above wherever a writer recorded
        bounds and no distinct count. It is what makes a surrogate foreign key
        estimate as the domain it is rather than as the row count.

        Unknown unless both bounds are recorded, `min <= max`, and the dtype
        stores integers of at most 64 bits: the integers, dates, times,
        timestamps, durations and the two narrow decimals, whose unscaled
        values are integers. A float has no such count, and a wider integer is
        not worth the arithmetic. Computed in 64-bit unsigned arithmetic, where
        the difference of two values of one type cannot wrap, and saturated
        before the `+ 1`, which could.

        Counts no NULL, like `ndv`: right for a join key, which matches no
        NULL, and corrected for in `Estimate.grouped`, where NULL is a group.
        """
        if not (self.min.is_valid() and self.max.is_valid()):
            return Approx.unknown()
        var dtype = self.min.type()
        if not dtype.is_primitive():
            return Approx.unknown()

        def arm[T: PrimitiveType](witness: T) raises {imm} -> Approx:
            comptime if T.native.is_integral() and bit_width_of[
                T.native
            ]() <= 64:
                if not self.max.isa[PrimitiveScalar[T]]():
                    return Approx.unknown()
                var lo = self.min.as_primitive[T]().value()
                var hi = self.max.as_primitive[T]().value()
                if lo > hi:
                    return Approx.unknown()
                var width = (
                    hi.cast[DType.int64]().cast[DType.uint64]()
                    - lo.cast[DType.int64]().cast[DType.uint64]()
                )
                if width >= UInt64(Int.MAX):
                    return Approx.estimated(Int.MAX)
                return Approx.estimated(Int(width) + 1)
            else:
                return Approx.unknown()

        try:
            return dtype.dispatch_primitive(arm)
        except:
            return Approx.unknown()

    def max_distinct(self, rows: Approx) -> Approx:
        """How many distinct values this column may hold, at most.

        The recorded `ndv` when there is one, otherwise the row count — a
        column cannot hold more distinct values than it holds values — and
        either one capped by `span` when the bounds give one. The fallback is
        an **upper bound presented as an estimate**, which is the honest
        reading: it is sound as a bound and usually far too large as a
        prediction, and the two callers that use it (`Aggregate`'s output
        cardinality and `Join`'s containment denominator) both want the bound.

        The span caps only when it is known: `at_most` with an unknown answers
        unknown, and a cap nobody can state must not blind every float and
        string key.
        """
        var cap = rows.to_estimate()
        var span = self.span()
        if span.is_known():
            cap = cap.at_most(span)
        if self.ndv.is_known():
            return self.ndv.at_most(cap)
        return cap

    def reduced(self, rows: Approx) -> Self:
        """This column after an operation that keeps some of its values and
        invents none — a filter, a join — leaving `rows` rows.

        The bounds and width survive. The counts survive as estimates capped
        by `rows`: what is left holds no more NULLs or distinct values than it
        did, nor more than it has rows — and a column proven free of NULLs
        stays so.
        """
        var out = self.copy()
        if self.nulls != Approx.exact(0):
            out.nulls = self.nulls.at_most(rows).to_estimate()
        out.ndv = self.ndv.at_most(rows).to_estimate()
        return out^

    def histogram(self, buckets: Int, rows: Approx) raises -> Index:
        """This column's range cut into at most `buckets` equal-width chunks,
        as a one-column index — the domain a filter's share is read over by
        its `mask`, values assumed spread evenly. No more buckets than the
        column has distinct values, so an equality keeps one value's share.
        No chunk unless the column has a `span`."""
        var span = self.span().known()
        if not span or span.value() == Int.MAX:
            return Index()
        var values = span.value()
        var k = buckets
        var distinct = self.max_distinct(rows).known()
        if distinct and distinct.value() < k:
            k = distinct.value() if distinct.value() > 0 else 1
        var n = k if k < values else values

        def arm[T: PrimitiveType](witness: T) raises {imm} -> Index:
            comptime if T.native.is_integral() and bit_width_of[
                T.native
            ]() <= 64:
                # Offsets from `min` in 64-bit unsigned arithmetic, where
                # the range of two values of one type cannot wrap.
                var lo = self.min.as_primitive[T]().value()
                var base = lo.cast[DType.int64]().cast[DType.uint64]()
                var mins = List[DynScalar](capacity=n)
                var maxes = List[DynScalar](capacity=n)
                for i in range(n):
                    var first = values // n * i + min(i, values % n)
                    var last = (
                        values // n * (i + 1) + min(i + 1, values % n) - 1
                    )
                    var a = (base + UInt64(first)).cast[DType.int64]()
                    var b = (base + UInt64(last)).cast[DType.int64]()
                    mins.append(
                        PrimitiveScalar[T](a.cast[T.native](), witness).to_dyn()
                    )
                    maxes.append(
                        PrimitiveScalar[T](b.cast[T.native](), witness).to_dyn()
                    )
                var columns = List[ColumnZones](capacity=1)
                columns.append(
                    ColumnZones(
                        self.name.copy(),
                        mins^,
                        maxes^,
                        List[Int](length=n, fill=-1),
                    )
                )
                return Index(List[Int](length=n, fill=-1), columns^)
            else:
                return Index()

        return self.min.type().dispatch_primitive(arm)

    def bounded_by(self, other: Self) -> Self:
        """This column with its bounds narrowed to `other`'s where those are
        tighter — sound for a key an inner join matched, since every value
        left is one both sides hold."""
        if not (other.min.is_valid() and other.max.is_valid()):
            return self.copy()
        if not (self.min.is_valid() and self.max.is_valid()):
            var out = self.copy()
            out.min = other.min.copy()
            out.max = other.max.copy()
            return out^
        var dtype = self.min.type()
        if not dtype.is_primitive() or other.min.type() != dtype:
            return self.copy()

        def arm[T: PrimitiveType](witness: T) raises {imm} -> Self:
            var out = self.copy()
            if not (
                other.min.isa[PrimitiveScalar[T]]()
                and self.min.isa[PrimitiveScalar[T]]()
            ):
                return out^
            var lo = self.min.as_primitive[T]().value()
            var hi = self.max.as_primitive[T]().value()
            var other_lo = other.min.as_primitive[T]().value()
            var other_hi = other.max.as_primitive[T]().value()
            if other_lo > lo:
                out.min = other.min.copy()
            if other_hi < hi:
                out.max = other.max.copy()
            return out^

        try:
            return dtype.dispatch_primitive(arm)
        except:
            return self.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(ndv=", self.ndv, ", nulls=", self.nulls, ")")


# ---------------------------------------------------------------------------
# Estimate -- one relation's output
# ---------------------------------------------------------------------------
struct Estimate(Copyable, Movable, Writable):
    """What a relation expects to produce: a row count and a summary per column,
    in schema order where a schema was available, so `Join` can concatenate two
    sides' summaries in the order it concatenates their fields. `column(name)`
    answers `None` for a column it does not hold.
    """

    var rows: Approx
    var columns: List[ColumnEstimate]

    def __init__(
        out self,
        rows: Approx = Approx(),
        var columns: List[ColumnEstimate] = [],
    ):
        self.rows = rows
        self.columns = columns^

    @staticmethod
    def unknown(schema: Schema) -> Self:
        """Nothing known, but one summary per field, so positional reads line
        up with the schema."""
        var cols = List[ColumnEstimate](capacity=len(schema.fields))
        for ref f in schema.fields:
            cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
        return Self(Approx.unknown(), cols^)

    @staticmethod
    def empty(schema: Schema) -> Self:
        """Exactly no rows — so every count is exactly zero, and no bound
        exists because there is no value to bound."""
        var cols = List[ColumnEstimate](capacity=len(schema.fields))
        for ref f in schema.fields:
            cols.append(
                ColumnEstimate(
                    f.name.copy(),
                    nulls=Approx.exact(0),
                    ndv=Approx.exact(0),
                    width=ColumnEstimate.width_of(f.dtype),
                )
            )
        return Self(Approx.exact(0), cols^)

    def index_of(self, name: String) -> Int:
        for i in range(len(self.columns)):
            if self.columns[i].name == name:
                return i
        return -1

    def column(self, name: String) -> Optional[ColumnEstimate]:
        """This column's summary, or `None` when the relation has no such
        column."""
        var i = self.index_of(name)
        if i < 0:
            return None
        return self.columns[i].copy()

    def row_width(self) -> Approx:
        """Bytes per output row — the sum of the column widths, unknown as soon
        as one column's width is."""
        var total = Approx.exact(0)
        for ref c in self.columns:
            total = total + c.width
        return total

    def size(self) -> Size:
        """Its rows and their width — what a `Cost` formula reads."""
        return Size(self.rows, self.row_width())

    def to_index(self) raises -> Index:
        """This estimate as a one-chunk index, so that a predicate's `mask` can
        decide it the way a scan decides a chunk. The row and null counts are
        carried only when exact, because `Index.defined` treats them as proof.
        """
        var columns = List[ColumnZones](capacity=len(self.columns))
        for ref c in self.columns:
            var nulls = c.nulls.known().or_else(
                -1
            ) if c.nulls.is_exact() else -1
            columns.append(
                ColumnZones(
                    c.name.copy(), [c.min.copy()], [c.max.copy()], [nulls]
                )
            )
        var rows = self.rows.known().or_else(-1) if self.rows.is_exact() else -1
        return Index([rows], columns^)

    @staticmethod
    def from_index(index: Index, schema: Schema) raises -> Self:
        """A source's estimate, read off the index it already built.

        Row and null counts are exact when every chunk recorded its part — a
        sum over some of the chunks is not the total — and the bounds are the
        extremes over the chunks that recorded one, compared in the column's
        own dtype: `date32` and `int32` share a representation but not an
        order.

        The distinct count is an estimate: the sum when every chunk recorded
        one and their ranges are pairwise disjoint — a key written in order, a
        chunk per range — since then no value is in two chunks. Otherwise the
        largest, which bounds the count in neither direction: a value can
        repeat across chunks, and a chunk's figure is a dictionary size that
        can include values no row uses. The maximum rather than the sum,
        because the sum overcounts a key every chunk repeats by up to the
        number of chunks, and an overcounted distinct count underestimates a
        join.
        """
        var rows = Self._total(index.rows, index.chunks())
        var cols = List[ColumnEstimate](capacity=len(schema.fields))
        for ref f in schema.fields:
            var i = index.find(f.name)
            if i < 0 or index.chunks() == 0:
                cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
                continue
            ref zones = index.columns[i]
            var bounds = Self._bounds(zones)
            var ndv = Approx.unknown()
            var best = -1
            for d in zones.distinct_counts:
                best = max(best, d)
            if best >= 0:
                ndv = Approx.estimated(best)
                var every = Self._total(zones.distinct_counts, index.chunks())
                if every.is_known() and bounds[2]:
                    ndv = every.to_estimate()
            cols.append(
                ColumnEstimate(
                    f.name.copy(),
                    bounds[0].copy(),
                    bounds[1].copy(),
                    nulls=Self._total(zones.null_counts, index.chunks()),
                    ndv=ndv,
                    width=ColumnEstimate.width_of(f.dtype),
                )
            )
        return Self(rows, cols^)

    @staticmethod
    def _total(counts: List[Int], chunks: Int) -> Approx:
        """The sum of one count per chunk, exact; unknown unless every chunk
        has one."""
        if chunks == 0 or len(counts) != chunks:
            return Approx.unknown()
        var total = 0
        for c in counts:
            if c < 0:
                return Approx.unknown()
            total += c
        return Approx.exact(total)

    @staticmethod
    def _bounds(zones: ColumnZones) raises -> Tuple[DynScalar, DynScalar, Bool]:
        """The smallest of `zones`' minimums and the largest of its maximums —
        nulls when no chunk recorded one — and whether the chunks' ranges are
        pairwise disjoint, which needs every chunk to have recorded both."""
        if not zones.dtype.is_primitive():
            return (NullScalar().to_dyn(), NullScalar().to_dyn(), False)

        def arm[
            T: PrimitiveType
        ](witness: T) raises {imm} -> Tuple[DynScalar, DynScalar, Bool]:
            var lo = zones.statistics[T](witness, False)
            var hi = zones.statistics[T](witness, True)
            var bounds = ColumnEstimate.bounds_of(lo, hi, witness)
            var disjoint = lo.null_count() == 0 and hi.null_count() == 0
            if disjoint:
                var order = List[Int](capacity=len(lo))
                for c in range(len(lo)):
                    var at = len(order)
                    while at > 0 and lo[order[at - 1]].value() > lo[c].value():
                        at -= 1
                    order.insert(at, c)
                for i in range(1, len(order)):
                    if hi[order[i - 1]].value() >= lo[order[i]].value():
                        disjoint = False
            return (bounds[0].copy(), bounds[1].copy(), disjoint)

        return zones.dtype.dispatch_primitive(arm)

    # -----------------------------------------------------------------------
    # Per-node formulas
    # -----------------------------------------------------------------------
    def filtered(self, kept: Selectivity) -> Self:
        """This estimate after filters keeping `kept`: untouched when they
        provably keep every row, otherwise `reduced` to what they leave."""
        if kept.whole:
            return self.copy()
        return self.reduced(kept.apply(self.rows))

    def reduced(self, rows: Approx) -> Self:
        """This estimate reduced to `rows` rows by an operation that keeps some
        of them and invents none. A filter can only remove values, so
        `[min, max]` still contains whatever is left, and the counts survive as
        estimates no larger than they were or than `rows`."""
        var cols = List[ColumnEstimate](capacity=len(self.columns))
        for ref c in self.columns:
            cols.append(c.reduced(rows))
        return Self(rows, cols^)

    def select(self, sources: List[String], output: Schema) -> Self:
        """This estimate's rows under `output`'s fields, field `i` carrying
        the summary of this estimate's column `sources[i]` under its own name
        — a rename keeps the values it renames — and, where there is no such
        column, only its dtype's width. What a node answers that keeps its
        rows and picks, renames or computes columns."""
        var cols = List[ColumnEstimate](capacity=len(output.fields))
        for i in range(len(output.fields)):
            ref f = output.fields[i]
            var c = self.column(sources[i])
            if c:
                cols.append(c.take())
                cols[i].name = f.name.copy()
            else:
                cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
        return Self(self.rows, cols^)

    def limited(self, offset: Int, length: Int) -> Self:
        """This estimate sliced to at most `length` rows starting at `offset`.

        Its row count is exact whenever the input's is: a limit is arithmetic
        on a cardinality, not an assumption about data. A limit that provably
        keeps every row — an exact count, no offset, no shorter — changes
        nothing at all.
        """
        var rows = (self.rows - Approx.exact(offset)).at_most(
            Approx.exact(length)
        )
        if offset == 0 and self.rows.is_exact() and rows == self.rows:
            return self.copy()
        return self.reduced(rows)

    def grouped(self, keys: List[String], output: Schema) raises -> Self:
        """An aggregate's output: exactly one row without keys, otherwise one per
        distinct key combination.

        The count is the product of the keys' distinct counts, capped by the input's
        row count, and an estimate even from exact inputs because it assumes the
        keys independent; a key with no distinct count contributes the row count,
        which leaves just the cap. A key that may hold NULL contributes one more,
        since NULL forms a group that no distinct count includes. Output columns match by position, keys first: a
        key keeps its bounds and an aggregate keeps none. Not by name, because an
        aggregate may share a key's name, and giving it the key's bounds would let
        a predicate above prune a group it has to read.
        """
        if len(keys) == 0:
            var one = List[ColumnEstimate](capacity=len(output.fields))
            for ref f in output.fields:
                one.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
            return Self(Approx.exact(1), one^)

        var groups = Approx.exact(1)
        for ref k in keys:
            var key = self.column(k)
            if Bool(key):
                var distinct = key.value().max_distinct(self.rows)
                # NULL is a group of its own, and neither a distinct count
                # nor a span counts it — unless the key provably has none.
                if key.value().nulls != Approx.exact(0):
                    distinct = distinct + Approx.exact(1)
                groups = groups * distinct
            else:
                groups = Approx.unknown()
        groups = groups.at_most(self.rows).to_estimate()

        var cols = List[ColumnEstimate](capacity=len(output.fields))
        for i in range(len(output.fields)):
            ref f = output.fields[i]
            var summary = Optional[ColumnEstimate](None)
            if i < len(keys):
                summary = self.column(keys[i])
            if Bool(summary):
                # A key keeps its bounds and width; its counts were not kept.
                ref key = summary.value()
                cols.append(
                    ColumnEstimate(
                        key.name.copy(),
                        key.min.copy(),
                        key.max.copy(),
                        ndv=groups,
                        width=key.width,
                    )
                )
            else:
                cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
        return Self(groups, cols^)

    @staticmethod
    def joined(
        left: Estimate,
        right: Estimate,
        left_keys: List[String],
        right_keys: List[String],
        kind: JoinKind,
        strictness: UInt8 = JOIN_ALL,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) -> Self:
        """An equijoin's output under the containment assumption,
        `|L| * |R| / max(ndv(L.key), ndv(R.key))`.

        Assuming the smaller key domain is contained in the larger is what makes a
        foreign-key join come out at `|fact|`. With several key pairs the most
        selective pair decides, since a composite join's keys are rarely
        independent. Either side having exactly zero rows makes the matched count
        exactly zero. An inner `JOIN_ANY` keeps at most one match per row of its
        probe side, the one `build_side` does not name. An outer kind raises the
        matched count to each side it preserves, counting shared matches once
        for FULL; an existence filter caps it at the side it projects, and one
        that `negates` keeps the rest — never estimated at zero, which only a
        proof answers.

        Every column keeps its distinct count, capped by the output's rows — a
        join repeats and drops values but never invents one, and NULL padding
        adds none that is counted. Where every row matched, a key column's
        count is the smaller of the two sides' and its null count is exactly
        zero.

        Static, because a kind and its mirror over exchanged sides must estimate
        identically: a cardinality that moved with the build side would let
        the join search change its own input.
        """
        debug_assert(
            len(left_keys) == len(right_keys),
            "joined: the key lists differ in length",
        )
        var inner: Approx
        if left.rows == Approx.exact(0) or right.rows == Approx.exact(0):
            inner = Approx.exact(0)
        else:
            var domain = Approx.exact(0)
            for i in range(len(left_keys)):
                var li = left.index_of(left_keys[i])
                var ri = right.index_of(right_keys[i])
                if li >= 0 and ri >= 0:
                    domain = domain.at_least(
                        left.columns[li].max_distinct(left.rows)
                    ).at_least(right.columns[ri].max_distinct(right.rows))
                else:
                    domain = Approx.unknown()
            inner = ((left.rows * right.rows) // domain).to_estimate()
        if strictness != JOIN_ALL and kind == JOIN_INNER:
            var probe = right.rows if build_side == BUILD_LEFT else left.rows
            inner = inner.at_most(probe.to_estimate())

        var rows: Approx
        if not (kind.emits_left_columns() and kind.emits_right_columns()):
            var side = left.rows if kind.emits_left_columns() else right.rows
            var matched = inner.at_most(side.to_estimate())
            if kind.negates():
                rows = (side.to_estimate() - matched).to_estimate()
                if side != Approx.exact(0):
                    rows = rows.at_least(Approx.estimated(1))
            else:
                rows = matched
        elif kind.emits_unmatched_left() and kind.emits_unmatched_right():
            var l = inner.at_least(left.rows.to_estimate())
            var r = inner.at_least(right.rows.to_estimate())
            rows = (l + r) - inner
        elif kind.emits_unmatched_left():
            rows = inner.at_least(left.rows.to_estimate())
        elif kind.emits_unmatched_right():
            rows = inner.at_least(right.rows.to_estimate())
        else:
            rows = inner

        # A side the join pads with NULLs gains some, so its null counts are
        # no longer known.
        var cols = List[ColumnEstimate]()
        if kind.emits_left_columns():
            for ref c in left.columns:
                cols.append(c.reduced(rows))
                if kind.emits_unmatched_right():
                    cols[len(cols) - 1].nulls = Approx.unknown()
        if kind.emits_right_columns():
            for ref c in right.columns:
                cols.append(c.reduced(rows))
                if kind.emits_unmatched_left():
                    cols[len(cols) - 1].nulls = Approx.unknown()

        # Every row of these kinds matched, so each key value is on both
        # sides: its distinct count is the smaller side's, its bounds the
        # tighter of the two, and it is never NULL, since a NULL key matches
        # nothing. By position, because a key's name may appear on both sides.
        var matched_only = not (
            kind.negates()
            or kind.emits_unmatched_left()
            or kind.emits_unmatched_right()
        )
        if matched_only:
            var right_at = len(left.columns) if kind.emits_left_columns() else 0
            var seen = List[Bool](length=len(cols), fill=False)
            for i in range(len(left_keys)):
                var li = left.index_of(left_keys[i])
                var ri = right.index_of(right_keys[i])
                if li < 0 or ri < 0:
                    continue
                var both = (
                    left.columns[li]
                    .max_distinct(left.rows)
                    .at_most(right.columns[ri].max_distinct(right.rows))
                    .at_most(rows)
                    .to_estimate()
                )
                var keyed = List[Int]()
                if kind.emits_left_columns():
                    cols[li] = cols[li].bounded_by(right.columns[ri])
                    keyed.append(li)
                if kind.emits_right_columns():
                    cols[right_at + ri] = cols[right_at + ri].bounded_by(
                        left.columns[li]
                    )
                    keyed.append(right_at + ri)
                # The smaller count when a column keys more than one pair.
                for at in keyed:
                    cols[at].ndv = cols[at].ndv.at_most(both) if seen[
                        at
                    ] else both
                    cols[at].nulls = Approx.exact(0)
                    seen[at] = True
        return Self(rows, cols^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Estimate(rows=", self.rows)
        for ref c in self.columns:
            writer.write(", ", c)
        writer.write(")")


# ---------------------------------------------------------------------------
# Cost -- what running a plan should take
# ---------------------------------------------------------------------------
@fieldwise_init
struct Size(Copyable, ImplicitlyCopyable, Movable):
    """An input as a `Cost` formula reads it: its rows, and bytes per row."""

    var rows: Approx
    var width: Approx

    def bytes(self) -> Approx:
        return self.rows * self.width


struct Cost(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Rows moved, comparisons made and bytes held, each an `Approx`, weighed by
    `total()`. Kept apart so a printed cost shows which term dominates. An
    unknown term makes `total()` unknown rather than zero, so a plan nobody can
    estimate never scores best.
    """

    var rows: Approx
    """Rows moved through an operator."""
    var compares: Approx
    """Key comparisons, hash insertions and probes."""
    var bytes: Approx
    """Bytes a pipeline breaker has to materialise."""

    def __init__(
        out self,
        rows: Approx = Approx.exact(0),
        compares: Approx = Approx.exact(0),
        bytes: Approx = Approx.exact(0),
    ):
        self.rows = rows
        self.compares = compares
        self.bytes = bytes

    @staticmethod
    def unknown() -> Self:
        """A node whose work nothing can bound."""
        return Self(Approx.unknown(), Approx.unknown(), Approx.unknown())

    @staticmethod
    def source(size: Size) -> Self:
        """Producing `size`'s rows.

        Charged as both movement and materialisation: a source hands its rows
        on *and* had to build them, which is the difference between reading a
        column and passing one through.
        """
        return Self(rows=size.rows, bytes=size.bytes())

    @staticmethod
    def per_row(rows: Approx) -> Self:
        """Touching each of `rows` rows once and holding nothing — a filter, a
        projection, a limit."""
        return Self(rows=rows)

    @staticmethod
    def hash_build(size: Size) -> Self:
        """Hashing `size`'s rows into a table and keeping them.

        One comparison per row for the hash and probe of the insert, and the
        whole side materialised — which is the term that should make a plan
        prefer the smaller build side, once something reads it.
        """
        return Self(rows=size.rows, compares=size.rows, bytes=size.bytes())

    @staticmethod
    def hash_probe(rows: Approx) -> Self:
        """Probing an existing table once per row."""
        return Self(rows=rows, compares=rows)

    @staticmethod
    def hash_join(
        left: Size, right: Size, build_side: JoinBuildSide, kind: JoinKind
    ) -> Self:
        """One hash join over inputs of these sizes, its inputs' own cost
        excluded: hashing and holding the build side, probing with the other —
        and holding that one too when `kind`, built on `build_side`, answers
        only once every probe row has arrived, as a join emitting the build
        side's unmatched rows or only its matched ones does. Over bare sizes,
        so a search pricing a join it has not built uses the formula a built
        one is priced by."""
        var build = left if build_side == BUILD_LEFT else right
        var probe = right if build_side == BUILD_LEFT else left
        var out = Self.hash_build(build) + Self.hash_probe(probe.rows)
        try:
            var physical = build_side.physical(kind)
            if (
                physical.emits_unmatched_left()
                or not physical.emits_right_columns()
            ):
                out = out + Self(bytes=probe.bytes())
        except:
            pass
        return out

    @staticmethod
    def sort(size: Size) -> Self:
        """`n log n` comparisons, with the whole input held while they happen.
        The log is rounded up to an integer: the term only has to grow faster
        than a filter's, not by precisely that factor."""
        var compares = Approx.unknown()
        var n = size.rows.known()
        if n:
            # ceil(log2(n)), and at least 1: a one-row sort still does work.
            var log = (
                Int(bit_width(UInt64(n.value() - 1))) if n.value() > 1 else 1
            )
            compares = size.rows.times(log).to_estimate()
        return Self(rows=size.rows, compares=compares, bytes=size.bytes())

    def __add__(self, other: Self) -> Self:
        return Self(
            self.rows + other.rows,
            self.compares + other.compares,
            self.bytes + other.bytes,
        )

    def total(self) -> Approx:
        """The three counters weighted into one comparable number."""
        return (
            self.rows.times(ROW_WEIGHT)
            + self.compares.times(COMPARE_WEIGHT)
            + self.bytes.times(BYTE_WEIGHT)
        )

    def __eq__(self, other: Self) -> Bool:
        return (
            self.rows == other.rows
            and self.compares == other.compares
            and self.bytes == other.bytes
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Cost(rows=",
            self.rows,
            ", compares=",
            self.compares,
            ", bytes=",
            self.bytes,
            ", total=",
            self.total(),
            ")",
        )
