"""What a plan node expects to produce, and what producing it should cost.

`index.mojo` describes what a source knows before it is read; this describes
what each node's output is expected to be, propagated up the plan, so that a
plan can be scored before it runs. `SelectBuildSide` and `JoinReassociation`
spend the score.

Counts are `Approx`: exact, estimated or unknown, with unknown absorbing, so
no figure is invented where nothing is known. Column bounds are `DynScalar`s
with null for unrecorded; a bound is only ever claimed to contain the data, so
it is sound wherever present, which lets a filter prove itself empty from an
estimated input. A filter's selectivity is `Value.mask` over a one-chunk index
built from its input's estimate, the comparisons row-group pruning uses. A
`Cost` weighs rows moved, comparisons made and bytes held.
"""

from std.bit import bit_width

from ..dtypes import DynType, PrimitiveType
from ..kernels.join import JoinKind
from ..scalars import DynScalar, NullScalar, PrimitiveScalar
from ..schema import Schema
from .index import ColumnZones, Index, ZoneMaps


comptime DEFAULT_SELECTIVITY = 20
"""Percent of its input a filter is assumed to keep when nothing can be proven.

The figure is DataFusion's `default_filter_selectivity`, taken rather than
invented because a made-up constant with no provenance is exactly what this
module is supposed to make visible. It applies **only** when the predicate
could not be decided against the child's bounds; a decided predicate answers
exactly zero and never comes here.
"""

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

    def __add__(self, other: Self) -> Self:
        return self._combine(other, self._value + other._value)

    def __sub__(self, other: Self) -> Self:
        """Saturating at zero — see `exact`."""
        return self._combine(other, self._value - other._value)

    def __mul__(self, other: Self) -> Self:
        return self._combine(other, self._value * other._value)

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
        """This count multiplied by a known integer, keeping its state."""
        if not self.is_known():
            return Self.unknown()
        if self.is_exact():
            return Self.exact(self._value * factor)
        return Self.estimated(self._value * factor)

    def scaled(self, percent: Int) -> Self:
        """`percent`% of this count, always an *estimate*.

        Rounded to nearest, then floored at one whenever the input held a row
        and the percentage is not zero: **zero is reserved for provably
        empty**. A filter that merely might not match must not report the same
        cardinality as one proven to match nothing, because the two are read
        identically downstream and only one of them is a proof.
        """
        if not self.is_known():
            return Self.unknown()
        var scaled = (self._value * percent + 50) // 100
        if scaled == 0 and self._value > 0 and percent > 0:
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
    """Distinct values, never more than estimated: `Index.distinct_count`
    reduces a footer's per-chunk counts, and most writers record none, so
    `max_distinct` supplies the fallback where one is needed."""
    var width: Approx
    """Average bytes per value.

    Exact for a fixed-width dtype, where it is a property of the type rather
    than of the data. Unknown for anything variable-width: a string column's
    average length is a fact about the data that nothing here has read.
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
        """Bytes per value for a fixed-width dtype, unknown otherwise.

        `DynType.byte_width` answers 0 for a variable-width dtype, which is not a
        width. A bool is charged a whole byte rather than a bit, erring toward the
        larger cost. Defined here rather than on `DynType`, which must not depend on
        the estimator.
        """
        var width = dtype.byte_width()
        if width > 0:
            return Approx.exact(width)
        if dtype.is_bool():
            return Approx.exact(1)
        return Approx.unknown()

    def max_distinct(self, rows: Approx) -> Approx:
        """How many distinct values this column may hold, at most.

        The recorded `ndv` when there is one, otherwise the row count — a
        column cannot hold more distinct values than it holds values. The
        fallback is an **upper bound presented as an estimate**, which is the
        honest reading: it is sound as a bound and usually far too large as a
        prediction, and the two callers that use it (`Aggregate`'s output
        cardinality and `Join`'s containment denominator) both want the bound.
        """
        if self.ndv.is_known():
            return self.ndv.at_most(rows.to_estimate())
        return rows.to_estimate()

    def to_bounds_only(self) -> Self:
        """This column with its counts dropped, its bounds and width kept.

        What every row-reducing node answers for a column it passes through: a
        bound still contains the data after rows are removed, where a null
        count and a distinct count do not survive an operation that did not
        count them. The width survives for a third reason again — it is a
        property of the dtype, which no row-reducing node changes.
        """
        return Self(
            self.name.copy(),
            self.min.copy(),
            self.max.copy(),
            width=self.width,
        )

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

    def byte_size(self) -> Approx:
        """Bytes this relation's whole output occupies, `rows * row_width`."""
        return self.rows * self.row_width()

    def to_index(self) raises -> Index:
        """This estimate as a one-chunk index, so that a predicate's `mask` can decide
        it the way a scan decides a chunk. The row and null counts are carried only
        when exact, because `Index.defined` treats them as proof.
        """
        var zones = ZoneMaps(capacity=len(self.columns))
        for ref c in self.columns:
            # An estimated null count is carried as unrecorded because it is
            # read as a proof; a distinct count never is, so it is carried
            # whenever it is known.
            var nulls = -1
            if c.nulls.is_exact():
                nulls = c.nulls.known().or_else(-1)
            var distinct = c.ndv.known().or_else(-1)
            zones.add(
                ColumnZones(
                    c.name.copy(),
                    [c.min.copy()],
                    [c.max.copy()],
                    [nulls],
                    [distinct],
                )
            )
        var rows = List[Int]()
        if self.rows.is_exact():
            rows.append(self.rows.known().or_else(0))
        return Index(chunks=1, zones=zones^, rows=rows^)

    @staticmethod
    def from_index(index: Index, schema: Schema) raises -> Self:
        """A source's estimate, read off the index it already built.

        Row and null counts are exact wherever the index has them, bounds are
        the recorded extremes, and the distinct count is always an estimate.
        """
        var rows = Approx.unknown()
        var num_rows = index.num_rows()
        if num_rows:
            rows = Approx.exact(num_rows.value())

        var cols = List[ColumnEstimate](capacity=len(schema.fields))
        for ref f in schema.fields:
            var nulls = Approx.unknown()
            var null_count = index.null_count(f.name)
            if null_count:
                nulls = Approx.exact(null_count.value())
            var ndv = Approx.unknown()
            var distinct = index.distinct_count(f.name)
            if distinct:
                ndv = Approx.estimated(distinct.value())
            cols.append(
                ColumnEstimate(
                    f.name.copy(),
                    index.extreme(f.name, upper=False),
                    index.extreme(f.name, upper=True),
                    nulls=nulls,
                    ndv=ndv,
                    width=ColumnEstimate.width_of(f.dtype),
                )
            )
        return Self(rows, cols^)

    # -----------------------------------------------------------------------
    # Per-node formulas
    # -----------------------------------------------------------------------
    def filtered(self, kept: Approx) -> Self:
        """This estimate reduced to `kept` rows by a predicate.

        Bounds survive and counts do not. A filter can only remove values, so
        `[min, max]` still contains whatever is left — while a null count and a
        distinct count are facts about a population that just changed, and
        nothing counted the new one. Distinct counts are additionally capped by
        the new row count, since a column cannot hold more distinct values than
        rows.
        """
        var cols = List[ColumnEstimate](capacity=len(self.columns))
        for ref c in self.columns:
            var out = c.to_bounds_only()
            out.ndv = c.ndv.at_most(kept).to_estimate()
            cols.append(out^)
        return Self(kept, cols^)

    def carried(self, output: Schema) -> Self:
        """This estimate's summaries matched by name onto `output`'s fields, for a
        node that keeps its rows and appends columns (`Window`). A field with no
        match is a new column and keeps only its width. By name rather than
        position because a projection can rename a column; `Project` uses
        `passes_through` instead, which also rejects a rename.
        """
        var cols = List[ColumnEstimate](capacity=len(output.fields))
        for ref f in output.fields:
            var i = self.index_of(f.name)
            if i >= 0:
                cols.append(self.columns[i].copy())
            else:
                cols.append(ColumnEstimate.unknown(f.name.copy(), f.dtype))
        return Self(self.rows, cols^)

    def limited(self, offset: Int, length: Int) -> Self:
        """This estimate sliced to at most `length` rows starting at `offset`.

        Exact whenever the row count is: a limit is arithmetic on a
        cardinality, not an assumption about data, which makes it the one
        row-reducing node that does not degrade precision.
        """
        var rows = (self.rows - Approx.exact(offset)).at_most(
            Approx.exact(length)
        )
        var out = self.filtered(rows)
        return out^

    def grouped(self, keys: List[String], output: Schema) raises -> Self:
        """An aggregate's output: exactly one row without keys, otherwise one per
        distinct key combination.

        The count is the product of the keys' distinct counts, capped by the input's
        row count, and an estimate even from exact inputs because it assumes the
        keys independent; a key with no distinct count contributes the row count,
        which leaves just the cap. Output columns match by position, keys first: a
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
                groups = groups * key.value().max_distinct(self.rows)
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
                var out = summary.value().to_bounds_only()
                out.ndv = groups
                cols.append(out^)
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
    ) -> Self:
        """An equijoin's output under the containment assumption,
        `|L| * |R| / max(ndv(L.key), ndv(R.key))`.

        Assuming the smaller key domain is contained in the larger is what makes a
        foreign-key join come out at `|fact|`. With several key pairs the most
        selective pair decides, since a composite join's keys are rarely
        independent. Either side having exactly zero rows makes the matched count
        exactly zero. An outer kind raises the matched count to each side it
        preserves, counting shared matches once for FULL; an existence filter caps
        it at the side it projects, and one that `negates` keeps the rest.

        Static, because a kind and its mirror over exchanged sides must estimate
        identically: a cardinality that moved with the build side would let
        `SelectBuildSide` change its own input.
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
                var l = left.column(left_keys[i])
                var r = right.column(right_keys[i])
                if Bool(l) and Bool(r):
                    domain = domain.at_least(
                        l.value().max_distinct(left.rows)
                    ).at_least(r.value().max_distinct(right.rows))
                else:
                    domain = Approx.unknown()
            inner = ((left.rows * right.rows) // domain).to_estimate()

        var rows: Approx
        if not (kind.emits_left_columns() and kind.emits_right_columns()):
            var side = left.rows if kind.emits_left_columns() else right.rows
            var matched = inner.at_most(side.to_estimate())
            if kind.negates():
                rows = (side.to_estimate() - matched).to_estimate()
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

        var cols = List[ColumnEstimate]()
        if kind.emits_left_columns():
            for ref c in left.columns:
                cols.append(c.to_bounds_only())
        if kind.emits_right_columns():
            for ref c in right.columns:
                cols.append(c.to_bounds_only())
        return Self(rows, cols^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Estimate(rows=", self.rows)
        for ref c in self.columns:
            writer.write(", ", c)
        writer.write(")")


# ---------------------------------------------------------------------------
# Cost -- what running a plan should take
# ---------------------------------------------------------------------------
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
    def source(rows: Approx, width: Approx) -> Self:
        """Producing `rows` rows of `width` bytes each.

        Charged as both movement and materialisation: a source hands its rows
        on *and* had to build them, which is the difference between reading a
        column and passing one through.
        """
        return Self(rows=rows, bytes=rows * width)

    @staticmethod
    def per_row(rows: Approx) -> Self:
        """Touching each of `rows` rows once and holding nothing — a filter, a
        projection, a limit."""
        return Self(rows=rows)

    @staticmethod
    def hash_build(rows: Approx, width: Approx) -> Self:
        """Hashing `rows` rows into a table and keeping them.

        One comparison per row for the hash and probe of the insert, and the
        whole side materialised — which is the term that should make a plan
        prefer the smaller build side, once something reads it.
        """
        return Self(rows=rows, compares=rows, bytes=rows * width)

    @staticmethod
    def hash_probe(rows: Approx) -> Self:
        """Probing an existing table once per row."""
        return Self(rows=rows, compares=rows)

    @staticmethod
    def sort(rows: Approx, width: Approx) -> Self:
        """`n log n` comparisons, with the whole input held while they happen.
        The log is rounded up to an integer: the term only has to grow faster
        than a filter's, not by precisely that factor."""
        var compares = Approx.unknown()
        var n = rows.known()
        if n:
            # ceil(log2(n)), and at least 1: a one-row sort still does work.
            var log = (
                Int(bit_width(UInt64(n.value() - 1))) if n.value() > 1 else 1
            )
            compares = rows.times(log).to_estimate()
        return Self(rows=rows, compares=compares, bytes=rows * width)

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
