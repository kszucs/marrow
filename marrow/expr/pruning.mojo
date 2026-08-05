"""Statistics-based predicate pruning for the expression layer.

A *pruning* evaluation answers, for a whole row group or data page: *could this
predicate be true for some row, given only each column's `[min, max]` bounds?*
When the answer is a definite no, the row group / page can be skipped without
decoding it. The evaluation is intentionally conservative — whenever a bound is
unknown or an operator is not modelled, it returns "maybe true", so a caller can
only ever skip data it has proven cannot match. Correctness never depends on the
result: a scan still applies the exact predicate to whatever it decodes.

`PruneStats` is the per-column bounds view (fed by a row-group's ColumnStatistics
or a page's ColumnIndex entry). `PruneBound` is the result of evaluating one node
against it: an interval `[lo, hi]` for a numeric sub-expression, or `maybe_true`
for a boolean predicate. Both `DynValue` (the runtime lane) and the fused comptime `Value`
nodes (static) implement `prune(stats)`, so either kind of expression can be
evaluated against the index.
"""

from ..scalars import DynScalar
from ..schema import Schema

from .. import dtypes as dt


struct PruneBound(Copyable, Movable):
    """Result of pruning one expression node: numeric nodes fill the interval
    `[lo, hi]` (`None` bound = unknown); boolean predicates fill `maybe_true`.
    """

    var lo: Optional[DynScalar]
    var hi: Optional[DynScalar]
    var maybe_true: Bool

    def __init__(
        out self,
        var lo: Optional[DynScalar],
        var hi: Optional[DynScalar],
        maybe_true: Bool,
    ):
        self.lo = lo^
        self.hi = hi^
        self.maybe_true = maybe_true

    @staticmethod
    def unknown() -> Self:
        """A numeric node with unknown bounds (conservative)."""
        return Self(None, None, True)

    @staticmethod
    def interval(
        var lo: Optional[DynScalar], var hi: Optional[DynScalar]
    ) -> Self:
        return Self(lo^, hi^, True)

    @staticmethod
    def boolean(maybe: Bool) -> Self:
        return Self(None, None, maybe)

    # Comparison rules — "could `self <op> other` be true for some pair of values
    # drawn from the two intervals?". Each is the min/max test used by both the
    # fused nodes and the runtime lane's `DynValue`, so they share one definition.
    def maybe_gt(self, other: Self) raises -> Bool:
        """`self > other` — possible iff max(self) > min(other)."""
        var c = Self._cmp_bounds(self.hi, other.lo)
        return not c or c.value() > 0

    def maybe_ge(self, other: Self) raises -> Bool:
        var c = Self._cmp_bounds(self.hi, other.lo)
        return not c or c.value() >= 0

    def maybe_lt(self, other: Self) raises -> Bool:
        """`self < other` — possible iff min(self) < max(other)."""
        var c = Self._cmp_bounds(self.lo, other.hi)
        return not c or c.value() < 0

    def maybe_le(self, other: Self) raises -> Bool:
        var c = Self._cmp_bounds(self.lo, other.hi)
        return not c or c.value() <= 0

    def maybe_eq(self, other: Self) raises -> Bool:
        """`self == other` — possible iff the two intervals overlap."""
        return self.maybe_le(other) and self.maybe_ge(other)

    # --- comparison primitives (conservative: unknown / incomparable → None) ---

    @staticmethod
    def _cmp_bounds(
        a: Optional[DynScalar], b: Optional[DynScalar]
    ) raises -> Optional[Int]:
        """Three-way compare two interval bounds, or None when either is unknown
        (missing) or incomparable — callers treat None as "maybe", staying
        conservative."""
        if not a or not b:
            return None
        return Self._cmp_scalar(a.value(), b.value())

    @staticmethod
    def _cmp[T: DType](x: Scalar[T], y: Scalar[T]) -> Int:
        return -1 if x < y else (1 if x > y else 0)

    @staticmethod
    def _cmp_scalar(a: DynScalar, b: DynScalar) raises -> Optional[Int]:
        """Three-way compare two valid scalars of the same type; None if either
        is null, the types differ, or the type is not orderable here.

        The numeric arm reuses ``DynType.dispatch_numeric`` — the same
        runtime-dtype → comptime-``NumericType`` selector the cast/compare
        kernels use — so the per-dtype comparison is written once here instead
        of a hand-rolled 11-way switch."""
        if not a.is_valid() or not b.is_valid():
            return None
        var t = a.type()
        if t != b.type():
            return None
        if t.is_primitive():
            # `PrimitiveType`, not `NumericType` -- the same widening M1.0 made
            # in `NumericCompareKernel.dispatch`, and for the same reason. While
            # this said `is_numeric`, a predicate on a date, timestamp or decimal
            # column compared no statistics at all, so **no row group and no page
            # was ever pruned on one**. ClickBench filters on `EventDate` and
            # `EventTime`; they were getting zero pushdown.
            @parameter
            def cmp_typed[T: dt.PrimitiveType](witness: T) raises -> Int:
                return Self._cmp(
                    a.as_primitive[T]().value(), b.as_primitive[T]().value()
                )

            return t.dispatch_primitive[cmp_typed]()
        elif t.is_string():
            var x = a.as_string().to_string()
            var y = b.as_string().to_string()
            var xb = x.as_bytes()
            var yb = y.as_bytes()
            var n = min(len(xb), len(yb))
            for i in range(n):
                if xb[i] != yb[i]:
                    return -1 if xb[i] < yb[i] else 1
            return Self._cmp(len(xb), len(yb))
        else:
            return None


struct PruneStats(Copyable, Movable):
    """Per-column `[min, max]` bounds for one row group or page. A `None` bound
    means the column has no usable statistic (treated as unknown)."""

    var schema: Schema
    var mins: List[Optional[DynScalar]]
    var maxs: List[Optional[DynScalar]]

    def __init__(
        out self,
        var schema: Schema,
        var mins: List[Optional[DynScalar]],
        var maxs: List[Optional[DynScalar]],
    ):
        self.schema = schema^
        self.mins = mins^
        self.maxs = maxs^

    def by_index(
        self, i: Int
    ) -> Tuple[Optional[DynScalar], Optional[DynScalar]]:
        if i < 0 or i >= len(self.mins):
            return (None, None)
        return (self.mins[i].copy(), self.maxs[i].copy())

    def by_name(
        self, name: String
    ) raises -> Tuple[Optional[DynScalar], Optional[DynScalar]]:
        return self.by_index(self.schema.get_field_index(name))
