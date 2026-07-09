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
for a boolean predicate. Both `DynValue` (dynamic) and the fused comptime `Value`
nodes (static) implement `prune_bound(stats)`, so either kind of expression can be
evaluated against the index.
"""

from ..scalars import AnyScalar
from ..schema import Schema
from .. import dtypes as dt


struct PruneBound(Copyable, Movable):
    """Result of pruning one expression node: numeric nodes fill the interval
    `[lo, hi]` (`None` bound = unknown); boolean predicates fill `maybe_true`.
    """

    var lo: Optional[AnyScalar]
    var hi: Optional[AnyScalar]
    var maybe_true: Bool

    def __init__(
        out self,
        var lo: Optional[AnyScalar],
        var hi: Optional[AnyScalar],
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
        var lo: Optional[AnyScalar], var hi: Optional[AnyScalar]
    ) -> Self:
        return Self(lo^, hi^, True)

    @staticmethod
    def boolean(maybe: Bool) -> Self:
        return Self(None, None, maybe)

    # Comparison rules — "could `self <op> other` be true for some pair of values
    # drawn from the two intervals?". Each is the min/max test used by both the
    # fused nodes and the DynValue interpreter, so they share one definition.
    def maybe_gt(self, other: Self) raises -> Bool:
        """`self > other` — possible iff max(self) > min(other)."""
        return gt_or_unknown(self.hi, other.lo)

    def maybe_ge(self, other: Self) raises -> Bool:
        return ge_or_unknown(self.hi, other.lo)

    def maybe_lt(self, other: Self) raises -> Bool:
        """`self < other` — possible iff min(self) < max(other)."""
        return lt_or_unknown(self.lo, other.hi)

    def maybe_le(self, other: Self) raises -> Bool:
        return le_or_unknown(self.lo, other.hi)

    def maybe_eq(self, other: Self) raises -> Bool:
        """`self == other` — possible iff the two intervals overlap."""
        return le_or_unknown(self.lo, other.hi) and ge_or_unknown(
            self.hi, other.lo
        )


struct PruneStats(Copyable, Movable):
    """Per-column `[min, max]` bounds for one row group or page. A `None` bound
    means the column has no usable statistic (treated as unknown)."""

    var schema: Schema
    var mins: List[Optional[AnyScalar]]
    var maxs: List[Optional[AnyScalar]]

    def __init__(
        out self,
        var schema: Schema,
        var mins: List[Optional[AnyScalar]],
        var maxs: List[Optional[AnyScalar]],
    ):
        self.schema = schema^
        self.mins = mins^
        self.maxs = maxs^

    def by_index(
        self, i: Int
    ) -> Tuple[Optional[AnyScalar], Optional[AnyScalar]]:
        if i < 0 or i >= len(self.mins):
            return (None, None)
        return (self.mins[i].copy(), self.maxs[i].copy())

    def by_name(
        self, name: String
    ) raises -> Tuple[Optional[AnyScalar], Optional[AnyScalar]]:
        return self.by_index(self.schema.get_field_index(name))


# ---------------------------------------------------------------------------
# Scalar comparison — exact per-type; returns None when incomparable so callers
# stay conservative (never skip on an uncertain comparison).
# ---------------------------------------------------------------------------


def _c[T: DType](x: Scalar[T], y: Scalar[T]) -> Int:
    return -1 if x < y else (1 if x > y else 0)


def _cmp_scalar(a: AnyScalar, b: AnyScalar) raises -> Optional[Int]:
    """Three-way compare two valid scalars of the same type; None if either is
    null, the types differ, or the type is not orderable here."""
    if not a.is_valid() or not b.is_valid():
        return None
    var t = a.type()
    if t != b.type():
        return None
    if t == dt.int8:
        return _c(a.as_int8().value(), b.as_int8().value())
    elif t == dt.int16:
        return _c(a.as_int16().value(), b.as_int16().value())
    elif t == dt.int32:
        return _c(a.as_int32().value(), b.as_int32().value())
    elif t == dt.int64:
        return _c(a.as_int64().value(), b.as_int64().value())
    elif t == dt.uint8:
        return _c(a.as_uint8().value(), b.as_uint8().value())
    elif t == dt.uint16:
        return _c(a.as_uint16().value(), b.as_uint16().value())
    elif t == dt.uint32:
        return _c(a.as_uint32().value(), b.as_uint32().value())
    elif t == dt.uint64:
        return _c(a.as_uint64().value(), b.as_uint64().value())
    elif t == dt.float32:
        return _c(a.as_float32().value(), b.as_float32().value())
    elif t == dt.float64:
        return _c(a.as_float64().value(), b.as_float64().value())
    elif t.is_string():
        var x = a.as_string().to_string()
        var y = b.as_string().to_string()
        var xb = x.as_bytes()
        var yb = y.as_bytes()
        var n = min(len(xb), len(yb))
        for i in range(n):
            if xb[i] != yb[i]:
                return -1 if xb[i] < yb[i] else 1
        return _c(len(xb), len(yb))
    else:
        return None


# `a <op> b`, or True when either bound is unknown / incomparable (conservative).
def gt_or_unknown(
    a: Optional[AnyScalar], b: Optional[AnyScalar]
) raises -> Bool:
    if not a or not b:
        return True
    var c = _cmp_scalar(a.value(), b.value())
    return True if not c else c.value() > 0


def ge_or_unknown(
    a: Optional[AnyScalar], b: Optional[AnyScalar]
) raises -> Bool:
    if not a or not b:
        return True
    var c = _cmp_scalar(a.value(), b.value())
    return True if not c else c.value() >= 0


def lt_or_unknown(
    a: Optional[AnyScalar], b: Optional[AnyScalar]
) raises -> Bool:
    if not a or not b:
        return True
    var c = _cmp_scalar(a.value(), b.value())
    return True if not c else c.value() < 0


def le_or_unknown(
    a: Optional[AnyScalar], b: Optional[AnyScalar]
) raises -> Bool:
    if not a or not b:
        return True
    var c = _cmp_scalar(a.value(), b.value())
    return True if not c else c.value() <= 0
