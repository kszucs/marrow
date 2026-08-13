"""Interval kernels — marrow's operators interpreted over `[lo, hi]` bounds
instead of over values.

Every comparison and boolean operator has two readings. `LtKernel.core` answers
`a < b` for a SIMD lane of concrete values; `LtInterval.apply` answers *could*
`a < b` be true for **some** pair drawn from two intervals. Same operator, a
different domain — so it is a different kernel, not a method on the SIMD one.

That separation is deliberate and has a precedent here.
`NumericCompareKernel` used to carry a `comptime StringKernel` naming its
string counterpart, and it was removed because "which family `a < b` means is a
question about the operands, and it belongs to whoever is interpreting the
operator, not to the SIMD kernel". An interval reading is the same kind of
claim, so it lives beside the SIMD kernels rather than inside them, and the
expression node pairs the two.

The evaluation is deliberately **conservative**: whenever a bound is unknown or
an operator has no useful interval rule, the answer is "maybe true". A caller
may only ever skip data it has *proven* cannot match, so a wrong "maybe" costs
time and a wrong "no" costs correctness.

`marrow.expr.pruning` is the consumer: it turns a row group's or page's column
statistics into `Interval`s and drives these kernels over an expression tree.
"""

from .core import Kernel
from ..scalars import DynScalar
from .. import dtypes as dt


struct Interval(Copyable, Movable):
    """What one expression node's value can be, given only column bounds.

    A numeric sub-expression fills `[lo, hi]` (a `None` bound is unknown); a
    boolean predicate fills `maybe_true`. One type carries both because a
    predicate's operands are intervals and its result feeds boolean operators.
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
        """Unknown bounds — the conservative answer."""
        return Self(None, None, True)

    @staticmethod
    def bounds(
        var lo: Optional[DynScalar], var hi: Optional[DynScalar]
    ) -> Self:
        """A value known to lie in `[lo, hi]`."""
        return Self(lo^, hi^, True)

    @staticmethod
    def truth(maybe: Bool) -> Self:
        """A predicate that may (or provably cannot) be true."""
        return Self(None, None, maybe)

    # --- the interval algebra ------------------------------------------------
    #
    # "Could `self <op> other` hold for some pair drawn from the two intervals?"
    # Each is a min/max test, and each is what the corresponding kernel below
    # is a one-line naming of.
    def maybe_gt(self, other: Self) raises -> Bool:
        """`self > other` — possible iff max(self) > min(other)."""
        var c = Self.compare(self.hi, other.lo)
        return not c or c.value() > 0

    def maybe_ge(self, other: Self) raises -> Bool:
        var c = Self.compare(self.hi, other.lo)
        return not c or c.value() >= 0

    def maybe_lt(self, other: Self) raises -> Bool:
        """`self < other` — possible iff min(self) < max(other)."""
        var c = Self.compare(self.lo, other.hi)
        return not c or c.value() < 0

    def maybe_le(self, other: Self) raises -> Bool:
        var c = Self.compare(self.lo, other.hi)
        return not c or c.value() <= 0

    def maybe_eq(self, other: Self) raises -> Bool:
        """`self == other` — possible iff the two intervals overlap."""
        return self.maybe_le(other) and self.maybe_ge(other)

    # --- comparison primitives (conservative: unknown / incomparable → None) ---

    @staticmethod
    def compare(
        a: Optional[DynScalar], b: Optional[DynScalar]
    ) raises -> Optional[Int]:
        """Three-way compare two bounds, or None when either is unknown
        (missing) or incomparable — callers treat None as "maybe", staying
        conservative."""
        if not a or not b:
            return None
        return Self._compare_scalar(a.value(), b.value())

    @staticmethod
    def _three_way[T: DType](x: Scalar[T], y: Scalar[T]) -> Int:
        return -1 if x < y else (1 if x > y else 0)

    @staticmethod
    def _compare_scalar(a: DynScalar, b: DynScalar) raises -> Optional[Int]:
        """Three-way compare two valid scalars of the same type; None if either
        is null, the types differ, or the type is not orderable here.

        The numeric arm reuses ``DynType.dispatch_primitive`` — the same
        runtime-dtype → comptime-type selector the cast/compare kernels use —
        so the per-dtype comparison is written once here instead of a
        hand-rolled 11-way switch."""
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
                return Self._three_way(
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
            return Self._three_way(len(xb), len(yb))
        else:
            return None


trait IntervalKernel(Kernel):
    """One operator, read over intervals: *could* it be true for some pair?

    `apply` never raises "false" unless it has proven the operator cannot hold
    anywhere in the two intervals. An operator with no useful rule conforms by
    answering `True` — see `NeInterval` and `XorInterval`.
    """

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        ...


# --- comparison ------------------------------------------------------------


struct LtInterval(IntervalKernel):
    comptime name = "less"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_lt(r)


struct LeInterval(IntervalKernel):
    comptime name = "less_equal"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_le(r)


struct GtInterval(IntervalKernel):
    comptime name = "greater"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_gt(r)


struct GeInterval(IntervalKernel):
    comptime name = "greater_equal"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_ge(r)


struct EqInterval(IntervalKernel):
    comptime name = "equal"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_eq(r)


struct NeInterval(IntervalKernel):
    """`!=` carries no usable interval rule: two identical point intervals are
    the only case it could rule out, and that is not worth a special case."""

    comptime name = "not_equal"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return True


# --- boolean ---------------------------------------------------------------


struct AndInterval(IntervalKernel):
    comptime name = "and"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_true and r.maybe_true


struct OrInterval(IntervalKernel):
    comptime name = "or"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return l.maybe_true or r.maybe_true


struct XorInterval(IntervalKernel):
    """`xor` tells us nothing about a row group: both operands being possible
    says nothing about them differing on any single row."""

    comptime name = "xor"

    @staticmethod
    def apply(l: Interval, r: Interval) raises -> Bool:
        return True
