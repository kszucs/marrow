"""Fused numeric operators: nodes built from other numeric nodes.

Each is generic over its operands' *types*, so the whole subtree is one type
and the driver inlines it into one loop. `Add[Column[Int64Type],
Literal[Int64Type]]` names its kernel, its operand types and its output type at
compile time; nothing is looked up when it runs.

The kernel is a parameter rather than a field for the same reason `_eval` is a
pointer in the runtime lane: routing on a name would put every arithmetic
kernel in every binary that builds any expression.
"""

from ...dtypes import DataType, DynType, NumericType
from ...kernels.interval import (
    GtInterval,
    Interval,
    IntervalKernel,
    LtInterval,
)
from ...kernels.numeric import (
    AddKernel,
    GtKernel,
    LtKernel,
    NumericCompareKernel,
    BinaryNumericKernel,
    MulKernel,
    SubKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ...buffers import Bitmap
from ..core import Datum, Shape
from ..pruning import PruneStats
from .core import BoolValue, NumericValue
from .rules import promote, wider, widest_shape


struct NumericBinary[
    K: BinaryNumericKernel, L: NumericValue, R: NumericValue
](NumericValue):
    """A binary arithmetic node over two fused operands."""

    comptime Type = promote[Self.L.Type, Self.R.Type]
    """The wider operand wins: `Add(int32, int64)` is `int64`."""

    comptime shape = widest_shape[Self.L, Self.R]

    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Analyzable ---------------------------------------------------------

    def columns(self) -> List[String]:
        var out = self.l.columns()
        for ref n in self.r.columns():
            var seen = False
            for ref have in out:
                if have == n:
                    seen = True
                    break
            if not seen:
                out.append(n.copy())
        return out^

    def name(self) -> String:
        # Computed, so it has no name of its own — the planner supplies one.
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    def interval(self, stats: PruneStats) raises -> Interval:
        # Arithmetic over intervals is a kernel family of its own and is not
        # wired here yet. `unknown` is always sound: a caller only skips data it
        # has *proven* cannot match, so this costs a decode and never a row.
        return Interval.unknown()

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return (self.l.bind(batch), self.r.bind(batch))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Arithmetic is null-in, null-out: the result is valid where both
        operands are. A missing bitmap means "cannot be null", so an absent
        side contributes nothing to the intersection."""
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return Self.K.core[Self.Type.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.Type.native](),
            self.r.lane[W](bound[1], idx).cast[Self.Type.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


# ---------------------------------------------------------------------------
# Named shapes
# ---------------------------------------------------------------------------
# Partial specialisations, so a call site writes `Add[L, R]` rather than
# restating the kernel. The kernel is still a comptime parameter — these are
# names for a binding, not a registry, and an unused one costs nothing.
comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]


struct NumericCompare[
    K: NumericCompareKernel,
    P: IntervalKernel,
    L: NumericValue,
    R: NumericValue,
](BoolValue):
    """A comparison over two numeric operands, producing packed bits.

    Carries **two** kernels: `K` compares fixed-width lanes, `P` is the same
    operator read over `[min, max]` intervals. Two parameters rather than one
    because `interval` must not re-derive the operator from `K` — `expr/` did
    that by matching `K.name` against five string literals, and when a kernel
    was renamed the match fell through to `unknown()`, which is *sound*. So
    pruning switched itself off with no error and no failing test.

    `Type` is not declared: `BoolValue` fixes it, because a comparison has no
    choice about what it produces.
    """

    comptime shape = widest_shape[Self.L, Self.R]

    comptime ArgType = promote[Self.L.Type, Self.R.Type]
    """The common type both operands are cast to before comparing.

    `int32 < int64` compares in `int64`; without this the narrower operand
    would be compared against a truncated view of the wider one.
    """

    comptime NativeType = wider[Self.L.Type.native, Self.R.Type.native]
    """Sizes the lane. The *operands*' width, not the output's — see
    `BoolValue.NativeType`."""

    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Analyzable ---------------------------------------------------------

    def columns(self) -> List[String]:
        var out = self.l.columns()
        for ref n in self.r.columns():
            var seen = False
            for ref have in out:
                if have == n:
                    seen = True
                    break
            if not seen:
                out.append(n.copy())
        return out^

    def name(self) -> String:
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    def interval(self, stats: PruneStats) raises -> Interval:
        """Can this comparison be true, given the operands' bounds?

        This is where statistics pruning actually happens: everything else in
        the interval chain exists to feed this one answer. A definite "no"
        skips a row group without decoding it, and an imprecise "maybe" costs
        that decode and nothing else.
        """
        return Interval.truth(
            Self.P.apply(self.l.interval(stats), self.r.interval(stats))
        )

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return (self.l.bind(batch), self.r.bind(batch))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """A comparison is null-in, null-out: valid where both operands are.

        The data bit is computed regardless — a SIMD lane compares whatever is
        in the slot — so this bitmap is the *only* thing that records the
        result is meaningless there. Reading the data bit without it is the
        defect that made NULL join keys match each other.
        """
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return Self.K.core[Self.ArgType.native, W](
            self.l.lane[W](bound[0], idx).cast[Self.ArgType.native](),
            self.r.lane[W](bound[1], idx).cast[Self.ArgType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime Gt = NumericCompare[GtKernel, GtInterval, _, _]
comptime Lt = NumericCompare[LtKernel, LtInterval, _, _]
