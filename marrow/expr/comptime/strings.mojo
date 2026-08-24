"""Fused string operators.

The family that cannot vectorise, kept honest about it. A `StringValue`'s
`lane` answers one `String`, so every node here loops where a numeric node
issues one SIMD instruction. Fusion still pays: it removes *dispatch*, not
just width, so `name = 'x' AND amount > 10` is still one pass with no
per-node materialisation.
"""

from ...arrays import StructArray
from ...buffers import Bitmap
from ...dtypes import DynType, StringLikeType
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, merged
from ..params import Bindings
from ..physical import Datum

from .rules import widest_shape
from .core import BoolValue, StringValue, Unnamed


trait StringCompareKernel:
    """A two-string predicate.

    Local to this module rather than reused from `kernels.numeric`: those
    kernels are `core[dtype, W](SIMD, SIMD) -> SIMD[bool, W]`, a shape a
    variable-width encoding cannot satisfy. The parameter stays a *type* for
    the same reason it does there — routing on a name would put every string
    predicate into every binary that builds any expression.
    """

    comptime name: StaticString

    @staticmethod
    def core(a: String, b: String) -> Bool:
        ...


struct StrEqKernel(StringCompareKernel):
    comptime name = StaticString("str_eq")

    @staticmethod
    def core(a: String, b: String) -> Bool:
        return a == b


struct StrNeKernel(StringCompareKernel):
    comptime name = StaticString("str_ne")

    @staticmethod
    def core(a: String, b: String) -> Bool:
        return a != b


struct StrLtKernel(StringCompareKernel):
    comptime name = StaticString("str_lt")

    @staticmethod
    def core(a: String, b: String) -> Bool:
        return a < b


struct StrGtKernel(StringCompareKernel):
    comptime name = StaticString("str_gt")

    @staticmethod
    def core(a: String, b: String) -> Bool:
        return a > b


struct StringCompare[K: StringCompareKernel, L: StringValue, R: StringValue](
    BoolValue, Unnamed
):
    """A comparison over two string operands, producing packed bits.

    `NativeType` sizes the SIMD lane the bool driver iterates with, and there is
    no honest answer for strings — the operands have no width. `uint64` is
    chosen to keep `W` small (2 on a 128-bit register): the lane body is a loop
    of `W` string comparisons, so a wide `W` would only mean a longer loop per
    call with no vectorisation to show for it. The *output* is still bit-packed
    exactly as a numeric comparison's is, which is what lets a string predicate
    feed `And`/`Or` and `Filter` unchanged.
    """

    comptime NativeType = DType.uint64
    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Null-in, null-out: valid exactly where both operands are.

        As in `NumericCompare`, the data bit is computed regardless of validity
        — the loop compares whatever bytes are there — so this bitmap is the
        only record that the result is meaningless in that row.
        """
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        var out = SIMD[DType.bool, W](fill=False)
        for j in range(W):
            out[j] = Self.K.core(
                self.l.lane(bound[0], idx + j),
                self.r.lane(bound[1], idx + j),
            )
        return out

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime StrEq = StringCompare[StrEqKernel, _, _]
comptime StrNe = StringCompare[StrNeKernel, _, _]
comptime StrLt = StringCompare[StrLtKernel, _, _]
comptime StrGt = StringCompare[StrGtKernel, _, _]
