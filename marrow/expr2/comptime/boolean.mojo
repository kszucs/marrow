"""Three-valued logic: `AND`, `OR`, `XOR`, `NOT`.

**These do not fuse, and that is a measured decision rather than an omission.**
Every other comptime node compiles into one SIMD loop; a Kleene node
materialises its operands and calls the kernel. The reason is that Kleene nulls
are *data-dependent*: `NULL AND FALSE` is `FALSE`, so the result's validity
follows the operands' values, not merely their validity.

A per-lane formulation of that rule was written and measured on 2026-08-22, and
it lost by 4-10x (`kernels/tests/bench_boolean.mojo`). `Bitmap`'s `&`, `|` and
`difference` process **64 bits per instruction**, while a SIMD bool lane carries
one bit per byte. Materialising two bitmaps — one bit per row — and running the
kernel's whole-bitmap algebra beats fusing, and it keeps exactly one
implementation of Kleene in the tree, in `kernels.boolean._kleene`.

So the operands here are bound on `ComptimeValue` and not on `BoolValue`: this
node asks them for a column, never for a lane. That is also what lets an `AND`
take another `AND` as an operand — the fusing families and this one share only
the base, and only the part of it that matters (`Type`).
"""

from ...arrays import BoolArray, DynArray
from ...dtypes import BoolType, DynType
from ...kernels.boolean import (
    AndKernel,
    BoolBinaryKernel,
    NotKernel,
    OrKernel,
    XorKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, merged
from ..params import Bindings
from ..physical import Datum
from .core import ComptimeValue


def _as_bool(d: Datum, n: Int) raises -> BoolArray:
    """Narrow an operand's result to the packed column the kernel takes.

    **The dtype check is the reason this exists**, not the narrowing. Operands
    are bound on `ComptimeValue`, so an `int64` operand type-checks here and is
    only wrong at run time. `DynArray.as_bool()` is `as_type[BoolArray]()`,
    whose failure is a `debug_assert` — *"as_type: wrong type, holds int64"* —
    which aborts the process rather than raising. Verified 2026-08-22 by
    deleting the check: the abort took down the whole test runner, failing all
    seven cases in the file rather than the one that was wrong.

    So this converts an unrecoverable abort into a catchable error naming the
    dtype it got. Without it the two lines would read the same and behave
    completely differently under a mistake.

    A boolean operand can still be `Shape.scalar` — `lit(True)` — so this is
    also where such an operand stops being lazy; `Datum.to_array` owns that
    broadcast.
    """
    var arr = d.to_array(n)
    if arr.dtype() != DynType(BoolType()):
        raise Error(
            String("boolean operator: expected bool operand, got ")
            + String(arr.dtype())
        )
    return arr.as_bool().copy()


struct BoolBinary[K: BoolBinaryKernel, L: ComptimeValue, R: ComptimeValue](
    ComptimeValue
):
    """`AND` / `OR` / `XOR` over two boolean operands."""

    comptime Type = BoolType
    comptime shape = Shape.columnar
    """Columnar even when both operands are scalar.

    A fusing node takes `widest_shape` of its operands so a constant subtree
    stays lazy. This one materialises through the kernel regardless, so it
    reports what it does rather than what its operands are.
    """

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    def name(self) -> String:
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: RecordBatch, bindings: Bindings) raises -> Datum:
        """Materialise both operands, then let the kernel decide the nulls.

        `K.apply` is Kleene-correct for `AND` and `OR` — a known-false operand
        forces a valid `FALSE`, a known-true one forces a valid `TRUE`. Nothing
        here re-derives that rule; deriving it twice is how two copies drift.
        """
        var n = batch.num_rows()
        var lhs = _as_bool(self.l.evaluate(batch, bindings), n)
        var rhs = _as_bool(self.r.evaluate(batch, bindings), n)
        return Self.K.apply(lhs, rhs).to_dyn()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]


struct Not[A: ComptimeValue](ComptimeValue):
    """`NOT`, which is three-valued too: `NOT NULL` is `NULL`.

    Unlike `AND`/`OR`, negation's validity *is* structural — the result is null
    exactly where the operand is. It sits here rather than in the fusing family
    only because it shares the operand bound that lets the families mix; a
    fused version would be a legitimate, separate optimisation.

    **Not parameterised on its kernel**, unlike `BoolBinary`. `NotKernel` is
    the only `BoolUnaryKernel` in the tree, so a `K` parameter would be generic
    over something that cannot vary — it would read as a choice where there is
    none. `BoolBinary` earns its `K` with three (`and`, `or`, `xor`). If a
    second unary boolean kernel ever lands, generalising is mechanical.
    """

    comptime Type = BoolType
    comptime shape = Shape.columnar

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    def columns(self) -> List[String]:
        return self.a.columns()

    def name(self) -> String:
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    def evaluate(self, batch: RecordBatch, bindings: Bindings) raises -> Datum:
        var n = batch.num_rows()
        return Datum(
            NotKernel.apply(
                _as_bool(self.a.evaluate(batch, bindings), n)
            ).to_dyn()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(NotKernel.name, "(", self.a, ")")
