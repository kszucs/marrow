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

from ...arrays import StructArray, BoolArray, DynArray
from ...dtypes import BoolType, DynType
from ...kernels.boolean import (
    AndKernel,
    BoolBinaryKernel,
    IsNullKernel,
    NotKernel,
    IsInfKernel,
    IsNanKernel,
    NotNullKernel,
    OrKernel,
    UnaryPredicateKernel,
    ValuePredicateKernel,
    XorKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, merged
from ..params import Bindings
from ..pruning import PruneStats, Truth
from ..physical import Datum
from .core import (
    BoolValue,
    ColumnBound,
    ComptimeValue,
    NumericValue,
    Unnamed,
)


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
    ComptimeValue, Unnamed
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

    def dtype(self, schema: Schema) raises -> DynType:
        """Spelled out, where the fusing bool nodes inherit it from
        `BoolValue`. A Kleene operator produces `bool` but does not fuse, so it
        conforms to `ComptimeValue` directly and never sees that default."""
        return DynType(Self.Type())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """Materialise both operands, then let the kernel decide the nulls.

        `K.apply` is Kleene-correct for `AND` and `OR` — a known-false operand
        forces a valid `FALSE`, a known-true one forces a valid `TRUE`. Nothing
        here re-derives that rule; deriving it twice is how two copies drift.
        """
        var n = len(batch)
        var lhs = _as_bool(self.l.evaluate(batch, bindings), n)
        var rhs = _as_bool(self.r.evaluate(batch, bindings), n)
        return Self.K.apply(lhs, rhs).to_dyn()

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """`AND` is provably false as soon as either conjunct is; `OR` only
        when both disjuncts are. `XOR` prunes nothing — both operands being
        possible says nothing about them differing on any single row, and a
        one-sided domain cannot say more."""
        comptime if Self.K.name == AndKernel.name:
            return self.l.prune(stats, bindings) & self.r.prune(stats, bindings)
        elif Self.K.name == OrKernel.name:
            return self.l.prune(stats, bindings) | self.r.prune(stats, bindings)
        else:
            return Truth.maybe

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]


struct Not[A: ComptimeValue](ComptimeValue, Unnamed):
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

    def dtype(self, schema: Schema) raises -> DynType:
        """Spelled out, where the fusing bool nodes inherit it from
        `BoolValue`. A Kleene operator produces `bool` but does not fuse, so it
        conforms to `ComptimeValue` directly and never sees that default."""
        return DynType(Self.Type())

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        var n = len(batch)
        return Datum(
            NotKernel.apply(
                _as_bool(self.a.evaluate(batch, bindings), n)
            ).to_dyn()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(NotKernel.name, "(", self.a, ")")


# ---------------------------------------------------------------------------
# NullPredicate — is_null / not_null
# ---------------------------------------------------------------------------
struct NullPredicate[K: UnaryPredicateKernel, A: ComptimeValue](
    BoolValue, ColumnBound, Unnamed
):
    """`is_null` / `not_null` — the only nodes that read an operand's *validity*
    rather than its values.

    The operand is bound on `ComptimeValue`, not on a family, for the same
    reason `BoolBinary`'s are: this asks its operand for a column and never for
    a lane, so any family will do. That is also why it is a breaker —
    `UnaryPredicateKernel.apply` takes a `DynArray`, because reading a bitmap
    is one operation whatever the dtype underneath it is.

    **The result is never null**, which is the whole content of a validity
    predicate: `is_null(NULL)` is `TRUE`, a perfectly good value. The kernel
    produces an all-valid `BoolArray` and `ColumnBound` reports it, so nothing
    here has to state the rule twice.

    Only the two validity predicates are aliased below. `is_nan` / `is_inf` are
    `UnaryPredicateKernel`s too, but they read *values* and are meaningful only
    over floating operands, so they need a narrower operand bound than
    `ComptimeValue` and therefore their own node.
    """

    comptime NativeType = DType.int32
    """Sizes the bit-pack driver's lane. The operand has no relevant width —
    `lane` reads bits out of a `BoolArray` the kernel already produced."""

    comptime shape = Shape.columnar
    """Always columnar: `bind` materialises a length-N `BoolArray` whatever the
    operand's shape was. Propagating the operand's shape would claim a scalar
    result this never produces."""

    comptime Bound = BoolArray

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(
            self.a.evaluate(batch, bindings).to_array(len(batch))
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime IsNull = NullPredicate[IsNullKernel, _]
comptime NotNull = NullPredicate[NotNullKernel, _]


# ---------------------------------------------------------------------------
# ValuePredicate — is_nan / is_inf
# ---------------------------------------------------------------------------
struct ValuePredicate[K: ValuePredicateKernel, A: NumericValue](
    BoolValue, ColumnBound, Unnamed
):
    """`is_nan` / `is_inf` — the predicates that read an operand's *values*.

    Its own node rather than another `NullPredicate` alias, which
    `NullPredicate`'s docstring already anticipates: these are meaningful only
    over a floating operand, so they need a narrower operand bound than
    `ComptimeValue`. The bound is `NumericValue` and the *floating* half is a
    `comptime assert` in `__init__` — narrowing the parameter to a hypothetical
    `FloatingValue` would need a fifth family trait and a fifth leaf, for one
    node.

    **The null rule is the point of these existing at all.** A null is not a
    NaN: `is_nan(NULL)` is NULL, not FALSE, where `is_null(NULL)` is TRUE.
    `ValuePredicateKernel.apply` propagates the operand's validity into the
    result, and `ColumnBound` reads that back, so the rule is stated once — in
    the kernel — rather than restated here.

    A breaker for the same reason `NullPredicate` is: the kernel takes a
    `DynArray` and produces a whole `BoolArray`, and `lane` reads bits back out
    of it.
    """

    comptime NativeType = DType.int32
    """Sizes the bit-pack driver's lane. The operand's own width is irrelevant
    — `lane` reads bits out of a `BoolArray` the kernel already produced."""

    comptime shape = Shape.columnar
    comptime Bound = BoolArray

    var a: Self.A

    def __init__(out self, var a: Self.A):
        comptime assert (
            Self.A.Type.native.is_floating_point()
        ), "is_nan / is_inf need a floating-point operand"
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(
            self.a.evaluate(batch, bindings).to_array(len(batch))
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime IsNan = ValuePredicate[IsNanKernel, _]
comptime IsInf = ValuePredicate[IsInfKernel, _]
