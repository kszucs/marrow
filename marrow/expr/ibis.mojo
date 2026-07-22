"""ibis-like typed expression system — TYPE ARCHITECTURE (no execution yet).

Value families are traits, operation nodes are structs, and each node statically
conforms to the family of its *output*. Execution is out of scope: ops are
mimicked by zero-size markers so we can construct trees and prove family
conformance / composition purely in the typesystem.

Promotion is encoded entirely in the value hierarchy, exactly like ibis: every
operation node has a SINGLE fixed value family and declares its own output dtype
(`comptime OutType`), just as each ibis op sets `dtype = rlz.numeric_like` /
`dt.float64` / `dt.boolean`. A `Kernel` carries no type information at all — it is
only a name grouping compute methods. There is one node struct per (family,
output-dtype rule): `NumericBinary` (widening), `FloatBinary` (float64),
`NumericUnary` (preserving), `CountingUnary` (int32), `BoolBinary`/`BoolUnary`
(bool). Reusable dtype rules (`highest_precedence`, `dtype_like`) mirror ibis's
`rlz.*`.

The family follows the *result*, not the input — a `StringValue` operand can yield
a `NumericValue` node (`length()`) or a `BoolValue` node (`startswith()`):

    Add(col("a", int32), col("b", int64))    # NumericValue, OutType = int64 (widening)
    col("a", int64) / col("b", int64)        # NumericValue, OutType = float64
    Less(a, b)                               # BoolValue
    (a + b) < a                              # numeric input -> BoolValue
    (a < b) & (b < c)                        # BoolValue & BoolValue -> BoolValue
    col("s", string).length()               # StringValue  -> NumericValue (int32)
    col("s", string).startswith(t)           # StringValue  -> BoolValue
"""

from std.sys import bit_width_of
from std.builtin.rebind import downcast

from .. import dtypes as dt
from ..dtypes import (
    DataType,
    NumericType,
    IntegerType,
    FloatingType,
    BoolType,
    StringLikeType,
    ListLikeType,
)
from ..scalars import PrimitiveScalar, StringScalar


# ---------------------------------------------------------------------------
# Promotion rules — named, reusable parametric comptime aliases (ibis rlz-style)
# ---------------------------------------------------------------------------
#
# A rule maps operand value types to an output dtype *type*. A value node names
# one of these directly as its `OutType` (like ibis ops: `dtype = rlz.numeric_like`
# / `rlz.dtype_like`). Kernels are not involved — promotion lives in the node.


def _rank[T: DataType]() -> Int:
    """Promotion rank from bit width; floats outrank same-width ints."""
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return bit_width_of[N.native]() + (
            1000 if N.native.is_floating_point() else 0
        )
    else:
        return 0


comptime dtype_like[L: Value, R: Value] = L.OutType
"""Output dtype follows the (left) operand — e.g. a unary op preserving type."""

comptime highest_precedence[L: Value, R: Value] = L.OutType if (
    _rank[L.OutType]() >= _rank[R.OutType]()
) else R.OutType
"""Output dtype is the wider operand (bit-width rank) — `Add(int32, int64) → int64`."""


def _is_float[T: DataType]() -> Bool:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return N.native.is_floating_point()
    else:
        return False


comptime sum_result[A: Value] = dt.Float64Type if _is_float[
    A.OutType
]() else dt.Int64Type
"""Reduction that widens to 64-bit to avoid overflow — floats → float64, ints →
int64 (like ibis `Sum`). (Unsigned is treated as signed int64 for now.)"""


# ---------------------------------------------------------------------------
# Kernel — the interface the expression expects a "kernel" to conform to
# ---------------------------------------------------------------------------


trait Kernel:
    """A mimic kernel — just a named *grouping* of an operation's compute methods
    (`name()` for now; `core`/`apply` later). It carries NO type information: the
    output dtype (promotion) is declared entirely by the value node (see the node
    structs' `OutType`), never by the kernel."""

    @staticmethod
    def name() -> String:
        ...


struct AddKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "add"


struct SubKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "subtract"


struct MulKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "multiply"


struct DivKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "divide"


struct NegKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "negate"


struct AbsKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "abs"


struct LtKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "less"


struct LeKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "less_equal"


struct GtKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "greater"


struct GeKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "greater_equal"


struct EqKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "equal"


struct NeKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "not_equal"


struct AndKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "and"


struct OrKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "or"


struct NotKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "not"


struct LengthKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "length"


struct StartsWithKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "startswith"


struct ModKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "modulo"


struct SqrtKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "sqrt"


struct IsNullKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "is_null"


struct UpperKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "upper"


struct LowerKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "lower"


struct PowKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "power"


struct CeilKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "ceil"


struct FloorKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "floor"


struct RoundKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "round"


struct SignKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "sign"


struct ExpKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "exp"


struct LnKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "ln"


struct XorKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "xor"


struct EndsWithKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "endswith"


struct ContainsKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "contains"


struct ReverseKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "reverse"


struct ArrayLengthKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "array_length"


struct ArrayContainsKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "array_contains"


struct SumKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "sum"


struct MeanKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "mean"


struct MinKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "min"


struct MaxKernel(Kernel):
    @staticmethod
    def name() -> String:
        return "max"


# ---------------------------------------------------------------------------
# Value — base trait; family sub-traits carry the operator surface
# ---------------------------------------------------------------------------


trait Value(Copyable, ImplicitlyDeletable, Movable, Writable):
    """Every expression node. Carries the associated output dtype.

    Copies are explicit (`.copy()`) — like arrays/scalars, nodes are not
    `ImplicitlyCopyable`, so a `Literal` can hold a typed (non-implicitly-copyable)
    scalar."""

    comptime OutType: DataType

    def name(self) -> String:
        return String()

    def isnull(self) -> BoolUnary[IsNullKernel, Self]:
        """Null predicate — any value in any family yields a `BoolValue` (ibis
        `IsNull`). The result family follows the op, not the operand."""
        return BoolUnary[IsNullKernel, Self](self.copy())


trait NumericValue(Value):
    """Numeric-typed nodes: arithmetic + comparison operator surface."""

    def __add__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[AddKernel, Self, Rhs]:
        return NumericBinary[AddKernel, Self, Rhs](self.copy(), o.copy())

    def __sub__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[SubKernel, Self, Rhs]:
        return NumericBinary[SubKernel, Self, Rhs](self.copy(), o.copy())

    def __mul__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[MulKernel, Self, Rhs]:
        return NumericBinary[MulKernel, Self, Rhs](self.copy(), o.copy())

    def __truediv__[
        Rhs: NumericValue
    ](self, o: Rhs) -> FloatBinary[DivKernel, Self, Rhs]:
        return FloatBinary[DivKernel, Self, Rhs](self.copy(), o.copy())

    def __mod__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[ModKernel, Self, Rhs]:
        return NumericBinary[ModKernel, Self, Rhs](self.copy(), o.copy())

    def __pow__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[PowKernel, Self, Rhs]:
        return NumericBinary[PowKernel, Self, Rhs](self.copy(), o.copy())

    def __neg__(self) -> NumericUnary[NegKernel, Self]:
        return NumericUnary[NegKernel, Self](self.copy())

    def abs(self) -> NumericUnary[AbsKernel, Self]:
        return NumericUnary[AbsKernel, Self](self.copy())

    def ceil(self) -> NumericUnary[CeilKernel, Self]:
        return NumericUnary[CeilKernel, Self](self.copy())

    def floor(self) -> NumericUnary[FloorKernel, Self]:
        return NumericUnary[FloorKernel, Self](self.copy())

    def round(self) -> NumericUnary[RoundKernel, Self]:
        return NumericUnary[RoundKernel, Self](self.copy())

    def sign(self) -> NumericUnary[SignKernel, Self]:
        return NumericUnary[SignKernel, Self](self.copy())

    def sqrt(self) -> FloatUnary[SqrtKernel, Self]:
        return FloatUnary[SqrtKernel, Self](self.copy())

    def exp(self) -> FloatUnary[ExpKernel, Self]:
        return FloatUnary[ExpKernel, Self](self.copy())

    def ln(self) -> FloatUnary[LnKernel, Self]:
        return FloatUnary[LnKernel, Self](self.copy())

    # reductions (N -> 1); the node stays a NumericValue carrying the result dtype
    def sum(self) -> SumUnary[SumKernel, Self]:
        return SumUnary[SumKernel, Self](self.copy())

    def mean(self) -> FloatUnary[MeanKernel, Self]:
        return FloatUnary[MeanKernel, Self](self.copy())

    def min(self) -> NumericUnary[MinKernel, Self]:
        return NumericUnary[MinKernel, Self](self.copy())

    def max(self) -> NumericUnary[MaxKernel, Self]:
        return NumericUnary[MaxKernel, Self](self.copy())

    def __lt__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[LtKernel, Self, Rhs]:
        return BoolBinary[LtKernel, Self, Rhs](self.copy(), o.copy())

    def __le__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[LeKernel, Self, Rhs]:
        return BoolBinary[LeKernel, Self, Rhs](self.copy(), o.copy())

    def __gt__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[GtKernel, Self, Rhs]:
        return BoolBinary[GtKernel, Self, Rhs](self.copy(), o.copy())

    def __ge__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[GeKernel, Self, Rhs]:
        return BoolBinary[GeKernel, Self, Rhs](self.copy(), o.copy())

    def __eq__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[EqKernel, Self, Rhs]:
        return BoolBinary[EqKernel, Self, Rhs](self.copy(), o.copy())

    def __ne__[
        Rhs: NumericValue
    ](self, o: Rhs) -> BoolBinary[NeKernel, Self, Rhs]:
        return BoolBinary[NeKernel, Self, Rhs](self.copy(), o.copy())


trait BoolValue(Value):
    """Boolean-typed nodes: logical operator surface."""

    def __and__[
        Rhs: BoolValue
    ](self, o: Rhs) -> BoolBinary[AndKernel, Self, Rhs]:
        return BoolBinary[AndKernel, Self, Rhs](self.copy(), o.copy())

    def __or__[Rhs: BoolValue](self, o: Rhs) -> BoolBinary[OrKernel, Self, Rhs]:
        return BoolBinary[OrKernel, Self, Rhs](self.copy(), o.copy())

    def __xor__[
        Rhs: BoolValue
    ](self, o: Rhs) -> BoolBinary[XorKernel, Self, Rhs]:
        return BoolBinary[XorKernel, Self, Rhs](self.copy(), o.copy())

    def __invert__(self) -> BoolUnary[NotKernel, Self]:
        return BoolUnary[NotKernel, Self](self.copy())


trait StringValue(Value):
    """String-typed nodes. Its methods cross families — `length()` yields a
    `NumericValue`, `startswith()` a `BoolValue` — because each node declares its
    own output family, independent of the (string) operand family.
    """

    def length(self) -> CountingUnary[LengthKernel, Self]:
        return CountingUnary[LengthKernel, Self](self.copy())

    def upper(self) -> StringUnary[UpperKernel, Self]:
        return StringUnary[UpperKernel, Self](self.copy())

    def lower(self) -> StringUnary[LowerKernel, Self]:
        return StringUnary[LowerKernel, Self](self.copy())

    def reverse(self) -> StringUnary[ReverseKernel, Self]:
        return StringUnary[ReverseKernel, Self](self.copy())

    def startswith[
        Rhs: StringValue
    ](self, o: Rhs) -> BoolBinary[StartsWithKernel, Self, Rhs]:
        return BoolBinary[StartsWithKernel, Self, Rhs](self.copy(), o.copy())

    def endswith[
        Rhs: StringValue
    ](self, o: Rhs) -> BoolBinary[EndsWithKernel, Self, Rhs]:
        return BoolBinary[EndsWithKernel, Self, Rhs](self.copy(), o.copy())

    def contains[
        Rhs: StringValue
    ](self, o: Rhs) -> BoolBinary[ContainsKernel, Self, Rhs]:
        return BoolBinary[ContainsKernel, Self, Rhs](self.copy(), o.copy())

    def __eq__[
        Rhs: StringValue
    ](self, o: Rhs) -> BoolBinary[EqKernel, Self, Rhs]:
        return BoolBinary[EqKernel, Self, Rhs](self.copy(), o.copy())

    def __ne__[
        Rhs: StringValue
    ](self, o: Rhs) -> BoolBinary[NeKernel, Self, Rhs]:
        return BoolBinary[NeKernel, Self, Rhs](self.copy(), o.copy())


trait ListValue(Value):
    """List-typed nodes (the nested family). Like `StringValue`, its methods cross
    families — `length()` yields a `NumericValue` (element count), `contains()` a
    `BoolValue` — the node's family follows the *result*, not the list operand.
    """

    def length(self) -> CountingUnary[ArrayLengthKernel, Self]:
        return CountingUnary[ArrayLengthKernel, Self](self.copy())

    def contains[
        E: Value
    ](self, elem: E) -> BoolBinary[ArrayContainsKernel, Self, E]:
        return BoolBinary[ArrayContainsKernel, Self, E](
            self.copy(), elem.copy()
        )


# ---------------------------------------------------------------------------
# Operation nodes — one struct per (family, output-dtype rule)
# ---------------------------------------------------------------------------
#
# Each node has a SINGLE, fixed value family and declares its own `OutType`
# directly (mirroring ibis, where every op class sets `dtype = …`). Promotion
# thus lives entirely in the value hierarchy; the kernel `K` is only a name.
# Because each node's family is fixed, there is no conditional conformance here —
# the family sub-trait is listed unconditionally.


# Nodes carry no `write_to` — the reflection default (which prints the struct name
# with its kernel type parameter, e.g. `NumericBinary[…AddKernel…](left=…)`) is a
# complete, unambiguous repr, so an explicit one would be pure boilerplate.


@fieldwise_init
struct NumericBinary[K: Kernel, L: Value, R: Value](NumericValue):
    """Arithmetic binary widening to the higher-precedence operand — `Add`, `Sub`,
    `Mul` (ibis `rlz.numeric_like`)."""

    comptime OutType = highest_precedence[Self.L, Self.R]

    var left: Self.L
    var right: Self.R


@fieldwise_init
struct FloatBinary[K: Kernel, L: Value, R: Value](NumericValue):
    """Binary numeric op whose result is always float64 — `Div` (ibis
    `Divide.dtype = dt.float64`)."""

    comptime OutType = dt.Float64Type

    var left: Self.L
    var right: Self.R


@fieldwise_init
struct NumericUnary[K: Kernel, A: Value](NumericValue):
    """Unary numeric op preserving the operand dtype — `Neg`, `Abs` (ibis
    `rlz.dtype_like`)."""

    comptime OutType = dtype_like[Self.A, Self.A]

    var arg: Self.A


@fieldwise_init
struct FloatUnary[K: Kernel, A: Value](NumericValue):
    """Unary numeric op whose result is always float64 — `sqrt()` (ibis
    `MathUnary`, roughly `higher_precedence(arg, float64)`)."""

    comptime OutType = dt.Float64Type

    var arg: Self.A


@fieldwise_init
struct CountingUnary[K: Kernel, A: Value](NumericValue):
    """Unary op whose result is always int32 — `length()` (ibis
    `StringLength.dtype = dt.int32`). Its operand is a `StringValue`/`ListValue`
    but the node is a `NumericValue`: family follows the *result*, not the input.
    """

    comptime OutType = dt.Int32Type

    var arg: Self.A


@fieldwise_init
struct SumUnary[K: Kernel, A: Value](NumericValue):
    """Reduction whose result widens to 64-bit (`sum()`, ibis `Sum`)."""

    comptime OutType = sum_result[Self.A]

    var arg: Self.A


@fieldwise_init
struct StringBinary[K: Kernel, L: Value, R: Value](StringValue):
    """Binary op whose result preserves a string operand dtype — e.g. `concat`
    (ibis strings `dtype = dt.string`)."""

    comptime OutType = dtype_like[Self.L, Self.R]

    var left: Self.L
    var right: Self.R


@fieldwise_init
struct StringUnary[K: Kernel, A: Value](StringValue):
    """Unary op whose result preserves the string operand dtype — `upper()`,
    `lower()` (ibis `Uppercase`/`Lowercase`, `dtype = dt.string`)."""

    comptime OutType = dtype_like[Self.A, Self.A]

    var arg: Self.A


@fieldwise_init
struct BoolBinary[K: Kernel, L: Value, R: Value](BoolValue):
    """Binary op whose result is always bool — comparisons, `And`/`Or`,
    `startswith()` (ibis `Comparison`/`LogicalBinary`, `dtype = dt.boolean`)."""

    comptime OutType = dt.BoolType

    var left: Self.L
    var right: Self.R


@fieldwise_init
struct BoolUnary[K: Kernel, A: Value](BoolValue):
    """Unary op whose result is always bool — `Not`, `isnull()` (ibis
    `IsNull`/`Not`, `dtype = dt.boolean`)."""

    comptime OutType = dt.BoolType

    var arg: Self.A


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Pow = NumericBinary[PowKernel, _, _]
comptime Div = FloatBinary[DivKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Ceil = NumericUnary[CeilKernel, _]
comptime Floor = NumericUnary[FloorKernel, _]
comptime Round = NumericUnary[RoundKernel, _]
comptime Sign = NumericUnary[SignKernel, _]
comptime Sqrt = FloatUnary[SqrtKernel, _]
comptime Exp = FloatUnary[ExpKernel, _]
comptime Ln = FloatUnary[LnKernel, _]

comptime Sum = SumUnary[SumKernel, _]
comptime Mean = FloatUnary[MeanKernel, _]
comptime Min = NumericUnary[MinKernel, _]
comptime Max = NumericUnary[MaxKernel, _]

comptime Less = BoolBinary[LtKernel, _, _]
comptime LessEqual = BoolBinary[LeKernel, _, _]
comptime Greater = BoolBinary[GtKernel, _, _]
comptime GreaterEqual = BoolBinary[GeKernel, _, _]
comptime Equal = BoolBinary[EqKernel, _, _]
comptime NotEqual = BoolBinary[NeKernel, _, _]
comptime StartsWith = BoolBinary[StartsWithKernel, _, _]
comptime EndsWith = BoolBinary[EndsWithKernel, _, _]
comptime Contains = BoolBinary[ContainsKernel, _, _]

comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]
comptime Not = BoolUnary[NotKernel, _]
comptime IsNull = BoolUnary[IsNullKernel, _]

comptime Length = CountingUnary[LengthKernel, _]
comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]

comptime ArrayLength = CountingUnary[ArrayLengthKernel, _]
comptime ArrayContains = BoolBinary[ArrayContainsKernel, _, _]


# ---------------------------------------------------------------------------
# Leaves — Column / Literal (conditionally a family by their dtype)
# ---------------------------------------------------------------------------


# `Value` must be listed first so the base traits (Copyable/Movable/…) resolve
# via its unconditional path; the `where`-guarded families would otherwise reach
# those ancestors with conflicting constraints. `fmt: off` stops the formatter
# from alphabetically sorting `Value` to the end and reintroducing that error.
# fmt: off
struct Column[T: DataType](
    Value,
    NumericValue where conforms_to(T, NumericType),
    StringValue where conforms_to(T, StringLikeType),
    ListValue where conforms_to(T, ListLikeType),
):
    # fmt: on
    """Named column reference — conditionally the value family of its dtype:
    `col("a", int64)` is a `NumericValue`, `col("s", string)` a `StringValue`.
    """

    comptime OutType = Self.T

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()


# The stored scalar is the dtype's companion `T.ScalarType` (declared on
# `DataType`): `PrimitiveScalar[Int64Type]` for `lit(2, int64)`, `StringScalar`
# for `lit("x", string)`.
# Scalar builders bound on the *provider* traits (`NumericType`/`StringLikeType`),
# where `T.ScalarType` reduces to the concrete companion. Each RETURNS `T.ScalarType`
# — so its result unifies with `Literal`'s `Self.T.ScalarType` field even though
# `Literal` is bound on the abstract `DataType`. This is what lets a generic leaf
# construct `Self.T.ScalarType` without any `rebind`.
def _numeric_scalar[T: NumericType](value: SIMD[T.native, 1]) -> T.ScalarType:
    return PrimitiveScalar[T](value)


def _string_scalar[T: StringLikeType](var value: String) -> T.ScalarType:
    return StringScalar(value^)


# fmt: off  (see `Column` — keep `Value` first for base-trait resolution)
struct Literal[T: DataType](
    Value,
    NumericValue where conforms_to(T, NumericType),
    StringValue where conforms_to(T, StringLikeType),
):
    # fmt: on
    """A constant leaf holding the dtype's companion typed scalar `T.ScalarType`,
    built in place from a raw value. `lit` is an alias for it — `lit(2, int64)`,
    `lit(1.5, float64)`, `lit("x", string)`. The `dtype` argument only pins `T`.
    """

    comptime OutType = Self.T

    var value: Self.T.ScalarType

    def __init__(
        out self, value: Int, dtype: Self.T
    ) where conforms_to(Self.T, IntegerType):
        self.value = _numeric_scalar[Self.T](SIMD[Self.T.native, 1](value))

    def __init__(
        out self, value: Float64, dtype: Self.T
    ) where conforms_to(Self.T, FloatingType):
        self.value = _numeric_scalar[Self.T](SIMD[Self.T.native, 1](value))

    def __init__(
        out self, var value: String, dtype: Self.T
    ) where conforms_to(Self.T, StringLikeType):
        self.value = _string_scalar[Self.T](value^)


def col[T: DataType](var name: String, dtype: T) -> Column[T]:
    """Reference a column by name — `col("a", int64)` / `col("s", string)`."""
    return Column[T](name^)


comptime lit = Literal
