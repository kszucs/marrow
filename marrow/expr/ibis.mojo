"""ibis-like typed expression system — TYPE ARCHITECTURE (no execution yet).

Value families are traits, operation nodes are structs, and each node statically
conforms to the family of its *output*. Execution is out of scope: ops are
mimicked by zero-size markers so we can construct trees and prove family
conformance / composition purely in the typesystem.

Each `Kernel` has two facets:
  * a **result-family marker** it conforms to (`NumericResult` / `BoolResult`) —
    this drives the node's family via a constraint-safe `conforms_to(K, …)`;
  * an **output-dtype rule** `Out[L, R]` — a named, reusable parametric `comptime`
    alias (à la ibis `rlz`: `dtype_like`, `highest_precedence`, `float_result`,
    `boolean`) that computes the exact `OutType`.
Two facets because a `where` constraint can only evaluate direct projections /
constants, not a conditional rule like `highest_precedence` — so the family uses
the marker while the exact dtype uses the rich rule.

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
from ..dtypes import DataType, NumericType, BoolType, StringLikeType


# ---------------------------------------------------------------------------
# Promotion rules — named, reusable parametric comptime aliases (ibis rlz-style)
# ---------------------------------------------------------------------------
#
# A rule maps operand value types to an output dtype *type*. Ops bind one of
# these as their `Out` rule; the node's family + OutType both derive from it.

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

comptime float_result[L: Value, R: Value] = dt.Float64Type
"""Output dtype is always float64 — e.g. `Divide`."""

comptime boolean[L: Value, R: Value] = BoolType
"""Output dtype is boolean — comparisons and logical ops."""

comptime int32_result[L: Value, R: Value] = dt.Int32Type
"""Output dtype is int32 — e.g. `StringLength` (a string→numeric op)."""


# ---------------------------------------------------------------------------
# Kernel — the interface the expression expects a "kernel" to conform to
# ---------------------------------------------------------------------------


trait Kernel:
    """A mimic kernel the expression can build a node from (mimics a kernel; no
    execution). `Out[L, R]` is its output-dtype promotion rule; the result-family
    sub-traits below declare which value family the node lands in.

    Two facets, because the `where` constraint solver can only evaluate direct
    projections/constants — not a conditional rule like `highest_precedence`. So
    the *family* comes from a marker the kernel conforms to (constraint-safe),
    while the *exact* output dtype comes from the rich `Out` rule (used in the
    comptime `OutType`, where a conditional/function rule is fine)."""

    comptime Out[L: Value, R: Value]: DataType

    @staticmethod
    def name() -> String:
        ...


trait NumericResult(Kernel):
    """A kernel whose result is numeric — nodes over it are `NumericValue`."""

    pass


trait BoolResult(Kernel):
    """A kernel whose result is boolean — nodes over it are `BoolValue`."""

    pass


struct AddKernel(NumericResult):
    comptime Out[L: Value, R: Value] = highest_precedence[L, R]

    @staticmethod
    def name() -> String:
        return "add"


struct SubKernel(NumericResult):
    comptime Out[L: Value, R: Value] = highest_precedence[L, R]

    @staticmethod
    def name() -> String:
        return "subtract"


struct MulKernel(NumericResult):
    comptime Out[L: Value, R: Value] = highest_precedence[L, R]

    @staticmethod
    def name() -> String:
        return "multiply"


struct DivKernel(NumericResult):
    comptime Out[L: Value, R: Value] = float_result[L, R]

    @staticmethod
    def name() -> String:
        return "divide"


struct NegKernel(NumericResult):
    comptime Out[L: Value, R: Value] = dtype_like[L, R]

    @staticmethod
    def name() -> String:
        return "negate"


struct AbsKernel(NumericResult):
    comptime Out[L: Value, R: Value] = dtype_like[L, R]

    @staticmethod
    def name() -> String:
        return "abs"


struct LtKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "less"


struct LeKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "less_equal"


struct GtKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "greater"


struct GeKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "greater_equal"


struct EqKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "equal"


struct NeKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "not_equal"


struct AndKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "and"


struct OrKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "or"


struct NotKernel(BoolResult):
    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "not"


# cross-family kernels: string operands, non-string result --------------------


struct LengthKernel(NumericResult):
    """string → int32. The node lands in `NumericValue` even though its operand
    is a `StringValue` — the family follows the *result*, not the input."""

    comptime Out[L: Value, R: Value] = int32_result[L, R]

    @staticmethod
    def name() -> String:
        return "length"


struct StartsWithKernel(BoolResult):
    """string × string → bool."""

    comptime Out[L: Value, R: Value] = boolean[L, R]

    @staticmethod
    def name() -> String:
        return "startswith"


# ---------------------------------------------------------------------------
# Value — base trait; family sub-traits carry the operator surface
# ---------------------------------------------------------------------------


trait Value(Copyable, ImplicitlyCopyable, ImplicitlyDeletable, Movable, Writable):
    """Every expression node. Carries the associated output dtype."""

    comptime OutType: DataType

    def name(self) -> String:
        return String()


trait NumericValue(Value):
    """Numeric-typed nodes: arithmetic + comparison operator surface."""

    def __add__[Rhs: NumericValue](self, o: Rhs) -> Binary[AddKernel, Self, Rhs]:
        return Binary[AddKernel, Self, Rhs](self.copy(), o.copy())

    def __sub__[Rhs: NumericValue](self, o: Rhs) -> Binary[SubKernel, Self, Rhs]:
        return Binary[SubKernel, Self, Rhs](self.copy(), o.copy())

    def __mul__[Rhs: NumericValue](self, o: Rhs) -> Binary[MulKernel, Self, Rhs]:
        return Binary[MulKernel, Self, Rhs](self.copy(), o.copy())

    def __truediv__[Rhs: NumericValue](self, o: Rhs) -> Binary[DivKernel, Self, Rhs]:
        return Binary[DivKernel, Self, Rhs](self.copy(), o.copy())

    def __neg__(self) -> Unary[NegKernel, Self]:
        return Unary[NegKernel, Self](self.copy())

    def __lt__[Rhs: NumericValue](self, o: Rhs) -> Binary[LtKernel, Self, Rhs]:
        return Binary[LtKernel, Self, Rhs](self.copy(), o.copy())

    def __le__[Rhs: NumericValue](self, o: Rhs) -> Binary[LeKernel, Self, Rhs]:
        return Binary[LeKernel, Self, Rhs](self.copy(), o.copy())

    def __gt__[Rhs: NumericValue](self, o: Rhs) -> Binary[GtKernel, Self, Rhs]:
        return Binary[GtKernel, Self, Rhs](self.copy(), o.copy())

    def __ge__[Rhs: NumericValue](self, o: Rhs) -> Binary[GeKernel, Self, Rhs]:
        return Binary[GeKernel, Self, Rhs](self.copy(), o.copy())

    def __eq__[Rhs: NumericValue](self, o: Rhs) -> Binary[EqKernel, Self, Rhs]:
        return Binary[EqKernel, Self, Rhs](self.copy(), o.copy())

    def __ne__[Rhs: NumericValue](self, o: Rhs) -> Binary[NeKernel, Self, Rhs]:
        return Binary[NeKernel, Self, Rhs](self.copy(), o.copy())


trait BoolValue(Value):
    """Boolean-typed nodes: logical operator surface."""

    def __and__[Rhs: BoolValue](self, o: Rhs) -> Binary[AndKernel, Self, Rhs]:
        return Binary[AndKernel, Self, Rhs](self.copy(), o.copy())

    def __or__[Rhs: BoolValue](self, o: Rhs) -> Binary[OrKernel, Self, Rhs]:
        return Binary[OrKernel, Self, Rhs](self.copy(), o.copy())

    def __invert__(self) -> Unary[NotKernel, Self]:
        return Unary[NotKernel, Self](self.copy())


trait StringValue(Value):
    """String-typed nodes. Its methods cross families — `length()` yields a
    `NumericValue`, `startswith()` a `BoolValue` — driven by the kernel's rule."""

    def length(self) -> Unary[LengthKernel, Self]:
        return Unary[LengthKernel, Self](self.copy())

    def startswith[
        Rhs: StringValue
    ](self, o: Rhs) -> Binary[StartsWithKernel, Self, Rhs]:
        return Binary[StartsWithKernel, Self, Rhs](self.copy(), o.copy())


# ---------------------------------------------------------------------------
# Binary / Unary — ONE node each; family follows the op's promotion rule
# ---------------------------------------------------------------------------


@fieldwise_init
struct Binary[K: Kernel, L: Value, R: Value](
    Value,
    NumericValue where conforms_to(K, NumericResult),
    BoolValue where conforms_to(K, BoolResult),
):
    """A binary op node. Its family and output dtype both come from the op's
    promotion rule `K.Out[L, R]`."""

    comptime OutType = Self.K.Out[Self.L, Self.R]

    var left: Self.L
    var right: Self.R

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name(), "(")
        self.left.write_to(writer)
        writer.write(", ")
        self.right.write_to(writer)
        writer.write(")")


@fieldwise_init
struct Unary[K: Kernel, A: Value](
    Value,
    NumericValue where conforms_to(K, NumericResult),
    BoolValue where conforms_to(K, BoolResult),
):
    """A unary op node, mirroring `Binary` (the rule is applied as `Out[A, A]`)."""

    comptime OutType = Self.K.Out[Self.A, Self.A]

    var arg: Self.A

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name(), "(")
        self.arg.write_to(writer)
        writer.write(")")


comptime Add = Binary[AddKernel, _, _]
comptime Sub = Binary[SubKernel, _, _]
comptime Mul = Binary[MulKernel, _, _]
comptime Div = Binary[DivKernel, _, _]

comptime Less = Binary[LtKernel, _, _]
comptime LessEqual = Binary[LeKernel, _, _]
comptime Greater = Binary[GtKernel, _, _]
comptime GreaterEqual = Binary[GeKernel, _, _]
comptime Equal = Binary[EqKernel, _, _]
comptime NotEqual = Binary[NeKernel, _, _]

comptime And = Binary[AndKernel, _, _]
comptime Or = Binary[OrKernel, _, _]

comptime Neg = Unary[NegKernel, _]
comptime Not = Unary[NotKernel, _]


# ---------------------------------------------------------------------------
# Leaves — NumericColumn, Literal, col()
# ---------------------------------------------------------------------------


struct NumericColumn[T: NumericType](NumericValue):
    """Named numeric column reference."""

    comptime OutType = Self.T

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", self._name, "]")


struct Literal[T: NumericType](NumericValue):
    """A numeric constant leaf — `lit(2, int64)`."""

    comptime OutType = Self.T

    var value: Int

    def __init__(out self, value: Int, dtype: Self.T):
        self.value = value

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Lit[", self.value, "]")


struct StringColumn[T: StringLikeType](StringValue):
    """Named string column reference."""

    comptime OutType = Self.T

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", self._name, "]")


def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column by name — `col("a", int64)`."""
    return NumericColumn[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """Reference a string column by name — `col("s", string)`."""
    return StringColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> Literal[T]:
    """Build a numeric constant — `lit(2, int64)`."""
    return Literal[T](value, dtype)
