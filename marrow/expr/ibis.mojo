"""ibis-like typed expression system with fused execution.

Value families are traits, operations are node structs, and each node statically
conforms to the family of its *output*. The numeric family additionally
*executes*: it is hooked to the real `marrow.kernels` (which supply the `core[T,W]`
SIMD functors) and fuses lane-computable chains into a single vectorized pass.

Layers:
  * `Value.execute(batch) -> Self.OutType.ArrayType` — the uniform verb (abstract
    on `Value` so `AnyValue` can call it on any boxed node). It returns the dtype's
    companion array (the dtype→array associated type on `DataType`). The numeric
    family fuses; every other concrete node stubs it (raises) until wired. `AnyValue`
    boxes any node and `.execute(batch)`s it to an `AnyArray` (`.to_any()`).
  * `NumericValue` **is** the numeric lane: it refines `OutType` to `NumericType`,
    adds the `core[W]` SIMD primitive, and its `execute` vectorizes `core` across
    the whole tree — composite nodes call the kernel's `core` on their children's
    `core`, so the compiler inlines the entire chain (zero intermediate arrays).
  * Promotion lives in the value nodes (`OutType`); compute lives in the kernel.
    Every op node is parameterized by a real `marrow.kernels` kernel — arithmetic /
    compare / boolean / aggregate are implemented; string (`kernels.string`) and
    list (`kernels.nested`) markers plus `xor`/`is_null` are not-implemented stubs
    (just `comptime name`) until their compute lands. No kernels are defined here.
  * `BoolValue` / `StringValue` / `ListValue` are the type architecture (execution
    is future work — their `execute` inherits the raising default). Cross-family
    numeric-producing boundaries (`length`, reductions) are non-lane `Value` nodes
    that materialize (not yet wired) rather than fuse.

Dedicated per-family leaves (`NumericColumn` / `StringColumn` / `ListColumn`,
`NumericLiteral` / `StringLiteral`) keep `core`/`NativeType` unconditional and the
hierarchy sharp; `col` / `lit` overload by dtype family.
"""

from std.sys import bit_width_of, size_of
from std.sys.info import simd_byte_width
from std.builtin.rebind import downcast
from std.builtin.simd import Scalar
from std.utils.index import IndexList
from std.algorithm.backend.vectorize import vectorize

from .. import dtypes as dt
from ..dtypes import (
    DataType,
    NumericType,
    IntegerType,
    FloatingType,
    BoolType,
    StringLikeType,
    ListLikeType,
    DType,
)
from std.memory import ArcPointer

from ..scalars import PrimitiveScalar, StringScalar
from ..buffers import Buffer
from ..arrays import PrimitiveArray, AnyArray
from ..tabular import RecordBatch
from ..kernels.helpers import Kernel
from ..kernels.arithmetic import (
    BinaryKernel,
    UnaryKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    ModKernel,
    PowKernel,
    NegKernel,
    AbsKernel,
    CeilKernel,
    FloorKernel,
    RoundKernel,
    SignKernel,
    TruncKernel,
    SqrtKernel,
    ExpKernel,
    Exp2Kernel,
    LogKernel,
    Log2Kernel,
    Log10Kernel,
    Log1pKernel,
    SinKernel,
    CosKernel,
)
from ..kernels.compare import (
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from ..kernels.boolean import (
    AndKernel,
    OrKernel,
    NotKernel,
    XorKernel,
    IsNullKernel,
    NotNullKernel,
    IsNanKernel,
    IsInfKernel,
)
from ..kernels.aggregate import SumKernel, MeanKernel, MinKernel, MaxKernel
from ..kernels.string import (
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    LengthKernel,
    UpperKernel,
    LowerKernel,
    ReverseKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    CapitalizeKernel,
)
from ..kernels.nested import ArrayLengthKernel, ArrayContainsKernel


# ---------------------------------------------------------------------------
# Promotion rules — reusable parametric comptime aliases (ibis rlz-style)
# ---------------------------------------------------------------------------


def _rank[T: DataType]() -> Int:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return bit_width_of[N.native]() + (
            1000 if N.native.is_floating_point() else 0
        )
    else:
        return 0


def _is_float[T: DataType]() -> Bool:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return N.native.is_floating_point()
    else:
        return False


comptime dtype_like[L: Value, R: Value] = L.OutType
"""Output dtype follows the (left) operand — preserving unary/binary."""

comptime highest_precedence[L: NumericValue, R: NumericValue] = L.OutType if (
    _rank[L.OutType]() >= _rank[R.OutType]()
) else R.OutType
"""Output dtype is the wider operand — `Add(int32, int64) → int64`. Bound on
`NumericValue` so the result is a `NumericType` (has `.native`)."""

comptime sum_result[A: Value] = dt.Float64Type if _is_float[
    A.OutType
]() else dt.Int64Type
"""Reduction widens to 64-bit — floats → float64, ints → int64 (ibis `Sum`)."""


def _numeric_array[
    T: NumericType
](var buffer: Buffer[mut=False], length: Int) -> T.ArrayType:
    """Wrap a filled buffer as the dtype's companion array. Bound on `NumericType`
    so `PrimitiveArray[T]` reduces to `T.ArrayType` (won't unify in the generic
    `execute` default otherwise)."""
    return PrimitiveArray[T](
        dtype=T(),
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buffer^,
    )


def _not_wired[T: DataType]() raises -> T.ArrayType:
    """Raising stub typed as `T.ArrayType` — lets a family/boundary `execute`
    default satisfy the return type before its execution is wired."""
    raise Error("execute: not wired for this node yet")


# ---------------------------------------------------------------------------
# Value — base trait; `execute` is the uniform verb
# ---------------------------------------------------------------------------


trait Value(Copyable, ImplicitlyDeletable, Movable, Writable):
    """Every expression node. `execute` returns the dtype's companion array.
    Copies are explicit (`.copy()`); nodes are not `ImplicitlyCopyable`."""

    comptime OutType: DataType

    # Abstract — implemented by the numeric family (fused vectorize) and by every
    # other concrete node (raising `_not_wired` stub) so a boxed `Value` in
    # `AnyValue` can `.execute(batch).to_any()` generically. Declared here (not
    # only per-family) so `AnyValue`'s trampoline can call it on any `V: Value`.
    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        ...

    def name(self) -> String:
        return String()

    def isnull(self) -> BoolUnary[IsNullKernel, Self]:
        """Null predicate — any value in any family yields a `BoolValue`."""
        return BoolUnary[IsNullKernel, Self](self.copy())

    def notnull(self) -> BoolUnary[NotNullKernel, Self]:
        """Non-null predicate — any value in any family yields a `BoolValue`."""
        return BoolUnary[NotNullKernel, Self](self.copy())


trait NumericValue(Value):
    """The numeric lane: refines `OutType` to `NumericType`, carries the `core[W]`
    SIMD fusion primitive + a fusing `execute`, and the arithmetic/comparison
    operator surface. Arithmetic nodes hook to the real kernels."""

    comptime OutType: NumericType
    comptime NativeType: DType

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        ...

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank],) -> None:
            var i = idx[0]
            buf.view[native](i, length).store[W](0, self.core[W](batch, i))

        @always_inline
        def lane[W: Int](i: Int):
            fill[W, rank=1](IndexList[1](i))

        vectorize[width](length, lane)
        return _numeric_array[Self.OutType](buf.to_immutable(), length)

    # --- arithmetic (fusable, real kernels) --------------------------------

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

    def __mod__[
        Rhs: NumericValue
    ](self, o: Rhs) -> NumericBinary[ModKernel, Self, Rhs]:
        return NumericBinary[ModKernel, Self, Rhs](self.copy(), o.copy())

    def __truediv__[
        Rhs: NumericValue
    ](self, o: Rhs) -> FloatBinary[DivKernel, Self, Rhs]:
        return FloatBinary[DivKernel, Self, Rhs](self.copy(), o.copy())

    def __pow__[
        Rhs: NumericValue
    ](self, o: Rhs) -> FloatBinary[PowKernel, Self, Rhs]:
        return FloatBinary[PowKernel, Self, Rhs](self.copy(), o.copy())

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

    def trunc(self) -> NumericUnary[TruncKernel, Self]:
        return NumericUnary[TruncKernel, Self](self.copy())

    # transcendental unary -> float64 (fused via the real kernels)
    def sqrt(self) -> FloatUnary[SqrtKernel, Self]:
        return FloatUnary[SqrtKernel, Self](self.copy())

    def exp(self) -> FloatUnary[ExpKernel, Self]:
        return FloatUnary[ExpKernel, Self](self.copy())

    def exp2(self) -> FloatUnary[Exp2Kernel, Self]:
        return FloatUnary[Exp2Kernel, Self](self.copy())

    def ln(self) -> FloatUnary[LogKernel, Self]:
        return FloatUnary[LogKernel, Self](self.copy())

    def log2(self) -> FloatUnary[Log2Kernel, Self]:
        return FloatUnary[Log2Kernel, Self](self.copy())

    def log10(self) -> FloatUnary[Log10Kernel, Self]:
        return FloatUnary[Log10Kernel, Self](self.copy())

    def log1p(self) -> FloatUnary[Log1pKernel, Self]:
        return FloatUnary[Log1pKernel, Self](self.copy())

    def sin(self) -> FloatUnary[SinKernel, Self]:
        return FloatUnary[SinKernel, Self](self.copy())

    def cos(self) -> FloatUnary[CosKernel, Self]:
        return FloatUnary[CosKernel, Self](self.copy())

    # numeric -> bool predicates (type-only until bool execution is wired)
    def isnan(self) -> BoolUnary[IsNanKernel, Self]:
        return BoolUnary[IsNanKernel, Self](self.copy())

    def isinf(self) -> BoolUnary[IsInfKernel, Self]:
        return BoolUnary[IsInfKernel, Self](self.copy())

    # --- reductions (N -> 1, boundary; non-lane `Value` result nodes) -------

    def sum(self) -> Sum[SumKernel, Self]:
        return Sum[SumKernel, Self](self.copy())

    def mean(self) -> Mean[MeanKernel, Self]:
        return Mean[MeanKernel, Self](self.copy())

    def min(self) -> MinMax[MinKernel, Self]:
        return MinMax[MinKernel, Self](self.copy())

    def max(self) -> MinMax[MaxKernel, Self]:
        return MinMax[MaxKernel, Self](self.copy())

    # --- comparisons (-> BoolValue) ----------------------------------------

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
    """Boolean-typed nodes: logical operator surface (type architecture)."""

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
    """String-typed nodes. Cross-family methods follow the *result*: `length()`
    yields a numeric boundary, `startswith()` a `BoolValue`."""

    def length(self) -> Counting[LengthKernel, Self]:
        return Counting[LengthKernel, Self](self.copy())

    def upper(self) -> StringUnary[UpperKernel, Self]:
        return StringUnary[UpperKernel, Self](self.copy())

    def lower(self) -> StringUnary[LowerKernel, Self]:
        return StringUnary[LowerKernel, Self](self.copy())

    def reverse(self) -> StringUnary[ReverseKernel, Self]:
        return StringUnary[ReverseKernel, Self](self.copy())

    def strip(self) -> StringUnary[StripKernel, Self]:
        return StringUnary[StripKernel, Self](self.copy())

    def lstrip(self) -> StringUnary[LStripKernel, Self]:
        return StringUnary[LStripKernel, Self](self.copy())

    def rstrip(self) -> StringUnary[RStripKernel, Self]:
        return StringUnary[RStripKernel, Self](self.copy())

    def capitalize(self) -> StringUnary[CapitalizeKernel, Self]:
        return StringUnary[CapitalizeKernel, Self](self.copy())

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
    """List-typed nodes (nested family). `length()` yields a numeric boundary,
    `contains()` a `BoolValue`."""

    def length(self) -> Counting[ArrayLengthKernel, Self]:
        return Counting[ArrayLengthKernel, Self](self.copy())

    def contains[
        E: Value
    ](self, elem: E) -> BoolBinary[ArrayContainsKernel, Self, E]:
        return BoolBinary[ArrayContainsKernel, Self, E](
            self.copy(), elem.copy()
        )


# ---------------------------------------------------------------------------
# Numeric lane nodes — real kernels + fused `core`
# ---------------------------------------------------------------------------


@fieldwise_init
struct NumericBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Arithmetic binary widening to the higher-precedence operand — `Add`, `Sub`,
    `Mul`, `Mod`. Operands cast to the promoted `NativeType`, then `K.core`."""

    comptime OutType = highest_precedence[Self.L, Self.R]
    comptime NativeType = Self.OutType.native

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx).cast[Self.NativeType]()
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct FloatBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Binary op whose result is always float64 — `Div`, `Pow`."""

    comptime OutType = dt.Float64Type
    comptime NativeType = DType.float64

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx).cast[Self.NativeType]()
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct NumericUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary numeric op preserving the operand dtype — `Neg`, `Abs`, `Ceil`, ….
    """

    comptime OutType = Self.A.OutType
    comptime NativeType = Self.A.NativeType

    var arg: Self.A

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return Self.K.core[Self.NativeType, W](self.arg.core[W](batch, idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct FloatUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary op whose result is always float64 — `sqrt`, `exp`, `ln`."""

    comptime OutType = dt.Float64Type
    comptime NativeType = DType.float64

    var arg: Self.A

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var a = self.arg.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](a)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


# ---------------------------------------------------------------------------
# Boundary nodes — numeric-producing but non-lane (`Value`); materialize (future)
# ---------------------------------------------------------------------------


@fieldwise_init
struct Counting[K: Kernel, A: Value](Value):
    """Unary op whose result is int32 — `length()` (string/list element count).
    A boundary: its operand is not numeric, so it materializes rather than fuses.
    """

    comptime OutType = dt.Int32Type
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct Sum[K: Kernel, A: Value](Value):
    """Reduction widening to 64-bit — `sum()`."""

    comptime OutType = sum_result[Self.A]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct Mean[K: Kernel, A: Value](Value):
    """Reduction to float64 — `mean()`."""

    comptime OutType = dt.Float64Type
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct MinMax[K: Kernel, A: Value](Value):
    """Reduction preserving the operand dtype — `min()`, `max()`."""

    comptime OutType = dtype_like[Self.A, Self.A]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


# ---------------------------------------------------------------------------
# Type-only nodes — bool / string families (execution is future work)
# ---------------------------------------------------------------------------


@fieldwise_init
struct BoolBinary[K: Kernel, L: Value, R: Value](BoolValue):
    comptime OutType = dt.BoolType
    var left: Self.L
    var right: Self.R

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct BoolUnary[K: Kernel, A: Value](BoolValue):
    comptime OutType = dt.BoolType
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct StringBinary[K: Kernel, L: Value, R: Value](StringValue):
    comptime OutType = dtype_like[Self.L, Self.R]
    var left: Self.L
    var right: Self.R

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


@fieldwise_init
struct StringUnary[K: Kernel, A: Value](StringValue):
    comptime OutType = dtype_like[Self.A, Self.A]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Div = FloatBinary[DivKernel, _, _]
comptime Pow = FloatBinary[PowKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Ceil = NumericUnary[CeilKernel, _]
comptime Floor = NumericUnary[FloorKernel, _]
comptime Round = NumericUnary[RoundKernel, _]
comptime Sign = NumericUnary[SignKernel, _]
comptime Trunc = NumericUnary[TruncKernel, _]
comptime Sqrt = FloatUnary[SqrtKernel, _]
comptime Exp = FloatUnary[ExpKernel, _]
comptime Exp2 = FloatUnary[Exp2Kernel, _]
comptime Ln = FloatUnary[LogKernel, _]
comptime Log2 = FloatUnary[Log2Kernel, _]
comptime Log10 = FloatUnary[Log10Kernel, _]
comptime Log1p = FloatUnary[Log1pKernel, _]
comptime Sin = FloatUnary[SinKernel, _]
comptime Cos = FloatUnary[CosKernel, _]

comptime Min = MinMax[MinKernel, _]
comptime Max = MinMax[MaxKernel, _]

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
comptime NotNull = BoolUnary[NotNullKernel, _]
comptime IsNan = BoolUnary[IsNanKernel, _]
comptime IsInf = BoolUnary[IsInfKernel, _]

comptime Length = Counting[LengthKernel, _]
comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]

comptime ArrayLength = Counting[ArrayLengthKernel, _]
comptime ArrayContains = BoolBinary[ArrayContainsKernel, _, _]


# ---------------------------------------------------------------------------
# Leaves — dedicated per-family (single-family → unconditional core/NativeType)
# ---------------------------------------------------------------------------


struct NumericColumn[T: NumericType](NumericValue):
    """A named numeric column, resolved by name against `batch.schema` per pass.
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def name(self) -> String:
        return self._name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct NumericLiteral[T: NumericType](NumericValue):
    """A numeric constant, broadcast into every lane."""

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var _value: Scalar[Self.NativeType]

    def __init__(out self, value: Int):
        self._value = Scalar[Self.NativeType](value)

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return SIMD[Self.NativeType, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct StringColumn[T: StringLikeType](StringValue):
    """A named string column (type architecture; execution is future work)."""

    comptime OutType = Self.T

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct StringLiteral[T: StringLikeType](StringValue):
    """A string constant leaf holding a `StringScalar`."""

    comptime OutType = Self.T

    var _value: StringScalar

    def __init__(out self, var value: String):
        self._value = StringScalar(value^)

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct ListColumn[T: DataType & ListLikeType](ListValue):
    """A named list column (nested family; execution is future work)."""

    comptime OutType = Self.T

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        return _not_wired[Self.OutType]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


# ---------------------------------------------------------------------------
# col / lit — overload by dtype family
# ---------------------------------------------------------------------------


def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column — `col("a", int64)`."""
    return NumericColumn[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """Reference a string column — `col("s", string)`."""
    return StringColumn[T](name^)


def col[
    T: DataType & ListLikeType
](var name: String, dtype: T) -> ListColumn[T]:
    """Reference a list column — `col("l", list_(int64))`."""
    return ListColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """A numeric constant — `lit(2, int64)`."""
    return NumericLiteral[T](value)


def lit[T: StringLikeType](var value: String, dtype: T) -> StringLiteral[T]:
    """A string constant — `lit("x", string)`."""
    return StringLiteral[T](value^)


# ---------------------------------------------------------------------------
# AnyValue — type-erased handle: box any `Value`, then `.execute(batch)` it
# ---------------------------------------------------------------------------


struct AnyValue(Copyable, Movable, Writable):
    """Type-erased handle over any expression node — the one box that lets
    runtime-typed / heterogeneous code hold a value regardless of its family and
    still `.execute(batch)` it. Erasure is via per-boxed-type trampolines that
    `rebind` the node back and forward to its `Value` methods; every typed result
    array converts to `AnyArray` via `.to_any()`."""

    var _boxed: ArcPointer[NoneType]
    var _execute: def(ArcPointer[NoneType], RecordBatch) thin raises -> AnyArray
    var _name_fn: def(ArcPointer[NoneType]) thin -> String
    var _write_fn: def(ArcPointer[NoneType]) thin -> String

    @staticmethod
    def _execute_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
        return rebind[ArcPointer[V]](ptr)[].execute(batch).to_any()

    @staticmethod
    def _name_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[V]](ptr)[].name()

    @staticmethod
    def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[V]](ptr)[].write_to(s)
        return s^

    @implicit
    def __init__[V: Value](out self, value: V):
        """Box any `Value` — a column, a fused numeric expression, a boundary.
        """
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._execute = Self._execute_tramp[V]
        self._name_fn = Self._name_tramp[V]
        self._write_fn = Self._write_tramp[V]

    def execute(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate the boxed node against `batch`, erased to `AnyArray`."""
        return self._execute(self._boxed, batch)

    def name(self) -> String:
        return self._name_fn(self._boxed)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write_fn(self._boxed))
