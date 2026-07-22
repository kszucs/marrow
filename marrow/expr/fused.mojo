"""marrow.expr.fused — typed expressions with fused execution (approach B).

A clean trait/type hierarchy that *executes*, hooked directly to `marrow.kernels`
(no local kernel markers — the real kernels already supply `core[T, W]`):

- `Value.execute(batch)` is the uniform verb; it returns the dtype's companion
  `Self.OutType.ArrayType` (the dtype→array associated type on `DataType`).
- `NumericValue` **is** the numeric lane: it refines `OutType` to `NumericType`,
  adds the `core[W]` SIMD primitive, and its `execute` default vectorizes `core`
  across the whole expression tree in a **single fused pass** — zero intermediate
  arrays. Composite nodes call the kernel's `core` on their children's `core`, so
  the compiler inlines the entire chain.
- Promotion lives in the value nodes (`OutType`); compute lives in the kernel.

Dedicated per-family leaves (`NumericColumn` / `NumericLiteral`) keep `core` /
`NativeType` unconditional and the hierarchy sharp. `col` / `lit` build them.

This is the executing counterpart to the type-architecture in `marrow.expr.ibis`.
"""

from std.sys import size_of, bit_width_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList
from std.algorithm.backend.vectorize import vectorize
from std.builtin.simd import Scalar

from .. import dtypes as dt
from ..dtypes import DataType, NumericType, DType
from ..arrays import PrimitiveArray
from ..buffers import Buffer
from ..tabular import RecordBatch
from ..kernels.arithmetic import (
    BinaryKernel,
    UnaryKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    NegKernel,
    AbsKernel,
    SqrtKernel,
)


# ---------------------------------------------------------------------------
# Promotion rules (value layer) — same shape as marrow.expr.ibis, but keyed on
# NumericValue whose OutType is already a NumericType (no conforms_to guard).
# ---------------------------------------------------------------------------


def _rank[T: NumericType]() -> Int:
    """Bit-width rank; floats outrank same-width ints."""
    return bit_width_of[T.native]() + (
        1000 if T.native.is_floating_point() else 0
    )


comptime highest_precedence[L: NumericValue, R: NumericValue] = (
    L.OutType if _rank[L.OutType]() >= _rank[R.OutType]() else R.OutType
)
"""Widen to the higher-precedence operand — `Add(int32, int64) → int64`."""


def _numeric_array[
    T: NumericType
](var buffer: Buffer[mut=False], length: Int) -> T.ArrayType:
    """Wrap a filled buffer as the dtype's companion array. Bound on `NumericType`
    so `PrimitiveArray[T]` reduces to `T.ArrayType` (which won't unify in the
    generic `execute` default — same trick as `Literal`'s scalar helper)."""
    return PrimitiveArray[T](
        dtype=T(),
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buffer^,
    )


# ---------------------------------------------------------------------------
# Value / NumericValue traits
# ---------------------------------------------------------------------------


trait Value(Copyable, ImplicitlyDeletable, Movable, Writable):
    """Every expression node. `execute` returns the dtype's companion array."""

    comptime OutType: DataType

    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType:
        ...

    def name(self) -> String:
        return String()


trait NumericValue(Value):
    """Numeric lane: refines `OutType` to `NumericType`, carries the `core[W]`
    SIMD fusion primitive, and supplies a fusing `execute` default (all numeric
    nodes share it — only `core`/`OutType`/`NativeType` differ)."""

    comptime OutType: NumericType
    comptime NativeType: DType

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        """One SIMD lane at `idx`, resolving leaves from `batch`."""
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

    # --- arithmetic operator surface (fusable, real kernels) ---------------

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

    def __neg__(self) -> NumericUnary[NegKernel, Self]:
        return NumericUnary[NegKernel, Self](self.copy())

    def abs(self) -> NumericUnary[AbsKernel, Self]:
        return NumericUnary[AbsKernel, Self](self.copy())

    def sqrt(self) -> FloatUnary[SqrtKernel, Self]:
        return FloatUnary[SqrtKernel, Self](self.copy())


# ---------------------------------------------------------------------------
# Nodes — parameterized by the REAL kernels; promotion stays here
# ---------------------------------------------------------------------------


@fieldwise_init
struct NumericBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Widening binary op fused via `K.core`. Operands are cast to the promoted
    `NativeType` per lane, then combined — one SIMD op, no intermediate array.
    """

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
    """Binary op whose result is always float64 — `Div`. Operands cast to float.
    """

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
    """Unary numeric op preserving the operand dtype — `Neg`, `Abs`."""

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
    """Unary op whose result is always float64 — `sqrt`. Operand cast to float.
    """

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


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Div = FloatBinary[DivKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Sqrt = FloatUnary[SqrtKernel, _]


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


def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column — `col("a", int64)`."""
    return NumericColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """A numeric constant — `lit(2, int64)`."""
    return NumericLiteral[T](value)
