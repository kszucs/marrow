"""Element-wise arithmetic kernels — CPU SIMD and GPU via ``elementwise``.

Three tiers per operation:

- **Tier 0 (core)** — ``KernelStruct.core[T: DType, W: Int]``: raw SIMD functor,
  no allocation; called directly by expression-node ``exec_core[W](idx)`` for
  kernel fusion.
- **Tier 1 (apply)** — ``KernelStruct.apply[T: PrimitiveType]``: allocates an
  output buffer, propagates null bitmaps, dispatches CPU/GPU via ``apply()``.
  Also exported as a standalone typed function for convenience.
- **Tier 2 (dispatch)** — ``KernelStruct.dispatch(AnyArray)``: runtime-typed entry
  point; dispatches to the typed ``apply`` overload via the dtype switch in
  ``helpers.py``. Also exported as a standalone function.

Structural kernels (filter, sort, concat, …) operate on array layout rather than
element values and are **not** part of this tier scheme.
"""

import std.math as math

from ..arrays import PrimitiveArray, AnyArray
from ..buffers import Buffer
from ..views import apply
from ..dtypes import (
    PrimitiveType,
    FloatingType,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
)
from .helpers import Kernel, bitmap_and
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Kernel traits — 3-tier interface
# ---------------------------------------------------------------------------


trait BinaryKernel(Kernel):
    """Base for element-wise binary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        if len(left) != len(right):
            raise Error(
                t"{Self.name} arrays must have the same length, got {len(left)} and"
                t" {len(right)}"
            )
        comptime native = T.native
        var length = len(left)
        var bm = bitmap_and(left.bitmap, right.bitmap)
        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[native](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            left.values(), right.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=left.dtype.copy(),
            length=length,
            nulls=length - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=buf.to_immutable(),
        )


trait BinaryNumericKernel(BinaryKernel):
    """Binary kernel dispatching over all numeric dtypes."""

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if left.dtype() != right.dtype():
            raise Error(
                t"{Self.name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
            )
        if left.dtype() == int8:
            return Self.apply(left.as_int8(), right.as_int8(), ctx).to_any()
        elif left.dtype() == int16:
            return Self.apply(left.as_int16(), right.as_int16(), ctx).to_any()
        elif left.dtype() == int32:
            return Self.apply(left.as_int32(), right.as_int32(), ctx).to_any()
        elif left.dtype() == int64:
            return Self.apply(left.as_int64(), right.as_int64(), ctx).to_any()
        elif left.dtype() == uint8:
            return Self.apply(left.as_uint8(), right.as_uint8(), ctx).to_any()
        elif left.dtype() == uint16:
            return Self.apply(left.as_uint16(), right.as_uint16(), ctx).to_any()
        elif left.dtype() == uint32:
            return Self.apply(left.as_uint32(), right.as_uint32(), ctx).to_any()
        elif left.dtype() == uint64:
            return Self.apply(left.as_uint64(), right.as_uint64(), ctx).to_any()
        elif left.dtype() == float16:
            return Self.apply(left.as_float16(), right.as_float16(), ctx).to_any()
        elif left.dtype() == float32:
            return Self.apply(left.as_float32(), right.as_float32(), ctx).to_any()
        elif left.dtype() == float64:
            return Self.apply(left.as_float64(), right.as_float64(), ctx).to_any()
        raise Error(t"{Self.name}: unsupported dtype {left.dtype()}")


trait BinaryFloatKernel(BinaryKernel):
    """Binary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if left.dtype() != right.dtype():
            raise Error(
                t"{Self.name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
            )
        if left.dtype() == float16:
            return Self.apply(left.as_float16(), right.as_float16(), ctx).to_any()
        elif left.dtype() == float32:
            return Self.apply(left.as_float32(), right.as_float32(), ctx).to_any()
        elif left.dtype() == float64:
            return Self.apply(left.as_float64(), right.as_float64(), ctx).to_any()
        raise Error(
            t"{Self.name}: unsupported dtype {left.dtype()}, expected float type"
        )


trait UnaryKernel(Kernel):
    """Base for element-wise unary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        comptime native = T.native
        var length = len(array)
        var buf: Buffer[mut=True]
        if ctx.is_gpu():
            buf = Buffer.alloc_device[native](ctx.device.value(), length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            array.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=array.dtype.copy(),
            length=length,
            nulls=length
            - array.bitmap.value().view().count_set_bits() if array.bitmap else 0,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )


trait UnaryNumericKernel(UnaryKernel):
    """Unary kernel dispatching over all numeric dtypes."""

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if array.dtype() == int8:
            return Self.apply(array.as_int8(), ctx).to_any()
        elif array.dtype() == int16:
            return Self.apply(array.as_int16(), ctx).to_any()
        elif array.dtype() == int32:
            return Self.apply(array.as_int32(), ctx).to_any()
        elif array.dtype() == int64:
            return Self.apply(array.as_int64(), ctx).to_any()
        elif array.dtype() == uint8:
            return Self.apply(array.as_uint8(), ctx).to_any()
        elif array.dtype() == uint16:
            return Self.apply(array.as_uint16(), ctx).to_any()
        elif array.dtype() == uint32:
            return Self.apply(array.as_uint32(), ctx).to_any()
        elif array.dtype() == uint64:
            return Self.apply(array.as_uint64(), ctx).to_any()
        elif array.dtype() == float16:
            return Self.apply(array.as_float16(), ctx).to_any()
        elif array.dtype() == float32:
            return Self.apply(array.as_float32(), ctx).to_any()
        elif array.dtype() == float64:
            return Self.apply(array.as_float64(), ctx).to_any()
        raise Error(t"{Self.name}: unsupported dtype {array.dtype()}")


trait UnaryFloatKernel(UnaryKernel):
    """Unary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        if array.dtype() == float16:
            return Self.apply(array.as_float16(), ctx).to_any()
        elif array.dtype() == float32:
            return Self.apply(array.as_float32(), ctx).to_any()
        elif array.dtype() == float64:
            return Self.apply(array.as_float64(), ctx).to_any()
        raise Error(
            t"{Self.name}: unsupported dtype {array.dtype()}, expected float type"
        )



# ---------------------------------------------------------------------------
# Kernel structs — BinaryKernel
# ---------------------------------------------------------------------------


struct AddKernel(BinaryNumericKernel):
    comptime name = "add"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct SubKernel(BinaryNumericKernel):
    comptime name = "subtract"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a - b


struct MulKernel(BinaryNumericKernel):
    comptime name = "multiply"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b


struct DivKernel(BinaryNumericKernel):
    comptime name = "divide"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        # Replace zeros with 1 to avoid SIGFPE; null positions are masked by bitmap.
        return a / b.eq(0).select(SIMD[T, W](1), b)


struct FloordivKernel(BinaryNumericKernel):
    comptime name = "floordiv"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a // b.eq(0).select(SIMD[T, W](1), b)


struct ModKernel(BinaryNumericKernel):
    comptime name = "mod"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a % b.eq(0).select(SIMD[T, W](1), b)


struct MinKernel(BinaryNumericKernel):
    comptime name = "min_element_wise"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxKernel(BinaryNumericKernel):
    comptime name = "max_element_wise"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryKernel
# ---------------------------------------------------------------------------


struct NegKernel(UnaryNumericKernel):
    comptime name = "neg"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__neg__()


struct AbsKernel(UnaryNumericKernel):
    comptime name = "abs_"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__abs__()


struct SignKernel(UnaryNumericKernel):
    comptime name = "sign"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.gt(SIMD[T, W](0)).cast[T]() - a.lt(SIMD[T, W](0)).cast[T]()


struct FloorKernel(UnaryNumericKernel):
    comptime name = "floor"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__floor__()


struct CeilKernel(UnaryNumericKernel):
    comptime name = "ceil"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__ceil__()


struct TruncKernel(UnaryNumericKernel):
    comptime name = "trunc"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__trunc__()


struct RoundKernel(UnaryNumericKernel):
    comptime name = "round"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__round__()


# ---------------------------------------------------------------------------
# Kernel structs — BinaryFloatKernel
# ---------------------------------------------------------------------------


struct PowKernel(BinaryFloatKernel):
    comptime name = "pow_"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.pow(a, b)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryFloatKernel
# ---------------------------------------------------------------------------


struct SqrtKernel(UnaryFloatKernel):
    comptime name = "sqrt"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return math.sqrt(a)


struct ExpKernel(UnaryFloatKernel):
    comptime name = "exp"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.exp(a)


struct Exp2Kernel(UnaryFloatKernel):
    comptime name = "exp2"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.exp2(a)


struct LogKernel(UnaryFloatKernel):
    comptime name = "log"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log(a)


struct Log2Kernel(UnaryFloatKernel):
    comptime name = "log2"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log2(a)


struct Log10Kernel(UnaryFloatKernel):
    comptime name = "log10"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.log10(a)


struct Log1pKernel(UnaryFloatKernel):
    comptime name = "log1p"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        # std.math.log1p upcasts to float64 in recent nightlies — use log(1+a) instead.
        comptime assert T.is_floating_point()
        return math.log(a + 1)


struct SinKernel(UnaryFloatKernel):
    comptime name = "sin"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.sin(a)


struct CosKernel(UnaryFloatKernel):
    comptime name = "cos"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_floating_point()
        return math.cos(a)


# ---------------------------------------------------------------------------
# Public API — typed thin wrappers
# ---------------------------------------------------------------------------


def add[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise addition."""
    return AddKernel.apply[T](left, right, ctx)


def subtract[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise subtraction."""
    return SubKernel.apply[T](left, right, ctx)


def multiply[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise multiplication."""
    return MulKernel.apply[T](left, right, ctx)


def divide[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise true division."""
    return DivKernel.apply[T](left, right, ctx)


def floordiv[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise floor division."""
    return FloordivKernel.apply[T](left, right, ctx)


def mod[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise modulo."""
    return ModKernel.apply[T](left, right, ctx)


def min_element_wise[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise minimum."""
    return MinKernel.apply[T](left, right, ctx)


def max_element_wise[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise maximum."""
    return MaxKernel.apply[T](left, right, ctx)


def pow_[
    T: FloatingType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise power: result[i] = left[i] ** right[i]."""
    return PowKernel.apply[T](left, right, ctx)


def neg[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise negation."""
    return NegKernel.apply[T](array, ctx)


def abs_[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise absolute value."""
    return AbsKernel.apply[T](array, ctx)


def sign[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise sign: -1, 0, or 1."""
    return SignKernel.apply[T](array, ctx)


def sqrt[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise square root."""
    return SqrtKernel.apply[T](array, ctx)


def exp[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural exponential (e^x)."""
    return ExpKernel.apply[T](array, ctx)


def exp2[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 exponential (2^x)."""
    return Exp2Kernel.apply[T](array, ctx)


def log[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural logarithm."""
    return LogKernel.apply[T](array, ctx)


def log2[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 logarithm."""
    return Log2Kernel.apply[T](array, ctx)


def log10[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-10 logarithm."""
    return Log10Kernel.apply[T](array, ctx)


def log1p[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise log(1 + x)."""
    return Log1pKernel.apply[T](array, ctx)


def floor[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise floor."""
    return FloorKernel.apply[T](array, ctx)


def ceil[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise ceiling."""
    return CeilKernel.apply[T](array, ctx)


def trunc[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise truncation toward zero."""
    return TruncKernel.apply[T](array, ctx)


def round[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise rounding to nearest integer."""
    return RoundKernel.apply[T](array, ctx)


def sin[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise sine."""
    return SinKernel.apply[T](array, ctx)


def cos[
    T: FloatingType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise cosine."""
    return CosKernel.apply[T](array, ctx)


# ---------------------------------------------------------------------------
# Runtime dispatch — AnyArray-typed thin wrappers
# ---------------------------------------------------------------------------


def add(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed add."""
    return AddKernel.dispatch(left, right, ctx)


def subtract(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed subtract."""
    return SubKernel.dispatch(left, right, ctx)


def multiply(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed multiply."""
    return MulKernel.dispatch(left, right, ctx)


def divide(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed divide."""
    return DivKernel.dispatch(left, right, ctx)


def floordiv(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed floordiv."""
    return FloordivKernel.dispatch(left, right, ctx)


def mod(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed mod."""
    return ModKernel.dispatch(left, right, ctx)


def min_element_wise(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed min_element_wise."""
    return MinKernel.dispatch(left, right, ctx)


def max_element_wise(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed max_element_wise."""
    return MaxKernel.dispatch(left, right, ctx)


def neg(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed neg."""
    return NegKernel.dispatch(array, ctx)


def abs_(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed abs_."""
    return AbsKernel.dispatch(array, ctx)


def sign(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sign."""
    return SignKernel.dispatch(array, ctx)


def pow_(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed pow_."""
    return PowKernel.dispatch(left, right, ctx)


def sqrt(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sqrt."""
    return SqrtKernel.dispatch(array, ctx)


def exp(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed exp."""
    return ExpKernel.dispatch(array, ctx)


def exp2(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed exp2."""
    return Exp2Kernel.dispatch(array, ctx)


def log(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log."""
    return LogKernel.dispatch(array, ctx)


def log2(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log2."""
    return Log2Kernel.dispatch(array, ctx)


def log10(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log10."""
    return Log10Kernel.dispatch(array, ctx)


def log1p(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log1p."""
    return Log1pKernel.dispatch(array, ctx)


def floor(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed floor."""
    return FloorKernel.dispatch(array, ctx)


def ceil(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed ceil."""
    return CeilKernel.dispatch(array, ctx)


def trunc(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed trunc."""
    return TruncKernel.dispatch(array, ctx)


def round(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed round."""
    return RoundKernel.dispatch(array, ctx)


def sin(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sin."""
    return SinKernel.dispatch(array, ctx)


def cos(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed cos."""
    return CosKernel.dispatch(array, ctx)
