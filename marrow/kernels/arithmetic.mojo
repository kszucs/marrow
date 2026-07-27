"""Element-wise arithmetic kernels — CPU SIMD and GPU via ``elementwise``.

Three tiers per operation:

- **Tier 0 (core)** — ``KernelStruct.core[T: DType, W: Int]``: raw SIMD functor,
  no allocation; called directly by expression-node ``exec_core[W](idx)`` for
  kernel fusion.
- **Tier 1 (apply)** — ``KernelStruct.apply[T: PrimitiveType]``: allocates an
  output buffer, propagates null bitmaps, dispatches CPU/GPU via ``apply()``.
- **Tier 2 (dispatch)** — ``KernelStruct.dispatch(AnyArray)``: runtime-typed entry
  point; resolves the runtime dtype to the typed ``apply`` overload via
  `AnyDataType.dispatch_numeric` / `.dispatch_floating`.

Structural kernels (filter, sort, concat, …) operate on array layout rather than
element values and are **not** part of this tier scheme.
"""

import std.math as math

from ..arrays import PrimitiveArray, AnyArray
from ..buffers import Buffer, Bitmap
from ..views import apply
from ..dtypes import (
    PrimitiveType,
    NumericType,
    FloatingType,
)
from .core import Kernel
from .cast import cast
from .execution import ExecutionContext
from ..utils import GPU_ENABLED


# ---------------------------------------------------------------------------
# Kernel traits — 3-tier interface
# ---------------------------------------------------------------------------


trait BinaryKernel(Kernel):
    """Base for element-wise binary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        Self.expect_same_length(len(left), len(right))
        comptime native = T.native
        var length = len(left)
        var bm = Bitmap.intersect(left.bitmap.copy(), right.bitmap.copy())
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[native](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_zeroed[native](length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            left.values(), right.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=left.dtype.copy(),
            length=length,
            nulls=bm.value().unset_count() if bm else 0,
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
        # Compute in the promoted domain of BOTH operands, exactly as the fused
        # `NumericBinary` does — otherwise `int64 + float64` means one thing
        # fused and raises interpreted. `cast` is identity when the dtype
        # already matches, so the same-dtype case is untouched.
        var dt = left.dtype().promote(right.dtype())
        var l = cast(left, dt, ctx=ctx)
        var r = cast(right, dt, ctx=ctx)

        @parameter
        def leaf[T: NumericType](d: T) raises -> AnyArray:
            return Self.apply(
                l.as_primitive[T](), r.as_primitive[T](), ctx
            ).to_any()

        return dt.dispatch_numeric[leaf]()


trait BinaryFloatKernel(BinaryKernel):
    """Binary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        var dt = left.dtype().promote(right.dtype())
        var l = cast(left, dt, ctx=ctx)
        var r = cast(right, dt, ctx=ctx)

        @parameter
        def leaf[T: FloatingType](d: T) raises -> AnyArray:
            return Self.apply(
                l.as_primitive[T](), r.as_primitive[T](), ctx
            ).to_any()

        return dt.dispatch_floating[leaf]()


trait UnaryKernel(Kernel):
    """Base for element-wise unary kernels: ``core`` (abstract) + ``apply`` (default).

    Concrete structs define ``comptime name`` and ``core``; subtraits add ``dispatch``.
    """

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        comptime native = T.native
        var length = len(array)
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[native](ctx.device.value(), length)
            else:
                buf = Buffer.alloc_zeroed[native](length)
        else:
            buf = Buffer.alloc_zeroed[native](length)
        apply[native, native, Self.core[native, _]](
            array.values(), buf.view[native](0, length), ctx
        )
        return PrimitiveArray[T](
            dtype=array.dtype.copy(),
            length=length,
            nulls=array.null_count(),
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
        @parameter
        def leaf[T: NumericType](d: T) raises -> AnyArray:
            return Self.apply(array.as_primitive[T](), ctx).to_any()

        return array.dtype().dispatch_numeric[leaf]()


trait UnaryFloatKernel(UnaryKernel):
    """Unary kernel dispatching over floating-point dtypes only."""

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        @parameter
        def leaf[T: FloatingType](d: T) raises -> AnyArray:
            return Self.apply(array.as_primitive[T](), ctx).to_any()

        return array.dtype().dispatch_floating[leaf]()


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
