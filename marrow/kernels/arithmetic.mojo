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
from ..dtypes import PrimitiveType
from .helpers import (
    Kernel,
    bitmap_and,
    binary_array_dispatch,
    binary_float_dispatch,
    unary_numeric_dispatch,
    unary_float_dispatch,
)
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Generic kernel wrappers — buffer allocation + null propagation
# ---------------------------------------------------------------------------


def _unary[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W]) thin -> SIMD[T.native, W],
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Unary kernel: allocates output, resolves views, calls elementwise."""
    comptime native = T.native
    var length = len(array)
    var buf: Buffer[mut=True]
    if ctx.is_gpu():
        buf = Buffer.alloc_device[native](ctx.device.value(), length)
    else:
        buf = Buffer.alloc_zeroed[native](length)
    apply[native, native, func](
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


def _binary[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) thin -> SIMD[
        T.native, W
    ],
    name: String = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Binary kernel: allocates output, resolves views, calls elementwise."""
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
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
    apply[native, native, func](
        left.values(),
        right.values(),
        buf.view[native](0, length),
        ctx,
    )
    return PrimitiveArray[T](
        dtype=left.dtype.copy(),
        length=length,
        nulls=length - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=buf.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Kernel traits — 3-tier interface
# ---------------------------------------------------------------------------


trait BinaryKernel(Kernel):
    """Element-wise binary kernel on numeric arrays.

    Concrete structs define ``comptime name`` and ``core``; ``apply`` and
    ``dispatch`` have default implementations.
    """

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        return _binary[T, func=Self.core[T.native, _], name=Self.name](left, right, ctx)

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return binary_array_dispatch[Self.name, Self.apply[_]](left, right, ctx)


trait UnaryKernel(Kernel):
    """Element-wise unary kernel on numeric arrays.

    Concrete structs define ``comptime name`` and ``core``; ``apply`` and
    ``dispatch`` have default implementations.
    """

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=Self.core[T.native, _]](array, ctx)

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return unary_numeric_dispatch[Self.name, Self.apply[_]](array, ctx)


trait BinaryFloatKernel(Kernel):
    """Element-wise binary kernel on floating-point arrays.

    ``core`` is not in the trait (Mojo doesn't support ``where`` on trait
    methods). Concrete structs define ``core`` with ``where T.is_floating_point()``
    and ``apply`` explicitly; ``dispatch`` has a default.
    """

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext,
    ) raises -> PrimitiveArray[T]: ...

    @staticmethod
    def dispatch(
        left: AnyArray,
        right: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return binary_float_dispatch[Self.name, Self.apply[_]](left, right, ctx)


trait UnaryFloatKernel(Kernel):
    """Element-wise unary kernel on floating-point arrays.

    ``core`` is not in the trait (Mojo doesn't support ``where`` on trait
    methods). Concrete structs define ``core`` with ``where T.is_floating_point()``
    and ``apply`` explicitly; ``dispatch`` has a default.
    """

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T],
        ctx: ExecutionContext,
    ) raises -> PrimitiveArray[T]: ...

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyArray:
        return unary_float_dispatch[Self.name, Self.apply[_]](array, ctx)


# ---------------------------------------------------------------------------
# Kernel structs — BinaryKernel
# ---------------------------------------------------------------------------


struct AddKernel(BinaryKernel):
    comptime name = "add"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct SubKernel(BinaryKernel):
    comptime name = "subtract"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a - b


struct MulKernel(BinaryKernel):
    comptime name = "multiply"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b


struct DivKernel(BinaryKernel):
    comptime name = "divide"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        # Replace zeros with 1 to avoid SIGFPE; null positions are masked by bitmap.
        return a / b.eq(0).select(SIMD[T, W](1), b)


struct FloordivKernel(BinaryKernel):
    comptime name = "floordiv"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a // b.eq(0).select(SIMD[T, W](1), b)


struct ModKernel(BinaryKernel):
    comptime name = "mod"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a % b.eq(0).select(SIMD[T, W](1), b)


struct MinKernel(BinaryKernel):
    comptime name = "min_element_wise"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxKernel(BinaryKernel):
    comptime name = "max_element_wise"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryKernel
# ---------------------------------------------------------------------------


struct NegKernel(UnaryKernel):
    comptime name = "neg"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__neg__()


struct AbsKernel(UnaryKernel):
    comptime name = "abs_"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__abs__()


struct SignKernel(UnaryKernel):
    comptime name = "sign"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.gt(SIMD[T, W](0)).cast[T]() - a.lt(SIMD[T, W](0)).cast[T]()


struct FloorKernel(UnaryKernel):
    comptime name = "floor"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__floor__()


struct CeilKernel(UnaryKernel):
    comptime name = "ceil"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__ceil__()


struct TruncKernel(UnaryKernel):
    comptime name = "trunc"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__trunc__()


struct RoundKernel(UnaryKernel):
    comptime name = "round"

    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return a.__round__()


# ---------------------------------------------------------------------------
# Kernel structs — BinaryFloatKernel
# ---------------------------------------------------------------------------


struct PowKernel(BinaryFloatKernel):
    comptime name = "pow_"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.pow(a, b)

    @staticmethod
    def apply[T: PrimitiveType](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveArray[T]:
        return _binary[T, func=PowKernel.core[T.native, _], name="pow_"](left, right, ctx)


# ---------------------------------------------------------------------------
# Kernel structs — UnaryFloatKernel
# ---------------------------------------------------------------------------


struct SqrtKernel(UnaryFloatKernel):
    comptime name = "sqrt"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.sqrt(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=SqrtKernel.core[T.native, _]](array, ctx)


struct ExpKernel(UnaryFloatKernel):
    comptime name = "exp"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.exp(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=ExpKernel.core[T.native, _]](array, ctx)


struct Exp2Kernel(UnaryFloatKernel):
    comptime name = "exp2"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.exp2(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=Exp2Kernel.core[T.native, _]](array, ctx)


struct LogKernel(UnaryFloatKernel):
    comptime name = "log"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.log(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=LogKernel.core[T.native, _]](array, ctx)


struct Log2Kernel(UnaryFloatKernel):
    comptime name = "log2"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.log2(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=Log2Kernel.core[T.native, _]](array, ctx)


struct Log10Kernel(UnaryFloatKernel):
    comptime name = "log10"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.log10(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=Log10Kernel.core[T.native, _]](array, ctx)


struct Log1pKernel(UnaryFloatKernel):
    comptime name = "log1p"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        # std.math.log1p internally upcasts to float64 in recent Mojo nightlies,
        # which is unsupported on Metal GPU. Use log(1 + a) instead.
        return math.log(a + 1)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=Log1pKernel.core[T.native, _]](array, ctx)


struct SinKernel(UnaryFloatKernel):
    comptime name = "sin"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.sin(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=SinKernel.core[T.native, _]](array, ctx)


struct CosKernel(UnaryFloatKernel):
    comptime name = "cos"

    @staticmethod
    def core[T: DType, W: Int](
        a: SIMD[T, W]
    ) -> SIMD[T, W] where T.is_floating_point():
        return math.cos(a)

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> PrimitiveArray[T]:
        return _unary[T, func=CosKernel.core[T.native, _]](array, ctx)


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
    T: PrimitiveType
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
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise square root."""
    return SqrtKernel.apply[T](array, ctx)


def exp[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural exponential (e^x)."""
    return ExpKernel.apply[T](array, ctx)


def exp2[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 exponential (2^x)."""
    return Exp2Kernel.apply[T](array, ctx)


def log[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural logarithm."""
    return LogKernel.apply[T](array, ctx)


def log2[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 logarithm."""
    return Log2Kernel.apply[T](array, ctx)


def log10[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-10 logarithm."""
    return Log10Kernel.apply[T](array, ctx)


def log1p[
    T: PrimitiveType
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
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise sine."""
    return SinKernel.apply[T](array, ctx)


def cos[
    T: PrimitiveType
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
