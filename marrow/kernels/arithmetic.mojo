"""Element-wise arithmetic kernels — CPU SIMD and GPU via ``elementwise``.

Each public function dispatches based on the optional `ctx` argument:
  - CPU (default): SIMD vectorization via ``elementwise``.
  - GPU (ctx provided): kernel dispatch via ``elementwise[target="gpu"]``.
"""

import std.math as math

from ..arrays import PrimitiveArray, AnyArray
from ..buffers import Buffer
from ..views import apply
from ..dtypes import PrimitiveType
from .helpers import (
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
    name: StringLiteral = "",
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
# SIMD helpers — shared by CPU and GPU paths
# ---------------------------------------------------------------------------

# Binary


def _add[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


def _sub[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a - b


def _mul[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a * b


def _div[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    # Replace zeros with 1 to avoid SIGFPE; null positions are masked by bitmap.
    return a / b.eq(0).select(SIMD[T, W](1), b)


def _floordiv[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a // b.eq(0).select(SIMD[T, W](1), b)


def _mod[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a % b.eq(0).select(SIMD[T, W](1), b)


def _min[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.min(a, b)


def _max[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.max(a, b)


# Unary


def _neg_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__neg__()


def _abs_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__abs__()


def _sign_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.gt(SIMD[T, W](0)).cast[T]() - a.lt(SIMD[T, W](0)).cast[T]()


# Float-only unary


def _sqrt_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.sqrt(a)


def _exp_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.exp(a)


def _exp2_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.exp2(a)


def _log_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log(a)


def _log2_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log2(a)


def _log10_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log10(a)


def _log1p_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    # std.math.log1p internally upcasts to float64 in recent Mojo nightlies,
    # which is unsupported on Metal GPU. Use log(1 + a) instead.
    return math.log(a + 1)


def _floor_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__floor__()


def _ceil_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__ceil__()


def _trunc_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__trunc__()


def _round_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.__round__()


def _sin_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.sin(a)


def _cos_fn[
    T: DType, W: Int
](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.cos(a)


def _pow_fn[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.pow(a, b)


# ---------------------------------------------------------------------------
# Public API — binary kernels
# ---------------------------------------------------------------------------


def add[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise addition."""
    return _binary[T, func=_add[T.native, _], name="add"](left, right, ctx)


def subtract[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise subtraction."""
    return _binary[T, func=_sub[T.native, _], name="subtract"](left, right, ctx)


def multiply[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise multiplication."""
    return _binary[T, func=_mul[T.native, _], name="multiply"](left, right, ctx)


def divide[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise true division."""
    return _binary[T, func=_div[T.native, _], name="divide"](left, right, ctx)


def floordiv[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise floor division."""
    return _binary[T, func=_floordiv[T.native, _], name="floordiv"](
        left, right, ctx
    )


def mod[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise modulo."""
    return _binary[T, func=_mod[T.native, _], name="mod"](left, right, ctx)


def min_element_wise[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise minimum."""
    return _binary[T, func=_min[T.native, _], name="min_element_wise"](
        left, right, ctx
    )


def max_element_wise[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise maximum."""
    return _binary[T, func=_max[T.native, _], name="max_element_wise"](
        left, right, ctx
    )


def pow_[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise power: result[i] = left[i] ** right[i]."""
    comptime assert (
        T.native.is_floating_point()
    ), "pow_ requires a floating-point type"
    return _binary[T, func=_pow_fn[T.native, _], name="pow_"](left, right, ctx)


# ---------------------------------------------------------------------------
# Public API — unary kernels
# ---------------------------------------------------------------------------


def neg[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise negation."""
    return _unary[T, _neg_fn[T.native, _]](array, ctx)


def abs_[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise absolute value."""
    return _unary[T, _abs_fn[T.native, _]](array, ctx)


def sign[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise sign: -1, 0, or 1."""
    return _unary[T, _sign_fn[T.native, _]](array, ctx)


def sqrt[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise square root."""
    comptime assert (
        T.native.is_floating_point()
    ), "sqrt requires a floating-point type"
    return _unary[T, _sqrt_fn[T.native, _]](array, ctx)


def exp[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural exponential (e^x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "exp requires a floating-point type"
    return _unary[T, _exp_fn[T.native, _]](array, ctx)


def exp2[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 exponential (2^x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "exp2 requires a floating-point type"
    return _unary[T, _exp2_fn[T.native, _]](array, ctx)


def log[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise natural logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log requires a floating-point type"
    return _unary[T, _log_fn[T.native, _]](array, ctx)


def log2[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-2 logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log2 requires a floating-point type"
    return _unary[T, _log2_fn[T.native, _]](array, ctx)


def log10[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise base-10 logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log10 requires a floating-point type"
    return _unary[T, _log10_fn[T.native, _]](array, ctx)


def log1p[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise log(1 + x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "log1p requires a floating-point type"
    return _unary[T, _log1p_fn[T.native, _]](array, ctx)


def floor[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise floor."""
    return _unary[T, _floor_fn[T.native, _]](array, ctx)


def ceil[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise ceiling."""
    return _unary[T, _ceil_fn[T.native, _]](array, ctx)


def trunc[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise truncation toward zero."""
    return _unary[T, _trunc_fn[T.native, _]](array, ctx)


def round[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise rounding to nearest integer."""
    return _unary[T, _round_fn[T.native, _]](array, ctx)


def sin[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise sine."""
    comptime assert (
        T.native.is_floating_point()
    ), "sin requires a floating-point type"
    return _unary[T, _sin_fn[T.native, _]](array, ctx)


def cos[
    T: PrimitiveType
](
    array: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Element-wise cosine."""
    comptime assert (
        T.native.is_floating_point()
    ), "cos requires a floating-point type"
    return _unary[T, _cos_fn[T.native, _]](array, ctx)


# ---------------------------------------------------------------------------
# Runtime dispatch — AnyArray-typed overloads
# ---------------------------------------------------------------------------


def add(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed add."""
    return binary_array_dispatch["add", add[_]](left, right, ctx)


def subtract(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed subtract."""
    return binary_array_dispatch["subtract", subtract[_]](left, right, ctx)


def multiply(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed multiply."""
    return binary_array_dispatch["multiply", multiply[_]](left, right, ctx)


def divide(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed divide."""
    return binary_array_dispatch["divide", divide[_]](left, right, ctx)


def floordiv(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed floordiv."""
    return binary_array_dispatch["floordiv", floordiv[_]](left, right, ctx)


def mod(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed mod."""
    return binary_array_dispatch["mod", mod[_]](left, right, ctx)


def min_element_wise(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed min_element_wise."""
    return binary_array_dispatch["min_element_wise", min_element_wise[_]](
        left, right, ctx
    )


def max_element_wise(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed max_element_wise."""
    return binary_array_dispatch["max_element_wise", max_element_wise[_]](
        left, right, ctx
    )


def neg(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed neg."""
    return unary_numeric_dispatch["neg", neg[_]](array, ctx)


def abs_(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed abs_."""
    return unary_numeric_dispatch["abs_", abs_[_]](array, ctx)


def sign(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sign."""
    return unary_numeric_dispatch["sign", sign[_]](array, ctx)


def pow_(
    left: AnyArray,
    right: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed pow_."""
    return binary_float_dispatch["pow_", pow_[_]](left, right, ctx)


def sqrt(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sqrt."""
    return unary_float_dispatch["sqrt", sqrt[_]](array, ctx)


def exp(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed exp."""
    return unary_float_dispatch["exp", exp[_]](array, ctx)


def exp2(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed exp2."""
    return unary_float_dispatch["exp2", exp2[_]](array, ctx)


def log(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log."""
    return unary_float_dispatch["log", log[_]](array, ctx)


def log2(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log2."""
    return unary_float_dispatch["log2", log2[_]](array, ctx)


def log10(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log10."""
    return unary_float_dispatch["log10", log10[_]](array, ctx)


def log1p(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed log1p."""
    return unary_float_dispatch["log1p", log1p[_]](array, ctx)


def floor(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed floor."""
    return unary_numeric_dispatch["floor", floor[_]](array, ctx)


def ceil(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed ceil."""
    return unary_numeric_dispatch["ceil", ceil[_]](array, ctx)


def trunc(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed trunc."""
    return unary_numeric_dispatch["trunc", trunc[_]](array, ctx)


def round(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed round."""
    return unary_numeric_dispatch["round", round[_]](array, ctx)


def sin(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed sin."""
    return unary_float_dispatch["sin", sin[_]](array, ctx)


def cos(
    array: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Runtime-typed cos."""
    return unary_float_dispatch["cos", cos[_]](array, ctx)
