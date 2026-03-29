"""Element-wise arithmetic kernels — CPU SIMD and GPU via ``elementwise``.

Each public function dispatches based on the optional `ctx` argument:
  - CPU (default): SIMD vectorization via ``elementwise[use_blocking_impl=True]``.
  - GPU (ctx provided): kernel dispatch via ``elementwise[target="gpu"]``.

Pointers are passed as function parameters to ``_elementwise_binary`` /
``_elementwise_unary`` so that DevicePassable conversion works correctly
during GPU offload (closure captures of raw UnsafePointer don't transfer
to device; function parameters do).
"""

import std.math as math
from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext, get_gpu_target
from std.sys import size_of
from std.sys.info import simd_byte_width, simd_width_of
from std.utils.index import IndexList

from ..arrays import PrimitiveArray, AnyArray
from ..buffers import Buffer
from ..views import BitmapView, BufferView
from ..dtypes import DataType, numeric_dtypes, float_dtypes
from . import (
    bitmap_and,
    binary_array_dispatch,
    binary_float_dispatch,
    unary_numeric_dispatch,
    unary_float_dispatch,
)
from .helpers import has_accelerator_support


# ---------------------------------------------------------------------------
# Elementwise dispatch — pointers as params for GPU DevicePassable
# ---------------------------------------------------------------------------


def _elementwise_unary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W]) -> SIMD[T.native, W],
    out_origin: Origin[mut=True],
](
    output: BufferView[T.native, out_origin],
    input: BufferView[T.native, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a unary SIMD function element-wise via ``elementwise``."""

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        output.store[W](i, func[W](input.load[W](i)))

    if ctx:
        comptime if has_accelerator_support[T.native]():
            comptime gpu_width = simd_width_of[
                T.native, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("_elementwise_unary: type not supported on GPU")
    else:
        comptime cpu_width = simd_byte_width() // size_of[Scalar[T.native]]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


def _elementwise_binary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        T.native, W
    ],
    out_origin: Origin[mut=True],
](
    output: BufferView[T.native, out_origin],
    lhs: BufferView[T.native, _],
    rhs: BufferView[T.native, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply a binary SIMD function element-wise via ``elementwise``."""

    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        output.store[W](i, func[W](lhs.load[W](i), rhs.load[W](i)))

    if ctx:
        comptime if has_accelerator_support[T.native]():
            comptime gpu_width = simd_width_of[
                T.native, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("_elementwise_binary: type not supported on GPU")
    else:
        comptime cpu_width = simd_byte_width() // size_of[Scalar[T.native]]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


# ---------------------------------------------------------------------------
# Generic kernel wrappers — buffer allocation + null propagation
# ---------------------------------------------------------------------------


def _unary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W]) -> SIMD[T.native, W],
](
    array: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Unary kernel: allocates output, resolves views, calls elementwise."""
    comptime native = T.native
    var length = len(array)

    if ctx:
        var buf = Buffer.alloc_device[native](ctx.value(), length)
        _elementwise_unary[T, func](
            buf.device_view[native](),
            array.buffer.device_view[native](array.offset),
            length,
            ctx,
        )
        return PrimitiveArray[T](
            length=length,
            nulls=length
            - array.bitmap.value().view().count_set_bits() if array.bitmap else 0,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )
    else:
        var buf = Buffer.alloc_zeroed[native](length)
        _elementwise_unary[T, func](
            buf.view[native](),
            array.buffer.view[native](array.offset),
            length,
        )
        return PrimitiveArray[T](
            length=length,
            nulls=length
            - array.bitmap.value().view().count_set_bits() if array.bitmap else 0,
            offset=0,
            bitmap=array.bitmap,
            buffer=buf.to_immutable(),
        )


def _binary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        T.native, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
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

    if ctx:
        # FIXME: cannot use aligned pointers since the operands
        # may end up with different alignments; need to switch to
        # unaligned loads/stores
        var buf = Buffer.alloc_device[native](ctx.value(), length)
        _elementwise_binary[T, func](
            buf.device_view[native](),
            left.buffer.device_view[native](left.offset),
            right.buffer.device_view[native](right.offset),
            length,
            ctx,
        )
        return PrimitiveArray[T](
            length=length,
            nulls=length - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=buf.to_immutable(),
        )
    else:
        # FIXME: use alloc_uninit to spare the zeroing of the output buffer
        var buf = Buffer.alloc_zeroed[native](length)
        _elementwise_binary[T, func](
            buf.view[native](),
            left.buffer.view[native](left.offset),
            right.buffer.view[native](right.offset),
            length,
        )
        return PrimitiveArray[T](
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
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise addition."""
    return _binary[T, func=_add[T.native, _], name="add"](left, right, ctx)


def sub[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise subtraction."""
    return _binary[T, func=_sub[T.native, _], name="sub"](left, right, ctx)


def mul[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise multiplication."""
    return _binary[T, func=_mul[T.native, _], name="mul"](left, right, ctx)


def div[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise true division."""
    return _binary[T, func=_div[T.native, _], name="div"](left, right, ctx)


def floordiv[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise floor division."""
    return _binary[T, func=_floordiv[T.native, _], name="floordiv"](
        left, right, ctx
    )


def mod[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise modulo."""
    return _binary[T, func=_mod[T.native, _], name="mod"](left, right, ctx)


def min_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise minimum."""
    return _binary[T, func=_min[T.native, _], name="min_"](left, right, ctx)


def max_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise maximum."""
    return _binary[T, func=_max[T.native, _], name="max_"](left, right, ctx)


def pow_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[
    T
]:
    """Element-wise power: result[i] = left[i] ** right[i]."""
    comptime assert (
        T.native.is_floating_point()
    ), "pow_ requires a floating-point type"
    return _binary[T, func=_pow_fn[T.native, _], name="pow_"](left, right)


# ---------------------------------------------------------------------------
# Public API — unary kernels
# ---------------------------------------------------------------------------


def neg[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise negation."""
    return _unary[T, _neg_fn[T.native, _]](array)


def abs_[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise absolute value."""
    return _unary[T, _abs_fn[T.native, _]](array)


def sign[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise sign: -1, 0, or 1."""
    return _unary[T, _sign_fn[T.native, _]](array)


def sqrt[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise square root."""
    comptime assert (
        T.native.is_floating_point()
    ), "sqrt requires a floating-point type"
    return _unary[T, _sqrt_fn[T.native, _]](array)


def exp[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise natural exponential (e^x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "exp requires a floating-point type"
    return _unary[T, _exp_fn[T.native, _]](array)


def exp2[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise base-2 exponential (2^x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "exp2 requires a floating-point type"
    return _unary[T, _exp2_fn[T.native, _]](array)


def log[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise natural logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log requires a floating-point type"
    return _unary[T, _log_fn[T.native, _]](array)


def log2[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise base-2 logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log2 requires a floating-point type"
    return _unary[T, _log2_fn[T.native, _]](array)


def log10[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise base-10 logarithm."""
    comptime assert (
        T.native.is_floating_point()
    ), "log10 requires a floating-point type"
    return _unary[T, _log10_fn[T.native, _]](array)


def log1p[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise log(1 + x)."""
    comptime assert (
        T.native.is_floating_point()
    ), "log1p requires a floating-point type"
    return _unary[T, _log1p_fn[T.native, _]](array)


def floor[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise floor."""
    return _unary[T, _floor_fn[T.native, _]](array)


def ceil[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise ceiling."""
    return _unary[T, _ceil_fn[T.native, _]](array)


def trunc[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise truncation toward zero."""
    return _unary[T, _trunc_fn[T.native, _]](array)


def round[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise rounding to nearest integer."""
    return _unary[T, _round_fn[T.native, _]](array)


def sin[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise sine."""
    comptime assert (
        T.native.is_floating_point()
    ), "sin requires a floating-point type"
    return _unary[T, _sin_fn[T.native, _]](array)


def cos[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise cosine."""
    comptime assert (
        T.native.is_floating_point()
    ), "cos requires a floating-point type"
    return _unary[T, _cos_fn[T.native, _]](array)


# ---------------------------------------------------------------------------
# Runtime dispatch — AnyArray-typed overloads
# ---------------------------------------------------------------------------


def add(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed add."""
    return binary_array_dispatch["add", add[_]](left, right)


def sub(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed sub."""
    return binary_array_dispatch["sub", sub[_]](left, right)


def mul(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed mul."""
    return binary_array_dispatch["mul", mul[_]](left, right)


def div(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed div."""
    return binary_array_dispatch["div", div[_]](left, right)


def floordiv(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed floordiv."""
    return binary_array_dispatch["floordiv", floordiv[_]](left, right)


def mod(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed mod."""
    return binary_array_dispatch["mod", mod[_]](left, right)


def min_(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed min_."""
    return binary_array_dispatch["min_", min_[_]](left, right)


def max_(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed max_."""
    return binary_array_dispatch["max_", max_[_]](left, right)


def neg(array: AnyArray) raises -> AnyArray:
    """Runtime-typed neg."""
    return unary_numeric_dispatch["neg", neg[_]](array)


def abs_(array: AnyArray) raises -> AnyArray:
    """Runtime-typed abs_."""
    return unary_numeric_dispatch["abs_", abs_[_]](array)


def sign(array: AnyArray) raises -> AnyArray:
    """Runtime-typed sign."""
    return unary_numeric_dispatch["sign", sign[_]](array)


def pow_(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed pow_."""
    return binary_float_dispatch["pow_", pow_[_]](left, right)


def sqrt(array: AnyArray) raises -> AnyArray:
    """Runtime-typed sqrt."""
    return unary_float_dispatch["sqrt", sqrt[_]](array)


def exp(array: AnyArray) raises -> AnyArray:
    """Runtime-typed exp."""
    return unary_float_dispatch["exp", exp[_]](array)


def exp2(array: AnyArray) raises -> AnyArray:
    """Runtime-typed exp2."""
    return unary_float_dispatch["exp2", exp2[_]](array)


def log(array: AnyArray) raises -> AnyArray:
    """Runtime-typed log."""
    return unary_float_dispatch["log", log[_]](array)


def log2(array: AnyArray) raises -> AnyArray:
    """Runtime-typed log2."""
    return unary_float_dispatch["log2", log2[_]](array)


def log10(array: AnyArray) raises -> AnyArray:
    """Runtime-typed log10."""
    return unary_float_dispatch["log10", log10[_]](array)


def log1p(array: AnyArray) raises -> AnyArray:
    """Runtime-typed log1p."""
    return unary_float_dispatch["log1p", log1p[_]](array)


def floor(array: AnyArray) raises -> AnyArray:
    """Runtime-typed floor."""
    return unary_numeric_dispatch["floor", floor[_]](array)


def ceil(array: AnyArray) raises -> AnyArray:
    """Runtime-typed ceil."""
    return unary_numeric_dispatch["ceil", ceil[_]](array)


def trunc(array: AnyArray) raises -> AnyArray:
    """Runtime-typed trunc."""
    return unary_numeric_dispatch["trunc", trunc[_]](array)


def round(array: AnyArray) raises -> AnyArray:
    """Runtime-typed round."""
    return unary_numeric_dispatch["round", round[_]](array)


def sin(array: AnyArray) raises -> AnyArray:
    """Runtime-typed sin."""
    return unary_float_dispatch["sin", sin[_]](array)


def cos(array: AnyArray) raises -> AnyArray:
    """Runtime-typed cos."""
    return unary_float_dispatch["cos", cos[_]](array)
