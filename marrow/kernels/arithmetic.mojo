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
from std.gpu.host import DeviceContext
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList

from ..arrays import PrimitiveArray, Array
from ..buffers import Buffer, BufferBuilder
from ..dtypes import DataType, numeric_dtypes, float_dtypes
from . import bitmap_and, binary_array_dispatch, unary_numeric_dispatch, unary_float_dispatch


# ---------------------------------------------------------------------------
# Elementwise dispatch — pointers as params for GPU DevicePassable
# ---------------------------------------------------------------------------


def _elementwise_unary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W]) -> SIMD[T.native, W],
    simd_width: Int,
](
    output: UnsafePointer[Scalar[T.native], MutAnyOrigin],
    input: UnsafePointer[Scalar[T.native], ImmutAnyOrigin],
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
        (output + i).store(func[W](input.load[width=W](i)))

    if ctx:
        elementwise[process, simd_width, target="gpu"](length, ctx.value())
    else:
        elementwise[process, simd_width, target="cpu", use_blocking_impl=True](
            length
        )


def _elementwise_binary[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        T.native, W
    ],
    simd_width: Int,
](
    output: UnsafePointer[Scalar[T.native], MutAnyOrigin],
    lhs: UnsafePointer[Scalar[T.native], ImmutAnyOrigin],
    rhs: UnsafePointer[Scalar[T.native], ImmutAnyOrigin],
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
        (output + i).store(func[W](lhs.load[width=W](i), rhs.load[width=W](i)))

    if ctx:
        elementwise[process, simd_width, target="gpu"](length, ctx.value())
    else:
        elementwise[process, simd_width, target="cpu", use_blocking_impl=True](
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
    """Unary kernel: allocates output, resolves pointers, calls elementwise."""
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()
    var length = len(array)

    var buf: BufferBuilder
    var in_ptr: UnsafePointer[Scalar[native], ImmutAnyOrigin]
    if ctx:
        buf = BufferBuilder.alloc_device[native](ctx.value(), length)
        in_ptr = array.buffer.device_ptr[native](array.offset)
    else:
        buf = BufferBuilder.alloc[native](length)
        in_ptr = array.buffer.unsafe_ptr[native](array.offset)

    _elementwise_unary[T, func, width](
        buf.ptr.bitcast[Scalar[native]](), in_ptr, length, ctx
    )

    return PrimitiveArray[T](
        length=length,
        nulls=length
        - array.bitmap.value().count_set_bits() if array.bitmap else 0,
        offset=0,
        bitmap=array.bitmap,
        buffer=buf.finish(),
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
    """Binary kernel: allocates output, resolves pointers, calls elementwise."""
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()
    var length = len(left)
    var bm = bitmap_and(left.bitmap, right.bitmap)

    var buf: BufferBuilder
    var out_ptr: UnsafePointer[Scalar[native], MutAnyOrigin]
    var lhs_ptr: UnsafePointer[Scalar[native], ImmutAnyOrigin]
    var rhs_ptr: UnsafePointer[Scalar[native], ImmutAnyOrigin]
    if ctx:
        buf = BufferBuilder.alloc_device[native](ctx.value(), length)
        out_ptr = buf.ptr.bitcast[Scalar[native]]()
        lhs_ptr = left.buffer.device_ptr[native](left.offset)
        rhs_ptr = right.buffer.device_ptr[native](right.offset)
    else:
        buf = BufferBuilder.alloc[native](length)
        out_ptr = buf.ptr.bitcast[Scalar[native]]()
        lhs_ptr = left.buffer.unsafe_ptr[native](left.offset)
        rhs_ptr = right.buffer.unsafe_ptr[native](right.offset)

    _elementwise_binary[T, func, width](out_ptr, lhs_ptr, rhs_ptr, length, ctx)

    return PrimitiveArray[T](
        length=length,
        nulls=length - bm.value().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=buf.finish(),
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
    return -a


def _abs_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return abs(a)


def _sign_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return a.gt(SIMD[T, W](0)).cast[T]() - a.lt(SIMD[T, W](0)).cast[T]()


# Float-only unary


def _sqrt_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.sqrt(a)


def _exp_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.exp(a)


def _exp2_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.exp2(a)


def _log_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log(a)


def _log2_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log2(a)


def _log10_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log10(a)


def _log1p_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.log1p(a)


def _floor_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.floor(a)


def _ceil_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.ceil(a)


def _trunc_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.trunc(a)


def _round_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.round(a)


def _sin_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.sin(a)


def _cos_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.cos(a)


def _tan_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.tan(a)


def _asin_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.asin(a)


def _acos_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.acos(a)


def _atan_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.atan(a)


def _sinh_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.sinh(a)


def _cosh_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.cosh(a)


def _tanh_fn[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
    return math.tanh(a)


# Float-only binary


def _pow_fn[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W] where T.is_floating_point():
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
) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise power: result[i] = left[i] ** right[i]."""
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


def sqrt[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise square root."""
    return _unary[T, _sqrt_fn[T.native, _]](array)


def exp[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise natural exponential (e^x)."""
    return _unary[T, _exp_fn[T.native, _]](array)


def exp2[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise base-2 exponential (2^x)."""
    return _unary[T, _exp2_fn[T.native, _]](array)


def log[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise natural logarithm."""
    return _unary[T, _log_fn[T.native, _]](array)


def log2[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise base-2 logarithm."""
    return _unary[T, _log2_fn[T.native, _]](array)


def log10[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise base-10 logarithm."""
    return _unary[T, _log10_fn[T.native, _]](array)


def log1p[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise log(1 + x)."""
    return _unary[T, _log1p_fn[T.native, _]](array)


def floor[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise floor."""
    return _unary[T, _floor_fn[T.native, _]](array)


def ceil[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise ceiling."""
    return _unary[T, _ceil_fn[T.native, _]](array)


def trunc[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise truncation toward zero."""
    return _unary[T, _trunc_fn[T.native, _]](array)


def round[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise rounding to nearest integer."""
    return _unary[T, _round_fn[T.native, _]](array)


def sin[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise sine."""
    return _unary[T, _sin_fn[T.native, _]](array)


def cos[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise cosine."""
    return _unary[T, _cos_fn[T.native, _]](array)


def tan[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise tangent."""
    return _unary[T, _tan_fn[T.native, _]](array)


def asin[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise arcsine."""
    return _unary[T, _asin_fn[T.native, _]](array)


def acos[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise arccosine."""
    return _unary[T, _acos_fn[T.native, _]](array)


def atan[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise arctangent."""
    return _unary[T, _atan_fn[T.native, _]](array)


def sinh[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise hyperbolic sine."""
    return _unary[T, _sinh_fn[T.native, _]](array)


def cosh[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise hyperbolic cosine."""
    return _unary[T, _cosh_fn[T.native, _]](array)


def tanh[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T] where T.is_floating_point():
    """Element-wise hyperbolic tangent."""
    return _unary[T, _tanh_fn[T.native, _]](array)


# ---------------------------------------------------------------------------
# Runtime dispatch — Array-typed overloads
# ---------------------------------------------------------------------------


def add(left: Array, right: Array) raises -> Array:
    """Runtime-typed add."""
    return binary_array_dispatch["add", add[_]](left, right)


def sub(left: Array, right: Array) raises -> Array:
    """Runtime-typed sub."""
    return binary_array_dispatch["sub", sub[_]](left, right)


def mul(left: Array, right: Array) raises -> Array:
    """Runtime-typed mul."""
    return binary_array_dispatch["mul", mul[_]](left, right)


def div(left: Array, right: Array) raises -> Array:
    """Runtime-typed div."""
    return binary_array_dispatch["div", div[_]](left, right)


def floordiv(left: Array, right: Array) raises -> Array:
    """Runtime-typed floordiv."""
    return binary_array_dispatch["floordiv", floordiv[_]](left, right)


def mod(left: Array, right: Array) raises -> Array:
    """Runtime-typed mod."""
    return binary_array_dispatch["mod", mod[_]](left, right)


def min_(left: Array, right: Array) raises -> Array:
    """Runtime-typed min_."""
    return binary_array_dispatch["min_", min_[_]](left, right)


def max_(left: Array, right: Array) raises -> Array:
    """Runtime-typed max_."""
    return binary_array_dispatch["max_", max_[_]](left, right)


def neg(array: Array) raises -> Array:
    """Runtime-typed neg."""
    return unary_numeric_dispatch["neg", neg[_]](array)


def abs_(array: Array) raises -> Array:
    """Runtime-typed abs_."""
    return unary_numeric_dispatch["abs_", abs_[_]](array)


def sign(array: Array) raises -> Array:
    """Runtime-typed sign."""
    return unary_numeric_dispatch["sign", sign[_]](array)


def pow_(left: Array, right: Array) raises -> Array:
    """Runtime-typed pow_."""
    return binary_array_dispatch["pow_", pow_[_]](left, right)


def sqrt(array: Array) raises -> Array:
    """Runtime-typed sqrt."""
    return unary_float_dispatch["sqrt", sqrt[_]](array)


def exp(array: Array) raises -> Array:
    """Runtime-typed exp."""
    return unary_float_dispatch["exp", exp[_]](array)


def exp2(array: Array) raises -> Array:
    """Runtime-typed exp2."""
    return unary_float_dispatch["exp2", exp2[_]](array)


def log(array: Array) raises -> Array:
    """Runtime-typed log."""
    return unary_float_dispatch["log", log[_]](array)


def log2(array: Array) raises -> Array:
    """Runtime-typed log2."""
    return unary_float_dispatch["log2", log2[_]](array)


def log10(array: Array) raises -> Array:
    """Runtime-typed log10."""
    return unary_float_dispatch["log10", log10[_]](array)


def log1p(array: Array) raises -> Array:
    """Runtime-typed log1p."""
    return unary_float_dispatch["log1p", log1p[_]](array)


def floor(array: Array) raises -> Array:
    """Runtime-typed floor."""
    return unary_float_dispatch["floor", floor[_]](array)


def ceil(array: Array) raises -> Array:
    """Runtime-typed ceil."""
    return unary_float_dispatch["ceil", ceil[_]](array)


def trunc(array: Array) raises -> Array:
    """Runtime-typed trunc."""
    return unary_float_dispatch["trunc", trunc[_]](array)


def round(array: Array) raises -> Array:
    """Runtime-typed round."""
    return unary_float_dispatch["round", round[_]](array)


def sin(array: Array) raises -> Array:
    """Runtime-typed sin."""
    return unary_float_dispatch["sin", sin[_]](array)


def cos(array: Array) raises -> Array:
    """Runtime-typed cos."""
    return unary_float_dispatch["cos", cos[_]](array)


def tan(array: Array) raises -> Array:
    """Runtime-typed tan."""
    return unary_float_dispatch["tan", tan[_]](array)


def asin(array: Array) raises -> Array:
    """Runtime-typed asin."""
    return unary_float_dispatch["asin", asin[_]](array)


def acos(array: Array) raises -> Array:
    """Runtime-typed acos."""
    return unary_float_dispatch["acos", acos[_]](array)


def atan(array: Array) raises -> Array:
    """Runtime-typed atan."""
    return unary_float_dispatch["atan", atan[_]](array)


def sinh(array: Array) raises -> Array:
    """Runtime-typed sinh."""
    return unary_float_dispatch["sinh", sinh[_]](array)


def cosh(array: Array) raises -> Array:
    """Runtime-typed cosh."""
    return unary_float_dispatch["cosh", cosh[_]](array)


def tanh(array: Array) raises -> Array:
    """Runtime-typed tanh."""
    return unary_float_dispatch["tanh", tanh[_]](array)
