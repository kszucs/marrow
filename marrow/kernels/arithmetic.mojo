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
from ..dtypes import DataType, numeric_dtypes
from . import (
    bitmap_and,
    binary_array_dispatch
)


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


# ---------------------------------------------------------------------------
# Public API — unary kernels
# ---------------------------------------------------------------------------


def neg[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise negation."""
    return _unary[T, _neg_fn[T.native, _]](array)


def abs_[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Element-wise absolute value."""
    return _unary[T, _abs_fn[T.native, _]](array)


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
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(neg[dtype](PrimitiveArray[dtype](data=array)))
    raise Error(t"neg: unsupported dtype {array.dtype}")


def abs_(array: Array) raises -> Array:
    """Runtime-typed abs_."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(abs_[dtype](PrimitiveArray[dtype](data=array)))
    raise Error(t"abs_: unsupported dtype {array.dtype}")
