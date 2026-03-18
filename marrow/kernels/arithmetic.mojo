"""Element-wise arithmetic kernels — CPU SIMD and GPU specializations.

Each public function dispatches based on the optional `ctx` argument:
  - CPU (default): operates on PrimitiveArray[T] using SIMD vectorization.
  - GPU (ctx provided): operates on device-resident PrimitiveArray[T] via a GPU kernel.

The SIMD helper functions (def[W: Int](SIMD[T, W], ...) -> SIMD[T, W]) are shared
between CPU and GPU paths. Since Scalar[T] = SIMD[T, 1], the GPU kernel calls
each helper with W=1 per thread.
"""

import std.math as math
from std.gpu.host import DeviceContext

from ..arrays import PrimitiveArray, Array
from ..dtypes import DataType, numeric_dtypes
from . import (
    binary_simd,
    binary_not_null,
    binary_gpu,
    unary_simd,
    binary_array_dispatch,
)


# ---------------------------------------------------------------------------
# SIMD helpers — shared by CPU and GPU paths
# ---------------------------------------------------------------------------

# Binary: def[W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[T, W]


def _add[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


def _sub[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a - b


def _mul[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a * b


def _div[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a / b


def _floordiv[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a // b


def _mod[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a % b


def _min[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    # TODO(kszucs): consider return (a < b).select(a, b)
    return math.min(a, b)


def _max[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    # TODO(kszucs): consider return (a > b).select(a, b)
    return math.max(a, b)


# Unary: def[W: Int](SIMD[T, W]) -> SIMD[T, W]


def _neg[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return -a


def _abs[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return abs(a)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def add[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise addition of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_add[T.native, _], name="add"](
                left, right, ctx.value()
            )
        else:
            raise Error("add: no GPU accelerator available on this system")
    return binary_simd[T, T, func=_add[T.native, _], name="add"](left, right)


def add(left: Array, right: Array) raises -> Array:
    """Runtime-typed add."""
    return binary_array_dispatch["add", add[_]](left, right)


# ---------------------------------------------------------------------------
# sub
# ---------------------------------------------------------------------------


def sub[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise subtraction of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] - right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_sub[T.native, _], name="sub"](
                left, right, ctx.value()
            )
        else:
            raise Error("sub: no GPU accelerator available on this system")
    return binary_simd[T, T, func=_sub[T.native, _], name="sub"](left, right)


def sub(left: Array, right: Array) raises -> Array:
    """Runtime-typed sub."""
    return binary_array_dispatch["sub", sub[_]](left, right)


# ---------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------


def mul[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise multiplication of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] * right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_mul[T.native, _], name="mul"](
                left, right, ctx.value()
            )
        else:
            raise Error("mul: no GPU accelerator available on this system")
    return binary_simd[T, T, func=_mul[T.native, _], name="mul"](left, right)


def mul(left: Array, right: Array) raises -> Array:
    """Runtime-typed mul."""
    return binary_array_dispatch["mul", mul[_]](left, right)


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------


def div[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise true division of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] / right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_div[T.native, _], name="div"](
                left, right, ctx.value()
            )
        else:
            raise Error("div: no GPU accelerator available on this system")
    return binary_not_null[T, func=_div[T.native, _], name="div"](left, right)


def div(left: Array, right: Array) raises -> Array:
    """Runtime-typed div."""
    return binary_array_dispatch["div", div[_]](left, right)


# ---------------------------------------------------------------------------
# floordiv
# ---------------------------------------------------------------------------


def floordiv[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise floor division of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] // right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_floordiv[T.native, _], name="floordiv"](
                left, right, ctx.value()
            )
        else:
            raise Error("floordiv: no GPU accelerator available on this system")
    return binary_not_null[T, func=_floordiv[T.native, _], name="floordiv"](
        left, right
    )


def floordiv(left: Array, right: Array) raises -> Array:
    """Runtime-typed floordiv."""
    return binary_array_dispatch["floordiv", floordiv[_]](left, right)


# ---------------------------------------------------------------------------
# mod
# ---------------------------------------------------------------------------


def mod[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise modulo of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] % right[i].
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_mod[T.native, _], name="mod"](
                left, right, ctx.value()
            )
        else:
            raise Error("mod: no GPU accelerator available on this system")
    return binary_not_null[T, func=_mod[T.native, _], name="mod"](left, right)


def mod(left: Array, right: Array) raises -> Array:
    """Runtime-typed mod."""
    return binary_array_dispatch["mod", mod[_]](left, right)


# ---------------------------------------------------------------------------
# min_
# ---------------------------------------------------------------------------


def min_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise minimum of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = min(left[i], right[i]).
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_min[T.native, _], name="min_"](
                left, right, ctx.value()
            )
        else:
            raise Error("min_: no GPU accelerator available on this system")
    return binary_simd[T, T, func=_min[T.native, _], name="min_"](left, right)


def min_(left: Array, right: Array) raises -> Array:
    """Runtime-typed min_."""
    return binary_array_dispatch["min_", min_[_]](left, right)


# ---------------------------------------------------------------------------
# max_
# ---------------------------------------------------------------------------


def max_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise maximum of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = max(left[i], right[i]).
        Null if either input is null at that position.
    """
    if ctx:
        comptime if has_accelerator():
            return binary_gpu[T, func=_max[T.native, _], name="max_"](
                left, right, ctx.value()
            )
        else:
            raise Error("max_: no GPU accelerator available on this system")
    return binary_simd[T, T, func=_max[T.native, _], name="max_"](left, right)


def max_(left: Array, right: Array) raises -> Array:
    """Runtime-typed max_."""
    return binary_array_dispatch["max_", max_[_]](left, right)


# ---------------------------------------------------------------------------
# neg
# ---------------------------------------------------------------------------


def neg[T: DataType](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """Element-wise negation.

    Args:
        array: Input array.

    Returns:
        A new PrimitiveArray where result[i] = -array[i].
        Null if the input is null at that position.
    """
    return unary_simd[T, func=_neg[T.native, _]](array)


def neg(array: Array) raises -> Array:
    """Runtime-typed neg."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(neg[dtype](PrimitiveArray[dtype](data=array)))
    raise Error(t"neg: unsupported dtype {array.dtype}")


# ---------------------------------------------------------------------------
# abs_
# ---------------------------------------------------------------------------


def abs_[T: DataType](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """Element-wise absolute value.

    Args:
        array: Input array.

    Returns:
        A new PrimitiveArray where result[i] = |array[i]|.
        Null if the input is null at that position.
    """
    return unary_simd[T, func=_abs[T.native, _]](array)


def abs_(array: Array) raises -> Array:
    """Runtime-typed abs_."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return Array(abs_[dtype](PrimitiveArray[dtype](data=array)))
    raise Error(t"abs_: unsupported dtype {array.dtype}")
