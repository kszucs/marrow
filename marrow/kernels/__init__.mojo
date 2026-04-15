"""Shared helpers for compute kernels.

Provides:
  - `bitmap_and` — null bitmap propagation (bitwise AND of two validity bitmaps).
  - `binary_array_dispatch` — runtime-typed dispatch over numeric dtypes.
  - `unary_numeric_dispatch` — runtime-typed unary dispatch over numeric dtypes.
  - `unary_float_dispatch` — runtime-typed unary dispatch over float dtypes.

Kernel implementations live in their respective modules:
  - `arithmetic.mojo` — binary arithmetic, unary math, GPU dispatch via ``elementwise``
  - `compare.mojo` — comparison kernels producing bit-packed bool output
  - `aggregate.mojo` — reductions using ``std.algorithm`` (sum, min, max, etc.)
  - `filter.mojo` — selection/filter kernels
  - `groupby.mojo` — fused groupby with aggregation (sum, min, max, count, mean)
  - `hashing.mojo` — hash_ for PrimitiveArray, StringArray, StructArray, AnyArray
"""

from std.gpu.host import DeviceContext

from marrow.arrays import BoolArray, PrimitiveArray, AnyArray
from marrow.buffers import Bitmap
from marrow.views import BitmapView
from marrow.dtypes import (
    PrimitiveType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bool_ as bool_dt,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)


# ---------------------------------------------------------------------------
# Null bitmap kernel
# ---------------------------------------------------------------------------


def bitmap_and(
    a: Optional[Bitmap[]], b: Optional[Bitmap[]]
) raises -> Optional[Bitmap[]]:
    """Compute the output validity bitmap as the bitwise AND of two input bitmaps.

    Output bit i is True iff both a[i] and b[i] are True (valid).
    None represents an all-valid bitmap.

    Args:
        a: First input bitmap (None = all valid).
        b: Second input bitmap (None = all valid).

    Returns:
        None if both are all-valid; otherwise the AND of the two bitmaps.
    """
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a
    return (a.value().view() & b.value().view()).to_immutable()


# ---------------------------------------------------------------------------
# Runtime-typed dispatch helpers
# ---------------------------------------------------------------------------


def binary_array_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) thin raises -> PrimitiveArray[T],
](
    left: AnyArray,
    right: AnyArray,
    ctx: Optional[DeviceContext] = None,
) raises -> AnyArray:
    """Runtime-typed binary dispatch: checks dtype match, loops over numeric types.

    Parameters:
        name: Operation name used in error messages.
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed AnyArray).
        right: Right operand (runtime-typed AnyArray).
        ctx: GPU device context, forwarded to the typed kernel.

    Returns:
        A new AnyArray with the element-wise result.
    """
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    if left.dtype() == int8:
        return func(left.as_int8(), right.as_int8(), ctx).to_any()
    elif left.dtype() == int16:
        return func(left.as_int16(), right.as_int16(), ctx).to_any()
    elif left.dtype() == int32:
        return func(left.as_int32(), right.as_int32(), ctx).to_any()
    elif left.dtype() == int64:
        return func(left.as_int64(), right.as_int64(), ctx).to_any()
    elif left.dtype() == uint8:
        return func(left.as_uint8(), right.as_uint8(), ctx).to_any()
    elif left.dtype() == uint16:
        return func(left.as_uint16(), right.as_uint16(), ctx).to_any()
    elif left.dtype() == uint32:
        return func(left.as_uint32(), right.as_uint32(), ctx).to_any()
    elif left.dtype() == uint64:
        return func(left.as_uint64(), right.as_uint64(), ctx).to_any()
    elif left.dtype() == float16:
        return func(left.as_float16(), right.as_float16(), ctx).to_any()
    elif left.dtype() == float32:
        return func(left.as_float32(), right.as_float32(), ctx).to_any()
    elif left.dtype() == float64:
        return func(left.as_float64(), right.as_float64(), ctx).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def binary_array_dispatch[
    name: StringLiteral,
    OutT: PrimitiveType,
    func: def[T: PrimitiveType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) thin raises -> PrimitiveArray[OutT],
](
    left: AnyArray,
    right: AnyArray,
    ctx: Optional[DeviceContext] = None,
) raises -> AnyArray:
    """Runtime-typed binary dispatch with a fixed output type (e.g. comparisons).

    Parameters:
        name: Operation name used in error messages.
        OutT: Output DataType (e.g. ``bool_`` for comparisons).
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed AnyArray).
        right: Right operand (runtime-typed AnyArray).
        ctx: GPU device context, forwarded to the typed kernel.

    Returns:
        A new AnyArray wrapping ``PrimitiveArray[OutT]`` with the result.
    """
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    if left.dtype() == int8:
        return func(left.as_int8(), right.as_int8(), ctx).to_any()
    elif left.dtype() == int16:
        return func(left.as_int16(), right.as_int16(), ctx).to_any()
    elif left.dtype() == int32:
        return func(left.as_int32(), right.as_int32(), ctx).to_any()
    elif left.dtype() == int64:
        return func(left.as_int64(), right.as_int64(), ctx).to_any()
    elif left.dtype() == uint8:
        return func(left.as_uint8(), right.as_uint8(), ctx).to_any()
    elif left.dtype() == uint16:
        return func(left.as_uint16(), right.as_uint16(), ctx).to_any()
    elif left.dtype() == uint32:
        return func(left.as_uint32(), right.as_uint32(), ctx).to_any()
    elif left.dtype() == uint64:
        return func(left.as_uint64(), right.as_uint64(), ctx).to_any()
    elif left.dtype() == float16:
        return func(left.as_float16(), right.as_float16(), ctx).to_any()
    elif left.dtype() == float32:
        return func(left.as_float32(), right.as_float32(), ctx).to_any()
    elif left.dtype() == float64:
        return func(left.as_float64(), right.as_float64(), ctx).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def bool_array_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) thin raises -> BoolArray,
](
    left: AnyArray,
    right: AnyArray,
    ctx: Optional[DeviceContext] = None,
) raises -> AnyArray:
    """Runtime-typed binary dispatch producing a BoolArray result (e.g. comparisons).

    Parameters:
        name: Operation name used in error messages.
        func: The typed binary kernel to dispatch to (returns BoolArray).
    """
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    if left.dtype() == int8:
        return func(left.as_int8(), right.as_int8(), ctx).to_any()
    elif left.dtype() == int16:
        return func(left.as_int16(), right.as_int16(), ctx).to_any()
    elif left.dtype() == int32:
        return func(left.as_int32(), right.as_int32(), ctx).to_any()
    elif left.dtype() == int64:
        return func(left.as_int64(), right.as_int64(), ctx).to_any()
    elif left.dtype() == uint8:
        return func(left.as_uint8(), right.as_uint8(), ctx).to_any()
    elif left.dtype() == uint16:
        return func(left.as_uint16(), right.as_uint16(), ctx).to_any()
    elif left.dtype() == uint32:
        return func(left.as_uint32(), right.as_uint32(), ctx).to_any()
    elif left.dtype() == uint64:
        return func(left.as_uint64(), right.as_uint64(), ctx).to_any()
    elif left.dtype() == float16:
        return func(left.as_float16(), right.as_float16(), ctx).to_any()
    elif left.dtype() == float32:
        return func(left.as_float32(), right.as_float32(), ctx).to_any()
    elif left.dtype() == float64:
        return func(left.as_float64(), right.as_float64(), ctx).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def unary_numeric_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T]
    ) thin raises -> PrimitiveArray[T],
](array: AnyArray) raises -> AnyArray:
    """Runtime-typed unary dispatch over all numeric dtypes.

    Parameters:
        name: Operation name used in error messages.
        func: The typed unary kernel to dispatch to.

    Args:
        array: Input array (runtime-typed).

    Returns:
        A new AnyArray with the element-wise result.
    """
    if array.dtype() == int8:
        return func(array.as_int8()).to_any()
    elif array.dtype() == int16:
        return func(array.as_int16()).to_any()
    elif array.dtype() == int32:
        return func(array.as_int32()).to_any()
    elif array.dtype() == int64:
        return func(array.as_int64()).to_any()
    elif array.dtype() == uint8:
        return func(array.as_uint8()).to_any()
    elif array.dtype() == uint16:
        return func(array.as_uint16()).to_any()
    elif array.dtype() == uint32:
        return func(array.as_uint32()).to_any()
    elif array.dtype() == uint64:
        return func(array.as_uint64()).to_any()
    elif array.dtype() == float16:
        return func(array.as_float16()).to_any()
    elif array.dtype() == float32:
        return func(array.as_float32()).to_any()
    elif array.dtype() == float64:
        return func(array.as_float64()).to_any()
    raise Error(t"{name}: unsupported dtype {array.dtype()}")


def binary_float_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T], PrimitiveArray[T]
    ) thin raises -> PrimitiveArray[T],
](left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed binary dispatch restricted to floating-point dtypes."""
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    if left.dtype() == float16:
        return func(left.as_float16(), right.as_float16()).to_any()
    elif left.dtype() == float32:
        return func(left.as_float32(), right.as_float32()).to_any()
    elif left.dtype() == float64:
        return func(left.as_float64(), right.as_float64()).to_any()
    raise Error(
        t"{name}: unsupported dtype {left.dtype()}, expected float type"
    )


def unary_float_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T]
    ) thin raises -> PrimitiveArray[T],
](array: AnyArray) raises -> AnyArray:
    """Runtime-typed unary dispatch restricted to floating-point dtypes.

    Parameters:
        name: Operation name used in error messages.
        func: The typed unary kernel to dispatch to.

    Args:
        array: Input array (runtime-typed); must be float16, float32, or float64.

    Returns:
        A new AnyArray with the element-wise result.
    """
    if array.dtype() == float16:
        return func(array.as_float16()).to_any()
    elif array.dtype() == float32:
        return func(array.as_float32()).to_any()
    elif array.dtype() == float64:
        return func(array.as_float64()).to_any()
    raise Error(
        t"{name}: unsupported dtype {array.dtype()}, expected float type"
    )
