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
        return func[Int8Type](
            left.as_primitive[Int8Type](), right.as_primitive[Int8Type](), ctx
        ).to_any()
    elif left.dtype() == int16:
        return func[Int16Type](
            left.as_primitive[Int16Type](), right.as_primitive[Int16Type](), ctx
        ).to_any()
    elif left.dtype() == int32:
        return func[Int32Type](
            left.as_primitive[Int32Type](), right.as_primitive[Int32Type](), ctx
        ).to_any()
    elif left.dtype() == int64:
        return func[Int64Type](
            left.as_primitive[Int64Type](), right.as_primitive[Int64Type](), ctx
        ).to_any()
    elif left.dtype() == uint8:
        return func[UInt8Type](
            left.as_primitive[UInt8Type](), right.as_primitive[UInt8Type](), ctx
        ).to_any()
    elif left.dtype() == uint16:
        return func[UInt16Type](
            left.as_primitive[UInt16Type](),
            right.as_primitive[UInt16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint32:
        return func[UInt32Type](
            left.as_primitive[UInt32Type](),
            right.as_primitive[UInt32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint64:
        return func[UInt64Type](
            left.as_primitive[UInt64Type](),
            right.as_primitive[UInt64Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float16:
        return func[Float16Type](
            left.as_primitive[Float16Type](),
            right.as_primitive[Float16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float32:
        return func[Float32Type](
            left.as_primitive[Float32Type](),
            right.as_primitive[Float32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float64:
        return func[Float64Type](
            left.as_primitive[Float64Type](),
            right.as_primitive[Float64Type](),
            ctx,
        ).to_any()
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
        return func[Int8Type](
            left.as_primitive[Int8Type](), right.as_primitive[Int8Type](), ctx
        ).to_any()
    elif left.dtype() == int16:
        return func[Int16Type](
            left.as_primitive[Int16Type](), right.as_primitive[Int16Type](), ctx
        ).to_any()
    elif left.dtype() == int32:
        return func[Int32Type](
            left.as_primitive[Int32Type](), right.as_primitive[Int32Type](), ctx
        ).to_any()
    elif left.dtype() == int64:
        return func[Int64Type](
            left.as_primitive[Int64Type](), right.as_primitive[Int64Type](), ctx
        ).to_any()
    elif left.dtype() == uint8:
        return func[UInt8Type](
            left.as_primitive[UInt8Type](), right.as_primitive[UInt8Type](), ctx
        ).to_any()
    elif left.dtype() == uint16:
        return func[UInt16Type](
            left.as_primitive[UInt16Type](),
            right.as_primitive[UInt16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint32:
        return func[UInt32Type](
            left.as_primitive[UInt32Type](),
            right.as_primitive[UInt32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint64:
        return func[UInt64Type](
            left.as_primitive[UInt64Type](),
            right.as_primitive[UInt64Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float16:
        return func[Float16Type](
            left.as_primitive[Float16Type](),
            right.as_primitive[Float16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float32:
        return func[Float32Type](
            left.as_primitive[Float32Type](),
            right.as_primitive[Float32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float64:
        return func[Float64Type](
            left.as_primitive[Float64Type](),
            right.as_primitive[Float64Type](),
            ctx,
        ).to_any()
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
        return func[Int8Type](
            left.as_primitive[Int8Type](), right.as_primitive[Int8Type](), ctx
        ).to_any()
    elif left.dtype() == int16:
        return func[Int16Type](
            left.as_primitive[Int16Type](), right.as_primitive[Int16Type](), ctx
        ).to_any()
    elif left.dtype() == int32:
        return func[Int32Type](
            left.as_primitive[Int32Type](), right.as_primitive[Int32Type](), ctx
        ).to_any()
    elif left.dtype() == int64:
        return func[Int64Type](
            left.as_primitive[Int64Type](), right.as_primitive[Int64Type](), ctx
        ).to_any()
    elif left.dtype() == uint8:
        return func[UInt8Type](
            left.as_primitive[UInt8Type](), right.as_primitive[UInt8Type](), ctx
        ).to_any()
    elif left.dtype() == uint16:
        return func[UInt16Type](
            left.as_primitive[UInt16Type](),
            right.as_primitive[UInt16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint32:
        return func[UInt32Type](
            left.as_primitive[UInt32Type](),
            right.as_primitive[UInt32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == uint64:
        return func[UInt64Type](
            left.as_primitive[UInt64Type](),
            right.as_primitive[UInt64Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float16:
        return func[Float16Type](
            left.as_primitive[Float16Type](),
            right.as_primitive[Float16Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float32:
        return func[Float32Type](
            left.as_primitive[Float32Type](),
            right.as_primitive[Float32Type](),
            ctx,
        ).to_any()
    elif left.dtype() == float64:
        return func[Float64Type](
            left.as_primitive[Float64Type](),
            right.as_primitive[Float64Type](),
            ctx,
        ).to_any()
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
        return func[Int8Type](array.as_primitive[Int8Type]()).to_any()
    elif array.dtype() == int16:
        return func[Int16Type](array.as_primitive[Int16Type]()).to_any()
    elif array.dtype() == int32:
        return func[Int32Type](array.as_primitive[Int32Type]()).to_any()
    elif array.dtype() == int64:
        return func[Int64Type](array.as_primitive[Int64Type]()).to_any()
    elif array.dtype() == uint8:
        return func[UInt8Type](array.as_primitive[UInt8Type]()).to_any()
    elif array.dtype() == uint16:
        return func[UInt16Type](array.as_primitive[UInt16Type]()).to_any()
    elif array.dtype() == uint32:
        return func[UInt32Type](array.as_primitive[UInt32Type]()).to_any()
    elif array.dtype() == uint64:
        return func[UInt64Type](array.as_primitive[UInt64Type]()).to_any()
    elif array.dtype() == float16:
        return func[Float16Type](array.as_primitive[Float16Type]()).to_any()
    elif array.dtype() == float32:
        return func[Float32Type](array.as_primitive[Float32Type]()).to_any()
    elif array.dtype() == float64:
        return func[Float64Type](array.as_primitive[Float64Type]()).to_any()
    raise Error(t"{name}: unsupported dtype {array.dtype()}")


def binary_float_dispatch[
    name: StringLiteral,
    func: def[T: PrimitiveType](
        PrimitiveArray[T], PrimitiveArray[T]
    ) raises -> PrimitiveArray[T],
](left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed binary dispatch restricted to floating-point dtypes."""
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    if left.dtype() == float16:
        return func[Float16Type](
            left.as_primitive[Float16Type](), right.as_primitive[Float16Type]()
        ).to_any()
    elif left.dtype() == float32:
        return func[Float32Type](
            left.as_primitive[Float32Type](), right.as_primitive[Float32Type]()
        ).to_any()
    elif left.dtype() == float64:
        return func[Float64Type](
            left.as_primitive[Float64Type](), right.as_primitive[Float64Type]()
        ).to_any()
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
        return func[Float16Type](array.as_primitive[Float16Type]()).to_any()
    elif array.dtype() == float32:
        return func[Float32Type](array.as_primitive[Float32Type]()).to_any()
    elif array.dtype() == float64:
        return func[Float64Type](array.as_primitive[Float64Type]()).to_any()
    raise Error(
        t"{name}: unsupported dtype {array.dtype()}, expected float type"
    )
