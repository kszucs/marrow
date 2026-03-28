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
    DataType,
    bool_ as bool_dt,
    numeric_dtypes,
    float_dtypes,
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
    var n = len(a.value())
    return (BitmapView(a.value()) & BitmapView(b.value())).to_immutable()


# ---------------------------------------------------------------------------
# Runtime-typed dispatch helpers
# ---------------------------------------------------------------------------


def binary_array_dispatch[
    name: StringLiteral,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) raises -> PrimitiveArray[T],
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

    comptime for dtype in numeric_dtypes:
        if left.dtype() == dtype:
            return func[dtype](
                left.as_primitive[dtype](),
                right.as_primitive[dtype](),
                ctx,
            ).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def binary_array_dispatch[
    name: StringLiteral,
    OutT: DataType,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) raises -> PrimitiveArray[OutT],
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

    comptime for dtype in numeric_dtypes:
        if left.dtype() == dtype:
            return func[dtype](
                left.as_primitive[dtype](),
                right.as_primitive[dtype](),
                ctx,
            ).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def bool_array_dispatch[
    name: StringLiteral,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) raises -> BoolArray,
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

    comptime for dtype in numeric_dtypes:
        if left.dtype() == dtype:
            return func[dtype](
                left.as_primitive[dtype](),
                right.as_primitive[dtype](),
                ctx,
            ).to_any()
    raise Error(t"{name}: unsupported dtype {left.dtype()}")


def unary_numeric_dispatch[
    name: StringLiteral,
    func: def[T: DataType](PrimitiveArray[T]) raises -> PrimitiveArray[T],
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
    comptime for dtype in numeric_dtypes:
        if array.dtype() == dtype:
            return func[dtype](array.as_primitive[dtype]()).to_any()
    raise Error(t"{name}: unsupported dtype {array.dtype()}")


def binary_float_dispatch[
    name: StringLiteral,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T]
    ) raises -> PrimitiveArray[T],
](left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed binary dispatch restricted to floating-point dtypes."""
    if left.dtype() != right.dtype():
        raise Error(
            t"{name}: dtype mismatch: {left.dtype()} vs {right.dtype()}"
        )

    comptime for dtype in float_dtypes:
        if left.dtype() == dtype:
            return func[dtype](
                left.as_primitive[dtype](),
                right.as_primitive[dtype](),
            ).to_any()
    raise Error(
        t"{name}: unsupported dtype {left.dtype()}, expected float type"
    )


def unary_float_dispatch[
    name: StringLiteral,
    func: def[T: DataType](PrimitiveArray[T]) raises -> PrimitiveArray[T],
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
    comptime for dtype in float_dtypes:
        if array.dtype() == dtype:
            return func[dtype](array.as_primitive[dtype]()).to_any()
    raise Error(
        t"{name}: unsupported dtype {array.dtype()}, expected float type"
    )
