"""Shared helpers for compute kernels.

Provides:
  - `bitmap_and` — null bitmap propagation (bitwise AND of two validity bitmaps).
  - `binary_array_dispatch` — runtime-typed dispatch over numeric dtypes.

Kernel implementations live in their respective modules:
  - `arithmetic.mojo` — binary arithmetic, unary math, GPU dispatch via ``elementwise``
  - `compare.mojo` — comparison kernels producing bit-packed bool output
  - `aggregate.mojo` — reductions using ``std.algorithm`` (sum, min, max, etc.)
  - `filter.mojo` — selection/filter kernels
"""

from std.gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.bitmap import Bitmap
from marrow.dtypes import DataType, bool_ as bool_dt, numeric_dtypes


# ---------------------------------------------------------------------------
# Null bitmap kernel
# ---------------------------------------------------------------------------


def bitmap_and(
    a: Optional[Bitmap], b: Optional[Bitmap]
) raises -> Optional[Bitmap]:
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
    return a.value() & b.value()


# ---------------------------------------------------------------------------
# Runtime-typed dispatch helpers
# ---------------------------------------------------------------------------


def binary_array_dispatch[
    name: StringLiteral,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) raises -> PrimitiveArray[T],
](
    left: Array,
    right: Array,
    ctx: Optional[DeviceContext] = None,
) raises -> Array:
    """Runtime-typed binary dispatch: checks dtype match, loops over numeric types.

    Parameters:
        name: Operation name used in error messages.
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).
        ctx: GPU device context, forwarded to the typed kernel.

    Returns:
        A new Array with the element-wise result.
    """
    if left.dtype != right.dtype:
        raise Error(t"{name}: dtype mismatch: {left.dtype} vs {right.dtype}")

    comptime for dtype in numeric_dtypes:
        if left.dtype == dtype:
            return Array(
                func[dtype](
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                    ctx,
                )
            )
    raise Error(t"{name}: unsupported dtype {left.dtype}")


def binary_array_dispatch[
    name: StringLiteral,
    OutT: DataType,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T]
    ) raises -> PrimitiveArray[OutT],
](left: Array, right: Array,) raises -> Array:
    """Runtime-typed binary dispatch with a fixed output type (e.g. comparisons).

    Parameters:
        name: Operation name used in error messages.
        OutT: Output DataType (e.g. ``bool_`` for comparisons).
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array wrapping ``PrimitiveArray[OutT]`` with the result.
    """
    if left.dtype != right.dtype:
        raise Error(t"{name}: dtype mismatch: {left.dtype} vs {right.dtype}")

    comptime for dtype in numeric_dtypes:
        if left.dtype == dtype:
            return Array(
                func[dtype](
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                )
            )
    raise Error(t"{name}: unsupported dtype {left.dtype}")
