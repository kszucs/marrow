"""Boolean and bitwise kernels."""

from ..arrays import PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import DataType, numeric_dtypes, bool_ as bool_dt
from ..views import BitmapView


def count_true(array: PrimitiveArray[bool_dt]) raises -> Int:
    """Count True values in a bit-packed boolean array.

    Args:
        array: A bit-packed boolean array.

    Returns:
        Number of True (and non-null) elements.
    """
    var n = len(array)
    var data_bv = array.values_bitmap()
    if array.nulls > 0:
        var combined = data_bv & array.validity().value()
        return BitmapView[ImmutExternalOrigin](
            ptr=combined.ptr, offset=0, length=n
        ).count_set_bits()
    return data_bv.count_set_bits()


def and_(
    lhs: PrimitiveArray[bool_dt], rhs: PrimitiveArray[bool_dt]
) raises -> PrimitiveArray[bool_dt]:
    """Bitwise AND of two bit-packed bool arrays."""
    var length = len(lhs)
    if len(rhs) != length:
        raise Error("and_: input arrays must have equal length")
    var lhs_bv = lhs.values_bitmap()
    var rhs_bv = rhs.values_bitmap()
    var result_buf = lhs_bv & rhs_bv
    return PrimitiveArray[bool_dt](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=result_buf,
    )


def or_(
    lhs: PrimitiveArray[bool_dt], rhs: PrimitiveArray[bool_dt]
) raises -> PrimitiveArray[bool_dt]:
    """Bitwise OR of two bit-packed bool arrays."""
    var length = len(lhs)
    if len(rhs) != length:
        raise Error("or_: input arrays must have equal length")
    var lhs_bv = lhs.values_bitmap()
    var rhs_bv = rhs.values_bitmap()
    var result_buf = lhs_bv | rhs_bv
    return PrimitiveArray[bool_dt](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=result_buf,
    )


def not_(arr: PrimitiveArray[bool_dt]) raises -> PrimitiveArray[bool_dt]:
    """Bitwise NOT of a bit-packed bool array."""
    var length = len(arr)
    var bv = arr.values_bitmap()
    var result_buf = ~bv
    return PrimitiveArray[bool_dt](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=result_buf,
    )


def is_null[T: DataType](arr: PrimitiveArray[T]) -> PrimitiveArray[bool_dt]:
    """Return a bool array that is True where arr has a null value."""
    var length = len(arr)
    var builder = Bitmap.alloc(length)
    for i in range(length):
        builder.set_bit(i, not arr.is_valid(i))
    var bm = builder.finish(length)
    return PrimitiveArray[bool_dt](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=bm._buffer,
    )


def is_null(arr: AnyArray) raises -> AnyArray:
    """Runtime-typed is_null."""
    comptime for dtype in numeric_dtypes:
        if arr.dtype() == dtype:
            return is_null[dtype](arr.as_primitive[dtype]()).to_any()
    raise Error(t"is_null: unsupported dtype {arr.dtype()}")


def select[
    T: DataType
](
    mask: PrimitiveArray[bool_dt],
    then_: PrimitiveArray[T],
    else_: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    """Element-wise select: result[i] = then_[i] if mask[i] else else_[i]."""
    var length = len(then_)
    if len(mask) != length or len(else_) != length:
        raise Error("select: input arrays must have equal length")
    var builder = PrimitiveBuilder[T](length)
    var data_bv = mask.values_bitmap()
    for i in range(length):
        if data_bv.test(i):
            builder._buffer.unsafe_set[T.native](i, then_.unsafe_get(i))
        else:
            builder._buffer.unsafe_set[T.native](i, else_.unsafe_get(i))
    builder._length = length
    return builder.finish()


# TODO: use SIMD select instead of naive element-wise loop when possible
def select(mask: AnyArray, then_: AnyArray, else_: AnyArray) raises -> AnyArray:
    """Runtime-typed select."""
    if then_.dtype() != else_.dtype():
        raise Error(
            t"select: dtype mismatch: {then_.dtype()} vs {else_.dtype()}"
        )
    ref bool_mask = mask.as_primitive[bool_dt]()
    comptime for dtype in numeric_dtypes:
        if then_.dtype() == dtype:
            return select[dtype](
                bool_mask,
                then_.as_primitive[dtype](),
                else_.as_primitive[dtype](),
            ).to_any()
    raise Error(t"select: unsupported dtype {then_.dtype()}")
