"""Boolean and bitwise kernels."""

from std.bit import pop_count
from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import DataType, numeric_dtypes, bool_ as bool_dt
from ..views import BitmapView


def count_true(array: BoolArray) -> Int:
    """Count True (and non-null) values in a bit-packed boolean array."""
    var data_bv = array.values().slice(array.offset, array.length)
    if array.nulls == 0:
        return data_bv.count_set_bits()
    var validity_bv = BitmapView(array.bitmap.value()).slice(array.offset, array.length)
    var count = 0
    var n = array.length
    var i = 0
    while i + 64 <= n:
        count += Int(pop_count(data_bv.load_word(i) & validity_bv.load_word(i)))
        i += 64
    if i < n:
        var mask = (UInt64(1) << UInt64(n - i)) - 1
        count += Int(pop_count((data_bv.load_word(i) & validity_bv.load_word(i)) & mask))
    return count


def count_false(array: BoolArray) -> Int:
    """Count False (and non-null) values in a bit-packed boolean array."""
    return array.length - array.nulls - count_true(array)


def and_(lhs: BoolArray, rhs: BoolArray) raises -> BoolArray:
    """Bitwise AND of two bit-packed bool arrays."""
    var length = len(lhs)
    if len(rhs) != length:
        raise Error("and_: input arrays must have equal length")
    return BoolArray(
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=(lhs.values() & rhs.values()).to_immutable(),
    )


def or_(lhs: BoolArray, rhs: BoolArray) raises -> BoolArray:
    """Bitwise OR of two bit-packed bool arrays."""
    var length = len(lhs)
    if len(rhs) != length:
        raise Error("or_: input arrays must have equal length")
    return BoolArray(
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=(lhs.values() | rhs.values()).to_immutable(),
    )


def not_(arr: BoolArray) raises -> BoolArray:
    """Bitwise NOT of a bit-packed bool array."""
    var length = len(arr)
    return BoolArray(
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=(~arr.values()).to_immutable(),
    )


# TODO: it should return with the bitmap from the input array instead of creating a new one, but that requires
def is_null[T: DataType](arr: PrimitiveArray[T]) -> BoolArray:
    """Return a bool array that is True where arr has a null value."""
    var length = len(arr)
    var builder = Bitmap.alloc_zeroed(length)
    builder.length = length
    for i in range(length):
        if not arr.is_valid(i):
            builder.set(i)
        else:
            builder.clear(i)
    var bm = builder.to_immutable()
    return BoolArray(length=length, nulls=0, offset=0, bitmap=None, buffer=bm)


def is_null(arr: AnyArray) raises -> AnyArray:
    """Runtime-typed is_null."""
    comptime for dtype in numeric_dtypes:
        if arr.dtype() == dtype:
            return is_null[dtype](arr.as_primitive[dtype]()).to_any()
    raise Error(t"is_null: unsupported dtype {arr.dtype()}")


def select[
    T: DataType
](
    mask: BoolArray,
    then_: PrimitiveArray[T],
    else_: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    """Element-wise select: result[i] = then_[i] if mask[i] else else_[i]."""
    var length = len(then_)
    if len(mask) != length or len(else_) != length:
        raise Error("select: input arrays must have equal length")
    var builder = PrimitiveBuilder[T](length)
    var data_bv = mask.values()
    for i in range(length):
        if data_bv.test(mask.offset + i):
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
    ref bool_mask = mask.as_bool()
    comptime for dtype in numeric_dtypes:
        if then_.dtype() == dtype:
            return select[dtype](
                bool_mask,
                then_.as_primitive[dtype](),
                else_.as_primitive[dtype](),
            ).to_any()
    raise Error(t"select: unsupported dtype {then_.dtype()}")
