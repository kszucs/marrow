"""Boolean and bitwise kernels."""


from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import (
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
    bool_ as bool_dt,
)
from ..views import BitmapView


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


def and_(lhs: AnyArray, rhs: AnyArray) raises -> AnyArray:
    """Runtime-typed AND: dispatches to the typed BoolArray overload."""
    if lhs.dtype() != bool_dt or rhs.dtype() != bool_dt:
        raise Error("and_: inputs must be bool arrays")
    return and_(lhs.as_bool(), rhs.as_bool()).to_any()


def or_(lhs: AnyArray, rhs: AnyArray) raises -> AnyArray:
    """Runtime-typed OR: dispatches to the typed BoolArray overload."""
    if lhs.dtype() != bool_dt or rhs.dtype() != bool_dt:
        raise Error("or_: inputs must be bool arrays")
    return or_(lhs.as_bool(), rhs.as_bool()).to_any()


def not_(arr: AnyArray) raises -> AnyArray:
    """Runtime-typed NOT: dispatches to the typed BoolArray overload."""
    if arr.dtype() != bool_dt:
        raise Error("not_: input must be a bool array")
    return not_(arr.as_bool()).to_any()


# TODO: it should return with the bitmap from the input array instead of creating a new one, but that requires
def is_null[T: PrimitiveType](arr: PrimitiveArray[T]) -> BoolArray:
    """Return a bool array that is True where arr has a null value."""
    var length = len(arr)
    var builder = Bitmap.alloc_zeroed(length)
    for i in range(length):
        if not arr.is_valid(i):
            builder.set(i)
        else:
            builder.clear(i)
    var bm = builder.to_immutable()
    return BoolArray(length=length, nulls=0, offset=0, bitmap=None, buffer=bm)


def is_null(arr: AnyArray) raises -> AnyArray:
    """Runtime-typed is_null."""
    if arr.dtype() == int8:
        return is_null[Int8Type](arr.as_primitive[Int8Type]()).to_any()
    elif arr.dtype() == int16:
        return is_null[Int16Type](arr.as_primitive[Int16Type]()).to_any()
    elif arr.dtype() == int32:
        return is_null[Int32Type](arr.as_primitive[Int32Type]()).to_any()
    elif arr.dtype() == int64:
        return is_null[Int64Type](arr.as_primitive[Int64Type]()).to_any()
    elif arr.dtype() == uint8:
        return is_null[UInt8Type](arr.as_primitive[UInt8Type]()).to_any()
    elif arr.dtype() == uint16:
        return is_null[UInt16Type](arr.as_primitive[UInt16Type]()).to_any()
    elif arr.dtype() == uint32:
        return is_null[UInt32Type](arr.as_primitive[UInt32Type]()).to_any()
    elif arr.dtype() == uint64:
        return is_null[UInt64Type](arr.as_primitive[UInt64Type]()).to_any()
    elif arr.dtype() == float16:
        return is_null[Float16Type](arr.as_primitive[Float16Type]()).to_any()
    elif arr.dtype() == float32:
        return is_null[Float32Type](arr.as_primitive[Float32Type]()).to_any()
    elif arr.dtype() == float64:
        return is_null[Float64Type](arr.as_primitive[Float64Type]()).to_any()
    raise Error(t"is_null: unsupported dtype {arr.dtype()}")


def select[
    T: PrimitiveType
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
            builder.unsafe_set(i, then_.unsafe_get(i))
        else:
            builder.unsafe_set(i, else_.unsafe_get(i))
    builder.set_length(length)
    return builder.finish()


# TODO: use SIMD select instead of naive element-wise loop when possible
def select(mask: AnyArray, then_: AnyArray, else_: AnyArray) raises -> AnyArray:
    """Runtime-typed select."""
    if then_.dtype() != else_.dtype():
        raise Error(
            t"select: dtype mismatch: {then_.dtype()} vs {else_.dtype()}"
        )
    ref bool_mask = mask.as_bool()
    if then_.dtype() == int8:
        return select[Int8Type](
            bool_mask,
            then_.as_primitive[Int8Type](),
            else_.as_primitive[Int8Type](),
        ).to_any()
    elif then_.dtype() == int16:
        return select[Int16Type](
            bool_mask,
            then_.as_primitive[Int16Type](),
            else_.as_primitive[Int16Type](),
        ).to_any()
    elif then_.dtype() == int32:
        return select[Int32Type](
            bool_mask,
            then_.as_primitive[Int32Type](),
            else_.as_primitive[Int32Type](),
        ).to_any()
    elif then_.dtype() == int64:
        return select[Int64Type](
            bool_mask,
            then_.as_primitive[Int64Type](),
            else_.as_primitive[Int64Type](),
        ).to_any()
    elif then_.dtype() == uint8:
        return select[UInt8Type](
            bool_mask,
            then_.as_primitive[UInt8Type](),
            else_.as_primitive[UInt8Type](),
        ).to_any()
    elif then_.dtype() == uint16:
        return select[UInt16Type](
            bool_mask,
            then_.as_primitive[UInt16Type](),
            else_.as_primitive[UInt16Type](),
        ).to_any()
    elif then_.dtype() == uint32:
        return select[UInt32Type](
            bool_mask,
            then_.as_primitive[UInt32Type](),
            else_.as_primitive[UInt32Type](),
        ).to_any()
    elif then_.dtype() == uint64:
        return select[UInt64Type](
            bool_mask,
            then_.as_primitive[UInt64Type](),
            else_.as_primitive[UInt64Type](),
        ).to_any()
    elif then_.dtype() == float16:
        return select[Float16Type](
            bool_mask,
            then_.as_primitive[Float16Type](),
            else_.as_primitive[Float16Type](),
        ).to_any()
    elif then_.dtype() == float32:
        return select[Float32Type](
            bool_mask,
            then_.as_primitive[Float32Type](),
            else_.as_primitive[Float32Type](),
        ).to_any()
    elif then_.dtype() == float64:
        return select[Float64Type](
            bool_mask,
            then_.as_primitive[Float64Type](),
            else_.as_primitive[Float64Type](),
        ).to_any()
    raise Error(t"select: unsupported dtype {then_.dtype()}")
