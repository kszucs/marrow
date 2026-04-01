"""Boolean and bitwise kernels."""


from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..buffers import Bitmap
from ..builders import PrimitiveBuilder
from ..dtypes import PrimitiveType, numeric_types, bool_ as bool_dt
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
    comptime for dtype in numeric_types:
        if arr.dtype() == dtype():
            return is_null[dtype](arr.as_primitive[dtype]()).to_any()
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
    comptime for dtype in numeric_types:
        if then_.dtype() == dtype():
            return select[dtype](
                bool_mask,
                then_.as_primitive[dtype](),
                else_.as_primitive[dtype](),
            ).to_any()
    raise Error(t"select: unsupported dtype {then_.dtype()}")
