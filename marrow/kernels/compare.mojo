"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a ``BoolArray`` following the Arrow boolean layout.

Null propagation: if either input has a null at position ``i``, the output is
null at ``i`` (validity = ``bitmap_and(left.bitmap, right.bitmap)``).  Data
bits for null positions are set to the comparison result of the underlying
values (undefined per Arrow spec, but branch-free for performance).

Available kernels
-----------------
* ``equal``          — left[i] == right[i]
* ``not_equal``      — left[i] != right[i]
* ``less``           — left[i] <  right[i]
* ``less_equal``     — left[i] <= right[i]
* ``greater``        — left[i] >  right[i]
* ``greater_equal``  — left[i] >= right[i]

Each has a typed overload ``def[T: DataType](PrimitiveArray[T], PrimitiveArray[T])``
and a runtime-typed overload ``def(AnyArray, AnyArray)`` that dispatches via
``binary_array_dispatch``.
"""

from std.gpu.host import DeviceContext

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    AnyArray,
    StructArray,
)
from ..buffers import Bitmap
from ..views import apply
from ..dtypes import PrimitiveType, bool_ as bool_dt
from . import bitmap_and, bool_array_dispatch


# ---------------------------------------------------------------------------
# Generic comparison kernel — compare + bit-pack via apply
# ---------------------------------------------------------------------------


def _binary_cmp[
    T: PrimitiveType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        DType.bool, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Binary comparison kernel — compare + bit-pack via apply."""
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    comptime native = T.native
    var length = len(left)
    var bm = bitmap_and(left.bitmap, right.bitmap) if (
        left.bitmap or right.bitmap
    ) else Optional[Bitmap[]]()

    var result = Bitmap.alloc_device(
        ctx.value(), length
    ) if ctx else Bitmap.alloc_uninit(length)
    apply[native, func](left.values(), right.values(), result.view(), ctx)
    return BoolArray(
        length=length,
        nulls=length - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=result.to_immutable(),
    )


# ---------------------------------------------------------------------------
# SIMD predicates — def[T: DType, W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[bool, W]
# ---------------------------------------------------------------------------


def _eq[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.eq(b)


def _ne[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.ne(b)


def _lt[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.lt(b)


def _le[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.le(b)


def _gt[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.gt(b)


def _ge[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.ge(b)


# ---------------------------------------------------------------------------
# Typed public API
# ---------------------------------------------------------------------------


def equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise equality: result[i] = left[i] == right[i]."""
    return _binary_cmp[T, _eq[T.native, _], "equal"](left, right, ctx)


def not_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise inequality: result[i] = left[i] != right[i]."""
    return _binary_cmp[T, _ne[T.native, _], "not_equal"](left, right, ctx)


def less[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise less-than: result[i] = left[i] < right[i]."""
    return _binary_cmp[T, _lt[T.native, _], "less"](left, right, ctx)


def less_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise less-or-equal: result[i] = left[i] <= right[i]."""
    return _binary_cmp[T, _le[T.native, _], "less_equal"](left, right, ctx)


def greater[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise greater-than: result[i] = left[i] > right[i]."""
    return _binary_cmp[T, _gt[T.native, _], "greater"](left, right, ctx)


def greater_equal[
    T: PrimitiveType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise greater-or-equal: result[i] = left[i] >= right[i]."""
    return _binary_cmp[T, _ge[T.native, _], "greater_equal"](left, right, ctx)


# ---------------------------------------------------------------------------
# String overloads
# ---------------------------------------------------------------------------


def equal(left: StringArray, right: StringArray) raises -> BoolArray:
    """Element-wise string equality."""
    var n = len(left)
    if len(right) != n:
        raise Error("equal: string arrays must have the same length")
    var bm = bitmap_and(left.bitmap, right.bitmap)
    var bm_builder = Bitmap.alloc_zeroed(n)
    for i in range(n):
        var eq = String(left.unsafe_get(UInt(i))) == String(
            right.unsafe_get(UInt(i))
        )
        if eq:
            bm_builder.set(i)
    return BoolArray(
        length=n,
        nulls=n - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=bm_builder.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Runtime-typed overloads
# ---------------------------------------------------------------------------


def equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed equal."""
    if left.dtype().is_string():
        return equal(left.as_string(), right.as_string()).to_any()
    return bool_array_dispatch["equal", equal[_]](left, right)


def equal(left: StructArray, right: StructArray) raises -> BoolArray:
    """Element-wise struct equality: all corresponding columns must match.

    Returns a boolean array where element ``i`` is True iff
    ``left[i] == right[i]`` across every child column.
    """
    from .boolean import and_

    var n_keys = len(left.children)
    var mask = (
        equal(left.children[0].copy(), right.children[0].copy())
        .as_bool()
        .copy()
    )
    for k in range(1, n_keys):
        mask = and_(
            mask,
            equal(left.children[k].copy(), right.children[k].copy())
            .as_bool()
            .copy(),
        )
    return mask^


def not_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed not_equal."""
    return bool_array_dispatch["not_equal", not_equal[_]](left, right)


def less(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed less."""
    return bool_array_dispatch["less", less[_]](left, right)


def less_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed less_equal."""
    return bool_array_dispatch["less_equal", less_equal[_]](left, right)


def greater(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed greater."""
    return bool_array_dispatch["greater", greater[_]](left, right)


def greater_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed greater_equal."""
    return bool_array_dispatch["greater_equal", greater_equal[_]](left, right)
