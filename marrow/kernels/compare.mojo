"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a bit-packed ``PrimitiveArray[bool_]`` following the Arrow boolean layout.

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
and a runtime-typed overload ``def(Array, Array)`` that dispatches via
``binary_array_dispatch``.
"""

from ..arrays import PrimitiveArray, Array
from ..dtypes import DataType, bool_ as bool_dt
from . import binary_simd, binary_array_dispatch


# ---------------------------------------------------------------------------
# SIMD predicates — def[T: DType, W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[bool, W]
# ---------------------------------------------------------------------------


def _eq[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.eq(b))


def _ne[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.ne(b))


def _lt[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.lt(b))


def _le[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.le(b))


def _gt[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.gt(b))


def _ge[
    T: DType, W: Int
](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[bool_dt.native, W]:
    return rebind[SIMD[bool_dt.native, W]](a.ge(b))


# ---------------------------------------------------------------------------
# Typed public API
# ---------------------------------------------------------------------------


def equal[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise equality: result[i] = left[i] == right[i]."""
    return binary_simd[T, bool_dt, _eq[T.native, _], "equal"](left, right)


def not_equal[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise inequality: result[i] = left[i] != right[i]."""
    return binary_simd[T, bool_dt, _ne[T.native, _], "not_equal"](left, right)


def less[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise less-than: result[i] = left[i] < right[i]."""
    return binary_simd[T, bool_dt, _lt[T.native, _], "less"](left, right)


def less_equal[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise less-or-equal: result[i] = left[i] <= right[i]."""
    return binary_simd[T, bool_dt, _le[T.native, _], "less_equal"](left, right)


def greater[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise greater-than: result[i] = left[i] > right[i]."""
    return binary_simd[T, bool_dt, _gt[T.native, _], "greater"](left, right)


def greater_equal[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    bool_dt
]:
    """Element-wise greater-or-equal: result[i] = left[i] >= right[i]."""
    return binary_simd[T, bool_dt, _ge[T.native, _], "greater_equal"](
        left, right
    )


# ---------------------------------------------------------------------------
# Runtime-typed overloads
# ---------------------------------------------------------------------------


def equal(left: Array, right: Array) raises -> Array:
    """Runtime-typed equal."""
    return binary_array_dispatch["equal", bool_dt, equal[_]](left, right)


def not_equal(left: Array, right: Array) raises -> Array:
    """Runtime-typed not_equal."""
    return binary_array_dispatch["not_equal", bool_dt, not_equal[_]](
        left, right
    )


def less(left: Array, right: Array) raises -> Array:
    """Runtime-typed less."""
    return binary_array_dispatch["less", bool_dt, less[_]](left, right)


def less_equal(left: Array, right: Array) raises -> Array:
    """Runtime-typed less_equal."""
    return binary_array_dispatch["less_equal", bool_dt, less_equal[_]](
        left, right
    )


def greater(left: Array, right: Array) raises -> Array:
    """Runtime-typed greater."""
    return binary_array_dispatch["greater", bool_dt, greater[_]](left, right)


def greater_equal(left: Array, right: Array) raises -> Array:
    """Runtime-typed greater_equal."""
    return binary_array_dispatch["greater_equal", bool_dt, greater_equal[_]](
        left, right
    )
