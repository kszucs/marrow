"""Aggregate (reduction) kernels.

Each reduction has:
  - A typed overload: def[T: DataType](PrimitiveArray[T]) -> Scalar[T.native]
  - A runtime-typed overload (where applicable): def(Array) raises -> Scalar[float64]

Available reductions and their SIMD horizontal method:
  sum      — reduce_add,  identity = 0
  product  — reduce_mul,  identity = 1
  min_     — reduce_min,  identity = MAX_FINITE
  max_     — reduce_max,  identity = MIN_FINITE
  any_     — reduce_or,   identity = False  (bool arrays only)
  all_     — reduce_and,  identity = True   (bool arrays only)
"""

import std.math as math

from ..arrays import PrimitiveArray, Array
from ..dtypes import (
    DataType,
    numeric_dtypes,
    bool_ as bool_dt,
)
from . import reduce_simd


# ---------------------------------------------------------------------------
# SIMD helpers — combine[W] and horizontal[W] pairs
# ---------------------------------------------------------------------------


# sum
def _add[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


def _horizontal_add[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_add()


# product
def _mul[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a * b


def _horizontal_mul[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_mul()


# min_
def _vmin[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.min(a, b)


def _horizontal_min[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_min()


# max_
def _vmax[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.max(a, b)


def _horizontal_max[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_max()


# any_ / all_
def _or[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a | b


def _horizontal_or[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_or()


def _and[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a & b


def _horizontal_and[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
    return v.reduce_and()


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def sum[T: DataType](array: PrimitiveArray[T]) -> Scalar[T.native]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    return reduce_simd[
        T, combine=_add[T.native, _], horizontal=_horizontal_add[T.native, _]
    ](array, Scalar[T.native](0))


def sum(array: Array) raises -> Scalar[DType.float64]:
    """Runtime-typed sum, returns float64."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return sum[dtype](PrimitiveArray[dtype](data=array)).cast[
                DType.float64
            ]()
    raise Error("sum: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def product[T: DataType](array: PrimitiveArray[T]) -> Scalar[T.native]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null.
    """
    return reduce_simd[
        T, combine=_mul[T.native, _], horizontal=_horizontal_mul[T.native, _]
    ](array, Scalar[T.native](1))


def product(array: Array) raises -> Scalar[DType.float64]:
    """Runtime-typed product, returns float64."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return product[dtype](PrimitiveArray[dtype](data=array)).cast[
                DType.float64
            ]()
    raise Error("product: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# min_
# ---------------------------------------------------------------------------


def min_[T: DataType](array: PrimitiveArray[T]) -> Scalar[T.native]:
    """Minimum of all valid (non-null) elements.

    Returns Scalar[T].MAX_FINITE if empty or all null.
    """
    return reduce_simd[
        T, combine=_vmin[T.native, _], horizontal=_horizontal_min[T.native, _]
    ](array, Scalar[T.native].MAX_FINITE)


def min_(array: Array) raises -> Scalar[DType.float64]:
    """Runtime-typed min, returns float64."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return min_[dtype](PrimitiveArray[dtype](data=array)).cast[
                DType.float64
            ]()
    raise Error("min_: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# max_
# ---------------------------------------------------------------------------


def max_[T: DataType](array: PrimitiveArray[T]) -> Scalar[T.native]:
    """Maximum of all valid (non-null) elements.

    Returns Scalar[T].MIN_FINITE if empty or all null.
    """
    return reduce_simd[
        T, combine=_vmax[T.native, _], horizontal=_horizontal_max[T.native, _]
    ](array, Scalar[T.native].MIN_FINITE)


def max_(array: Array) raises -> Scalar[DType.float64]:
    """Runtime-typed max, returns float64."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return max_[dtype](PrimitiveArray[dtype](data=array)).cast[
                DType.float64
            ]()
    raise Error("max_: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# any_ / all_  (bool arrays)
# ---------------------------------------------------------------------------


def any_(array: PrimitiveArray[bool_dt]) -> Bool:
    """True if any valid element is True. False if empty or all null."""
    return Bool(
        reduce_simd[
            bool_dt,
            combine=_or[bool_dt.native, _],
            horizontal=_horizontal_or[bool_dt.native, _],
        ](array, Scalar[bool_dt.native](False))
    )


def all_(array: PrimitiveArray[bool_dt]) -> Bool:
    """True if all valid elements are True. True if empty or all null."""
    return Bool(
        reduce_simd[
            bool_dt,
            combine=_and[bool_dt.native, _],
            horizontal=_horizontal_and[bool_dt.native, _],
        ](array, Scalar[bool_dt.native](True))
    )
