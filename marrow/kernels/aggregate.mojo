"""Aggregate (reduction) kernels.

Each reduction has:
  - A typed overload: ``def[T](PrimitiveArray[T]) -> PrimitiveScalar[T]``
  - A runtime-typed overload: ``def(AnyArray) -> AnyScalar``

The typed overloads all delegate to ``_reduce[T, combine]``, which is generic
over a thin SIMD combine function — the same pattern as ``_binary[T, func]`` in
``arithmetic.mojo`` and ``apply[op]`` in ``views.mojo``.

To add a new aggregate kernel:
  1. Define a thin SIMD combine:
     ``def _op[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]``
  2. Add a typed overload calling ``_reduce[T, _op[T.native, _]](array, identity)``
  3. Add an AnyArray overload using ``unary_scalar_dispatch``
"""

import std.math as math
from std.sys.info import simd_width_of

from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..dtypes import *
from ..scalars import PrimitiveScalar, AnyScalar
from . import unary_scalar_dispatch


# ---------------------------------------------------------------------------
# SIMD combine functions — thin helpers passed as parameters to _reduce
# ---------------------------------------------------------------------------


def _add[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


def _mul[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a * b


def _min[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.min(a, b)


def _max[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return math.max(a, b)


# ---------------------------------------------------------------------------
# Generic reduction helper
# ---------------------------------------------------------------------------


def _reduce[
    T: PrimitiveType,
    combine: def[W: Int](
        SIMD[T.native, W], SIMD[T.native, W]
    ) thin -> SIMD[T.native, W],
](array: PrimitiveArray[T], identity: Scalar[T.native]) raises -> Scalar[
    T.native
]:
    """Vectorized reduction over a PrimitiveArray with a SIMD combine function.

    Bitmap-aware: null elements are replaced with `identity` so they
    contribute nothing to the result.
    """
    comptime native = T.native
    comptime W = simd_width_of[native]()
    var length = len(array)
    var vals = array.values()
    var acc = SIMD[native, W](identity)
    var i = 0

    if array.bitmap:
        var bm = array.validity().value()
        while i + W <= length:
            var data = vals.load[W](i)
            acc = combine[W](
                acc, bm.mask[W](i).select(data, SIMD[native, W](identity))
            )
            i += W
    else:
        while i + W <= length:
            acc = combine[W](acc, vals.load[W](i))
            i += W

    var out: Scalar[native] = acc[0]
    comptime for lane in range(1, W):
        out = combine[1](out, acc[lane])

    while i < length:
        if (not array.bitmap) or array.validity().value().test(i):
            out = combine[1](out, vals[i])
        i += 1

    return out


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def sum_[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    return PrimitiveScalar[T](
        _reduce[T, _add[T.native, _]](array, Scalar[T.native](0)),
        array.dtype.copy(),
    )


def sum_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed sum."""
    return unary_scalar_dispatch["sum_", sum_[_]](array)


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def product[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null."""
    return PrimitiveScalar[T](
        _reduce[T, _mul[T.native, _]](array, Scalar[T.native](1)),
        array.dtype.copy(),
    )


def product(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed product."""
    return unary_scalar_dispatch["product", product[_]](array)


# ---------------------------------------------------------------------------
# min_
# ---------------------------------------------------------------------------


def min_[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Minimum of all valid (non-null) elements.

    Returns MAX_FINITE if empty or all null.
    """
    return PrimitiveScalar[T](
        _reduce[T, _min[T.native, _]](array, Scalar[T.native].MAX_FINITE),
        array.dtype.copy(),
    )


def min_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed min."""
    return unary_scalar_dispatch["min_", min_[_]](array)


# ---------------------------------------------------------------------------
# max_
# ---------------------------------------------------------------------------


def max_[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Maximum of all valid (non-null) elements.

    Returns MIN_FINITE if empty or all null.
    """
    return PrimitiveScalar[T](
        _reduce[T, _max[T.native, _]](array, Scalar[T.native].MIN_FINITE),
        array.dtype.copy(),
    )


def max_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed max."""
    return unary_scalar_dispatch["max_", max_[_]](array)


# ---------------------------------------------------------------------------
# any_ / all_  (bool arrays) — implemented via SIMD bitmap operations
# ---------------------------------------------------------------------------


def any_(array: AnyArray) raises -> Bool:
    return any_(array.as_bool())


def any_(array: BoolArray) raises -> Bool:
    """True if any valid element is True. False if empty or all null."""
    var n = len(array)
    var data_bv = array.values()
    if not array.bitmap:
        return Bool(data_bv)
    var validity_bv = array.validity().value()
    var i = 0
    while i + 64 <= n:
        if (
            data_bv.load_bits[DType.uint64](i)
            & validity_bv.load_bits[DType.uint64](i)
        ) != 0:
            return True
        i += 64
    if i < n:
        var mask = (UInt64(1) << UInt64(n - i)) - 1
        if (
            data_bv.load_bits[DType.uint64](i)
            & validity_bv.load_bits[DType.uint64](i)
        ) & mask != 0:
            return True
    return False


def all_(array: AnyArray) raises -> Bool:
    return all_(array.as_bool())


def all_(array: BoolArray) raises -> Bool:
    """True if all valid elements are True. True if empty or all null."""
    var n = len(array)
    var data_bv = array.values()
    if not array.bitmap:
        return data_bv.all_set()
    var validity_bv = array.validity().value()
    var i = 0
    while i + 64 <= n:
        var v = validity_bv.load_bits[DType.uint64](i)
        if (data_bv.load_bits[DType.uint64](i) & v) != v:
            return False
        i += 64
    if i < n:
        var mask = (UInt64(1) << UInt64(n - i)) - 1
        var v = validity_bv.load_bits[DType.uint64](i) & mask
        if (data_bv.load_bits[DType.uint64](i) & v) != v:
            return False
    return True
