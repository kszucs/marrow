"""Aggregate (reduction) kernels using std.algorithm reductions.

Each reduction has:
  - A typed overload: ``def[T](PrimitiveArray[T]) -> PrimitiveScalar[T]``
  - A runtime-typed overload: ``def(AnyArray) -> AnyScalar``

Bitmap-aware loading is fused into the stdlib's `input_fn` callback:
null elements are replaced with the reduction's identity value so they
contribute nothing to the result (0 for sum, 1 for product, MAX for min, etc.).
"""

from std.algorithm.reduction import (
    sum as algo_sum,
    product as algo_product,
    min as algo_min,
    max as algo_max,
)
from std.utils.index import Index, IndexList

from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..dtypes import *
from ..scalars import PrimitiveScalar, AnyScalar
from . import unary_scalar_dispatch


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def sum_[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    comptime native = T.native
    var length = len(array)
    var vals = array.values()
    var out = Scalar[native](0)

    @always_inline
    @parameter
    def output_fn[w: Int, r: Int](idx: IndexList[r], val: SIMD[native, w]):
        out = val[0]

    if array.bitmap:
        var bm = array.validity().value()

        @always_inline
        @parameter
        def input_fn_nulls[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            var i = idx[0]
            var data = vals.load[w](i)
            return bm.mask[w](i).select(data, SIMD[native, w](0))

        algo_sum[native, input_fn_nulls, output_fn, True](
            Index(length), reduce_dim=0
        )
    else:

        @always_inline
        @parameter
        def input_fn[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            return vals.load[w](idx[0])

        algo_sum[native, input_fn, output_fn, True](Index(length), reduce_dim=0)

    return PrimitiveScalar[T](out, array.dtype.copy())


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
    comptime native = T.native
    var length = len(array)
    var vals = array.values()
    var out = Scalar[native](1)

    @always_inline
    @parameter
    def output_fn[w: Int, r: Int](idx: IndexList[r], val: SIMD[native, w]):
        out = val[0]

    if array.bitmap:
        var bm = array.validity().value()

        @always_inline
        @parameter
        def input_fn_nulls[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            var i = idx[0]
            var data = vals.load[w](i)
            return bm.mask[w](i).select(data, SIMD[native, w](1))

        algo_product[native, input_fn_nulls, output_fn, True](
            Index(length), reduce_dim=0
        )
    else:

        @always_inline
        @parameter
        def input_fn[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            return vals.load[w](idx[0])

        algo_product[native, input_fn, output_fn, True](
            Index(length), reduce_dim=0
        )

    return PrimitiveScalar[T](out, array.dtype.copy())


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
    comptime native = T.native
    comptime identity = Scalar[native].MAX_FINITE
    var length = len(array)
    var vals = array.values()
    var out = identity

    @always_inline
    @parameter
    def output_fn[w: Int, r: Int](idx: IndexList[r], val: SIMD[native, w]):
        out = val[0]

    if array.bitmap:
        var bm = array.validity().value()

        @always_inline
        @parameter
        def input_fn_nulls[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            var i = idx[0]
            var data = vals.load[w](i)
            return bm.mask[w](i).select(data, SIMD[native, w](identity))

        algo_min[native, input_fn_nulls, output_fn, True](
            Index(length), reduce_dim=0
        )
    else:

        @always_inline
        @parameter
        def input_fn[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            return vals.load[w](idx[0])

        algo_min[native, input_fn, output_fn, True](Index(length), reduce_dim=0)

    return PrimitiveScalar[T](out, array.dtype.copy())


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
    comptime native = T.native
    comptime identity = Scalar[native].MIN_FINITE
    var length = len(array)
    var vals = array.values()
    var out = identity

    @always_inline
    @parameter
    def output_fn[w: Int, r: Int](idx: IndexList[r], val: SIMD[native, w]):
        out = val[0]

    if array.bitmap:
        var bm = array.validity().value()

        @always_inline
        @parameter
        def input_fn_nulls[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            var i = idx[0]
            var data = vals.load[w](i)
            return bm.mask[w](i).select(data, SIMD[native, w](identity))

        algo_max[native, input_fn_nulls, output_fn, True](
            Index(length), reduce_dim=0
        )
    else:

        @always_inline
        @parameter
        def input_fn[w: Int, r: Int](idx: IndexList[r]) -> SIMD[native, w]:
            return vals.load[w](idx[0])

        algo_max[native, input_fn, output_fn, True](Index(length), reduce_dim=0)

    return PrimitiveScalar[T](out, array.dtype.copy())


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
