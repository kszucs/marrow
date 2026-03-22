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

from ..arrays import PrimitiveArray, AnyArray
from ..bitmap import Bitmap
from ..dtypes import (
    DataType,
    numeric_dtypes,
    float64,
    bool_ as bool_dt,
)
from ..scalars import PrimitiveScalar, AnyScalar


# ---------------------------------------------------------------------------
# Generic reduction helper (internal — returns Mojo Scalar for SIMD compat)
# ---------------------------------------------------------------------------


def _reduce[
    T: DataType, op: StringLiteral
](array: PrimitiveArray[T], identity: Scalar[T.native]) raises -> Scalar[
    T.native
]:
    """Reduce a primitive array using one of sum/product/min/max.

    Bitmap-aware: null elements are replaced with `identity` so they
    contribute nothing. The `op` parameter selects the stdlib reduction
    at compile time.
    """
    comptime native = T.native
    var length = len(array)
    var arr_off = array.offset
    var out = identity

    @always_inline
    @parameter
    def output_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank], val: SIMD[native, width]):
        out = val[0]

    if array.bitmap:
        var bitmap = array.bitmap.value()

        @always_inline
        @parameter
        def input_fn_nulls[
            width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[native, width]:
            var i = idx[0]
            var data = array.buffer.simd_load[native, width](arr_off + i)
            return bitmap.mask[native, width](arr_off + i).select(
                data, SIMD[native, width](identity)
            )

        comptime if op == "sum":
            algo_sum[
                native,
                input_fn_nulls,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "product":
            algo_product[
                native,
                input_fn_nulls,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "min":
            algo_min[
                native,
                input_fn_nulls,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "max":
            algo_max[
                native,
                input_fn_nulls,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
    else:

        @always_inline
        @parameter
        def input_fn[
            width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[native, width]:
            return array.buffer.simd_load[native, width](arr_off + idx[0])

        comptime if op == "sum":
            algo_sum[
                native,
                input_fn,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "product":
            algo_product[
                native,
                input_fn,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "min":
            algo_min[
                native,
                input_fn,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)
        elif op == "max":
            algo_max[
                native,
                input_fn,
                output_fn,
                single_thread_blocking_override=True,
            ](Index(length), reduce_dim=0)

    return out


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def sum_[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    return PrimitiveScalar[T](_reduce[T, "sum"](array, Scalar[T.native](0)))


def sum_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed sum."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return sum_[dtype](PrimitiveArray[dtype](data=array))
    raise Error("sum: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def product[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null.
    """
    return PrimitiveScalar[T](_reduce[T, "product"](array, Scalar[T.native](1)))


def product(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed product."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return product[dtype](PrimitiveArray[dtype](data=array))
    raise Error("product: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# min_
# ---------------------------------------------------------------------------


def min_[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Minimum of all valid (non-null) elements.

    Returns MAX_FINITE if empty or all null.
    """
    return PrimitiveScalar[T](
        _reduce[T, "min"](array, Scalar[T.native].MAX_FINITE)
    )


def min_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed min."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return min_[dtype](PrimitiveArray[dtype](data=array))
    raise Error("min_: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# max_
# ---------------------------------------------------------------------------


def max_[T: DataType](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Maximum of all valid (non-null) elements.

    Returns MIN_FINITE if empty or all null.
    """
    return PrimitiveScalar[T](
        _reduce[T, "max"](array, Scalar[T.native].MIN_FINITE)
    )


def max_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed max."""
    comptime for dtype in numeric_dtypes:
        if array.dtype == dtype:
            return max_[dtype](PrimitiveArray[dtype](data=array))
    raise Error("max_: unsupported dtype ", array.dtype)


# ---------------------------------------------------------------------------
# any_ / all_  (bool arrays) — implemented via SIMD bitmap operations
# ---------------------------------------------------------------------------


def any_(array: PrimitiveArray[bool_dt]) raises -> Bool:
    """True if any valid element is True. False if empty or all null."""
    var data_bm = Bitmap(array.buffer, array.offset, len(array))
    if array.bitmap:
        return (data_bm & array.bitmap.value()).any_set()
    else:
        return data_bm.any_set()


def all_(array: PrimitiveArray[bool_dt]) raises -> Bool:
    """True if all valid elements are True. True if empty or all null."""
    var data_bm = Bitmap(array.buffer, array.offset, len(array))
    if array.bitmap:
        var validity = array.bitmap.value()
        return (data_bm & validity) == validity
    else:
        return data_bm.all_set()
