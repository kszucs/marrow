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
from ..scalars import PrimitiveScalar, AnyScalar


# ---------------------------------------------------------------------------
# Generic reduction helper (internal — returns Mojo Scalar for SIMD compat)
# ---------------------------------------------------------------------------


def _reduce[
    T: PrimitiveType, op: StringLiteral
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
    var vals = array.values()
    var out = identity

    @always_inline
    @parameter
    def output_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank], val: SIMD[native, width]):
        out = val[0]

    if array.bitmap:
        var bm = array.validity().value()

        @always_inline
        @parameter
        def input_fn_nulls[
            width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[native, width]:
            var i = idx[0]
            var data = vals.load[width](i)
            return bm.mask[width](i).select(data, SIMD[native, width](identity))

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
            return vals.load[width](idx[0])

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


def sum_[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    return PrimitiveScalar[T](_reduce[T, "sum"](array, Scalar[T.native](0)))


def sum_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed sum."""
    if array.dtype() == int8:
        return sum_[Int8Type](array.as_primitive[Int8Type]())
    elif array.dtype() == int16:
        return sum_[Int16Type](array.as_primitive[Int16Type]())
    elif array.dtype() == int32:
        return sum_[Int32Type](array.as_primitive[Int32Type]())
    elif array.dtype() == int64:
        return sum_[Int64Type](array.as_primitive[Int64Type]())
    elif array.dtype() == uint8:
        return sum_[UInt8Type](array.as_primitive[UInt8Type]())
    elif array.dtype() == uint16:
        return sum_[UInt16Type](array.as_primitive[UInt16Type]())
    elif array.dtype() == uint32:
        return sum_[UInt32Type](array.as_primitive[UInt32Type]())
    elif array.dtype() == uint64:
        return sum_[UInt64Type](array.as_primitive[UInt64Type]())
    elif array.dtype() == float16:
        return sum_[Float16Type](array.as_primitive[Float16Type]())
    elif array.dtype() == float32:
        return sum_[Float32Type](array.as_primitive[Float32Type]())
    elif array.dtype() == float64:
        return sum_[Float64Type](array.as_primitive[Float64Type]())
    raise Error("sum: unsupported dtype ", array.dtype())


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def product[
    T: PrimitiveType
](array: PrimitiveArray[T]) raises -> PrimitiveScalar[T]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null.
    """
    return PrimitiveScalar[T](_reduce[T, "product"](array, Scalar[T.native](1)))


def product(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed product."""
    if array.dtype() == int8:
        return product[Int8Type](array.as_primitive[Int8Type]())
    elif array.dtype() == int16:
        return product[Int16Type](array.as_primitive[Int16Type]())
    elif array.dtype() == int32:
        return product[Int32Type](array.as_primitive[Int32Type]())
    elif array.dtype() == int64:
        return product[Int64Type](array.as_primitive[Int64Type]())
    elif array.dtype() == uint8:
        return product[UInt8Type](array.as_primitive[UInt8Type]())
    elif array.dtype() == uint16:
        return product[UInt16Type](array.as_primitive[UInt16Type]())
    elif array.dtype() == uint32:
        return product[UInt32Type](array.as_primitive[UInt32Type]())
    elif array.dtype() == uint64:
        return product[UInt64Type](array.as_primitive[UInt64Type]())
    elif array.dtype() == float16:
        return product[Float16Type](array.as_primitive[Float16Type]())
    elif array.dtype() == float32:
        return product[Float32Type](array.as_primitive[Float32Type]())
    elif array.dtype() == float64:
        return product[Float64Type](array.as_primitive[Float64Type]())
    raise Error("product: unsupported dtype ", array.dtype())


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
        _reduce[T, "min"](array, Scalar[T.native].MAX_FINITE)
    )


def min_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed min."""
    if array.dtype() == int8:
        return min_[Int8Type](array.as_primitive[Int8Type]())
    elif array.dtype() == int16:
        return min_[Int16Type](array.as_primitive[Int16Type]())
    elif array.dtype() == int32:
        return min_[Int32Type](array.as_primitive[Int32Type]())
    elif array.dtype() == int64:
        return min_[Int64Type](array.as_primitive[Int64Type]())
    elif array.dtype() == uint8:
        return min_[UInt8Type](array.as_primitive[UInt8Type]())
    elif array.dtype() == uint16:
        return min_[UInt16Type](array.as_primitive[UInt16Type]())
    elif array.dtype() == uint32:
        return min_[UInt32Type](array.as_primitive[UInt32Type]())
    elif array.dtype() == uint64:
        return min_[UInt64Type](array.as_primitive[UInt64Type]())
    elif array.dtype() == float16:
        return min_[Float16Type](array.as_primitive[Float16Type]())
    elif array.dtype() == float32:
        return min_[Float32Type](array.as_primitive[Float32Type]())
    elif array.dtype() == float64:
        return min_[Float64Type](array.as_primitive[Float64Type]())
    raise Error("min_: unsupported dtype ", array.dtype())


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
        _reduce[T, "max"](array, Scalar[T.native].MIN_FINITE)
    )


def max_(array: AnyArray) raises -> AnyScalar:
    """Runtime-typed max."""
    if array.dtype() == int8:
        return max_[Int8Type](array.as_primitive[Int8Type]())
    elif array.dtype() == int16:
        return max_[Int16Type](array.as_primitive[Int16Type]())
    elif array.dtype() == int32:
        return max_[Int32Type](array.as_primitive[Int32Type]())
    elif array.dtype() == int64:
        return max_[Int64Type](array.as_primitive[Int64Type]())
    elif array.dtype() == uint8:
        return max_[UInt8Type](array.as_primitive[UInt8Type]())
    elif array.dtype() == uint16:
        return max_[UInt16Type](array.as_primitive[UInt16Type]())
    elif array.dtype() == uint32:
        return max_[UInt32Type](array.as_primitive[UInt32Type]())
    elif array.dtype() == uint64:
        return max_[UInt64Type](array.as_primitive[UInt64Type]())
    elif array.dtype() == float16:
        return max_[Float16Type](array.as_primitive[Float16Type]())
    elif array.dtype() == float32:
        return max_[Float32Type](array.as_primitive[Float32Type]())
    elif array.dtype() == float64:
        return max_[Float64Type](array.as_primitive[Float64Type]())
    raise Error("max_: unsupported dtype ", array.dtype())


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
