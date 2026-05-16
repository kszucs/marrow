"""Aggregate (reduction) kernels.

Each reduction has:
  - A typed overload: ``def[T](PrimitiveArray[T], Optional[DeviceContext]) -> PrimitiveScalar[T]``
  - A runtime-typed overload: ``def(AnyArray, Optional[DeviceContext]) -> AnyScalar``

The typed overloads delegate to ``_reduce[T, combine]``, which extracts the
array's ``BufferView`` and optional validity ``BitmapView`` and forwards to
``reduce[T, combine]`` in ``views.mojo`` — the same infrastructure used by
``apply``.  GPU dispatch is handled there via ``_reduce_dispatch``.

To add a new aggregate kernel:
  1. Define a thin SIMD combine:
     ``def _op[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]``
  2. Add a typed overload calling ``_reduce[T, _op[T.native, _]](array, identity, ctx)``
  3. Add an AnyArray overload using ``unary_scalar_dispatch``
"""

import std.math as math

from ..arrays import BoolArray, PrimitiveArray, AnyArray
from ..dtypes import *
from ..scalars import PrimitiveScalar, AnyScalar
from ..views import reduce
from .helpers import Kernel, unary_scalar_dispatch
from .execution import ExecutionContext


# ---------------------------------------------------------------------------
# Generic reduction helper
# ---------------------------------------------------------------------------


def _reduce[
    T: PrimitiveType,
    combine: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) thin -> SIMD[
        T.native, W
    ],
](
    array: PrimitiveArray[T],
    identity: Scalar[T.native],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Scalar[T.native]:
    """Reduce a PrimitiveArray to a scalar using a SIMD combine function.

    Delegates to ``views.reduce``, which handles CPU/GPU dispatch via
    ``_reduce_dispatch``. Null elements are replaced with ``identity`` so
    they contribute nothing to the result.
    """
    comptime native = T.native
    if array.bitmap:
        return reduce[native, combine](
            array.values(), array.validity().value(), identity, ctx
        )
    else:
        return reduce[native, combine](array.values(), identity, ctx)


# ---------------------------------------------------------------------------
# Kernel trait
# ---------------------------------------------------------------------------


trait ReductionKernel(Kernel):
    """Reduction kernel: reduces a PrimitiveArray to a scalar.

    Concrete structs define ``comptime name``, ``combine``, and ``identity``;
    ``apply`` and ``dispatch`` have default implementations.
    """

    @staticmethod
    def combine[T: DType, W: Int](
        a: SIMD[T, W], b: SIMD[T, W]
    ) -> SIMD[T, W]: ...

    @staticmethod
    def identity[T: DType]() -> Scalar[T]: ...

    @staticmethod
    def apply[T: PrimitiveType](
        array: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[T]:
        return PrimitiveScalar[T](
            _reduce[T, combine=Self.combine[T.native, _]](
                array, Self.identity[T.native](), ctx
            ),
            array.dtype.copy(),
        )

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyScalar:
        return unary_scalar_dispatch[Self.name, Self.apply[_]](array, ctx)


# ---------------------------------------------------------------------------
# Kernel structs
# ---------------------------------------------------------------------------


struct SumKernel(ReductionKernel):
    comptime name = "sum"

    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)


struct ProductKernel(ReductionKernel):
    comptime name = "product"

    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](1)


struct MinAggKernel(ReductionKernel):
    comptime name = "min"

    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MAX_FINITE


struct MaxAggKernel(ReductionKernel):
    comptime name = "max"

    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MIN_FINITE


# ---------------------------------------------------------------------------
# Public API — thin wrappers
# ---------------------------------------------------------------------------


def sum[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Sum all valid (non-null) elements. Returns 0 if empty or all null."""
    return SumKernel.apply[T](array, ctx)


def sum(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Runtime-typed sum."""
    return SumKernel.dispatch(array, ctx)


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def product[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null."""
    return ProductKernel.apply[T](array, ctx)


def product(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Runtime-typed product."""
    return ProductKernel.dispatch(array, ctx)


# ---------------------------------------------------------------------------
# min
# ---------------------------------------------------------------------------


def min[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Minimum of all valid (non-null) elements.

    Returns MAX_FINITE if empty or all null.
    """
    return MinAggKernel.apply[T](array, ctx)


def min(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Runtime-typed min."""
    return MinAggKernel.dispatch(array, ctx)


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------


def max[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Maximum of all valid (non-null) elements.

    Returns MIN_FINITE if empty or all null.
    """
    return MaxAggKernel.apply[T](array, ctx)


def max(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Runtime-typed max."""
    return MaxAggKernel.dispatch(array, ctx)


# ---------------------------------------------------------------------------
# any / all  (bool arrays) — implemented via SIMD bitmap operations
# ---------------------------------------------------------------------------


def any(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Bool:
    return any(array.as_bool(), ctx)


def any(
    array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Bool:
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


def all(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Bool:
    return all(array.as_bool(), ctx)


def all(
    array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Bool:
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
