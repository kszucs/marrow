"""Aggregate kernels — scalar reductions and grouped aggregation.

Every aggregate is one ``AggKernel`` type: it defines the accumulator-dtype
algebra (``AccType``), an ``identity``, a SIMD ``combine``, and a ``finalize``.
Grouped aggregation runs through ``AggState[K, V]`` — a *fully typed* per-group
state (``update``/``finish`` carry no dtype dispatch). Whole-array reduction is
``AggKernel.reduce`` — the single-(full-)group case, so ``SELECT avg(col)`` /
``count(col)`` work like ``sum(col)``; it defaults to the general single-group
path (used by ``mean``/``count``), and same-type reductions
(``sum``/``min``/``max``/``product``) override it with a SIMD ``views.reduce``
fast path.

Runtime ``name -> kernel`` selection lives in the expression layer
(``marrow/expr``), mirroring ``DynValue``'s tag switch. The runtime *data* dtype
is resolved to the comptime ``V`` at the boundary via `AnyDataType.dispatch_numeric`
(``marrow.utils``), so ``AggState[K, V]`` itself is fully typed with no dispatch.
"""

import std.math as math

from ..arrays import BoolArray, PrimitiveArray, AnyArray, Int32Array, Int64Array
from ..builders import (
    PrimitiveBuilder,
    AnyBuilder,
    Int64Builder,
    Int32Builder,
    BinaryLikeBuilder,
)
from ..dtypes import *
from ..scalars import (
    PrimitiveScalar,
    AnyScalar,
    Int64Scalar,
    Float64Scalar,
    StringScalar,
)
from ..views import reduce
from .helpers import Kernel
from .execution import ExecutionContext


def _reduce_widened[
    K: AggKernel
](
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Erased entry point: resolve the runtime dtype, then run the typed reduce.

    Accumulates in the int64 / float64 accumulator
    (`K.AccType`), so narrow integer inputs don't overflow — matching the grouped
    path's widening. The widening is *fused* into the SIMD reduce (`reduce` casts
    each lane to `Acc` as it is loaded), so no widened copy of the input is
    materialized; when the input is already the accumulator width the per-lane
    cast is a compile-time no-op."""

    @parameter
    def run[V: NumericType](d: V) raises -> AnyScalar:
        return _reduce_widened_typed[K, V](
            array.as_primitive[V](), ctx
        ).to_any()

    return array.dtype().dispatch_numeric[run]()


def _reduce_widened_typed[
    K: AggKernel, V: NumericType
](
    array: PrimitiveArray[V], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[K.AccType[V]]:
    """Fully-typed counterpart of `_reduce_widened` — the input dtype `V` is known
    at comptime, so there is no `AnyDataType.dispatch_numeric` and no erased scalar. The
    lane cast to the `K.AccType[V]` accumulator is fused into the SIMD `reduce`.
    """
    comptime Acc = K.AccType[V].native
    var identity = K.identity[Acc]()
    var value: Scalar[Acc]
    if array.bitmap:
        value = reduce[V.native, K.combine, Acc](
            array.values(), array.validity().value(), identity, ctx
        )
    else:
        value = reduce[V.native, K.combine, Acc](array.values(), identity, ctx)
    return PrimitiveScalar[K.AccType[V]](value)


# ---------------------------------------------------------------------------
# AggKernel — one trait for every aggregate.
#
# A kernel is the pure algebra of a fold. Grouped aggregation is driven by
# `AggState[K, V]` (fully typed); whole-array reduction is `reduce` — the
# single-full-group case — which defaults to that same path but is overridden by
# `sum`/`min`/`max`/`product` with the SIMD `apply`/`dispatch` fast path. One
# SIMD `combine[T, W]` per kernel serves both: the horizontal reduce (same-type)
# and the grouped scatter (fold `combine[A, 1]` over each value cast to `A`).
# ---------------------------------------------------------------------------


trait AggKernel(Kernel):
    """An aggregate: the pure *algebra* of a fold — accumulator-dtype
    (`AccType`), `identity`, SIMD `combine`, and `finalize` — plus a default
    whole-array `reduce`.

    Grouped state + driver live in the fully typed `AggState[K, V]` (below); a
    kernel is a pure type, so any runtime `name -> kernel` selection lives in the
    expression layer, never here. The default per-group state is an accumulator
    column plus a valid-count column (the count drives NULL output for
    empty/all-null groups and the `mean` divisor); a richer aggregate can pair
    itself with a different state struct."""

    comptime AccType[V: NumericType]: NumericType
    """Per-group accumulator type for input `V` (also the output type). `sum`
    widens integers to int64; `min`/`max` keep `V`; `count` is int64; `mean` is
    float64."""

    @staticmethod
    def identity[A: DType]() -> Scalar[A]:
        """Initial accumulator value."""
        ...

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """Fold two same-type (SIMD) values. Used both by the vectorized
        whole-array reduce and, at `W == 1`, by the grouped scatter (each value
        is cast to the accumulator type `A` first)."""
        ...

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        """Finalize a non-empty group's accumulator into its output value."""
        ...

    # -- whole-array scalar reduction ----------------------------------------

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        """Reduce a whole array to one scalar — the single-(full-)group case.

        This default works for *any* kernel (it drives one `AggState[Self, V]`
        with every row in group 0), so `mean`/`count` reduce here too
        (`SELECT avg(col)`). Same-type reductions (`sum`/`min`/`max`/`product`)
        override it with the SIMD `dispatch` fast path."""
        var n = len(array)
        var gb = Int32Builder(n)
        for _ in range(n):
            gb.append(Scalar[int32.native](0))
        var gids = gb.finish()

        @parameter
        def job[V: NumericType](d: V) raises -> AnyScalar:
            var state = AggState[Self, V]()
            state.update(gids, array.as_primitive[V](), 1)
            return state.finish(1)[0]

        return array.dtype().dispatch_numeric[job]()

    @staticmethod
    def reduce[
        V: NumericType
    ](
        array: PrimitiveArray[V],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        """Fully-typed whole-array reduce — the input dtype is known at comptime,
        so the result is `PrimitiveScalar[Self.AccType[V]]` directly (no erased
        `AnyScalar`, no downcast). This general default drives one `AggState` over
        a single group (as the erased overload does); `sum`/`min`/`max`/`product`
        override it with the SIMD widened fast path."""
        var n = len(array)
        var gb = Int32Builder(n)
        for _ in range(n):
            gb.append(Scalar[int32.native](0))
        var gids = gb.finish()
        var state = AggState[Self, V]()
        state.update(gids, array, 1)
        return state.finish(1)[0]

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[T]:
        """SIMD-vectorized whole-array reduce to a same-type scalar via
        `views.reduce` — the fast path for same-type reductions (sum/min/max/
        product). Null elements are replaced by `identity`."""
        comptime native = T.native
        var identity = Self.identity[native]()
        var value: Scalar[native]
        if array.bitmap:
            value = reduce[native, Self.combine](
                array.values(), array.validity().value(), identity, ctx
            )
        else:
            value = reduce[native, Self.combine](array.values(), identity, ctx)
        return PrimitiveScalar[T](value, array.dtype.copy())

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyScalar:
        """Runtime-dtype entry to the SIMD `apply` (same-type reductions)."""

        @parameter
        def leaf[T: NumericType](d: T) raises -> AnyScalar:
            return Self.apply(array.as_primitive[T](), ctx)

        return array.dtype().dispatch_numeric[leaf]()


# ---------------------------------------------------------------------------
# Kernel structs — one SIMD `combine` each. sum/min/max/product override
# `reduce` with the SIMD whole-array fast path; count/mean use the default
# single-group `reduce`.
# ---------------------------------------------------------------------------


trait WideningOp:
    """The two things that distinguish `sum` from `product`: the fold's identity
    element and its lane-wise operator. Both widen integers to int64 and floats
    to float64, which is why they share one shell."""

    comptime name: String

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        ...

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        ...


struct SumOp(WideningOp):
    comptime name = "sum"

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct ProductOp(WideningOp):
    comptime name = "product"

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](1)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b


struct Widening[Op: WideningOp](AggKernel):
    """`sum`/`product` as one kernel: integers accumulate in int64 and floats in
    float64 so narrow inputs cannot overflow, the fold is `Op.combine`, and
    finalize is the identity."""

    comptime name = Self.Op.name
    comptime AccType[
        V: NumericType
    ] = Int64Type if V.native.is_integral() else Float64Type

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Self.Op.identity[T]()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return Self.Op.combine[T, W](a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        return _reduce_widened[Self](array, ctx)

    @staticmethod
    def reduce[
        V: NumericType
    ](
        array: PrimitiveArray[V],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        return _reduce_widened_typed[Self](array, ctx)


comptime SumKernel = Widening[SumOp]
comptime ProductKernel = Widening[ProductOp]


trait MinMaxOp:
    """The two things that distinguish `min` from `max`: which SIMD lane-wise
    extremum to take, and which sentinel is the fold's identity."""

    comptime name: String
    comptime is_min: Bool

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        ...

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        ...


struct MinOp(MinMaxOp):
    comptime name = "min"
    comptime is_min = True

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MAX_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxOp(MinMaxOp):
    comptime name = "max"
    comptime is_min = False

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MIN_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)


struct MinMax[Op: MinMaxOp](AggKernel):
    """`min`/`max` as one kernel: the accumulator keeps the input type, the fold
    is `Op.combine`, and finalize is the identity. Previously two structs that
    differed only in `name`, `identity`, `combine`, and the `is_min` flag passed
    to the string path."""

    comptime name = Self.Op.name
    comptime AccType[V: NumericType] = V

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Self.Op.identity[T]()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return Self.Op.combine[T, W](a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        var dt = array.dtype()
        if dt.is_temporal():
            return _minmax_temporal_scalar[Self](array, ctx)
        elif dt.is_string() or dt.is_large_string():
            return _minmax_string_scalar(array, is_min=Self.Op.is_min)
        return Self.dispatch(array, ctx)  # SIMD whole-array fast path

    @staticmethod
    def reduce[
        V: NumericType
    ](
        array: PrimitiveArray[V],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        return Self.apply(array, ctx)  # AccType == V → same-type SIMD reduce


comptime MinKernel = MinMax[MinOp]
comptime MaxKernel = MinMax[MaxOp]


struct CountKernel(AggKernel):
    """Counts valid (non-null) values. `combine` leaves the accumulator
    untouched — the result is the per-group valid count that every kernel keeps,
    returned by `finalize`."""

    comptime name = "count"
    comptime AccType[V: NumericType] = Int64Type

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return Scalar[A](count)

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        # Valid count is metadata — no scan.
        return Int64Scalar(Int64(len(array) - array.null_count())).to_any()

    @staticmethod
    def reduce[
        V: NumericType
    ](
        array: PrimitiveArray[V],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        # Valid count is metadata — no scan. `AccType` is always int64.
        return Int64Scalar(Int64(len(array) - array.null_count()))


struct MeanKernel(AggKernel):
    """Sums into a float64 accumulator; divides by the valid count on finish."""

    comptime name = "mean"
    comptime AccType[V: NumericType] = Float64Type

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc / Scalar[A](count)

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        # Vectorized sum (widened, SIMD) divided by the valid count.
        var cnt = len(array) - array.null_count()
        if cnt == 0:
            return Float64Scalar(None, float64).to_any()  # null
        var total = SumKernel.reduce(array, ctx)
        var s = (
            total.as_float64().value() if total.type()
            == float64 else Float64(total.as_int64().value())
        )
        return Float64Scalar(s / Float64(cnt)).to_any()

    @staticmethod
    def reduce[
        V: NumericType
    ](
        array: PrimitiveArray[V],
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        # Vectorized widened sum divided by the valid count; null on empty.
        var cnt = len(array) - array.null_count()
        if cnt == 0:
            return Float64Scalar(None, float64)
        var total = _reduce_widened_typed[SumKernel](array, ctx)
        return Float64Scalar(total.value().cast[DType.float64]() / Float64(cnt))


# ---------------------------------------------------------------------------
# min / max over string and temporal columns.
#
# `min`/`max` are order-preserving, so temporal types (date/time/timestamp/
# duration) reduce over their signed-integer backing and the result is
# reinterpreted back to the temporal dtype — reusing the whole numeric path
# (SIMD whole-array reduce + typed `AggState` scatter). Strings compare
# bytewise (lexicographic), matching Arrow's `hash_min`/`hash_max`; nulls are
# excluded (SQL semantics) and an empty / all-null group yields null.
# ---------------------------------------------------------------------------


def reinterpret_array(array: AnyArray, dt: AnyDataType) raises -> AnyArray:
    """View `array`'s buffers under a new (same-width) `dt` — used to read a
    temporal column as its integer backing and to relabel the integer result
    back to the temporal dtype, so min/max reuse the numeric aggregation path.
    """
    return array.view(dt.copy())


def temporal_backing_dtype(dt: AnyDataType) -> AnyDataType:
    """The integer dtype backing a temporal value — 32-bit for date32/time32,
    64-bit otherwise (date64/time64/timestamp/duration). Paired with
    `reinterpret_array` to route temporal columns through the numeric path."""
    return AnyDataType(int32) if (
        dt.is_date32() or dt.is_time32()
    ) else AnyDataType(int64)


def _mm_temporal_typed[
    K: AggKernel, T: TemporalType
](arr: PrimitiveArray[T], ctx: ExecutionContext) raises -> AnyScalar:
    """Whole-array min/max over one temporal array — the SIMD same-type reduce
    over the integer backing, preserving the runtime dtype (unit/tz). All-null /
    empty → null."""
    if len(arr) == arr.null_count():
        return PrimitiveScalar[T](None, arr.dtype.copy()).to_any()
    return K.apply(arr, ctx).to_any()


def _minmax_temporal_scalar[
    K: AggKernel
](array: AnyArray, ctx: ExecutionContext) raises -> AnyScalar:
    """Whole-array min/max over a temporal `AnyArray` → temporal `AnyScalar`."""
    var dt = array.dtype()
    if dt.is_date32():
        return _mm_temporal_typed[K](array.as_date32(), ctx)
    elif dt.is_date64():
        return _mm_temporal_typed[K](array.as_date64(), ctx)
    elif dt.is_time32():
        return _mm_temporal_typed[K](array.as_time32(), ctx)
    elif dt.is_time64():
        return _mm_temporal_typed[K](array.as_time64(), ctx)
    elif dt.is_timestamp():
        return _mm_temporal_typed[K](array.as_timestamp(), ctx)
    else:
        return _mm_temporal_typed[K](array.as_duration(), ctx)


def _minmax_string_scalar(array: AnyArray, is_min: Bool) raises -> AnyScalar:
    """Whole-array lexicographic (bytewise) min/max over a string/large_string
    array. Nulls excluded; empty / all-null → null `StringScalar`."""

    @parameter
    def leaf[T: StringLikeType](d: T) raises -> AnyScalar:
        ref sa = array.as_binary_like[T]()
        var n = len(sa)
        var has_null = sa.null_count() > 0
        var best = -1
        for i in range(n):
            if has_null and not sa.is_valid(i):
                continue
            if best == -1:
                best = i
            else:
                var a = sa.unsafe_get(UInt(i))
                var b = sa.unsafe_get(UInt(best))
                var take = (a < b) if is_min else (b < a)
                if take:
                    best = i
        if best == -1:
            return StringScalar.null().to_any()
        return StringScalar(String(sa.unsafe_get(UInt(best)))).to_any()

    return array.dtype().dispatch_stringlike[leaf]()


def min_max_string_grouped(
    gids: Int32Array, value: AnyArray, num_groups: Int, is_min: Bool
) raises -> AnyArray:
    """Per-group lexicographic (bytewise) min/max over a string/large_string
    column, over precomputed `gids`. Nulls excluded; an empty / all-null group
    yields null. Matches Arrow's bytewise ordering for `hash_min`/`hash_max`."""

    @parameter
    def leaf[T: StringLikeType](d: T) raises -> AnyArray:
        ref sa = value.as_binary_like[T]()
        var best = List[Int](length=num_groups, fill=-1)
        var gv = gids.values()
        var n = len(gids)
        var has_null = sa.null_count() > 0
        for i in range(n):
            if has_null and not sa.is_valid(i):
                continue
            var g = Int(gv[i])
            if best[g] == -1:
                best[g] = i
            else:
                var a = sa.unsafe_get(UInt(i))
                var b = sa.unsafe_get(UInt(best[g]))
                var take = (a < b) if is_min else (b < a)
                if take:
                    best[g] = i
        var out = BinaryLikeBuilder[T](capacity=num_groups)
        for g in range(num_groups):
            if best[g] == -1:
                out.append_null()
            else:
                out.append(sa.unsafe_get(UInt(best[g])))
        return out.finish().to_any()

    return value.dtype().dispatch_stringlike[leaf]()


def count_valid_grouped(
    gids: Int32Array, value: AnyArray, num_groups: Int
) raises -> Int64Array:
    """Per-group count of valid (non-null) rows over precomputed `gids`.

    `count` reads only validity, so — unlike the other folds — it is defined for
    *every* dtype. This is the non-numeric counterpart of
    `AggState[CountKernel, V]`, keeping `COUNT(col)` / `COUNT(*)` available over
    string, binary and nested columns that the typed numeric scatter can't
    resolve. An empty group counts 0 (never null), matching SQL."""
    var counts = List[Int64](length=num_groups, fill=0)
    var gv = gids.values()
    var has_null = value.null_count() > 0
    for i in range(len(gids)):
        if has_null and not value.is_valid(i):
            continue
        counts[Int(gv[i])] += 1
    var out = Int64Builder(num_groups)
    for g in range(num_groups):
        out.append(Scalar[int64.native](counts[g]))
    return out.finish()


# ---------------------------------------------------------------------------
# any / all — boolean reductions via SIMD bitmap operations.
#
# Not `AggKernel`s: they fold bit-packed masks (not the numeric
# accumulator/identity/combine/finalize algebra), so each is its own struct
# exposing a `reduce(BoolArray) -> Bool` (plus an `AnyArray` overload), matching
# the struct-per-kernel shape of the rest of the module.
# ---------------------------------------------------------------------------


trait BoolReduceKernel(Kernel):
    """A boolean whole-column fold to a single `Bool` — `any`/`all`. Not an
    `AggKernel` (it folds bit-packed masks, not the numeric accumulator algebra),
    so it exposes just `reduce(BoolArray) -> Bool`; the expression layer selects
    between the two by kernel type."""

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
        ...

    @staticmethod
    def dispatch(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
        """Runtime-dtype entry: fold a boolean `AnyArray` to a `Bool`."""
        return Self.reduce(array.as_bool(), ctx)


struct AnyKernel(BoolReduceKernel):
    """True if any valid element is True. False if empty or all null."""

    comptime name = "any"

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
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

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
        return Self.reduce(array.as_bool(), ctx)


struct AllKernel(BoolReduceKernel):
    """True if all valid elements are True. True if empty or all null."""

    comptime name = "all"

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
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

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> Bool:
        return Self.reduce(array.as_bool(), ctx)


# ---------------------------------------------------------------------------
# AggState — per-group state + driver
#
# The default aggregate state: an accumulator column plus a valid-count column.
# `update[K]` / `finish[K]` take the kernel `K` as a comptime parameter, so the
# state holds no kernel identity and the kernel layer needs no enum or vtable —
# the AOT path fixes `K` at `group_by[K]`, the runtime path resolves it once via
# the expression layer's tag dispatch. The runtime *data* dtype is resolved once
# per call by `_for_dtype`; the hot `_scatter` loop is branch-free.
#
# A richer aggregate (variance = sum+sumsq+count, distinct = hash set, ...) is
# added by pairing its `AggKernel` with a different state struct exposing the
# same `update`/`finish` shape.


# ---------------------------------------------------------------------------
# AggState — per-group state for a *fully typed* (kernel, input dtype) pair.
#
# `acc` is a real `PrimitiveBuilder[K.AccType[V]]` (not erased), so `update` /
# `finish` carry no dtype dispatch at all — the runtime dtype was resolved once
# at the boundary by `AnyDataType.dispatch_numeric`. The count column drives NULL output for
# empty/all-null groups and the `mean` divisor. A richer aggregate (variance,
# distinct, ...) pairs its kernel with a different state struct of this shape.
#
# The runtime processor stores its accumulators erased and, per batch, resolves
# `(K, V)` then wraps the shared builders into a transient `AggState[K, V]` —
# the builders are `ArcPointer`-shared, so the wrap mutates them in place.
# ---------------------------------------------------------------------------
struct AggState[K: AggKernel, V: NumericType](Movable):
    """Per-group state for a fully typed (kernel, input dtype) pair.

    Everything is typed: the accumulator is a `PrimitiveBuilder[Acc]`
    (`Acc = K.AccType[V]`), `update` takes a `PrimitiveArray[V]`, and `finish`
    returns a `PrimitiveArray[Acc]` — no `AnyBuilder`/`AnyArray`/`AnyScalar`
    anywhere, so the hot loops are fully monomorphized. The runtime dtype was
    resolved once at the boundary (`AnyDataType.dispatch_numeric`) before this type existed.
    A richer aggregate (variance, distinct, ...) pairs its kernel with a
    different state struct of this shape."""

    comptime Acc = Self.K.AccType[Self.V]

    var acc: PrimitiveBuilder[Self.Acc]
    var cnt: Int64Builder

    def __init__(out self):
        self.acc = PrimitiveBuilder[Self.Acc]()
        self.cnt = Int64Builder()

    def num_groups(self) -> Int:
        return self.acc.length()

    def update(
        mut self,
        group_ids: Int32Array,
        input: PrimitiveArray[Self.V],
        num_groups: Int,
    ) raises:
        """Grow to `num_groups` (new slots filled with `K.identity`), then
        scatter-fold this batch. No dtype dispatch — `Acc`/`V` are comptime."""
        comptime A = Self.Acc.native
        while self.acc.length() < num_groups:
            self.acc.append(Self.K.identity[A]())
            self.cnt.append(Scalar[int64.native](0))

        # Reads go through `BufferView`s; accumulator/count writes use the builder
        # element accessor (a builder has no mutable value view — this is
        # random-access scatter, not a sequential `views.apply`).
        var gids = group_ids.values()
        var vals = input.values()
        var n = len(group_ids)
        if input.null_count() > 0:
            var valid = input.validity().value()
            for i in range(n):
                if not valid[i]:
                    continue
                var g = Int(gids[i])
                self.acc.unsafe_set(
                    g,
                    Self.K.combine[A, 1](
                        self.acc.unsafe_get(g), vals[i].cast[A]()
                    ),
                )
                self.cnt.unsafe_set(g, self.cnt.unsafe_get(g) + 1)
        else:
            for i in range(n):
                var g = Int(gids[i])
                self.acc.unsafe_set(
                    g,
                    Self.K.combine[A, 1](
                        self.acc.unsafe_get(g), vals[i].cast[A]()
                    ),
                )
                self.cnt.unsafe_set(g, self.cnt.unsafe_get(g) + 1)

    def finish(mut self, num_groups: Int) raises -> PrimitiveArray[Self.Acc]:
        """Finalize into the typed output column (NULL for empty/all-null
        groups)."""
        comptime A = Self.Acc.native
        var b = PrimitiveBuilder[Self.Acc](num_groups)
        for g in range(num_groups):
            var c = Int(self.cnt.unsafe_get(g))
            if c > 0:
                b.append(Self.K.finalize[A](self.acc.unsafe_get(g), c))
            else:
                b.append_null()
        return b.finish()

    def into_partials(
        mut self,
    ) raises -> Tuple[PrimitiveArray[Self.Acc], Int64Array]:
        """Freeze the raw (non-finalized) per-group accumulator and valid-count
        columns — the partial state a parallel merge folds together. Consumes
        the builders."""
        return (self.acc.finish(), self.cnt.finish())

    def merge(
        mut self,
        group_ids: Int32Array,
        part_acc: PrimitiveArray[Self.Acc],
        part_cnt: Int64Array,
        num_groups: Int,
    ) raises:
        """Fold another partial's per-group `(acc, cnt)` into this state at
        remapped group ids: `group_ids[j]` is *this* state's group id for the
        partial's local group `j`. Grows to `num_groups` first (new slots =
        `K.identity` / count 0), then combines accumulators and sums counts.

        Combining is exact for every kernel because the accumulator is the raw
        fold (`sum`/`min`/`max`/`product`) and the count is carried separately —
        so `mean` merges as (Σsum, Σcount) and finalizes once at the end. An
        all-null local group contributes `identity` (a no-op under `combine`)
        and count 0, so it correctly leaves the target unchanged."""
        comptime A = Self.Acc.native
        while self.acc.length() < num_groups:
            self.acc.append(Self.K.identity[A]())
            self.cnt.append(Scalar[int64.native](0))

        var gids = group_ids.values()
        var acc = part_acc.values()
        var cnt = part_cnt.values()
        for j in range(len(group_ids)):
            var g = Int(gids[j])
            self.acc.unsafe_set(
                g, Self.K.combine[A, 1](self.acc.unsafe_get(g), acc[j])
            )
            self.cnt.unsafe_set(g, self.cnt.unsafe_get(g) + cnt[j])
