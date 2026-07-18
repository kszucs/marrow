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
(``marrow/expr``), mirroring ``DynValue``'s tag switch. The only runtime *data*
dtype switch is the boundary bridge ``for_value_dtype``.
"""

import std.math as math

from ..arrays import BoolArray, PrimitiveArray, AnyArray, Int32Array, Int64Array
from ..builders import PrimitiveBuilder, AnyBuilder, Int64Builder, Int32Builder
from ..dtypes import *
from ..scalars import PrimitiveScalar, AnyScalar
from ..views import reduce
from .helpers import Kernel
from .execution import ExecutionContext


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
        var box = List[AnyScalar]()

        @parameter
        def job[V: NumericType]() raises:
            var state = AggState[Self, V]()
            state.update(gids, array.as_primitive[V](), 1)
            box.append(state.finish(1)[0])

        for_value_dtype[job](array.dtype())
        return box[0].copy()

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
        comptime combine = Self.combine[native, _]
        var identity = Self.identity[native]()
        var value: Scalar[native]
        if array.bitmap:
            value = reduce[native, combine](
                array.values(), array.validity().value(), identity, ctx
            )
        else:
            value = reduce[native, combine](array.values(), identity, ctx)
        return PrimitiveScalar[T](value, array.dtype.copy())

    @staticmethod
    def dispatch(
        array: AnyArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> AnyScalar:
        """Runtime-dtype entry to the SIMD `apply` (same-type reductions)."""
        if array.dtype() == int8:
            return Self.apply(array.as_int8(), ctx)
        elif array.dtype() == int16:
            return Self.apply(array.as_int16(), ctx)
        elif array.dtype() == int32:
            return Self.apply(array.as_int32(), ctx)
        elif array.dtype() == int64:
            return Self.apply(array.as_int64(), ctx)
        elif array.dtype() == uint8:
            return Self.apply(array.as_uint8(), ctx)
        elif array.dtype() == uint16:
            return Self.apply(array.as_uint16(), ctx)
        elif array.dtype() == uint32:
            return Self.apply(array.as_uint32(), ctx)
        elif array.dtype() == uint64:
            return Self.apply(array.as_uint64(), ctx)
        elif array.dtype() == float16:
            return Self.apply(array.as_float16(), ctx)
        elif array.dtype() == float32:
            return Self.apply(array.as_float32(), ctx)
        elif array.dtype() == float64:
            return Self.apply(array.as_float64(), ctx)
        raise Error(t"{Self.name}: unsupported dtype {array.dtype()}")


# ---------------------------------------------------------------------------
# Kernel structs — one SIMD `combine` each. sum/min/max/product override
# `reduce` with the SIMD whole-array fast path; count/mean use the default
# single-group `reduce`.
# ---------------------------------------------------------------------------


struct SumKernel(AggKernel):
    comptime name = "sum"
    comptime AccType[
        V: NumericType
    ] = Int64Type if V.native.is_integral() else Float64Type

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
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        return Self.dispatch(array, ctx)  # SIMD whole-array fast path


struct ProductKernel(AggKernel):
    comptime name = "product"
    comptime AccType[
        V: NumericType
    ] = Int64Type if V.native.is_integral() else Float64Type

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](1)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a * b

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        return Self.dispatch(array, ctx)  # SIMD whole-array fast path


struct MinKernel(AggKernel):
    comptime name = "min"
    comptime AccType[V: NumericType] = V

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MAX_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        return Self.dispatch(array, ctx)  # SIMD whole-array fast path


struct MaxKernel(AggKernel):
    comptime name = "max"
    comptime AccType[V: NumericType] = V

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MIN_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce(
        array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
    ) raises -> AnyScalar:
        return Self.dispatch(array, ctx)  # SIMD whole-array fast path


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


# ---------------------------------------------------------------------------
# Runtime aggregate tags — the one place a runtime function *name* resolves to
# a comptime `AggKernel`. Used by any runtime, multi-aggregate driver (the
# expression layer's `AggregateProcessor`, the Python group-by binding).
# ---------------------------------------------------------------------------

comptime AGG_SUM: UInt8 = 0
comptime AGG_MIN: UInt8 = 1
comptime AGG_MAX: UInt8 = 2
comptime AGG_COUNT: UInt8 = 3
comptime AGG_MEAN: UInt8 = 4
comptime AGG_PRODUCT: UInt8 = 5


def agg_tag_from_name(name: String) raises -> UInt8:
    """Map an aggregate function name to its tag."""
    if name == "sum":
        return AGG_SUM
    elif name == "min":
        return AGG_MIN
    elif name == "max":
        return AGG_MAX
    elif name == "count":
        return AGG_COUNT
    elif name == "mean":
        return AGG_MEAN
    elif name == "product":
        return AGG_PRODUCT
    raise Error("unknown aggregate function: ", name)


def for_agg_tag[
    job: def[K: AggKernel]() raises capturing[_] -> None
](tag: UInt8) raises:
    """Resolve a runtime aggregate tag to its comptime kernel, run `job[K]`."""
    if tag == AGG_SUM:
        job[SumKernel]()
    elif tag == AGG_MIN:
        job[MinKernel]()
    elif tag == AGG_MAX:
        job[MaxKernel]()
    elif tag == AGG_COUNT:
        job[CountKernel]()
    elif tag == AGG_MEAN:
        job[MeanKernel]()
    elif tag == AGG_PRODUCT:
        job[ProductKernel]()
    else:
        raise Error("unknown aggregate tag ", Int(tag))


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
    return SumKernel.dispatch(array, ctx)


def product[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Multiply all valid (non-null) elements. Returns 1 if empty or all null.
    """
    return ProductKernel.apply[T](array, ctx)


def product(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    return ProductKernel.dispatch(array, ctx)


def min[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Minimum of all valid (non-null) elements.

    Returns MAX_FINITE if empty or all null.
    """
    return MinKernel.apply[T](array, ctx)


def min(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    return MinKernel.dispatch(array, ctx)


def max[
    T: PrimitiveType
](
    array: PrimitiveArray[T], ctx: ExecutionContext = ExecutionContext.serial()
) raises -> PrimitiveScalar[T]:
    """Maximum of all valid (non-null) elements.

    Returns MIN_FINITE if empty or all null.
    """
    return MaxKernel.apply[T](array, ctx)


def max(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    return MaxKernel.dispatch(array, ctx)


# ---------------------------------------------------------------------------
# mean — arithmetic mean as float64
# ---------------------------------------------------------------------------


def mean(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> AnyScalar:
    """Arithmetic mean of all valid (non-null) elements, as float64.

    A whole-array reduction like ``sum``/``min``/``max`` — the single-(full-)group
    case of ``MeanKernel``. Returns a null float64 scalar for an empty or all-null
    array. Mirrors ``pyarrow.compute.mean``.
    """
    return MeanKernel.reduce(array, ctx)


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
# for_value_dtype — the one runtime data-dtype -> comptime `V` bridge.
#
# The input array's dtype is a runtime fact, so *some* switch must cross into
# the typed world. It lives here (not inside `AggState`), invoked once at the
# boundary — `AggKernel.reduce`, `group_by[K]`, and the expression-layer
# processor — so `AggState[K, V]` itself is fully typed with no dispatch.
# ---------------------------------------------------------------------------


def for_value_dtype[
    job: def[V: NumericType]() raises capturing[_] -> None
](dtype: AnyDataType) raises:
    """Resolve a runtime numeric dtype to the comptime `V` and run `job[V]()`.
    """
    if dtype == int8:
        job[Int8Type]()
    elif dtype == int16:
        job[Int16Type]()
    elif dtype == int32:
        job[Int32Type]()
    elif dtype == int64:
        job[Int64Type]()
    elif dtype == uint8:
        job[UInt8Type]()
    elif dtype == uint16:
        job[UInt16Type]()
    elif dtype == uint32:
        job[UInt32Type]()
    elif dtype == uint64:
        job[UInt64Type]()
    elif dtype == float16:
        job[Float16Type]()
    elif dtype == float32:
        job[Float32Type]()
    elif dtype == float64:
        job[Float64Type]()
    else:
        raise Error("aggregate: unsupported input dtype ", dtype)


# ---------------------------------------------------------------------------
# AggState — per-group state for a *fully typed* (kernel, input dtype) pair.
#
# `acc` is a real `PrimitiveBuilder[K.AccType[V]]` (not erased), so `update` /
# `finish` carry no dtype dispatch at all — the runtime dtype was resolved once
# at the boundary by `for_value_dtype`. The count column drives NULL output for
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
    resolved once at the boundary (`for_value_dtype`) before this type existed.
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
