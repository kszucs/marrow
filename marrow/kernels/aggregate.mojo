"""Aggregate kernels — scalar reductions and grouped aggregation.

Two levels, and they answer different questions.

- **``FoldKernel``** — the pure *algebra* of a fold: the accumulator-dtype rule
  (``AccType``), an ``identity``, a SIMD ``combine`` and a ``finalize``. It
  knows nothing about arrays beyond the fully typed whole-array ``reduce`` /
  ``apply``, and grouped folding is ``AggState[K, V]`` — a fully typed
  per-group state with no dtype dispatch anywhere inside it. This is the level
  the comptime expression lane fuses into one SIMD loop, and it exists only for
  the aggregates that *are* folds.
- **``AggKernel``** — one aggregate, whole, over **erased** columns: the output
  dtype from the input dtypes, and one value per group. Every aggregate has one
  — ``Fold[K]`` wraps the level above, and ``StringExtremum``, ``ValidCount``
  and ``DistinctCount`` have no fold algebra to wrap.

Erasure at the second level is deliberate. It is what lets a single vocabulary
serve both expression lanes: the comptime lane names an ``AggKernel`` as a type
parameter, the runtime lane stores one as an ``AggregateFn`` pointer. The hot
loop is still fully typed either way, because the dispatch happens once, at the
boundary, before ``AggState`` exists.

No aggregate *name* is ever compared in this module. Mapping ``"sum"`` onto
``Fold[SumKernel]`` is the expression layer's job and lives there.
"""

import std.math as math

from ..arrays import (
    Array,
    BoolArray,
    BinaryLikeArray,
    PrimitiveArray,
    DynArray,
    Int32Array,
    Int64Array,
)
from ..builders import (
    PrimitiveBuilder,
    DynBuilder,
    Int64Builder,
    Int32Builder,
    BinaryLikeBuilder,
)
from ..dtypes import (
    DynType,
    Float64Type,
    Int32Type,
    Int64Type,
    IntegerType,
    NumericType,
    PrimitiveType,
    StringLikeType,
    TemporalType,
    UInt8Type,
    float64,
    int32,
    int64,
)
from ..scalars import (
    PrimitiveScalar,
    DynScalar,
    Int64Scalar,
    Float64Scalar,
)
from ..views import reduce
from .core import Kernel, Groups
from ..execution import ExecContext
from .distinct import (
    count_distinct,
    approx_count_distinct,
    count_distinct_grouped,
    approx_count_distinct_grouped,
)


# ---------------------------------------------------------------------------
# FoldKernel — one trait for every aggregate.
#
# A kernel is the pure algebra of a fold. Grouped aggregation is driven by
# `AggState[K, V]` (fully typed); whole-array reduction is `reduce` — the
# single-full-group case — which defaults to that same path but is overridden by
# `sum`/`min`/`max`/`product` with the SIMD `apply` fast path. One SIMD
# `combine[T, W]` per kernel serves both: the horizontal reduce (same-type) and
# the grouped scatter (fold `combine[A, 1]` over each value cast to `A`).
# ---------------------------------------------------------------------------


trait FoldKernel(Kernel):
    """The pure *algebra* of a fold — accumulator-dtype (`AccType`),
    `identity`, SIMD `combine`, and `finalize` — plus a default whole-array
    `reduce`.

    Not an aggregate: that is `AggKernel`, below. This is the algebra an
    aggregate may be *built from*, and only six are. `Fold[K]` is the one that
    turns it into one, and the comptime expression lane fuses it directly.

    Everything here is typed: `reduce` / `apply` take a `PrimitiveArray[V]` and
    return a `PrimitiveScalar`, so a kernel never sees a `DynArray` or a
    `DynType`. Binding one to a *runtime* dtype happens on the other side of
    that boundary.

    Grouped state + driver live in the fully typed `AggState[K, V]`. The default
    per-group state is an accumulator column plus a valid-count column (the count
    drives NULL output for empty/all-null groups and the `mean` divisor); a
    richer aggregate can pair itself with a different state struct."""

    comptime AccType[V: PrimitiveType]: PrimitiveType
    """Per-group accumulator type for input `V` (also the output type). `sum`
    widens integers to int64; `min`/`max` keep `V`; `count` is int64; `mean` is
    float64.

    Bound on `PrimitiveType` at both ends, not `NumericType`: `min`/`max` keep
    the input's type, and that input may be a timestamp or a decimal. What a
    kernel *requires* of its input is stated separately, by the domain markers
    (`OrderedAgg` / `ArithmeticAgg` / `IntegralAgg`) — one bound cannot say
    both "I can be read as a lane" and "I support addition"."""

    @staticmethod
    def check_domain[V: PrimitiveType]():
        """Compile-time gate: reject an input type this kernel's domain excludes.

        Non-raising, so it runs at comptime; a violation is a build error naming
        the domain, not an `Error` at run time. No marker means "accepts
        anything" — that is `CountKernel`, which reads validity and never touches
        a value, and `OrderedAgg`, which needs only a total order and gets one from
        every fixed-width type.
        """
        comptime if conforms_to(Self, IntegralAgg):
            comptime assert conforms_to(
                V, IntegerType
            ), "this aggregate is defined for integer columns only"
        elif conforms_to(Self, ArithmeticAgg):
            comptime assert conforms_to(V, NumericType), (
                "this aggregate needs arithmetic, so it is defined for numeric"
                " columns only -- not temporal, interval or decimal"
            )

    comptime empty_is_null: Bool = True
    """Whether a group with no valid rows has no answer. It usually does not —
    the minimum of nothing is NULL, not a sentinel — but `count` is the
    exception SQL calls out: counting nothing is 0."""

    comptime needs_count: Bool = False
    """Whether `finalize` reads the group's valid count. Only `mean` does (it is
    the divisor). Everything else needs to know *whether* a group was touched,
    not how often — and that is a flag, not a counter. See `AggState`."""

    @staticmethod
    def identity[A: DType]() -> Scalar[A]:
        """Initial accumulator value."""
        ...

    @staticmethod
    def acc_dtype[V: PrimitiveType](dtype: V) -> Self.AccType[V]:
        """The accumulator's dtype **as a value**, given the input column's.

        `AccType` names the type; this names the instance, and the two are not
        the same question. `NumericType` is `Defaultable`, so a numeric
        accumulator can be conjured from its type alone — `TemporalType` and
        `DecimalType` are not, because a timestamp carries a unit and timezone
        and a decimal a precision and scale. `min`/`max` keep the input's type,
        so their accumulator dtype is the input's dtype and can only come from
        the column. Every caller that builds an `AggState` goes through here."""
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
    def reduce[
        V: PrimitiveType
    ](
        array: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        """Whole-array reduce — the single-(full-)group case. The input dtype is
        known at comptime, so the result is `PrimitiveScalar[Self.AccType[V]]`
        directly (no erased `DynScalar`, no downcast).

        This general default works for *any* kernel — it drives one
        `AggState[Self, V]` with every row in group 0 — so `mean`/`count` reduce
        here too (`SELECT avg(col)`); `sum`/`min`/`max`/`product` override it
        with the SIMD widened fast path."""
        var n = len(array)
        Self.check_domain[V]()
        var gb = Int32Builder(n)
        for _ in range(n):
            gb.append(Scalar[int32.native](0))
        var gids = gb.finish()
        var state = AggState[Self, V](Self.acc_dtype[V](array.dtype))
        state.update(gids, array, 1)
        return state.finish(1)[0]

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        array: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
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


# ---------------------------------------------------------------------------
# Kernel structs — one SIMD `combine` each. sum/min/max/product override
# `reduce` with the SIMD whole-array fast path; count/mean use the default
# single-group `reduce`.
# ---------------------------------------------------------------------------


trait WideningOp(Kernel):
    """The two things that distinguish `sum` from `product`: the fold's identity
    element and its lane-wise operator. Both widen integers to int64 and floats
    to float64, which is why they share one shell."""

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


# ---------------------------------------------------------------------------
# Input domains — what a kernel requires of the column it folds
# ---------------------------------------------------------------------------
# `AccType` used to be bound on `NumericType`, and that bound was the
# **intersection of four different requirements**: `count` needs nothing of the
# type, `min`/`max` need an ordering, `sum` needs addition, `mean` needs
# division. One bound for four requirements meant the most permissive kernel
# was constrained by the least — `TemporalMinMax` existed only to work around
# it — and no kernel stated what it actually needs.
#
# The bound is now `PrimitiveType` and these markers say the rest. They are
# enforced by `FoldKernel.check_domain`, which every fold runs at compile
# time, so `sum(date)` is a build error rather than a silent nonsense or a runtime
# raise.
#
# **No marker means "accepts anything"** — that is `CountKernel`, which reads
# validity and never touches a value.
#
# Markers rather than a `comptime numeric_only: Bool` because a flag can be
# wrong silently: `numeric_only = False` on a summing kernel compiles and
# yields nonsense, whereas conformance is a claim the compiler checks and a
# `where` clause can dispatch on. Mojo permits neither narrowing a trait method
# with `where` (it becomes a different signature) nor narrowing an associated
# type's bound in a conformer, so this is the closest sound form.
#
# They constrain the **input domain** only. State shape — variance's
# sum+sumsq+count, a quantile sketch, `StringMinMax`'s per-group index — is a
# separate axis and stays a kernel paired with a different state struct.
#
# **What unblocked the widening**: `NumericType` is `Defaultable`;
# `TemporalType` and `DecimalType` are not, because a timestamp carries a unit
# and timezone and a decimal a precision and scale. `AggState` used to build
# its accumulator with `PrimitiveBuilder[Acc]()`, the no-dtype constructor that
# only exists for numeric types. It now holds the dtype as a *value*, supplied
# by `FoldKernel.acc_dtype(input_dtype)` — the one place that knows whether the
# accumulator keeps the input's dtype (`min`/`max`) or names its own
# (`sum` -> int64/float64, `count` -> int64, `mean` -> float64).
trait OrderedAgg(FoldKernel):
    """Needs an ordering and nothing more — any fixed-width type will do.

    `min`, `max`, and later `quantile` / `median`.
    """

    pass


trait ArithmeticAgg(FoldKernel):
    """Needs addition or division, so numeric input only.

    `sum`, `product`, `mean`, and later `variance` / `stddev`.
    """

    pass


trait IntegralAgg(ArithmeticAgg):
    """Needs bit operations, so integers only — narrower than arithmetic.

    Nothing conforms yet; `bitwise_and` / `_or` / `_xor` will. Declared with
    the others because it is what shows the domains form a lattice rather than
    a flag: a bitwise kernel accepts `int64` and rejects `float64`, which a
    Bool could not express.
    """

    pass


struct Widening[Op: WideningOp](ArithmeticAgg):
    """`sum`/`product` as one kernel: integers accumulate in int64 and floats in
    float64 so narrow inputs cannot overflow, the fold is `Op.combine`, and
    finalize is the identity."""

    comptime name = Self.Op.name
    comptime AccType[
        V: PrimitiveType
    ] = Int64Type if V.native.is_integral() else Float64Type

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Self.Op.identity[T]()

    @staticmethod
    def acc_dtype[V: PrimitiveType](dtype: V) -> Self.AccType[V]:
        # The accumulator names its own type, so the input's dtype says
        # nothing: int64 or float64, both `Defaultable`.
        return Self.AccType[V]()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return Self.Op.combine[T, W](a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce[
        V: PrimitiveType
    ](
        array: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        """Widened SIMD whole-array reduce: accumulate in `AccType[V]` so narrow
        integer inputs cannot overflow, matching the grouped path. The widening
        is *fused* into the reduce (each lane is cast to `Acc` as it is loaded),
        so no widened copy of the input is materialized; when the input is
        already the accumulator width the per-lane cast is a no-op."""
        Self.check_domain[V]()
        comptime Acc = Self.AccType[V].native
        var identity = Self.identity[Acc]()
        var value: Scalar[Acc]
        if array.bitmap:
            value = reduce[V.native, Self.combine, Acc](
                array.values(), array.validity().value(), identity, ctx
            )
        else:
            value = reduce[V.native, Self.combine, Acc](
                array.values(), identity, ctx
            )
        return PrimitiveScalar[Self.AccType[V]](value)


comptime SumKernel = Widening[SumOp]
comptime ProductKernel = Widening[ProductOp]


trait MinMaxOp(Kernel):
    """The two things that distinguish `min` from `max`: which SIMD lane-wise
    extremum to take, and which sentinel is the fold's identity."""

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


struct MinMax[Op: MinMaxOp](OrderedAgg):
    """`min`/`max` as one kernel: the accumulator keeps the input type, the fold
    is `Op.combine`, and finalize is the identity. Previously two structs that
    differed only in `name`, `identity`, `combine`, and the `is_min` flag passed
    to the string path."""

    comptime name = Self.Op.name
    comptime AccType[V: PrimitiveType] = V

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Self.Op.identity[T]()

    @staticmethod
    def acc_dtype[V: PrimitiveType](dtype: V) -> Self.AccType[V]:
        # The accumulator *is* the input column, so its unit, timezone,
        # precision and scale come from the column and nowhere else.
        return dtype.copy()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return Self.Op.combine[T, W](a, b)

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc

    @staticmethod
    def reduce[
        V: PrimitiveType
    ](
        array: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        # `AccType == V`, so this is the same-type SIMD reduce — but `apply`
        # folds nulls to `identity`, and `identity` is a sentinel
        # (`MAX_FINITE` / `MIN_FINITE`), not an answer. The minimum of nothing
        # is NULL. The grouped path says so through its valid count; the SIMD
        # fast path has no count, so it asks the column directly.
        if len(array) - array.null_count() == 0:
            return PrimitiveScalar[Self.AccType[V]](
                None, Self.acc_dtype[V](array.dtype)
            )
        else:
            return Self.apply(array, ctx)


comptime MinKernel = MinMax[MinOp]
comptime MaxKernel = MinMax[MaxOp]


struct CountKernel(FoldKernel):
    """Counts valid (non-null) values. `combine` leaves the accumulator
    untouched — the result is the per-group valid count that every kernel keeps,
    returned by `finalize`."""

    comptime name = "count"
    comptime AccType[V: PrimitiveType] = Int64Type
    comptime empty_is_null = False
    comptime needs_count = True  # the answer *is* the count

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @staticmethod
    def acc_dtype[V: PrimitiveType](dtype: V) -> Self.AccType[V]:
        return Int64Type()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return Scalar[A](count)

    @staticmethod
    def reduce[
        V: PrimitiveType
    ](
        array: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        # Valid count is metadata — no scan. `AccType` is always int64.
        return Int64Scalar(Int64(len(array) - array.null_count()))


struct MeanKernel(ArithmeticAgg):
    """Sums into a float64 accumulator; divides by the valid count on finish."""

    comptime name = "mean"
    comptime AccType[V: PrimitiveType] = Float64Type
    comptime needs_count = True  # the divisor

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @staticmethod
    def acc_dtype[V: PrimitiveType](dtype: V) -> Self.AccType[V]:
        return Float64Type()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc / Scalar[A](count)

    @staticmethod
    def reduce[
        V: PrimitiveType
    ](
        array: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveScalar[Self.AccType[V]]:
        Self.check_domain[V]()
        # Vectorized widened sum divided by the valid count; null on empty.
        var cnt = len(array) - array.null_count()
        if cnt == 0:
            return Float64Scalar(None, float64)
        var total = SumKernel.reduce(array, ctx)
        return Float64Scalar(total.value().cast[DType.float64]() / Float64(cnt))


# ---------------------------------------------------------------------------
# any / all — boolean reductions via SIMD bitmap operations.
#
# Not `FoldKernel`s: they fold bit-packed masks (not the numeric
# accumulator/identity/combine/finalize algebra), so each is its own struct
# exposing a `reduce(BoolArray) -> Bool` (plus an `DynArray` overload), matching
# the struct-per-kernel shape of the rest of the module.
# ---------------------------------------------------------------------------


trait BoolReduceKernel(Kernel):
    """A boolean whole-column fold to a single `Bool` — `any`/`all`. Not an
    `FoldKernel` (it folds bit-packed masks, not the numeric accumulator algebra),
    so it exposes just `reduce(BoolArray) -> Bool`; the expression layer selects
    between the two by kernel type."""

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> Bool:
        ...

    @staticmethod
    def dispatch(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> Bool:
        """Runtime-dtype entry: fold a boolean `DynArray` to a `Bool`."""
        return Self.reduce(array.as_bool(), ctx)


struct AnyKernel(BoolReduceKernel):
    """True if any valid element is True. False if empty or all null."""

    comptime name = "any"

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecContext = ExecContext.serial()
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
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> Bool:
        return Self.reduce(array.as_bool(), ctx)


struct AllKernel(BoolReduceKernel):
    """True if all valid elements are True. True if empty or all null."""

    comptime name = "all"

    @staticmethod
    def reduce(
        array: BoolArray, ctx: ExecContext = ExecContext.serial()
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
        array: DynArray, ctx: ExecContext = ExecContext.serial()
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
# added by pairing its `FoldKernel` with a different state struct exposing the
# same `update`/`finish` shape.


# ---------------------------------------------------------------------------
# AggState — per-group state for a *fully typed* (kernel, input dtype) pair.
#
# `acc` is a real `PrimitiveBuilder[K.AccType[V]]` (not erased), so `update` /
# `finish` carry no dtype dispatch at all — the runtime dtype was resolved once
# at the boundary by `DynType.dispatch_numeric`. The count column drives NULL output for
# empty/all-null groups and the `mean` divisor. A richer aggregate (variance,
# distinct, ...) pairs its kernel with a different state struct of this shape.
#
# The runtime processor stores its accumulators erased and, per batch, resolves
# `(K, V)` then wraps the shared builders into a transient `AggState[K, V]` —
# the builders are `ArcPointer`-shared, so the wrap mutates them in place.
# ---------------------------------------------------------------------------
struct AggState[K: FoldKernel, V: PrimitiveType](Movable):
    """Per-group state for a fully typed (kernel, input dtype) pair.

    Everything is typed: the accumulator is a `PrimitiveBuilder[Acc]`
    (`Acc = K.AccType[V]`), `update` takes a `PrimitiveArray[V]`, and `finish`
    returns a `PrimitiveArray[Acc]` — no `DynBuilder`/`DynArray`/`DynScalar`
    anywhere, so the hot loops are fully monomorphized. The runtime dtype was
    resolved once at the boundary (`DynType.dispatch_numeric`) before this type existed.
    A richer aggregate (variance, distinct, ...) pairs its kernel with a
    different state struct of this shape."""

    comptime Acc = Self.K.AccType[Self.V]

    comptime Seen = Int64Type if Self.K.needs_count else UInt8Type
    """How the second column is stored. A kernel that reads the count needs a
    real counter; the rest only ask *was this group touched*, so one byte per
    group is enough — and at 100k groups that is 100 KB against 800 KB, which
    is the difference between staying in cache and not, on the loop that does a
    random write per row."""

    var acc: PrimitiveBuilder[Self.Acc]
    var cnt: PrimitiveBuilder[Self.Seen]

    var dtype: Self.Acc
    """The accumulator's dtype, as a *value*.

    Held rather than conjured because a dtype is not always constructible from
    its type. `NumericType` is `Defaultable`, but `TemporalType` and
    `DecimalType` are not — a timestamp carries a unit and timezone, a decimal
    a precision and scale — and `min`/`max` keep the input's type as the
    accumulator's (`AccType[V] = V`). So the moment this state folds a temporal
    column, `PrimitiveBuilder[Acc]()` cannot build the column it must produce,
    and only the caller knows what to pass — `FoldKernel.acc_dtype(input_dtype)`
    is where every caller gets it.
    """

    def __init__(out self, dtype: Self.Acc):
        Self.K.check_domain[Self.V]()
        self.dtype = dtype
        self.acc = PrimitiveBuilder[Self.Acc](dtype)
        self.cnt = PrimitiveBuilder[Self.Seen]()

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
        self._grow(num_groups)
        comptime A = Self.Acc.native

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
                self._mark(g)
        else:
            for i in range(n):
                var g = Int(gids[i])
                self.acc.unsafe_set(
                    g,
                    Self.K.combine[A, 1](
                        self.acc.unsafe_get(g), vals[i].cast[A]()
                    ),
                )
                self._mark(g)

    @always_inline
    def _grow(mut self, num_groups: Int) raises:
        """Ensure `num_groups` slots exist, new ones seeded with `K.identity`.

        Called by `update` and by `finish`. `finish` needs it because an
        aggregate over **zero** batches never calls `update` at all, and its
        loop then reads slots that were never allocated — a `debug_assert` under
        `ASSERT=all` and a **silent out-of-bounds read in a release build**.
        Unreachable through `FoldKernel.reduce`, which always calls `update`
        once; reachable the moment an accumulator is driven by a plan that sees
        no rows.
        """
        comptime A = Self.Acc.native
        comptime S = Self.Seen.native
        while self.acc.length() < num_groups:
            self.acc.append(Self.K.identity[A]())
            self.cnt.append(Scalar[S](0))

    def combine_at(
        mut self, g: Int, value: Scalar[Self.Acc.native], count: Int
    ) raises:
        """Fold an already-reduced value into group `g`, crediting `count` rows.

        The entry point for a caller that folded in **registers** rather than
        scattering. An ungrouped aggregate reduces a whole morsel to one value
        and one count, then calls this once per morsel instead of scattering
        once per row — measured at 14.6x on 1M rows, where the scatter is a
        million serially dependent read-modify-writes through a builder slot.

        Additive, not a store, so the register accumulator stays *per-batch
        scratch* and this state remains the only thing crossing a batch
        boundary. `K.finalize`, `K.empty_is_null` and the count-is-zero rule
        therefore stay defined in exactly one place — `finish` — rather than
        being re-derived by every caller that folds its own way.
        """
        self._grow(g + 1)
        comptime A = Self.Acc.native
        comptime S = Self.Seen.native
        self.acc.unsafe_set(
            g, Self.K.combine[A, 1](self.acc.unsafe_get(g), value)
        )
        comptime if Self.K.needs_count:
            self.cnt.unsafe_set(g, self.cnt.unsafe_get(g) + Scalar[S](count))
        else:
            if count > 0:
                self.cnt.unsafe_set(g, Scalar[S](1))

    @always_inline
    def accumulate[
        W: Int
    ](
        mut self,
        groups: SIMD[DType.int32, W],
        values: SIMD[Self.Acc.native, W],
        mask: SIMD[DType.bool, W],
        num_groups: Int,
    ) raises:
        """Scatter-fold one SIMD lane of already-computed values.

        The entry point a **fused** caller needs: it takes values in registers
        rather than a materialised `PrimitiveArray`, so an expression like
        `sum(a * 2 + b)` folds without ever writing the intermediate column.

        It cannot be generic over the expression that produced the lane —
        `NumericValue` lives in `marrow.expr`, and kernels must not depend on
        it. Passing the lane by value is what keeps the dependency pointing one
        way, and this method is public because the alternative is the caller
        reaching `_mark`.

        The scatter stays scalar per lane and that is not an oversight: two
        lanes may carry the same group, and a vector read-modify-write would
        lose one of them without conflict detection. `W > 1` still pays,
        measured at 1.09-1.37x, because the *loads and arithmetic* feeding this
        vectorise even though the store does not.

        `mask` is validity: a false lane contributes nothing and is not counted,
        which is what keeps a null out of both the accumulator and `K.finalize`'s
        divisor.
        """
        self._grow(num_groups)
        comptime A = Self.Acc.native
        for j in range(W):
            if mask[j]:
                var g = Int(groups[j])
                self.acc.unsafe_set(
                    g, Self.K.combine[A, 1](self.acc.unsafe_get(g), values[j])
                )
                self._mark(g)

    def _mark(mut self, g: Int):
        """Record that group `g` saw a valid value: a counter bump when the
        kernel reads the count, otherwise a plain store — no read, no add."""
        comptime S = Self.Seen.native
        comptime if Self.K.needs_count:
            self.cnt.unsafe_set(g, self.cnt.unsafe_get(g) + 1)
        else:
            self.cnt.unsafe_set(g, Scalar[S](1))

    def finish(mut self, num_groups: Int) raises -> PrimitiveArray[Self.Acc]:
        """Finalize into the typed output column. A group with no valid rows is
        NULL unless the kernel says otherwise (`FoldKernel.empty_is_null`)."""
        comptime A = Self.Acc.native
        # An aggregate over zero batches never called `update`, so the slots may
        # not exist yet. Seeding them here is what makes `SUM` of nothing NULL
        # rather than an out-of-bounds read.
        self._grow(num_groups)
        var b = PrimitiveBuilder[Self.Acc](self.dtype, num_groups)
        for g in range(num_groups):
            var c = Int(self.cnt.unsafe_get(g))
            if c > 0:
                b.append(Self.K.finalize[A](self.acc.unsafe_get(g), c))
            else:
                comptime if Self.K.empty_is_null:
                    b.append_null()
                else:
                    # `count` of nothing is 0, not unknown — finalizing the
                    # untouched identity accumulator gives exactly that.
                    b.append(Self.K.finalize[A](Self.K.identity[A](), 0))
        return b.finish()

    def into_partials(
        mut self,
    ) raises -> Tuple[PrimitiveArray[Self.Acc], Int64Array]:
        """Freeze the raw (non-finalized) per-group accumulator and valid-count
        columns — the partial state a parallel merge folds together. Consumes
        the builders.

        The exchanged count is always int64, whatever this state stores: it
        crosses a thread boundary as data, and widening a flag to a count here
        is O(groups), not O(rows)."""
        var seen = self.cnt.finish()
        var counts = Int64Builder(len(seen))
        for g in range(len(seen)):
            counts.append(Int64(seen.unsafe_get(g)))
        return (self.acc.finish(), counts.finish())

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
        comptime S = Self.Seen.native
        while self.acc.length() < num_groups:
            self.acc.append(Self.K.identity[A]())
            self.cnt.append(Scalar[S](0))

        var gids = group_ids.values()
        var acc = part_acc.values()
        var cnt = part_cnt.values()
        for j in range(len(group_ids)):
            var g = Int(gids[j])
            self.acc.unsafe_set(
                g, Self.K.combine[A, 1](self.acc.unsafe_get(g), acc[j])
            )
            comptime if Self.K.needs_count:
                self.cnt.unsafe_set(
                    g, self.cnt.unsafe_get(g) + Scalar[S](cnt[j])
                )
            else:
                if cnt[j] > 0:
                    self.cnt.unsafe_set(g, Scalar[S](1))


# ---------------------------------------------------------------------------
# AggKernel — an aggregate, whole.
#
# `FoldKernel` above is the algebra of a *fold*: identity, combine, finalize,
# one lane at a time. It is what the comptime expression lane fuses, and it
# exists only for the aggregates that can be expressed that way.
#
# `AggKernel` is the aggregate itself. Every aggregate has one — the two that
# have no fold algebra at all (a distinct count keeps a hash set or a sketch, a
# string min/max keeps the index of the best row) as much as the four that do.
# Its inputs and its output are **erased**, and that is the whole point: it is
# one vocabulary that the comptime lane can name as a type parameter and the
# runtime lane can store as a plain function pointer.
#
# This replaced a second, *typed* copy of the same four aggregates. The pair
# `Aggregation` (typed, with `InArray`/`OutArray`/`from_any`/`to_dyn`) and
# `ColumnAggregation` (erased) described the same `sum`, `min`, `count` and
# `count_distinct` twice, and every consumer picked one column and ignored the
# other. Erasing costs an O(1) handle copy at the boundary; the fold's hot loop
# is still fully typed, because it runs inside `AggState[K, V]` after one
# dispatch.
# ---------------------------------------------------------------------------


comptime AggregateFn = def(Groups, List[DynArray]) thin raises -> DynArray
"""`AggKernel.grouped`, erased — the assignment, its columns, one value per
slot out.

Mojo has no dynamic dispatch, so this pointer is how a caller that resolved an
aggregate *at run time* stores which one it got. A caller that knows the type
writes `Agg.grouped` and pays nothing.

`List[DynArray]` rather than a single column from the start. No multi-input
aggregate is scheduled, but the signature is the expensive-to-change part —
every implementation plus every caller — and widening it costs nothing while
the operator owns the list and lends it.

`thin`, so it carries no captures and no identity: if the wrong arm is handed
over, nothing downstream can name which aggregate it holds. Whoever builds one
is the only place that pairing can be checked.
"""


trait AggKernel(Kernel):
    """One aggregate, over erased columns.

    Four implementations, named for what they compute rather than for what they
    consume: `Fold[K]` (any fixed-width column, via `K`'s lane algebra),
    `StringExtremum[Op]`, `ValidCount` and `DistinctCount[exact]`.

    No associated types, deliberately. That is what lets one vocabulary serve
    both expression lanes: the comptime lane takes the trait as a parameter and
    calls `grouped` directly, the runtime lane takes it as an `AggregateFn` and
    calls it through a pointer. A single associated type would break the second
    half, because an erased caller cannot name it.
    """

    comptime mergeable: Bool = False
    """Whether `partials` / `merge` are implemented — whether this aggregate can
    run as independent per-thread folds that are combined afterwards.

    A sketch-based aggregate cannot: two hash sets do not add. The grouper reads
    this before choosing a strategy, so an aggregate that says `False` is simply
    never asked."""

    @staticmethod
    def dtype(inputs: List[DynType]) raises -> DynType:
        """The column this produces, from its inputs' dtypes alone.

        Answered before any data exists — a plan's output schema is built from
        it — which is why it takes dtypes and not arrays. It is also the arity
        and domain gate: an aggregate that is not defined for these inputs
        raises here rather than at the first batch.
        """
        ...

    @staticmethod
    def grouped(groups: Groups, inputs: List[DynArray]) raises -> DynArray:
        """One value per slot.

        **`groups.is_single()` must be its first branch.** The one-slot
        assignment carries no ids, and a per-group body is a
        `for i in range(len(groups.ids))` loop, which over an empty id array
        does not execute at all — `[0]` or `[null]` where the whole-input
        answer belongs. A wrong answer, not a crash. It is also where the
        whole-column fast paths live: a vectorized reduce, an O(1) metadata
        read, a radix-parallel sketch.
        """
        ...

    @staticmethod
    def empty() raises -> Optional[DynArray]:
        """The one-row answer over an input that produced **no column at all**,
        or `None` when there is no such answer.

        A filter that keeps nothing answers with no batch rather than an empty
        one, so an aggregate above it never sees a dtype. `COUNT(DISTINCT x)`
        of nothing is still `0` — SQL's answer and PyArrow's — and needs no
        dtype to say so. An extremum does need one, so it declines here and the
        caller supplies a null from the plan's schema.
        """
        return None

    @staticmethod
    def partials(
        in_dtype: DynType, groups: Groups, inputs: List[DynArray]
    ) raises -> Tuple[DynArray, Int64Array]:
        """One thread's raw, *non-finalized* per-group accumulator plus its
        valid counts, for a later `merge`.

        `in_dtype` is passed rather than read off `inputs`, so that `merge` —
        which only ever sees accumulators — can be handed the same value. A
        widening fold loses the input type on the way out (`sum(int32)`
        accumulates in int64), so the accumulator column cannot answer what it
        was folded from.
        """
        raise Error(
            "aggregate '", Self.name, "' has no mergeable partial state"
        )

    @staticmethod
    def merge(
        in_dtype: DynType,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        """Fold every thread's partials at remapped group ids, then finalize."""
        raise Error(
            "aggregate '", Self.name, "' has no mergeable partial state"
        )


struct Fold[K: FoldKernel](AggKernel):
    """The aggregate expressible as a lane fold, over any fixed-width column —
    `sum`, `product`, `mean`, `min`, `max`, `count`.

    One dispatch on the input's runtime dtype, then the fully typed
    `AggState[K, V]` — the same state the comptime lane fuses, reached the slow
    way. Two things send an aggregate here rather than into the fused loop: the
    query named it with a string, or its input is temporal.

    `K.acc_dtype` decides the output dtype, so `sum(int32)` widens to int64,
    `mean` answers float64, and `min`/`max` keep the input's unit, timezone,
    precision and scale.
    """

    comptime name = Self.K.name
    comptime mergeable = True

    @staticmethod
    def _domain(inputs: List[DynType]) raises -> DynType:
        """The arity and domain gate every entry point shares.

        The runtime half of `AggState`'s compile-time domain assertion: an
        arithmetic fold is numeric-only, and instantiating one over a temporal
        column is a *build* error rather than a raise, so the two must agree or
        a query that should raise fails to compile instead.
        """
        if len(inputs) != 1:
            raise Self.error(t"takes exactly one input, got {len(inputs)}")
        ref dtype = inputs[0]
        comptime if conforms_to(Self.K, ArithmeticAgg):
            if not dtype.is_numeric():
                raise Self.error(
                    t"needs arithmetic, so it is not defined for"
                    t" {dtype} columns"
                )
        else:
            if not (dtype.is_numeric() or dtype.is_temporal()):
                raise Self.error(t"is not defined for {dtype} columns")
        return dtype.copy()

    @staticmethod
    def _domain(inputs: List[DynArray]) raises -> DynType:
        """The same gate, reached from the columns themselves."""
        var dtypes = List[DynType]()
        for i in range(len(inputs)):
            dtypes.append(inputs[i].dtype())
        return Self._domain(dtypes)

    @staticmethod
    def dtype(inputs: List[DynType]) raises -> DynType:
        var in_dtype = Self._domain(inputs)

        def numeric[V: NumericType](d: V) raises {imm} -> DynType:
            return DynType(Self.K.acc_dtype[V](d))

        comptime if conforms_to(Self.K, ArithmeticAgg):
            return in_dtype.dispatch_numeric(numeric)
        else:
            if in_dtype.is_numeric():
                return in_dtype.dispatch_numeric(numeric)

            def temporal[V: TemporalType](d: V) raises {imm} -> DynType:
                return DynType(Self.K.acc_dtype[V](d))

            return in_dtype.dispatch_temporal(temporal)

    @staticmethod
    def grouped(groups: Groups, inputs: List[DynArray]) raises -> DynArray:
        var in_dtype = Self._domain(inputs)

        # Numeric and temporal are separate dispatch arms rather than one
        # `dispatch_primitive`, so a kernel whose domain excludes temporal
        # columns is never instantiated over one — `AggState`'s domain
        # assertion would fail the build rather than raise.
        def numeric[V: NumericType](d: V) raises {imm} -> DynArray:
            return Self._one[V](groups, inputs[0])

        comptime if conforms_to(Self.K, ArithmeticAgg):
            return in_dtype.dispatch_numeric(numeric)
        else:
            if in_dtype.is_numeric():
                return in_dtype.dispatch_numeric(numeric)

            def temporal[V: TemporalType](d: V) raises {imm} -> DynArray:
                return Self._one[V](groups, inputs[0])

            return in_dtype.dispatch_temporal(temporal)

    @staticmethod
    def _one[
        V: PrimitiveType
    ](groups: Groups, input: DynArray) raises -> DynArray:
        """One fixed-width column folded by `K`, at one slot or at many."""
        var column = input.as_primitive[V]().copy()
        if groups.is_single():
            # The vectorized whole-array reduce, not the scatter loop over an
            # id array that does not exist. Serial: the SIMD reduce only
            # benefits from threads well above the sizes reached here, and that
            # gating belongs in the reduce primitive. `with_threads(1)` rather
            # than `serial()` — forcing one worker must not also discard a
            # device the reduce could run on.
            return (
                Self.K.reduce(column, ExecContext.auto().with_threads(1))
                .repeat(1)
                .to_dyn()
            )
        var state = AggState[Self.K, V](Self.K.acc_dtype[V](column.dtype))
        state.update(groups.ids, column, groups.num_groups)
        return state.finish(groups.num_groups).to_dyn()

    @staticmethod
    def partials(
        in_dtype: DynType, groups: Groups, inputs: List[DynArray]
    ) raises -> Tuple[DynArray, Int64Array]:
        var box = List[Tuple[DynArray, Int64Array]]()
        box.reserve(1)

        def run[V: PrimitiveType](d: V) raises {mut box, imm}:
            var column = inputs[0].as_primitive[V]().copy()
            var state = AggState[Self.K, V](Self.K.acc_dtype[V](column.dtype))
            state.update(groups.ids, column, groups.num_groups)
            var parts = state.into_partials()
            box.append((parts[0].copy().to_dyn(), parts[1].copy()))

        Self._dispatch(in_dtype, run)
        return box.pop()

    @staticmethod
    def merge(
        in_dtype: DynType,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        # Exact for every kernel: the accumulator is the raw fold and the count
        # is carried separately, so `mean` merges as (Σsum, Σcount) and
        # finalizes once at the end.
        if len(accs) == 0:
            raise Self.error("merge: no partial states to fold")
        var box = List[DynArray]()
        box.reserve(1)

        def run[V: PrimitiveType](d: V) raises {mut box, imm}:
            # The accumulator's dtype comes off the first partial rather than
            # from the type: `min`/`max` keep the input's, and a timestamp's
            # unit and timezone are not recoverable from `AccType` alone. Every
            # partial was folded from the same column, so any of them answers.
            comptime Acc = Self.K.AccType[V]
            var first = accs[0].as_primitive[Acc]().copy()
            var state = AggState[Self.K, V](first.dtype.copy())
            for t in range(len(remap)):
                state.merge(
                    remap[t],
                    accs[t].as_primitive[Acc]().copy(),
                    cnts[t],
                    num_groups,
                )
            box.append(state.finish(num_groups).to_dyn())

        Self._dispatch(in_dtype, run)
        return box.pop()

    @staticmethod
    def _dispatch[
        Job: def[V: PrimitiveType](V) raises -> None
    ](in_dtype: DynType, job: Job) raises:
        """Narrow the input dtype to the family this kernel's domain allows.

        The same numeric / temporal split `grouped` writes out, in the one
        shape the partial-fold path needs — a job that returns nothing and
        escapes its result through a captured box.
        """

        def numeric[V: NumericType](d: V) raises {imm}:
            job[V](d)

        comptime if conforms_to(Self.K, ArithmeticAgg):
            if not in_dtype.is_numeric():
                raise Self.error(t"is not defined for {in_dtype} columns")
            in_dtype.dispatch_numeric(numeric)
        else:
            if in_dtype.is_numeric():
                in_dtype.dispatch_numeric(numeric)
            elif in_dtype.is_temporal():

                def temporal[V: TemporalType](d: V) raises {imm}:
                    job[V](d)

                in_dtype.dispatch_temporal(temporal)
            else:
                raise Self.error(t"is not defined for {in_dtype} columns")


struct StringExtremum[Op: MinMaxOp](AggKernel):
    """`min`/`max` over a string column — a bytewise (lexicographic) scan,
    matching Arrow's `hash_min`/`hash_max`.

    Not a fold: there is no scalar accumulator, so the scan keeps the index of
    the best row per slot and materializes at the end. That is also why it is
    not mergeable — two best-row indices into different batches do not combine.

    Nulls are excluded (SQL semantics) and an empty or all-null slot yields
    null.
    """

    comptime name = Self.Op.name

    @staticmethod
    def dtype(inputs: List[DynType]) raises -> DynType:
        if len(inputs) != 1:
            raise Self.error(t"takes exactly one input, got {len(inputs)}")
        ref in_dtype = inputs[0]
        if not (in_dtype.is_string() or in_dtype.is_large_string()):
            raise Self.error(t"is not defined for {in_dtype} columns")
        # An extremum *is* one of the input's values, so it keeps its type.
        return in_dtype.copy()

    @staticmethod
    def grouped(groups: Groups, inputs: List[DynArray]) raises -> DynArray:
        var dtypes = List[DynType]()
        for i in range(len(inputs)):
            dtypes.append(inputs[i].dtype())
        # `dtype` is the arity and domain gate, and a string extremum keeps its
        # input's type, so it also answers which arm to dispatch into.
        var in_dtype = Self.dtype(dtypes)

        def stringly[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self._scan[T](groups, inputs[0])

        return in_dtype.dispatch_stringlike(stringly)

    @staticmethod
    @always_inline
    def _better[
        T: StringLikeType
    ](values: BinaryLikeArray[T], i: Int, best: Int) raises -> Bool:
        """Whether row `i` beats the current best row."""
        var a = values.unsafe_get(UInt(i))
        var b = values.unsafe_get(UInt(best))
        return (a < b) if Self.Op.is_min else (b < a)

    @staticmethod
    def _scan[
        T: StringLikeType
    ](groups: Groups, input: DynArray) raises -> DynArray:
        var values = input.as_binary_like[T]().copy()
        var has_null = values.null_count() > 0
        if groups.is_single():
            # The same comparison as the per-group loop, without the id load —
            # and without synthesising an all-zeros id array to get there.
            var best = -1
            for i in range(len(values)):
                if has_null and not values.is_valid(i):
                    continue
                if best == -1 or Self._better[T](values, i, best):
                    best = i
            var one = BinaryLikeBuilder[T](capacity=1)
            if best == -1:
                one.append_null()
            else:
                one.append(values.unsafe_get(UInt(best)))
            return one.finish().to_dyn()

        var best = List[Int](length=groups.num_groups, fill=-1)
        var gids = groups.ids.values()
        for i in range(len(groups.ids)):
            if has_null and not values.is_valid(i):
                continue
            var g = Int(gids[i])
            if best[g] == -1 or Self._better[T](values, i, best[g]):
                best[g] = i
        var out = BinaryLikeBuilder[T](capacity=groups.num_groups)
        for g in range(groups.num_groups):
            if best[g] == -1:
                out.append_null()
            else:
                out.append(values.unsafe_get(UInt(best[g])))
        return out.finish().to_dyn()


struct ValidCount(AggKernel):
    """`COUNT(x)` — the *non-null* values of `x`, over a column of any type.

    A validity scan and nothing else, so it is defined for every dtype and
    there is nothing to monomorphize on: the input stays erased. An empty slot
    counts 0 and is never null, matching SQL.

    Mergeable, because per-slot counts merge by addition — which is exactly what
    the shared `(accumulator, valid count)` partial format already carries, with
    the count as the accumulator.

    The comptime lane does **not** route numeric `count` here — it fuses
    `FusedAggregate[CountKernel, A]`, which pays one typed `bitmap.test()` per
    row where this pays a `DynArray._dispatch` walk: a linear `comptime for`
    over 37 variant arms plus an indirect call *per row*, inside an already
    cache-hostile random-write loop. Measured at 1M rows / 100k groups on a
    nullable column: 1.7159 ms (sd 0.0702) against 8.3555 ms (sd 0.2331). On a
    null-free column this skips validity entirely via its `has_null` guard and
    never loads a value, which is why it edges ahead there instead — 1.3710 ms
    against 1.4124 ms, a ~3% gap. Converging the two would mean paying the
    erased per-row cost on every numeric `count`, so the split stays. This is
    the path for the dtypes a fold cannot serve, and for a `count` named at run
    time.
    """

    comptime name = CountKernel.name
    comptime mergeable = True

    @staticmethod
    def dtype(inputs: List[DynType]) raises -> DynType:
        if len(inputs) != 1:
            raise Self.error(t"takes exactly one input, got {len(inputs)}")
        return DynType(int64)

    @staticmethod
    def grouped(groups: Groups, inputs: List[DynArray]) raises -> DynArray:
        ref column = inputs[0]
        if groups.is_single():
            # A valid count is metadata. Losing this branch would turn
            # `count(x)` with no GROUP BY from O(1) into O(n).
            return (
                Int64Scalar(Int64(len(column) - column.null_count()))
                .repeat(1)
                .to_dyn()
            )
        return Self._per_group(groups, column).to_dyn()

    @staticmethod
    def _per_group(groups: Groups, column: DynArray) raises -> Int64Array:
        var counts = List[Int64](length=groups.num_groups, fill=0)
        var gids = groups.ids.values()
        var has_null = column.null_count() > 0
        for i in range(len(groups.ids)):
            if has_null and not column.is_valid(i):
                continue
            counts[Int(gids[i])] += 1
        var out = Int64Builder(groups.num_groups)
        for g in range(groups.num_groups):
            out.append(Scalar[int64.native](counts[g]))
        return out.finish()

    @staticmethod
    def empty() raises -> Optional[DynArray]:
        return Int64Scalar(Int64(0)).repeat(1).to_dyn()

    @staticmethod
    def partials(
        in_dtype: DynType, groups: Groups, inputs: List[DynArray]
    ) raises -> Tuple[DynArray, Int64Array]:
        """A thread's per-slot counts, in both halves of the partial format: as
        the accumulator to merge, and as the valid count that says the slot was
        seen at all."""
        var counts = Self._per_group(groups, inputs[0])
        return (counts.copy().to_dyn(), counts.copy())

    @staticmethod
    def merge(
        in_dtype: DynType,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        var totals = List[Int64](length=num_groups, fill=0)
        for t in range(len(remap)):
            var gids = remap[t].values()
            var part = accs[t].as_int64().values()
            for j in range(len(remap[t])):
                totals[Int(gids[j])] += part[j]
        var out = Int64Builder(num_groups)
        for g in range(num_groups):
            out.append(Scalar[int64.native](totals[g]))
        return out.finish().to_dyn()


struct DistinctCount[exact: Bool](AggKernel):
    """`COUNT(DISTINCT x)` exactly, or a HyperLogLog estimate of it.

    Not a fold at all — the per-slot state is a hash set or a sketch, not a
    scalar accumulator — which is why there is no `FoldKernel` for it, no fused
    form to fall back to, and no merge: two sketches of the same rows do not
    add. Nulls are excluded (SQL semantics, PyArrow's `only_valid`).
    """

    comptime name = "count_distinct" if Self.exact else "approx_count_distinct"

    @staticmethod
    def dtype(inputs: List[DynType]) raises -> DynType:
        if len(inputs) != 1:
            raise Self.error(t"takes exactly one input, got {len(inputs)}")
        # A cardinality, whatever was counted.
        return DynType(int64)

    @staticmethod
    def grouped(groups: Groups, inputs: List[DynArray]) raises -> DynArray:
        comptime if Self.exact:
            if groups.is_single():
                # Not an optimisation: `count_distinct_grouped` loops over ids
                # there are none of. It is also where the whole-column
                # radix-partition-parallel path lives, which self-gates on size
                # — so the context passes straight through.
                return (
                    count_distinct(inputs[0], ExecContext.auto())
                    .repeat(1)
                    .to_dyn()
                )
            return count_distinct_grouped(groups, inputs[0]).to_dyn()
        else:
            if groups.is_single():
                return (
                    approx_count_distinct(inputs[0], ExecContext.auto())
                    .repeat(1)
                    .to_dyn()
                )
            return approx_count_distinct_grouped(groups, inputs[0]).to_dyn()

    @staticmethod
    def empty() raises -> Optional[DynArray]:
        return Int64Scalar(Int64(0)).repeat(1).to_dyn()
