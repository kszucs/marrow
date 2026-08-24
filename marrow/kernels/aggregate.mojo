"""Aggregate kernels — scalar reductions and grouped aggregation.

Two layers, both made of types:

- **``AggKernel``** — the pure *algebra* of a fold: the accumulator-dtype
  rule (``AccType``), an ``identity``, a SIMD ``combine`` and a ``finalize``.
  It knows nothing about arrays beyond the fully typed whole-array ``reduce`` /
  ``apply``. Grouped folding is ``AggState[K, V]`` — a fully typed per-group
  state with no dtype dispatch anywhere inside it.
- **``Aggregation``** — a fold *bound to a concrete input type*: the pair
  (kernel, input dtype) resolved down to one type that names its own
  ``InArray`` / ``OutArray`` and carries the whole per-column implementation
  (``grouped`` / ``whole`` / ``partials`` / ``merge``). Every aggregate
  behaviour that used to be selected by comparing an aggregate's *name* — the
  bytewise string ``min``/``max``, the temporal reinterpret, the validity-only
  ``count``, the distinct sketches — is a distinct ``Aggregation`` type
  instead, chosen once when the input dtype becomes known.

The named aggregate functions themselves (``Sum``, ``Min``, ``Count``, …) are
the expression layer's vocabulary, not this one's: they live in
``marrow.exprold.aggregates`` as ``AggFunction``s and resolve a runtime input dtype
onto one of the ``Aggregation`` types here. No aggregate *name* is ever compared
in this module.
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
# AggKernel — one trait for every aggregate.
#
# A kernel is the pure algebra of a fold. Grouped aggregation is driven by
# `AggState[K, V]` (fully typed); whole-array reduction is `reduce` — the
# single-full-group case — which defaults to that same path but is overridden by
# `sum`/`min`/`max`/`product` with the SIMD `apply` fast path. One SIMD
# `combine[T, W]` per kernel serves both: the horizontal reduce (same-type) and
# the grouped scatter (fold `combine[A, 1]` over each value cast to `A`).
# ---------------------------------------------------------------------------


trait AggKernel(Kernel):
    """An aggregate: the pure *algebra* of a fold — accumulator-dtype
    (`AccType`), `identity`, SIMD `combine`, and `finalize` — plus a default
    whole-array `reduce`.

    Everything here is typed: `reduce` / `apply` take a `PrimitiveArray[V]` and
    return a `PrimitiveScalar`, so a kernel never sees an `DynArray` or an
    `DynType`. Binding a kernel to a *runtime* dtype (and routing the
    non-numeric input types it can also serve) is `Aggregation`'s job, below.

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

    comptime Grouped[V: PrimitiveType]: Aggregation
    """The `Aggregation` that implements this kernel over a numeric column of
    type `V` — normally the typed `AggState` fold, `NumericAgg[Self, V]`. A
    kernel whose grouped form is *not* that fold (`count`, whose per-group state
    is a validity scan) names its own, so the choice is a type and never a
    comparison."""

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
        _check_domain[Self, V]()
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
# enforced by `_check_domain`, which every fold runs at compile time, so
# `sum(date)` is a build error rather than a silent nonsense or a runtime
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
# by `AggKernel.acc_dtype(input_dtype)` — the one place that knows whether the
# accumulator keeps the input's dtype (`min`/`max`) or names its own
# (`sum` -> int64/float64, `count` -> int64, `mean` -> float64).
trait OrderedAgg(AggKernel):
    """Needs an ordering and nothing more — any fixed-width type will do.

    `min`, `max`, and later `quantile` / `median`.
    """

    pass


trait ArithmeticAgg(AggKernel):
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


def _check_domain[K: AggKernel, V: PrimitiveType]():
    """Compile-time gate: reject an input type this kernel's domain excludes.

    Non-raising, so it runs at comptime; a violation is a build error naming
    the domain, not an `Error` at run time. No marker means "accepts
    anything" — that is `CountKernel`, which reads validity and never touches
    a value, and `OrderedAgg`, which needs only a total order and gets one from
    every fixed-width type.
    """
    comptime if conforms_to(K, IntegralAgg):
        comptime assert conforms_to(
            V, IntegerType
        ), "this aggregate is defined for integer columns only"
    elif conforms_to(K, ArithmeticAgg):
        comptime assert conforms_to(V, NumericType), (
            "this aggregate needs arithmetic, so it is defined for numeric"
            " columns only -- not temporal, interval or decimal"
        )


struct Widening[Op: WideningOp](ArithmeticAgg):
    """`sum`/`product` as one kernel: integers accumulate in int64 and floats in
    float64 so narrow inputs cannot overflow, the fold is `Op.combine`, and
    finalize is the identity."""

    comptime name = Self.Op.name
    comptime AccType[
        V: PrimitiveType
    ] = Int64Type if V.native.is_integral() else Float64Type
    comptime Grouped[V: PrimitiveType] = NumericAgg[Self, V]

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
        _check_domain[Self, V]()
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
    comptime Grouped[V: PrimitiveType] = NumericAgg[Self, V]

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


struct CountKernel(AggKernel):
    """Counts valid (non-null) values. `combine` leaves the accumulator
    untouched — the result is the per-group valid count that every kernel keeps,
    returned by `finalize`."""

    comptime name = "count"
    comptime AccType[V: PrimitiveType] = Int64Type
    comptime empty_is_null = False
    comptime needs_count = True  # the answer *is* the count
    # `K.Grouped[V]` is the validity scan for every numeric `V` — there is no
    # accumulator to fold, and the scan is mergeable (counts add). That is what
    # the AOT lane uses (`values.mojo`'s `K.Grouped[In.OutType]`). The runtime
    # lane picks differently: it resolves numeric columns to
    # `NumericAgg[CountKernel, V]` instead (`CountValid.resolve`,
    # `expr/aggregates.mojo`) and reaches `CountAgg` only for non-numeric
    # columns. See `CountAgg`'s docstring for why the two lanes disagree here.
    comptime Grouped[V: PrimitiveType] = CountAgg

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
    comptime Grouped[V: PrimitiveType] = NumericAgg[Self, V]
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
        _check_domain[Self, V]()
        # Vectorized widened sum divided by the valid count; null on empty.
        var cnt = len(array) - array.null_count()
        if cnt == 0:
            return Float64Scalar(None, float64)
        var total = SumKernel.reduce(array, ctx)
        return Float64Scalar(total.value().cast[DType.float64]() / Float64(cnt))


# ---------------------------------------------------------------------------
# any / all — boolean reductions via SIMD bitmap operations.
#
# Not `AggKernel`s: they fold bit-packed masks (not the numeric
# accumulator/identity/combine/finalize algebra), so each is its own struct
# exposing a `reduce(BoolArray) -> Bool` (plus an `DynArray` overload), matching
# the struct-per-kernel shape of the rest of the module.
# ---------------------------------------------------------------------------


trait BoolReduceKernel(Kernel):
    """A boolean whole-column fold to a single `Bool` — `any`/`all`. Not an
    `AggKernel` (it folds bit-packed masks, not the numeric accumulator algebra),
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
# added by pairing its `AggKernel` with a different state struct exposing the
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
struct AggState[K: AggKernel, V: PrimitiveType](Movable):
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
    and only the caller knows what to pass — `AggKernel.acc_dtype(input_dtype)`
    is where every caller gets it.
    """

    def __init__(out self, dtype: Self.Acc):
        _check_domain[Self.K, Self.V]()
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
        Unreachable through `AggKernel.reduce`, which always calls `update`
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
        NULL unless the kernel says otherwise (`AggKernel.empty_is_null`)."""
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
# Aggregation — a fold bound to a concrete input type.
#
# `AggKernel` is the algebra; an `Aggregation` is that algebra *applied to one
# input type*, and it is the unit the rest of the system passes around. It names
# its own `InArray` / `OutArray` and implements the whole per-column surface
# (`grouped` / `whole` / `partials` / `merge`) over those types, so nothing
# downstream has to ask what kind of aggregate it is holding: the routing that
# used to be a name comparison (bytewise string min/max, the temporal
# reinterpret, the validity-only count, the distinct sketches) is *which
# `Aggregation` type was chosen*.
#
# `from_any` / `to_dyn` are the only erasure points, both O(1) handle copies —
# a typed column enters, a typed column leaves.
# ---------------------------------------------------------------------------


trait Aggregation(Kernel):
    """One aggregate over one input type — the fully resolved, monomorphized
    unit of aggregation.

    Implementations are `NumericAgg[K, V]` (the typed `AggState` fold, over any
    fixed-width column — temporal `min`/`max` included), `StringMinMax[Op, T]`,
    `CountAgg` and `DistinctAgg[exact]`. Which one a runtime dtype maps to is
    decided once, by `AggFunction.resolve`."""

    comptime InArray: Copyable & Deinitable
    """The typed input column this aggregation consumes. `DynArray` for the two
    aggregations whose work *is* dtype-generic (validity scan, row hashing) —
    there is nothing to monomorphize on."""

    comptime OutArray: Array
    """The typed output column, always one value per group."""

    comptime is_mergeable: Bool
    """Whether `partials`/`merge` are implemented — i.e. whether this aggregate
    can run as thread-local partial folds plus a merge."""

    @staticmethod
    def from_any(value: DynArray) raises -> Self.InArray:
        """Narrow an erased column to this aggregation's input type (O(1))."""
        ...

    @staticmethod
    def to_dyn(values: Self.InArray) raises -> DynArray:
        """Widen back to an erased column — for the row-shuffling machinery
        (`take` / `slice` / `concat`), which is dtype-generic by nature."""
        ...

    @staticmethod
    def out_dtype(in_dtype: DynType) raises -> DynType:
        """This aggregation's output dtype. Takes the *runtime* input dtype
        because an order-preserving aggregate carries its parameters through
        (a timestamp's unit and timezone, a decimal's precision and scale)."""
        ...

    @staticmethod
    def grouped(groups: Groups, values: Self.InArray) raises -> Self.OutArray:
        """One aggregate column over precomputed group ids."""
        ...

    @staticmethod
    def whole(
        values: Self.InArray, ctx: ExecContext = ExecContext.auto()
    ) raises -> Self.OutArray:
        """The whole-table aggregate (no GROUP BY) as a one-row column.

        Defaults to the grouped path with every row in group 0 — which is what
        "no GROUP BY" means, and is correct for any aggregation. Override it
        only where there is a genuinely different route: a vectorized reduce, an
        O(1) answer, a whole-array sketch.

        Each aggregation still decides its own parallelism — the SIMD fold
        reductions run serial (threads only pay off well above the sizes where
        the reduce is the bottleneck), while the distinct sketches self-gate on
        size. That freedom is why this used to take a bare worker budget, but it
        never needed one: an implementation that wants serial says so with
        `ctx.with_threads(1)`, and keeps the device it was handed. The `Int`
        only ever discarded information — three of the four implementations
        ignored it outright, and the fourth rebuilt `ExecContext.parallel(n)`,
        dropping the caller's GPU device."""
        var n = len(Self.to_dyn(values))
        var zeros = Int32Builder(n)
        for _ in range(n):
            zeros.append(Int32(0))
        return Self.grouped(Groups(zeros.finish(), 1), values)

    @staticmethod
    def partials(
        groups: Groups, values: Self.InArray
    ) raises -> Tuple[Self.OutArray, Int64Array]:
        """A thread-local partial fold: the raw (non-finalized) per-group
        accumulator plus valid counts, for a later `merge`."""
        raise Error(
            "aggregate '", Self.name, "' has no mergeable partial state"
        )

    @staticmethod
    def merge(
        remap: List[Int32Array],
        accs: List[Self.OutArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> Self.OutArray:
        """Fold every thread's partials at remapped group ids and finalize."""
        raise Error(
            "aggregate '", Self.name, "' has no mergeable partial state"
        )


# ---------------------------------------------------------------------------
# AggFunction — an aggregate before its input type is known.
#
# The one dispatch left in the aggregate layer: map a *runtime input dtype* onto
# the `Aggregation` that implements this aggregate for it, and hand that type to
# a comptime `job`. Which dtypes an aggregate supports is stated by its own
# `resolve` — a new aggregate cannot forget the rule, and no central ladder has
# to know every aggregate that will ever exist.
#
# The functions themselves (`Sum`, `Min`, `Count`, …) are the frontend's
# vocabulary and live in `marrow.exprold.aggregates`, together with the one string
# comparison that maps a runtime name onto them. Nothing in this package turns a
# name into behaviour.
# ---------------------------------------------------------------------------


trait AggFunction(Kernel):
    """An aggregate *function*: a name plus the input dtypes it supports.

    The contract for resolving an aggregate against a runtime dtype — `resolve`
    picks the `Aggregation` that implements this function over a column of that
    type and hands the type to a comptime `job`. The catalog of functions
    (`Sum`, `Min`, `Count`, …) and their implementations live in
    `marrow.exprold.aggregates`; this layer only executes what it is given."""

    @staticmethod
    def resolve[
        Job: def[A: Aggregation]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        """Run `job[A]` with the `Aggregation` implementing this function over a
        `value_dtype` column. Raises if the aggregate is not defined for it."""
        ...


struct NumericAgg[K: AggKernel, V: PrimitiveType](Aggregation):
    """Kernel `K` over a fixed-width column of type `V` — the typed `AggState`
    fold.

    The fused leaf: `InArray`, `OutArray` and the accumulator are all fixed at
    compile time, so nothing is resolved at run time and the scatter loop is
    fully monomorphized.

    `V` is any `PrimitiveType`, so this is also `min`/`max` over a temporal or
    decimal column — the fold runs on the column's own storage and
    `K.acc_dtype` carries its unit, timezone, precision and scale through.
    Which kernels a non-numeric `V` is legal for is `_check_domain`'s
    question, answered at compile time."""

    comptime name = Self.K.name
    comptime InArray = PrimitiveArray[Self.V]
    comptime OutArray = PrimitiveArray[Self.K.AccType[Self.V]]
    comptime is_mergeable = True

    @staticmethod
    def from_any(value: DynArray) raises -> Self.InArray:
        return value.as_primitive[Self.V]().copy()

    @staticmethod
    def to_dyn(values: Self.InArray) raises -> DynArray:
        return values.copy().to_dyn()

    @staticmethod
    def out_dtype(in_dtype: DynType) raises -> DynType:
        # Through `acc_dtype`, so an order-preserving fold carries the input's
        # unit / timezone / precision / scale into the output column.
        _check_domain[Self.K, Self.V]()
        return DynType(Self.K.acc_dtype[Self.V](in_dtype.as_type[Self.V]()))

    @staticmethod
    def grouped(groups: Groups, values: Self.InArray) raises -> Self.OutArray:
        var state = AggState[Self.K, Self.V](
            Self.K.acc_dtype[Self.V](values.dtype)
        )
        state.update(groups.ids, values, groups.num_groups)
        return state.finish(groups.num_groups)

    @staticmethod
    def whole(
        values: Self.InArray, ctx: ExecContext = ExecContext.auto()
    ) raises -> Self.OutArray:
        # The vectorized whole-array reduce, broadcast to length 1. Serial: the
        # SIMD reduce only benefits from threads well above the sizes reached
        # here, and that gating belongs in the reduce primitive itself.
        # `with_threads(1)` rather than `serial()` — forcing one worker must not
        # also discard a device the reduce could run on.
        return Self.K.reduce(values, ctx.with_threads(1)).repeat(1)

    @staticmethod
    def partials(
        groups: Groups, values: Self.InArray
    ) raises -> Tuple[Self.OutArray, Int64Array]:
        var state = AggState[Self.K, Self.V](
            Self.K.acc_dtype[Self.V](values.dtype)
        )
        state.update(groups.ids, values, groups.num_groups)
        return state.into_partials()

    @staticmethod
    def merge(
        remap: List[Int32Array],
        accs: List[Self.OutArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> Self.OutArray:
        # Exact for every kernel: the accumulator is the raw fold and the count
        # is carried separately, so `mean` merges as (Σsum, Σcount) and
        # finalizes once at the end.
        #
        # The accumulator's dtype comes off the first partial rather than from
        # the type: `min`/`max` keep the input's, and a timestamp's unit and
        # timezone are not recoverable from `Acc` alone. Every partial was
        # folded from the same column, so any of them answers.
        if len(accs) == 0:
            raise Error("merge: no partial states to fold")
        var state = AggState[Self.K, Self.V](accs[0].dtype)
        for t in range(len(remap)):
            state.merge(remap[t], accs[t], cnts[t], num_groups)
        return state.finish(num_groups)


struct StringMinMax[Op: MinMaxOp, T: StringLikeType](Aggregation):
    """`min`/`max` over a string column — a bytewise (lexicographic) scan,
    matching Arrow's `hash_min`/`hash_max`.

    Not an `AggState` fold: there is no scalar accumulator, so the scan keeps the
    index of the best row per group and materializes at the end. Nulls are
    excluded (SQL semantics) and an empty / all-null group yields null."""

    comptime name = Self.Op.name
    comptime InArray = BinaryLikeArray[Self.T]
    comptime OutArray = BinaryLikeArray[Self.T]
    comptime is_mergeable = False

    @staticmethod
    def from_any(value: DynArray) raises -> Self.InArray:
        return value.as_binary_like[Self.T]().copy()

    @staticmethod
    def to_dyn(values: Self.InArray) raises -> DynArray:
        return values.copy().to_dyn()

    @staticmethod
    def out_dtype(in_dtype: DynType) raises -> DynType:
        return in_dtype.copy()

    @staticmethod
    @always_inline
    def _better(values: Self.InArray, i: Int, best: Int) raises -> Bool:
        """Whether row `i` beats the current best row."""
        var a = values.unsafe_get(UInt(i))
        var b = values.unsafe_get(UInt(best))
        return (a < b) if Self.Op.is_min else (b < a)

    @staticmethod
    def grouped(groups: Groups, values: Self.InArray) raises -> Self.OutArray:
        var best = List[Int](length=groups.num_groups, fill=-1)
        var gv = groups.ids.values()
        var has_null = values.null_count() > 0
        for i in range(len(groups.ids)):
            if has_null and not values.is_valid(i):
                continue
            var g = Int(gv[i])
            if best[g] == -1 or Self._better(values, i, best[g]):
                best[g] = i
        var out = BinaryLikeBuilder[Self.T](capacity=groups.num_groups)
        for g in range(groups.num_groups):
            if best[g] == -1:
                out.append_null()
            else:
                out.append(values.unsafe_get(UInt(best[g])))
        return out.finish()


struct CountAgg(Aggregation):
    """`count` over a non-numeric column — a validity-only scan.

    `count` reads validity and nothing else, so it is defined for *every* dtype
    and there is nothing to monomorphize on: the input stays erased. It does
    **not** serve numeric columns too — `CountValid.resolve`
    (`marrow/exprold/aggregates.mojo`) hands those `NumericAgg[CountKernel, V]`
    instead, the typed `AggState` fold. This type serves non-numeric columns on
    that runtime-dtype lane, plus the AOT lane's `K.Grouped` (`CountKernel.Grouped`
    names it there, `values.mojo`), which has no separate numeric/non-numeric
    split to begin with. The two lanes deliberately run different code for
    numeric `count`, measured rather than assumed: `CountAgg.grouped` takes a
    `DynArray` and calls `values.is_valid(i)` per row, which routes through
    `DynArray._dispatch` — a linear `comptime for` over 37 variant arms, with
    `Float64Array` at position 13 — paying roughly 13 sequential `isa[T]()`
    checks plus an indirect call *per row*, inside an already cache-hostile
    100k-group random-write loop. `AggState` pays one typed `bitmap.test()`
    instead. At 1M rows, g100k, nullable input: `AggState` 1.7159 ms (sd 0.0702)
    vs `CountAgg` 8.3555 ms (sd 0.2331) — roughly 4.9x, about 30x the stddev. On
    a null-free column `CountAgg` skips validity entirely via its `has_null`
    guard and never loads a value, which is why it edges ahead there instead:
    `AggState` 1.4124 ms (sd 0.0098) vs `CountAgg` 1.3710 ms (sd 0.0117) — a ~3%
    gap, only ~4x the stddev, and diluted by grouping overhead in the timed
    region. Converging the two would mean paying the erased per-row cost on
    every numeric `count`, so the split stays. An empty group counts 0 (never
    null), matching SQL.

    Mergeable, because per-group counts merge by addition — which is exactly
    what the shared `(accumulator, valid count)` partial format already carries,
    with the count as the accumulator."""

    comptime name = CountKernel.name
    comptime InArray = DynArray
    comptime OutArray = Int64Array
    comptime is_mergeable = True

    @staticmethod
    def from_any(value: DynArray) raises -> Self.InArray:
        return value.copy()

    @staticmethod
    def to_dyn(values: Self.InArray) raises -> DynArray:
        return values.copy()

    @staticmethod
    def out_dtype(in_dtype: DynType) raises -> DynType:
        return DynType(int64)

    @staticmethod
    def grouped(groups: Groups, values: Self.InArray) raises -> Self.OutArray:
        var counts = List[Int64](length=groups.num_groups, fill=0)
        var gv = groups.ids.values()
        var has_null = values.null_count() > 0
        for i in range(len(groups.ids)):
            if has_null and not values.is_valid(i):
                continue
            counts[Int(gv[i])] += 1
        var out = Int64Builder(groups.num_groups)
        for g in range(groups.num_groups):
            out.append(Scalar[int64.native](counts[g]))
        return out.finish()

    @staticmethod
    def whole(
        values: Self.InArray, ctx: ExecContext = ExecContext.auto()
    ) raises -> Self.OutArray:
        # Valid count is metadata — no scan.
        return Int64Scalar(Int64(len(values) - values.null_count())).repeat(1)

    @staticmethod
    def partials(
        groups: Groups, values: Self.InArray
    ) raises -> Tuple[Self.OutArray, Int64Array]:
        """A thread's per-group counts, in both slots of the partial format: as
        the accumulator to merge, and as the valid count that says the group was
        seen at all."""
        var counts = Self.grouped(groups, values)
        return (counts.copy(), counts.copy())

    @staticmethod
    def merge(
        remap: List[Int32Array],
        accs: List[Self.OutArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> Self.OutArray:
        var totals = List[Int64](length=num_groups, fill=0)
        for t in range(len(remap)):
            var gids = remap[t].values()
            var part = accs[t].values()
            for j in range(len(remap[t])):
                totals[Int(gids[j])] += part[j]
        var out = Int64Builder(num_groups)
        for g in range(num_groups):
            out.append(Scalar[int64.native](totals[g]))
        return out.finish()


struct DistinctAgg[exact: Bool](Aggregation):
    """`count_distinct` (exact) / `approx_count_distinct` (HyperLogLog).

    Not a fold at all — the per-group state is a hash set / HLL sketch rather
    than a scalar accumulator, which is why it is never mergeable. The work is
    row hashing, which is dtype-generic, so the input stays erased."""

    comptime name = "count_distinct" if Self.exact else "approx_count_distinct"
    comptime InArray = DynArray
    comptime OutArray = Int64Array
    comptime is_mergeable = False

    @staticmethod
    def from_any(value: DynArray) raises -> Self.InArray:
        return value.copy()

    @staticmethod
    def to_dyn(values: Self.InArray) raises -> DynArray:
        return values.copy()

    @staticmethod
    def out_dtype(in_dtype: DynType) raises -> DynType:
        return DynType(int64)

    @staticmethod
    def grouped(groups: Groups, values: Self.InArray) raises -> Self.OutArray:
        comptime if Self.exact:
            return count_distinct_grouped(groups, values)
        else:
            return approx_count_distinct_grouped(groups, values)

    @staticmethod
    def whole(
        values: Self.InArray, ctx: ExecContext = ExecContext.auto()
    ) raises -> Self.OutArray:
        # `count_distinct` self-gates on size, going radix-partition-parallel at
        # scale, so the context passes straight through — this is the one
        # implementation that ever used the worker budget, and rebuilding it as
        # `ExecContext.parallel(n)` is what dropped the device.
        comptime if Self.exact:
            return count_distinct(values, ctx).repeat(1)
        else:
            return approx_count_distinct(values, ctx).repeat(1)
