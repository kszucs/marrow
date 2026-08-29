"""Aggregate kernels — scalar reductions and grouped aggregation.

Two levels, and they answer different questions.

- **``FoldKernel``** — the pure *algebra* of a fold: the accumulator-dtype rule
  (``AccType``), an ``identity``, a SIMD ``combine`` and a ``finalize``. It
  knows nothing about arrays at all — no ``reduce``, no ``apply`` — and grouped
  folding is ``AggState[K, V]``, a fully typed per-group state with no dtype
  dispatch anywhere inside it. This is the level the comptime expression lane
  fuses into one SIMD loop, and it exists only for the aggregates that *are*
  folds.
- **``AggKernel``** — one aggregate, whole, **typed on what it consumes and
  produces** (``InArray`` / ``OutArray``): the output dtype from the input
  dtype, and one value per group. Every aggregate has one — ``Fold[K, V]``
  wraps the level above, and ``StringExtremum``, ``Dispersion``, ``ValidCount``
  and ``DistinctCount`` have no fold algebra to wrap.

Typing at the second level is deliberate, and it is what lets a single
vocabulary serve both expression lanes: the comptime lane names an
``AggKernel`` as a type parameter, and the runtime lane resolves a *name* onto
one through ``expr.runtime.aggregates.dispatch_agg`` before any state exists.
Neither lane ever holds an erased aggregate, so the hot loop is fully typed
both ways.

No aggregate *name* is ever compared in this module. The ten names themselves
live here, as the constants every kernel's ``Kernel.name`` is defined from and
as ``agg_vocabulary()``; mapping one onto ``Fold[SumKernel, V]`` needs a
runtime dtype and is the expression layer's job (``dispatch_agg``).
"""

import std.math as math

from ..arrays import (
    Array,
    Float64Array,
    BoolArray,
    BinaryLikeArray,
    PrimitiveArray,
    DynArray,
    Int32Array,
    Int64Array,
)
from ..builders import (
    Float64Builder,
    PrimitiveBuilder,
    DynBuilder,
    Int64Builder,
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
from ..scalars import PrimitiveScalar, DynScalar
from ..views import reduce
from .core import Kernel, Groups
from ..execution import ExecContext
from .distinct import (
    HLL_P_GROUPED,
    hll_estimate,
    hll_rho,
)
from .hashing import RapidHashKernel
from .hashtable import SwissHashTable
from ..utils import RapidHash64
from ..arrays import StructArray
from ..dtypes import Field, struct_


# ---------------------------------------------------------------------------
# FoldKernel — the algebra, and only the algebra.
#
# A kernel is the pure algebra of a fold: it names no array type and declares no
# entry point that takes one. Grouped and ungrouped folding are both driven by
# `AggState[K, V]` (fully typed), and one SIMD `combine[T, W]` per kernel serves
# both: the horizontal reduce over a register (same-type) and the grouped
# scatter (fold `combine[A, 1]` over each value cast to `A`).
# ---------------------------------------------------------------------------


trait FoldKernel(Kernel):
    """The pure *algebra* of a fold — accumulator-dtype (`AccType`),
    `identity`, SIMD `combine`, and `finalize`.

    Not an aggregate: that is `AggKernel`, below. This is the algebra an
    aggregate may be *built from*, and only six are. `Fold[K, V]` is the one
    that turns it into one, and the comptime expression lane fuses it directly.

    **No array-shaped member at all.** `reduce` and `apply` used to be
    documented here as a "default whole-array path" the extrema overrode; they
    were removed with the streaming rewrite and the docstring outlived them by
    long enough to mislead. Whole-column reduction is now the `is_single()` arm
    of `AggState.update`, reached through `Fold`, so a `FoldKernel` never sees a
    `PrimitiveArray` — only `SIMD` — and binding one to a *runtime* dtype
    happens on the other side of that boundary.

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
    (`ArithmeticAgg`) — one bound cannot say
    both "I can be read as a lane" and "I support addition"."""

    @staticmethod
    def check_domain[V: PrimitiveType]():
        """Compile-time gate: reject an input type this kernel's domain excludes.

        Non-raising, so it runs at comptime; a violation is a build error naming
        the domain, not an `Error` at run time. No marker means "accepts
        anything" — that is `CountKernel`, which reads validity and never touches
        a value, and `min`/`max`, which need only a total order and get one
        from every fixed-width type.
        """
        comptime if conforms_to(Self, ArithmeticAgg):
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


# ---------------------------------------------------------------------------
# The vocabulary — every aggregate name, once.
#
# **No aggregate name is ever *compared* in this module**, and that has not
# changed: mapping `"sum"` onto `Fold[SumKernel, V]` is a runtime-dtype
# question and lives in `expr/runtime/aggregates.mojo`, with `dispatch_agg`.
# What lives here is the *catalog* — the strings themselves — because they are
# the kernels' own `Kernel.name` values and every one below is defined from
# them.
#
# They were declared a second time in the expression layer, beside a
# hand-written `vocabulary()` listing the same ten. Nothing enforced that the
# two agreed, and a disagreement is not a build error: a name the resolver
# accepts but no kernel answers to raises `unknown aggregate` from inside
# `dispatch_agg` on the first morsel, and a name a kernel reports but the
# resolver rejects is simply unreachable. Deriving each `name` from the
# constant makes both unrepresentable.
# ---------------------------------------------------------------------------
comptime SUM = "sum"
comptime PRODUCT = "product"
comptime MEAN = "mean"
comptime COUNT = "count"
comptime COUNT_DISTINCT = "count_distinct"
comptime APPROX_COUNT_DISTINCT = "approx_count_distinct"
comptime VARIANCE = "variance"
comptime STDDEV = "stddev"
comptime MIN = "min"
comptime MAX = "max"


def agg_vocabulary() -> List[String]:
    """Every aggregate name an `AggKernel` answers to.

    The list a frontend validates against, and the reason the constants above
    are not simply inlined: a caller that wants to know whether `"cnt"` is an
    aggregate has one place to ask. `any` and `all` are deliberately absent —
    they are `BoolReduceKernel`s, not `AggKernel`s, and no expression node
    resolves to them.
    """
    var out = List[String](capacity=10)
    out.append(SUM)
    out.append(PRODUCT)
    out.append(MEAN)
    out.append(COUNT)
    out.append(COUNT_DISTINCT)
    out.append(APPROX_COUNT_DISTINCT)
    out.append(VARIANCE)
    out.append(STDDEV)
    out.append(MIN)
    out.append(MAX)
    return out^


# ---------------------------------------------------------------------------
# Kernel structs — one SIMD `combine` each, and nothing else. There is no
# per-kernel array fast path to override: `Fold.update` takes the whole-column
# route through `views.reduce` for every one of them.
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
    comptime name = SUM

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b


struct ProductOp(WideningOp):
    comptime name = PRODUCT

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
trait ArithmeticAgg(FoldKernel):
    """Needs addition or division, so numeric input only.

    `sum`, `product`, `mean`, and later `variance` / `stddev`.
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
    comptime name = MIN
    comptime is_min = True

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MAX_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.min(a, b)


struct MaxOp(MinMaxOp):
    comptime name = MAX
    comptime is_min = False

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T].MIN_FINITE

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return math.max(a, b)


struct MinMax[Op: MinMaxOp](FoldKernel):
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


comptime MinKernel = MinMax[MinOp]
comptime MaxKernel = MinMax[MaxOp]


struct CountKernel(FoldKernel):
    """Counts valid (non-null) values. `combine` leaves the accumulator
    untouched — the result is the per-group valid count that every kernel keeps,
    returned by `finalize`."""

    comptime name = COUNT
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


struct MeanKernel(ArithmeticAgg):
    """Sums into a float64 accumulator; divides by the valid count on finish."""

    comptime name = MEAN
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
# AggState — per-group state for a *fully typed* (kernel, input dtype) pair.
#
# The default aggregate state: an accumulator column plus a valid-count column.
# `acc` is a real `PrimitiveBuilder[K.AccType[V]]` (not erased), so `update` /
# `finish` carry no dtype dispatch at all — the runtime dtype was resolved once
# at the boundary, by `dispatch_agg` in the expression layer, before this type
# existed. The count column drives NULL output for empty/all-null groups and
# the `mean` divisor.
#
# `K` and `V` are struct parameters rather than per-call ones, so the state
# holds no kernel identity and the kernel layer needs no enum or vtable. A
# richer aggregate (variance = count+mean+M2, distinct = hash set, ...) is
# added by pairing its `FoldKernel` with a different state struct exposing the
# same `update`/`finish` shape — `Dispersion` and `DistinctCount` below are two
# worked examples, and neither goes through this struct.
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
        self.reserve(num_groups)
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
    def reserve(mut self, num_groups: Int) raises:
        """Ensure `num_groups` slots exist, new ones seeded with `K.identity`.

        Monotonic, and that is what makes this state the **single owner** of
        the group count: `finish` reads `acc.length()` rather than taking a
        count from its caller, so a caller and a state cannot disagree. `Fold`
        used to keep a parallel `_slots` field `max`'d at four sites and pass
        it back in here, which is exactly the shape `AggKernel.finish`'s
        docstring warns against.

        Public rather than private because `Fold` forwards `AggKernel.reserve`
        onto it: an aggregate over **zero** morsels never calls `update` at
        all, and the plan still owes one row per slot — `sum` of nothing is one
        NULL, not no rows.
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
        self.reserve(g + 1)
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
        self.reserve(num_groups)
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

    def finish(mut self) raises -> PrimitiveArray[Self.Acc]:
        """Finalize into the typed output column. A group with no valid rows is
        NULL unless the kernel says otherwise (`FoldKernel.empty_is_null`).

        **Takes no group count.** `reserve` is monotonic and every write path
        goes through it, so `acc.length()` *is* how far this state grew — and
        it is the only number that cannot be wrong. A caller that had to pass
        one could pass a stale one, which is the disagreement
        `AggKernel.finish` documents.
        """
        comptime A = Self.Acc.native
        var num_groups = self.acc.length()
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


# ---------------------------------------------------------------------------
# AggKernel — an aggregate, whole.
#
# `FoldKernel` above is the algebra of a *fold*: identity, combine, finalize,
# one lane at a time. It is what the comptime expression lane fuses, and it
# exists only for the aggregates that can be expressed that way.
#
# `AggKernel` is the aggregate itself. Every aggregate has one — the three that
# have no fold algebra at all (a distinct count keeps a hash set or a sketch, a
# string min/max keeps the best value, a dispersion keeps Welford's triple) as
# much as the ones that do. Its input and its output are **typed**
# (`InArray` / `OutArray`), and that is the whole point: it is one vocabulary
# the comptime lane names as a type parameter and the runtime lane resolves a
# *name* onto through `dispatch_agg`, so neither ever holds an erased
# aggregate.
#
# This replaced a second, *erased* copy of the same aggregates. The pair
# `Aggregation` (typed, with `InArray`/`OutArray`/`from_any`/`to_dyn`) and
# `ColumnAggregation` (erased) described the same `sum`, `min`, `count` and
# `count_distinct` twice, and every consumer picked one column and ignored the
# other. Narrowing an erased column at the boundary costs an O(1) handle copy;
# the fold's hot loop is fully typed either way, because it runs inside
# `AggState[K, V]` after that one narrowing.
# ---------------------------------------------------------------------------


trait AggKernel(Deinitable, Kernel, Movable):
    """One aggregate, **typed on what it consumes and produces**.

    Five implementations, named for what they compute rather than for what they
    consume: `Fold[K, V]`, `StringExtremum[Op, T]`, `Dispersion[ddof, root, V]`,
    `ValidCount` and `DistinctCount[exact]`.

    **Typed first, narrowed once at the boundary — the same shape as every
    other kernel family here.** `filter`, `take`, `cast` and `concat` all put
    the logic in typed overloads and expose one erased entry point that narrows
    and delegates. Aggregates briefly did the opposite: the contract itself
    spoke `List[DynArray]`, so every conformer re-narrowed on every morsel and
    the trait could not say what it ate. There is no erased face at all now —
    `Self.InArray(column.to_data())` at the one call site in
    `BufferedAggregateOperator.push` is the whole of it.

    **An aggregate is a state machine**, so `reserve` / `update` / `finish` are
    instance methods. `dtype` is not: it is a plan-time question, answered from
    dtypes alone before any state exists, which is what lets a plan build its
    output schema without touching data.
    """

    comptime InArray: Array
    """The typed column this consumes.

    Every conformer answers with a concrete array, including the two whose
    *algebra* is dtype-generic. `Fold[K, V]` eats a `PrimitiveArray[V]`,
    `StringExtremum[Op, T]` a `BinaryLikeArray[T]`, `Dispersion[…, V]` a
    `PrimitiveArray[V]` — and `ValidCount[A]` / `DistinctCount[exact, A]` eat
    an `A`, because a cardinality is dtype-generic in what it *computes* and
    not in how it *reads*: a valid count wants a validity bitmap and a distinct
    count wants to hash values, and both are faster knowing the array type than
    walking an erased one.

    `Array` rather than a looser bound, and that is what removes two methods
    from this trait: every `Array` constructs from `ArrayData` and answers
    `type()`, so a caller narrows with `Self.InArray(column.to_data())` and
    reads the dtype with `input.type()` — neither needs a per-conformer hook.

    An earlier shape let this be `DynArray` so the two cardinalities could sit
    under the same trait without naming an array type. It bought nothing: the
    bound had no readable members, so no kernel body was written against it,
    and with `DynArray` in the same bound as `PrimitiveArray[Int64Type]` the
    declaration said only "some column-ish thing". Resolving the array type at
    the *dispatch site* — where `dispatch_primitive` already binds a concrete
    `T` — is what lets the bound mean what it says.
    """

    comptime OutArray: Array
    """The typed column this produces, one value per slot."""

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        """The column this produces, from its input's dtype alone.

        Answered before any data exists — a plan's output schema is built from
        it — which is why it takes a dtype and not an array. It is also the
        domain gate: an aggregate that is not defined for this input raises
        here rather than at the first batch.

        **One dtype, not a list.** Every aggregate here reduces exactly one
        column, so a `List[DynType]` bought nothing but an arity check in each
        of five conformers — five chances to word the same error differently,
        and a shape that let a two-input call typecheck and fail at run time.
        A genuinely two-operand aggregate (`corr`, `covar`) is a different
        family with a different trait, not a longer list."""
        ...

    def __init__(out self, in_dtype: DynType) raises:
        """A fresh accumulator ready to absorb columns of `in_dtype`.

        **Construction, not a lifecycle step.** This was an `open` method for a
        while, on the theory that an operator is built before a schema is in
        hand. It no longer is: `Value.to_operator` takes the input schema, so
        the dtype is known wherever an aggregate is constructed. Keeping `open`
        cost every conformer an "am I open?" `Optional` — including `Fold`,
        whose accumulator is unwrapped once per SIMD lane in the fused loop —
        and four of the five bodies were a no-op or a domain check the type
        parameters already guaranteed.

        The dtype is an argument at all because marrow's dtypes are
        *parameterised values*, not fully reified types: `TimestampType` carries
        `var unit` and `var timezone`, so `V` says "timestamp" and cannot say
        "timestamp[us, UTC]". `Fold` needs that to size its accumulator; the
        others ignore it, which is why they get a one-line constructor.
        """
        ...

    def reserve(mut self, slots: Int) raises:
        """Ensure `slots` per-group slots exist, seeded with this aggregate's
        empty answer.

        **The zero-morsel seed, and nothing else.** `update` grows the state on
        its own; this exists for the one case `update` never sees — an
        aggregate over an input that produced no batch at all, which still owes
        one row per slot. `sum` of nothing is one NULL, `count` of nothing is
        one 0, and neither is "no rows".

        It replaced `Foldable.grow`, which said the same thing but only for the
        aggregates that fold: the seed is not lane machinery, so it does not
        belong on the lane protocol. Putting it here also unified the three
        per-conformer `_grow` length conventions that had drifted apart.

        Defaults to `pass`, so an aggregate whose state needs no per-slot
        allocation says nothing. All five conformers do need it and all five
        override; the default is for the sixth."""
        pass

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        """Absorb one morsel, already narrowed.

        **`groups.is_single()` must be its first branch.** The one-slot
        assignment carries no ids, and a per-group body is a
        `for i in range(len(groups.ids))` loop, which over an empty id array
        does not execute at all — leaving the state untouched and `finish`
        answering `[0]` or `[null]` where the whole-input answer belongs. A
        wrong answer, not a crash.

        `groups.num_groups` may grow between morsels, so every per-group
        container has to grow with it; ids already handed out are never
        renumbered, so growing in place is sound."""
        ...

    def finish(mut self) raises -> Self.OutArray:
        """One value per slot, from everything absorbed.

        Takes no group count: `update` has seen every `groups.num_groups` and
        the state is the only thing that knows how far it grew. Passing one
        here is how a caller and a state disagree."""
        ...

    @staticmethod
    def grouped(groups: Groups, input: Self.InArray) raises -> Self.OutArray:
        """The whole-input aggregate in one call — construct, `update`,
        `finish`.

        A default, for callers that genuinely have all the data — the tests,
        and any caller holding a whole column. Nothing overrides it, and an
        implementation that did would be a second copy of its own algebra.

        Static, where the rest of the contract is not: a `mut self` method
        cannot be called on a temporary, so an instance `grouped` would turn
        every `Fold[SumKernel, Int64Type].grouped(...)` into two statements. It
        needs no state of its own — it constructs one."""
        var state = Self(input.type())
        state.update(groups, input)
        return state.finish()


trait Foldable(AggKernel):
    """An `AggKernel` that wraps a lane algebra, and can name it.

    `Fold[K]` is the only conformer, and `Lane` is the `K` it was built from.
    `StringExtremum` and `DistinctCount` deliberately do not conform: a bytewise
    scan and a hash set have no identity, combine or finalize.

    This exists so the expression layer can ask *at compile time* whether an
    aggregate is capable of fusing, without a second parallel hierarchy of
    nodes. A node holds an `AggKernel`; `comptime if conforms_to(Agg, Foldable)`
    is what decides whether it can hand `Agg.Lane` to the fused operator.

    The two methods below are the **lane-facing** half of that: a fused
    driver folds in registers and needs somewhere to put the result, which
    `update(groups, inputs)` cannot express because it takes a materialised
    column. They are forwarding one-liners onto `AggState`, and they exist so
    the expression layer stops holding an `AggState[K, V]` field directly —
    reaching past the kernel into its own state struct was the leak.

    **`grow` used to be a third, and did not belong.** It was never called from
    a lane: its only caller was the fused operator's `drain`, seeding slots for
    an input that produced no rows. That is a property of *every* aggregate,
    not of the ones drivable from registers, so it moved up to
    `AggKernel.reserve` and this protocol became exactly "this aggregate can be
    driven from registers" — `Lane`, `Acc`, `scatter`, `combine_at`.
    """

    comptime Lane: FoldKernel
    """The lane algebra this aggregate folds with."""

    comptime Acc: DType
    """The accumulator's element dtype, for the SIMD the driver folds in."""

    def scatter[
        W: Int
    ](
        mut self,
        groups: SIMD[DType.int32, W],
        values: SIMD[Self.Acc, W],
        valid: SIMD[DType.bool, W],
        num_groups: Int,
    ) raises:
        """Fold `W` lanes into the groups their ids name."""
        ...

    def combine_at(
        mut self, slot: Int, value: Scalar[Self.Acc], count: Int
    ) raises:
        """Fold an already-reduced register value into `slot`, crediting
        `count` rows — the ungrouped path's once-per-morsel hand-off."""
        ...


struct Fold[K: FoldKernel, V: PrimitiveType](Foldable):
    """The aggregate expressible as a lane fold, over a column of type `V` —
    `sum`, `product`, `mean`, `min`, `max`, `count`.

    **Typed on its input, like every other kernel in this package.** It holds
    an `AggState[K, V]` as a plain field: no dispatch, no box, no trampoline.
    Narrowing the erased column it is handed is one O(1) handle copy, and the
    per-row loop inside `AggState.update` is fully monomorphized.

    `V` used to be absent, and `Fold[K]` resolved a *runtime* dtype on every
    call — which meant that once it had to hold state across morsels, that
    state could not be a typed field and ended up behind an `ArcPointer` and
    two thin trampolines. Erasure machinery inside a compute kernel is the
    wrong layer: mapping a runtime dtype onto a type is the expression layer's
    job, and it already does exactly that for every other kernel.

    `K.acc_dtype` decides the output dtype, so `sum(int32)` widens to int64,
    `mean` answers float64, and `min`/`max` keep the input's unit, timezone,
    precision and scale.
    """

    comptime name = Self.K.name
    comptime Lane = Self.K
    comptime Acc = Self.K.AccType[Self.V].native
    comptime InArray = PrimitiveArray[Self.V]
    comptime OutArray = PrimitiveArray[Self.K.AccType[Self.V]]

    var _state: AggState[Self.K, Self.V]
    """The accumulator, built at construction.

    Not an `Optional`. It was one while `open` was a separate lifecycle step,
    and the cost landed in the worst place: the fused loop unwraps this once
    per SIMD lane through `scatter`. Taking the dtype in the constructor is
    what removes both the branch and the wrapper.

    **The only owner of the group count.** A `_slots: Int` field beside it kept
    a parallel copy, `max`'d at four sites and handed back to
    `AggState.finish`, each `max` immediately followed by a call that grew the
    state to the same value — so the two were invariantly equal and the field
    was one more thing that could disagree."""

    def __init__(out self, in_dtype: DynType) raises:
        self._state = AggState[Self.K, Self.V](
            Self.K.acc_dtype[Self.V](Self._domain(in_dtype).as_type[Self.V]())
        )

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        var checked = Self._domain(in_dtype)
        return DynType(Self.K.acc_dtype[Self.V](checked.as_type[Self.V]()))

    @staticmethod
    def _domain(in_dtype: DynType) raises -> DynType:
        """Check that this column really is a `V`, and hand it back.

        **Not a family gate.** Whether an arithmetic fold accepts this column
        is settled twice before control arrives here: `AggState` asserts it at
        compile time, and every caller binds `V` *from* `in_dtype` through
        `dispatch_numeric`/`dispatch_primitive`, so a family check could only
        fire after a different caller had already resolved wrong. It used to be
        written out anyway, and in the comptime lane it could not fire at all —
        `Column[T].dtype` ignores the schema and answers `DynType(Self.T())`,
        so the dtype being validated was manufactured from `V` two frames
        earlier. The AOT binary still linked both arms and their format
        strings.

        This check is different, and *is* reachable: `TemporalColumn[T].dtype`
        genuinely reads the schema, so a `TemporalColumn[TimestampType]` over a
        `date32` field arrives here with a `V` that does not match. Without it
        that reaches `as_type` and **aborts the process** rather than raising —
        which under `ASSERT=all` fails every case in the file.
        """
        if not in_dtype.holds[Self.V]():
            raise Self.error(
                t"was resolved for a different column type than {in_dtype}"
            )
        return in_dtype.copy()

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        ref column = input
        if groups.is_single():
            # One slot: fold this morsel into a register with `views.reduce`
            # and hand the accumulator to slot 0 — the shape the fused loop
            # uses, once per morsel rather than once per row.
            #
            # **`views.reduce` with `K.combine`, not `K.reduce`.** The kernel's
            # own `reduce` returns the *finalized* value, and for `mean` that
            # is already the quotient; handing it to `combine_at` divided a
            # second time at `finish`.
            comptime Acc = Self.K.AccType[Self.V].native
            var identity = Self.K.identity[Acc]()
            var acc: Scalar[Acc]
            if column.bitmap:
                acc = reduce[Self.V.native, Self.K.combine, Acc](
                    column.values(),
                    column.validity().value(),
                    identity,
                    ExecContext.serial(),
                )
            else:
                acc = reduce[Self.V.native, Self.K.combine, Acc](
                    column.values(), identity, ExecContext.serial()
                )
            self._state.reserve(1)
            self._state.combine_at(0, acc, len(column) - column.null_count())
        else:
            self._state.update(groups.ids, column, groups.num_groups)

    def reserve(mut self, slots: Int) raises:
        self._state.reserve(slots)

    def finish(mut self) raises -> Self.OutArray:
        return self._state.finish()

    # -- Foldable: the lane-facing half, forwarded onto `AggState` -----------

    def scatter[
        W: Int
    ](
        mut self,
        groups: SIMD[DType.int32, W],
        values: SIMD[Self.Acc, W],
        valid: SIMD[DType.bool, W],
        num_groups: Int,
    ) raises:
        self._state.accumulate[W](groups, values, valid, num_groups)

    def combine_at(
        mut self, slot: Int, value: Scalar[Self.Acc], count: Int
    ) raises:
        self._state.combine_at(slot, value, count)


struct Dispersion[ddof: Int, root: Bool, V: NumericType](AggKernel):
    """`variance` / `stddev` — the second central moment, optionally rooted.

    **The worked example of an aggregate that is a fold but not a `Fold`.**
    Welford's recurrence is a textbook fold, yet its accumulator is a *triple*
    — running count, running mean, and the sum of squared deviations `M2` —
    and `AggState[K, V]` holds exactly one accumulator column plus one count
    column. So this cannot be a `FoldKernel`, does not conform to `Foldable`,
    and an `Aggregate` over it never fuses. `Fold` means *scalar* fold; the
    composite ones are siblings.

    `ddof` is the delta degrees of freedom: the divisor is `n - ddof`, so
    `ddof=0` is the population variance and `ddof=1` the sample variance.
    Zero is the default because it is Arrow's — `VarianceOptions(int ddof = 0)`
    in `arrow/compute/api_aggregate.h` — and therefore PyArrow's.

    Welford's online form rather than the naive `E[x^2] - E[x]^2`: the naive
    one subtracts two large nearly-equal numbers and loses every significant
    digit when the mean is large relative to the spread, reporting a small
    negative variance for data that plainly has none.

    Nulls are skipped. A slot with `n - ddof <= 0` yields null, which is what
    makes `variance` of one value answer `0.0` at `ddof=0` and null at
    `ddof=1` — both checked against PyArrow.

    **Not mergeable, though Welford famously is.** The parallel combination of
    two `(n, mean, M2)` triples is standard, but `partials`/`merge` carry one
    accumulator column plus counts, and a triple does not fit. The obstacle is
    the shape of that contract, not the mathematics.
    """

    comptime name = STDDEV if Self.root else VARIANCE
    comptime InArray = PrimitiveArray[Self.V]
    comptime OutArray = Float64Array

    var _n: List[Float64]
    var _mean: List[Float64]
    var _m2: List[Float64]
    """Welford's triple, one slot each. The reason this is not a `Fold`: an
    `AggState` holds one accumulator column plus one count."""

    def __init__(out self, in_dtype: DynType) raises:
        self._n = List[Float64]()
        self._mean = List[Float64]()
        self._m2 = List[Float64]()

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        """A dispersion is a real number whatever the input's width, and a
        rooted one is not even in the input's units squared.

        No domain check: `V: NumericType` is the domain, and every caller binds
        `V` *from* `in_dtype` through `dispatch_numeric`, so the two cannot
        disagree. The check that used to be here restated the bound at run
        time and could only fire if a caller had already resolved wrong."""
        return DynType(float64)

    def reserve(mut self, slots: Int) raises:
        while len(self._n) < slots:
            self._n.append(0.0)
            self._mean.append(0.0)
            self._m2.append(0.0)

    @always_inline
    def _absorb(mut self, g: Int, x: Float64):
        """One Welford step against slot `g`."""
        self._n[g] += 1.0
        var delta = x - self._mean[g]
        self._mean[g] += delta / self._n[g]
        self._m2[g] += delta * (x - self._mean[g])

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        self.reserve(1 if groups.is_single() else groups.num_groups)
        ref values = input
        var has_null = values.null_count() > 0
        var vals = values.values()
        # Two loops rather than one with a per-row branch on an invariant:
        # `groups.is_single()` holds no ids at all, so the grouped loop would
        # not execute and would leave the state untouched.
        if groups.is_single():
            for i in range(len(values)):
                if has_null and not values.is_valid(i):
                    continue
                self._absorb(0, Float64(vals[i].cast[DType.float64]()))
        else:
            var gids = groups.ids.values()
            for i in range(len(groups.ids)):
                if has_null and not values.is_valid(i):
                    continue
                self._absorb(
                    Int(gids[i]), Float64(vals[i].cast[DType.float64]())
                )

    def finish(mut self) raises -> Self.OutArray:
        var out = Float64Builder(len(self._n))
        for g in range(len(self._n)):
            var divisor = self._n[g] - Float64(Self.ddof)
            if divisor <= 0.0:
                out.append_null()
            else:
                var v = self._m2[g] / divisor
                out.append(math.sqrt(v) if Self.root else v)
        return out.finish()


struct StringExtremum[Op: MinMaxOp, T: StringLikeType](AggKernel):
    """`min`/`max` over a string column — a bytewise (lexicographic) scan,
    matching Arrow's `hash_min`/`hash_max`.

    Not a fold: there is no scalar accumulator to combine. That is also why it
    is not mergeable — but it *is* streamable, which the previous design got
    wrong. It used to keep the index of the best row, and an index is only
    meaningful inside the morsel it came from, so the whole column had to be
    concatenated first. Keeping the best **value** instead is O(groups) and
    survives a morsel boundary.

    Nulls are excluded and an empty or all-null slot yields null.
    """

    comptime name = Self.Op.name
    comptime InArray = BinaryLikeArray[Self.T]
    comptime OutArray = BinaryLikeArray[Self.T]

    var _best: List[Optional[String]]
    """The winning value per slot, not its row index. An index is only
    meaningful inside the morsel it came from, which is what used to make this
    unstreamable."""

    def __init__(out self, in_dtype: DynType) raises:
        self._best = List[Optional[String]]()

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        """An extremum *is* one of the input's values, so it keeps its type.

        Answered from `T`, not from the argument: `StringLikeType` is
        `Defaultable`, so `T` already carries everything the output type needs.
        The temporal and decimal kernels cannot do this — a timestamp's unit
        and timezone live in the *value* — which is the whole reason the trait
        still takes an `in_dtype` at all."""
        return DynType(Self.T())

    def reserve(mut self, slots: Int) raises:
        while len(self._best) < slots:
            self._best.append(None)

    @always_inline
    def _offer(mut self, g: Int, var candidate: String):
        """Keep `candidate` if it beats slot `g`'s incumbent."""
        if not self._best[g]:
            self._best[g] = candidate^
            return
        var better = (
            candidate < self._best[g].value()
        ) if Self.Op.is_min else (self._best[g].value() < candidate)
        if better:
            self._best[g] = candidate^

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        """No dispatch: `T` is the input's type, so the scan is monomorphized.
        The `dispatch_stringlike` this ran once per morsel is gone."""
        self.reserve(1 if groups.is_single() else groups.num_groups)
        var has_null = input.null_count() > 0
        if groups.is_single():
            for i in range(len(input)):
                if has_null and not input.is_valid(i):
                    continue
                self._offer(0, String(input.unsafe_get(UInt(i))))
        else:
            var gids = groups.ids.values()
            for i in range(len(groups.ids)):
                if has_null and not input.is_valid(i):
                    continue
                self._offer(Int(gids[i]), String(input.unsafe_get(UInt(i))))

    def finish(mut self) raises -> Self.OutArray:
        var out = BinaryLikeBuilder[Self.T](capacity=len(self._best))
        for g in range(len(self._best)):
            if self._best[g]:
                out.append(self._best[g].value())
            else:
                out.append_null()
        return out.finish()


struct ValidCount[A: Array](AggKernel):
    """`COUNT(x)` — the *non-null* values of `x`, over a column of any type.

    A validity scan and nothing else, so it is defined for every dtype and
    there is nothing to monomorphize on: the input stays erased. An empty slot
    counts 0 and is never null, matching SQL.

    Its state is one `Int64` per slot, so streaming it costs nothing over the
    one-shot form it replaced.

    The comptime lane does **not** route numeric `count` here — it fuses
    `Aggregate[Fold[CountKernel], A]` when its operand is numeric, which pays
    one typed `bitmap.test()` per row where this pays a `DynArray._dispatch`
    walk: a linear `comptime for` over 37 variant arms plus an indirect call
    *per row*, inside an already cache-hostile random-write loop. Measured at
    1M rows / 100k groups on a nullable column: 1.7159 ms (sd 0.0702) against
    8.3555 ms (sd 0.2331). On a null-free column this skips validity entirely
    via its `has_null` guard and never loads a value, which is why it edges
    ahead there instead — 1.3710 ms against 1.4124 ms, a ~3% gap. Converging
    the two would mean paying the erased per-row cost on every numeric `count`,
    so the split stays. This is the path for the dtypes a fold cannot serve,
    and for a `count` named at run time.
    """

    comptime name = CountKernel.name
    comptime InArray = Self.A
    comptime OutArray = Int64Array

    var _counts: List[Int64]
    """One running count per slot. Grows with `groups.num_groups`; ids already
    handed out are never renumbered, so growing in place is sound."""

    def __init__(out self, in_dtype: DynType) raises:
        self._counts = List[Int64]()

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        return DynType(int64)

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        ref column = input
        if groups.is_single():
            # A valid count is metadata. Losing this branch would turn
            # `count(x)` with no GROUP BY from O(1) into O(n).
            self.reserve(1)
            self._counts[0] += Int64(len(column) - column.null_count())
            return
        self.reserve(groups.num_groups)
        var gids = groups.ids.values()
        # The bitmap once, not `is_valid(i)` per row — on an erased column that
        # per-row call was a `_dispatch` walk and measured 4.9x against a typed
        # fold. Typed now, so it is a direct call either way; reading the view
        # once is still the cheaper shape and needs nothing from the dtype.
        var data = column.to_data()
        var v = data.validity()
        if v:
            var bits = v.value()
            for i in range(len(groups.ids)):
                if bits[i]:
                    self._counts[Int(gids[i])] += 1
        else:
            for i in range(len(groups.ids)):
                self._counts[Int(gids[i])] += 1

    def reserve(mut self, slots: Int) raises:
        while len(self._counts) < slots:
            self._counts.append(0)

    def finish(mut self) raises -> Self.OutArray:
        var out = Int64Builder(len(self._counts))
        for g in range(len(self._counts)):
            out.append(Scalar[int64.native](self._counts[g]))
        return out.finish()


struct DistinctCount[exact: Bool, A: Array](AggKernel):
    """`COUNT(DISTINCT x)` exactly, or a HyperLogLog estimate of it.

    Not a fold — the per-slot state is a hash set or a sketch, not a scalar
    accumulator — which is why there is no `FoldKernel` for it, no fused form
    to fall back to, and no merge: two sketches of the same rows do not add.
    Nulls are excluded (SQL semantics, PyArrow's `only_valid`).

    **Streaming, and this is the aggregate that most needed it.** Both states
    are naturally incremental: the exact form dedups `(group, value)` pairs in
    one `SwissHashTable` that simply persists across morsels, and HyperLogLog
    is definitionally updatable. The previous contract still made a query
    containing a `count_distinct` hold its entire input column in memory,
    because `grouped` was one-shot. Exact is now O(distinct pairs) and approx
    is O(groups * 2**11) — neither is O(rows).

    One thing was given up. The one-slot exact path used to call
    `count_distinct`, which goes radix-partition-parallel above 200k rows;
    a streaming state sees one morsel at a time and cannot. Whole-column
    parallelism and bounded memory are not simultaneously available through
    this interface, and bounded memory is the one an execution engine needs.
    `count_distinct` itself is unchanged for callers that have the whole
    column.

    Both field sets are declared, and one is empty in each instantiation: a
    `comptime if` cannot select a *field*, and an unused `SwissHashTable` or
    `List[UInt8]` is a few words, not per-group.
    """

    comptime name = COUNT_DISTINCT if Self.exact else APPROX_COUNT_DISTINCT
    comptime InArray = Self.A
    comptime OutArray = Int64Array

    comptime _p = HLL_P_GROUPED
    comptime _m = 1 << Self._p

    var _table: SwissHashTable[RapidHash64]
    """Exact: one table over `(group, value)` pairs, kept between morsels."""
    var _seen: List[Bool]
    """Exact: whether a table slot has already been counted."""
    var _registers: List[UInt8]
    """Approx: `2**11` HLL registers per slot."""
    var _counts: List[Int64]
    """Exact: the running distinct count per slot."""
    var _slots: Int

    def __init__(out self, in_dtype: DynType) raises:
        self._table = SwissHashTable[RapidHash64]()
        self._seen = List[Bool]()
        self._registers = List[UInt8]()
        self._counts = List[Int64]()
        self._slots = 0

    @staticmethod
    def dtype(in_dtype: DynType) raises -> DynType:
        # A cardinality, whatever was counted.
        return DynType(int64)

    def reserve(mut self, slots: Int) raises:
        if slots <= self._slots:
            return
        comptime if Self.exact:
            while len(self._counts) < slots:
                self._counts.append(0)
        else:
            while len(self._registers) < slots * Self._m:
                self._registers.append(0)
        self._slots = slots

    def update(mut self, groups: Groups, input: Self.InArray) raises:
        ref value = input
        # The exact arm builds a `(group, value)` pair as a `StructArray`, so
        # it needs the value column erased — once per morsel, to construct the
        # struct, not per row. The approximate arm hashes `input` directly.
        var erased = input.copy().to_dyn()
        var single = groups.is_single()
        self.reserve(1 if single else groups.num_groups)
        var n = len(value) if single else len(groups.ids)
        var has_null = value.null_count() > 0

        comptime if Self.exact:
            # The pair `(group, value)` is what makes one table serve every
            # slot: a row is new when its *pair* is new. At one slot the group
            # is constant, so the value alone identifies the pair and hashing
            # it is enough.
            # `List[DynArray]` because that is `StructArray`'s child layout,
            # not because this kernel is erased: a struct holds columns of
            # differing types, so its children cannot be one typed list. The
            # narrowing back out never happens — the struct goes straight to
            # the hasher.
            var children = List[DynArray]()
            if not single:
                children.append(groups.ids.copy().to_dyn())
            children.append(erased.copy())
            var fields = List[Field]()
            if not single:
                fields.append(Field("g", int32))
            fields.append(Field("v", erased.dtype().copy()))
            var pairs = StructArray(
                dtype=struct_(fields^),
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                children=children^,
            )
            var bids = self._table.insert_hashes(
                RapidHashKernel.apply(pairs, ExecContext.serial()),
                grow_adaptively=True,
            )
            while len(self._seen) < self._table.num_keys():
                self._seen.append(False)
            for i in range(n):
                if has_null and not value.is_valid(i):
                    continue
                var b = Int(bids.unsafe_get(i))
                if not self._seen[b]:
                    self._seen[b] = True
                    var g = 0 if single else Int(groups.ids.unsafe_get(i))
                    self._counts[g] += 1
        else:
            # `dispatch` and not `apply`, and the reason is a real limit
            # rather than leftover erasure: `apply` is overloaded per array
            # *family* and there is no `apply[A: Array]`, so a generic `A`
            # selects none of them. The narrowing happens once per morsel, not
            # per row. A generic overload in `hashing.mojo` would remove it.
            var hv = RapidHashKernel.dispatch(
                erased, ExecContext.serial()
            ).values()
            for i in range(n):
                if has_null and not value.is_valid(i):
                    continue
                var h = UInt64(hv[i])
                var g = 0 if single else Int(groups.ids.unsafe_get(i))
                var idx = g * Self._m + Int(h >> (64 - Self._p))
                var rho = hll_rho[Self._p](h)
                if rho > self._registers[idx]:
                    self._registers[idx] = rho

    def finish(mut self) raises -> Self.OutArray:
        var out = Int64Builder(self._slots)
        comptime if Self.exact:
            for g in range(self._slots):
                out.append(Scalar[int64.native](self._counts[g]))
        else:
            for g in range(self._slots):
                out.append(
                    Scalar[int64.native](
                        hll_estimate[Self._p](self._registers, g * Self._m)
                    )
                )
        return out.finish()


# ---------------------------------------------------------------------------
# Runtime dtype -> array type
# ---------------------------------------------------------------------------


def dispatch_agg_array[
    R: Movable, //, Func: def[A: Array]() raises -> R
](in_dtype: DynType, func: Func) raises -> R:
    """Resolve a runtime dtype to the **array type** that holds it, and run
    `func` at that type.

    The counterpart, for aggregates, of the `dispatch` static every value
    kernel exposes (`RapidHashKernel.dispatch`, `kernels/hashing.mojo`). Those
    narrow an erased *array* and run immediately, because they are stateless.
    An aggregate is a state machine that must exist before its first morsel, so
    this narrows a *dtype* instead and hands back whatever the caller builds at
    that type — an accumulator, or an operator holding one.

    It is what lets `ValidCount[A]` and `DistinctCount[exact, A]` be typed at
    all. Both are dtype-generic in what they *compute* — a cardinality is an
    int64 whatever was counted — but not in how they *read*: a valid count
    wants a validity bitmap and a distinct count wants to hash values, and both
    are faster knowing the array type than walking an erased one per row.
    Without this they had to declare `InArray = DynArray`, which made
    `AggKernel.InArray` mean "some column-ish thing" and put an unchecked
    reinterpret at every call site.

    Three arms rather than one ladder over every layout: each existing family
    dispatcher already knows its own array, and `dispatch_primitive` spans
    numeric, temporal, interval and decimal. A layout outside them raises here,
    at plan time, rather than at the first morsel.
    """
    if in_dtype.is_bool():
        return func[BoolArray]()
    elif in_dtype.is_string() or in_dtype.is_large_string():

        def stringly[T: StringLikeType](d: T) raises {imm func} -> R:
            return func[BinaryLikeArray[T]]()

        return in_dtype.dispatch_stringlike(stringly)
    elif in_dtype.is_primitive():

        def primitive[T: PrimitiveType](d: T) raises {imm func} -> R:
            return func[PrimitiveArray[T]]()

        return in_dtype.dispatch_primitive(primitive)
    else:
        raise Error("no array type for ", in_dtype, " columns")
