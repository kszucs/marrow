# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the operator binds the subtree itself and hands `lane[W]` to
the fold, which reads it straight into a register. Ungrouped that measured
14.6x over materialise-then-scatter; grouped it is at parity (2026-09-25).

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**The three executors live here too.** `ScatteredAggregateOperator`,
`RegisterAggregateOperator` and `BufferedAggregateOperator` are what the nodes
below become, and they sit beside them for the same reason `NumericColumn` sits beside
its own `bind`/`lane`: this lane is organised by **family**, not by
logical-versus-physical. That split is real one level up (`logical.mojo` /
`physical.mojo`); inside a lane the distinction that matters is lane-agnostic
versus lane-specific. `EvalOperator` is generic over any `Evaluable` and
belongs in `physical.mojo`; the two fused operators are parameterised on
`A: PrimitiveValue` and cannot go there without making the lane-agnostic layer
name a lane.

The fold owns its loops — `Foldable.reduce` and `Foldable.scatter`, in
`kernels` — and the fused operators here only bind the operand and hand those
loops its lanes, so the column path and the fused path fold alike.
`AggState[K, V]` is the accumulator, also in `kernels`, and it is the only thing
here that crosses a batch boundary. The ungrouped path folds into registers,
but those are *per-batch scratch* — the same status `Bound` has — and
`Fold.reduce` hands them to slot 0 once per morsel, through
`AggState.combine_reserved`. So `K.finalize`, `K.empty_is_null` and the
count-is-zero rule stay defined in exactly one place.

`FILTER (WHERE ...)` is a third type parameter defaulting to `Nothing`, so
an unfiltered aggregate compiles exactly as it did without one. A fused
fold applies the predicate to its operand's validity before the lane loop.
"""

from ...dtypes import DynType, NumericType

from ...arrays import BinaryLikeArray, PrimitiveArray, Int32Array, StructArray
from ...kernels.aggregate import (
    FoldKernel,
    AggKernel,
    Dispersion,
    ApproxDistinctCount,
    DistinctCount,
    Fold,
    Foldable,
    FoldPartial,
    Windowable,
    CountFold,
    DecimalMeanFold,
    DecimalSumFold,
    MaxFold,
    MeanFold,
    MinFold,
    ProductFold,
    LexicalExtremum,
    SumFold,
)
from ...buffers import Bitmap
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import (
    Nothing,
    References,
    Shape,
    Value,
    WindowFunction,
    is_filled,
    reject_non_boolean_filter,
)
from ..logical import Bindings
from ...execution import ExecContext
from ...kernels.filter import FilterKernel
from ...kernels.groupby import Groups
from ...kernels.window import WindowExtents, WindowFrame
from ...arrays import DynArray
from ..physical import (
    BufferedAggregateOperator,
    ColumnEvaluator,
    CombineAggregateOperator,
    FoldMode,
    emit_fold,
    FrameEvaluator,
    PerFrameEvaluator,
    Evaluable,
    EvalOperator,
)
from ...engine.core import Datum, DynOperator, Morsel, Operator

from .core import ComptimeValue, NumericValue, PrimitiveValue, StringValue


struct Aggregate[
    Agg: AggKernel,
    A: Evaluable & Value,
    P: Evaluable & Value = Nothing,
](Value, WindowFunction):
    """One aggregate: its operand, its `FILTER` predicate, and both ways of
    running.

    The predicate is the type parameter `P`, defaulting to `Nothing`, so an
    unfiltered aggregate is the same instantiation as before `P` existed and
    compiles no filtering code.

    **Whether this fuses is computed, not declared.** `to_operator` asks two
    independent comptime questions, and there is no runtime test:

    - does `Agg` have a lane algebra — `conforms_to(Agg, Foldable)`? A hash
      set and a bytewise scan do not.
    - is the operand lane-readable — `conforms_to(A, PrimitiveValue)`? A lane
      is `SIMD[Type.native, W]`, so it needs a fixed-width element — which is
      all it needs, temporal columns included.

    Both yes, and the fold is fused into the operand's loop: `sum(a * 2 + b)`
    never builds `a * 2 + b` — at parity grouped, **14.6x** ungrouped.
    Otherwise the operand is evaluated to a column and the aggregate runs over
    it.

    That is why this is one struct rather than two. `col("v", int64).min()` and
    `col("ts", timestamp(us)).min()` are the *same aggregate* — both spell
    `Aggregate[Fold[MinFold], _]` — and differ only in whether their operand
    can be read as a lane. The previous `FusedAggregate` / `BufferedAggregate`
    pair made that one difference into two node types, so every fluent method
    had to restate the split by hand and get it right.

    **The operand is never erased.** Only the aggregation step can
    materialise; `upper(region)` stays a fused string subtree that compiles to
    one loop even though `min` over it cannot fuse. Boxing the operand into a
    `DynValue` here would throw away the operand's fusion along with the
    aggregate's, which is a loss nothing forces. So `A` stays a type parameter
    and the operand builds its own `EvalOperator[A]`.

    **`A: Evaluable & Value` — the looser bound — is load-bearing.** It names
    concrete traits rather than a projection off `Agg`, and that is what makes
    `_input` storable: a trait-valued associated type (`A: Agg.Operand`) does
    reduce and its members do resolve, but it cannot type a *field*. See
    CLAUDE.md's associated-types section. The tight bound is recovered inside
    `to_operator`, where `comptime if conforms_to` is enough to hand `Self.A`
    to an operator that requires `PrimitiveValue` — verified 2026-08-29: Mojo
    accepts binding a parameter at a *tighter* trait bound at a struct
    *instantiation* site, not only at a method call.

    **An ordinary `Value`.** It conforms to exactly what `x + 1` conforms to
    and is boxed by the same `DynValue`. There is no `AggValue` trait: once
    every logical node answers `to_operator`, an aggregate stopped being a
    different *kind* of node and became one that answers from
    `Operator.drain` rather than from `Operator.push`. That difference is
    behavioural, not structural, and is deliberately not encoded as a marker
    trait — a trait constraining nothing documents nothing. When a planner
    needs to *check* it (to reject `project([col("a").sum()])` by inspection
    rather than by raising), the mechanism is a comptime `kind` on
    `Analyzable`, not a parallel hierarchy.
    """

    comptime aggregates = True
    """It answers from its operator's `drain`, never per batch. `Project`
    and `Filter` read this and raise, which is what makes
    `project([col("a").sum()])` a plan-time error naming the mistake."""

    comptime filters = is_filled[Self.P]
    """Whether this aggregate carries a `FILTER` predicate."""

    comptime fuses = conforms_to(Self.Agg, Foldable) and conforms_to(
        Self.A, PrimitiveValue
    )
    """Whether this instantiation folds lanes or evaluates a column.

    **Stated once, here, and nowhere else.** It used to be written twice —
    identically — once on this node and once on the operator, which is a
    two-place invariant with nothing enforcing it: had they drifted, this node
    would have picked the fused operator while that operator took its buffered
    arm, or the reverse. The operators are now two structs, and each one *is*
    the answer rather than recomputing it.

    **`PrimitiveValue`, not `NumericValue`.** The fused arm calls exactly
    `bind`, `validity` and `lane[W]` on its operand, and all three are
    `PrimitiveValue` members — so the narrower bound excluded temporal
    `min`/`max` from fusion for no reason the fused loop can see.
    `TemporalValue.min()` and `.max()` already build a `Fold[MinFold, V]`
    over a temporal `V`; the accumulator dtype comes from the `in_dtype:
    DynType` the fused operator is constructed with, not from `Self.A.Type()`,
    so the "a `TemporalType` is not `Defaultable`" obstacle that once justified
    the narrower bound is gone.
    """

    var _input: Self.A
    var _where: Self.P
    """The `FILTER` predicate, or `Nothing`. A typed field rather than an
    `Optional`, so the reflected `references` walk declares the columns the
    predicate reads and `ColumnPruning` keeps them."""

    var _alias: String
    """What `Value.name()` answers. The aggregate itself is `Agg`, a comptime
    parameter, so `alias` cannot possibly change which kernel runs — the
    two-field split `RuntimeAggregate` needs is structural here."""

    def __init__(out self, var input: Self.A, var predicate: Self.P):
        self._input = input^
        self._where = predicate^
        self._alias = String(Self.Agg.name)

    def __init__(
        out self, var input: Self.A, var predicate: Self.P, var name: String
    ):
        self._input = input^
        self._where = predicate^
        self._alias = name^

    # -- Value --------------------------------------------------------------

    def name(self) -> String:
        return self._alias.copy()

    def references(self, mut into: References):
        """The operand's reads, then the predicate's. Written out because
        `WindowFunction` requires it, which leaves `Value`'s derived walk
        unreachable here."""
        self._input.references(into)
        self._where.references(into)

    def dtype(self, schema: Schema) raises -> DynType:
        """Through `Agg.dtype`, from the *operand's* dtype.

        Not from `A.Type()`, even where this fuses: a temporal dtype is not
        constructible from its type — a timestamp carries a unit and a
        timezone — which is exactly why `TemporalColumn.dtype` reads the
        schema. `Agg.dtype` applies the same widening rule either way, so
        `sum(int32)` still answers int64.
        """
        return Self.Agg.dtype(self._input.dtype(schema))

    comptime shape = Shape.scalar
    """An aggregate yields one value per group, so it is scalar-shaped in the
    same sense a literal is: it does not produce a value per input row."""

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self,
        schema: Schema,
        grouped: Bool,
        bindings: Bindings = Bindings(),
        mode: FoldMode = FoldMode.final(),
    ) raises -> DynOperator:
        """Pick the machine and the placement, once, when the plan is built.

        Both decisions resolve to comptime *types*: whether the query has keys
        is a property of the plan, and whether this aggregate can fuse is a
        property of `Agg` and `A`. Resolving them here is what keeps them out
        of the inner loop.

        All three operators keep the operand as a parameter, so nothing this
        lane folds over is erased: the fused pair because the lanes they hand
        the fold read it directly, and `BufferedAggregateOperator` so its
        operand is evaluated by a direct call. The predicate is a parameter on
        the fused pair and a lowered `DynOperator` on the buffered one. All
        three are boxed once, here, so the return type does not depend on the
        branch.

        **Placement is a comptime choice for a fused fold and a runtime one
        for a buffered fold**, and that asymmetry is the measurement rather
        than an oversight. `ScatteredAggregateOperator` and
        `RegisterAggregateOperator` are different programs — a random write per
        row against a register accumulated across the whole morsel — and the
        gap between them is the 14.6x. `BufferedAggregateOperator` reads its
        placement once per morsel to build a `Groups`, never per row, so
        instantiating it twice would double the code for nothing — measured
        at +4.6%.

        A `FILTER` predicate is checked here, at plan time: it must be
        boolean and must not itself aggregate.
        """
        comptime if Self.filters:
            comptime assert not Self.P.aggregates, (
                "filter: an aggregate's FILTER predicate has no value per row;"
                " filter on the rows instead"
            )
            reject_non_boolean_filter(self._where.dtype(schema))
        if mode.is_combine():
            return CombineAggregateOperator[Self.Agg](
                mode.column, grouped, self._input.dtype(schema)
            )
        comptime if Self.fuses:
            if grouped:
                return ScatteredAggregateOperator[Self.Agg, Self.A, Self.P](
                    self._input.copy(),
                    self._where.copy(),
                    bindings.copy(),
                    self._input.dtype(schema),
                    mode.is_partial(),
                )
            return RegisterAggregateOperator[Self.Agg, Self.A, Self.P](
                self._input.copy(),
                self._where.copy(),
                bindings.copy(),
                self._input.dtype(schema),
                mode.is_partial(),
            )
        else:
            var predicate: Optional[DynOperator] = None
            comptime if Self.filters:
                predicate = Optional(
                    self._where.to_operator(schema, False, bindings.copy())
                )
            return BufferedAggregateOperator[Self.Agg, Self.A](
                self._input.copy(),
                predicate^,
                bindings.copy(),
                grouped,
                self._input.dtype(schema),
                mode.is_partial(),
            )

    comptime Evaluator = AggregateEvaluator[Self.Agg, Self.A, Self.P]

    def to_evaluator(
        self, schema: Schema, bindings: Bindings, frame: WindowFrame
    ) raises -> Self.Evaluator:
        """The window counterpart of `to_operator`. The `FILTER` predicate is
        checked here, at plan time, as `to_operator` checks it."""
        comptime if Self.filters:
            reject_non_boolean_filter(self._where.dtype(schema))
        return AggregateEvaluator[Self.Agg, Self.A, Self.P](
            self._input.copy(),
            self._where.copy(),
            schema.copy(),
            bindings.copy(),
            frame,
        )

    def alias(self, var name: String) -> Self:
        """Rename this aggregate. `col("x", int64).sum().alias("total")`.

        Returns a copy rather than mutating, so an aggregate stays a pure
        description and the same subtree can be named twice. The predicate
        rides along — `.filter(p).alias("n")` and `.alias("n").filter(p)` are
        the same node.
        """
        return Self(self._input.copy(), self._where.copy(), name^)

    def filter[
        Pred: Evaluable & Value
    ](self, var predicate: Pred) -> Aggregate[Self.Agg, Self.A, Pred]:
        """`FILTER (WHERE predicate)`: this aggregate sees only the rows the
        predicate answers TRUE for.

        `col("v", int64).sum().filter(col("v", int64) > lit(2, int64))`. Unlike
        `WHERE` it removes no rows from the query, so a group whose rows are
        all rejected still appears, with its kernel's empty answer (NULL for
        `sum`, 0 for `count`).
        """
        return Aggregate[Self.Agg, Self.A, Pred](
            self._input.copy(), predicate^, self._alias.copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.Agg.name, "(", self._input, ")")
        comptime if Self.filters:
            writer.write(" filter (", self._where, ")")


# ---------------------------------------------------------------------------
# The vocabulary — one alias per aggregate, partially applied on its operand.
#
# `Sum[Self]` rather than `Aggregate[Fold[SumFold], Self]` at every fluent
# method. The long form names three things to say one, and it puts the *kernel*
# vocabulary in `core.mojo`'s imports, where nothing else needs it: with these,
# `core.mojo` names no `AggKernel` at all.
#
# `Min` is the fixed-width extremum and `StringMin` the bytewise one, because
# they are genuinely different kernels — `NumericValue.min()` and
# `StringValue.min()` are two methods for a reason, and one alias covering both
# would have to hide a `conforms_to` that the fluent surface already answers by
# overload.
#
# Every fold alias names its operand, because `Fold[K, V]` is typed on its
# input and `V` is exactly `A.Type` — the operand already knows it. That is
# what removes the dtype dispatch from the comptime lane's folds entirely:
# `col("ts", timestamp(us)).min()` is `Fold[MinFold, TimestampType]`, not a
# runtime lookup. `StringMin`, `CountDistinct` and friends stay partially
# applied, since their kernels are genuinely dtype-generic.
# ---------------------------------------------------------------------------
comptime Sum[A: PrimitiveValue] = Aggregate[Fold[SumFold, A.Type], A]
comptime Product[A: PrimitiveValue] = Aggregate[Fold[ProductFold, A.Type], A]
comptime Min[A: PrimitiveValue] = Aggregate[Fold[MinFold, A.Type], A]
comptime Max[A: PrimitiveValue] = Aggregate[Fold[MaxFold, A.Type], A]
comptime Mean[A: PrimitiveValue] = Aggregate[Fold[MeanFold, A.Type], A]
comptime DecimalSum[A: PrimitiveValue] = Aggregate[
    Fold[DecimalSumFold, A.Type], A
]
comptime DecimalMean[A: PrimitiveValue] = Aggregate[
    Fold[DecimalMeanFold, A.Type], A
]
comptime Count[A: PrimitiveValue] = Aggregate[Fold[CountFold, A.Type], A]
comptime Variance[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, False, A.Type], A
]
comptime StdDev[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, True, A.Type], A
]
comptime StringMin[A: StringValue] = Aggregate[
    LexicalExtremum[True, BinaryLikeArray[A.Type]], A
]
comptime StringMax[A: StringValue] = Aggregate[
    LexicalExtremum[False, BinaryLikeArray[A.Type]], A
]
# The three cardinalities are dtype-generic in what they *compute* — an int64
# whatever was counted — but each still names the array it reads, because a
# validity scan and a hash are both faster typed. In this lane the operand
# knows it: a `PrimitiveValue` evaluates to a `PrimitiveArray[A.Type]` and a
# `StringValue` to a `BinaryLikeArray[A.Type]`, so each family's fluent method
# supplies the array and no `ArrayType` companion is needed on the node.
comptime CountDistinct[A: PrimitiveValue] = Aggregate[
    DistinctCount[PrimitiveArray[A.Type]], A
]
comptime ApproxCountDistinct[A: PrimitiveValue] = Aggregate[
    ApproxDistinctCount[PrimitiveArray[A.Type]], A
]
comptime StringCountDistinct[A: StringValue] = Aggregate[
    DistinctCount[BinaryLikeArray[A.Type]], A
]
comptime StringApproxCountDistinct[A: StringValue] = Aggregate[
    ApproxDistinctCount[BinaryLikeArray[A.Type]], A
]


# ---------------------------------------------------------------------------
# The three executors — one per way of feeding an aggregate's state.
#
# `ScatteredAggregateOperator` folds lanes into the slot each row's group id
# names. `RegisterAggregateOperator` folds a whole morsel in registers and
# hands the result to slot 0 once. `BufferedAggregateOperator` evaluates the
# operand to a column and gives the column to the kernel. That is the whole
# taxonomy, and it is named for *how the state is fed* rather than for what the
# aggregate is called.
#
# **The three stay separate, and the constraint is a language one.** A struct
# body admits no `comptime if`, so a merged operator would carry every arm's
# state in every instantiation.
# ---------------------------------------------------------------------------


@no_inline
def _admitted_validity[
    A: PrimitiveValue, P: Evaluable
](
    input: A,
    bound: A.Bound,
    predicate: P,
    batch: StructArray,
    bindings: Bindings,
) raises -> Optional[Bitmap[mut=False]]:
    """The operand's validity intersected with the rows `predicate` admits.

    What a `FILTER` costs a fused fold: one bitmap AND per morsel, before the
    lane loop, which then takes the arm it already has for a nullable column.
    A fold cannot compact instead, because row `i` is folded into the group
    `ids[i]` names. Not inlined, so the two operators that call it share one
    copy per operand and predicate type rather than each inlining the
    predicate's evaluation. Both bitmaps are offset-0, so the offset-unaware
    `Bitmap.intersect` applies.
    """
    var mask = predicate.evaluate(batch, bindings).to_array(len(batch))
    var admitted = FilterKernel.selection(mask.as_bool())
    return Bitmap.intersect(input.validity(bound), Optional(admitted^))


# A function rather than an operator method so `state` and `input` arrive as
# two arguments: a closure borrowing `self._input` cannot be passed while
# `self._state` is borrowed mutably.
@always_inline
def _bind_and_fold[
    Agg: Foldable, A: PrimitiveValue, P: Evaluable, //, scatters: Bool
](
    mut state: Agg, input: A, predicate: P, morsel: Morsel, bindings: Bindings
) raises:
    """Bind `input` to the morsel and hand its lanes to `state`: to `scatter`,
    into the morsel's groups, when `scatters`, else to `reduce`, into the one
    slot — masked by the operand's validity and the `FILTER`."""
    ref batch = morsel.batch
    var bound = input.bind(batch, bindings)
    var v: Optional[Bitmap[mut=False]]
    comptime if is_filled[P]:
        v = _admitted_validity(input, bound, predicate, batch, bindings)
    else:
        v = input.validity(bound)

    @always_inline
    def lanes[W: Int](i: Int) {imm input, imm bound} -> SIMD[Agg.Acc, W]:
        return input.lane[W](bound, i).cast[Agg.Acc]()

    comptime if scatters:
        debug_assert(
            len(morsel.groups.ids) == len(batch), "scatter: one id per row"
        )
    if v:
        ref vb = v.value()
        comptime if scatters:
            state.scatter(morsel.groups, lanes, vb.view())
        else:
            state.reduce(len(batch), lanes, vb.view())
    else:
        comptime if scatters:
            state.scatter(morsel.groups, lanes)
        else:
            state.reduce(len(batch), lanes)


struct ScatteredAggregateOperator[
    Agg: Foldable, A: PrimitiveValue, P: Evaluable = Nothing
](Operator):
    """A fused fold that scatters into the slot each row's group id names —
    `GROUP BY` with a lane-readable operand and a lane algebra.

    `sum(a * 2 + b)` never builds `a * 2 + b`: this binds the subtree and hands
    `lane[W]` to `Foldable.scatter`, whose loop reads it straight into a
    register before scattering. The loop is the kernel's — `Fold.update` runs
    the same one over a column's values — so this operator only binds.

    The scatter stays scalar per lane even at `W > 1`, and that is not an
    oversight — two lanes may carry the same group, and a vector
    read-modify-write would lose one of them without conflict detection. `W`
    still pays because the loads and the arithmetic feeding it vectorise.

    **Two operator structs over one helper, `_bind_and_fold`.** Both bind the
    operand alike and differ in where its lanes go: this one to
    `Foldable.scatter`, a random write per row, `RegisterAggregateOperator` to
    `Foldable.reduce`, which accumulates in registers and hands the morsel to
    slot 0 once. They stay two structs because a struct body admits no
    `comptime if`: one operator would carry both arms' state, the slots this
    one grows to every morsel's groups and the slot 0 the register one holds
    from the start.

    **The bounds are the honest ones.** `Agg: Foldable` because this calls
    `scatter`; `A: PrimitiveValue` because it calls `bind`, `validity` and
    `lane[W]`. `Aggregate.to_operator` recovers both inside
    `comptime if Self.fuses`, where `conforms_to` has already established them.
    """

    var _input: Self.A
    var _where: Self.P
    """The `FILTER (WHERE ...)` predicate, or the empty slot."""

    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype. Not an
    `Optional`: `to_operator` takes the input schema, so the dtype is in hand
    before the first morsel, which is what let `AggKernel.open` and the
    `_opened` latch that guarded it disappear."""

    var _partial: Bool
    """Whether `drain` answers the state rather than the aggregate."""
    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Self.P,
        var bindings: Bindings,
        in_dtype: DynType,
        partial: Bool,
    ) raises:
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype, True)
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Bind the operand and hand its lanes to `Foldable.scatter`, masked
        by its validity and the `FILTER`. A morsel with no rows still grows the
        state to its groups: `scatter` reserves before it loops."""
        _bind_and_fold[scatters=True](
            self._state, self._input, self._where, morsel, self._bindings
        )
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return emit_fold(self._state, self._partial)


struct AggregateEvaluator[
    Agg: AggKernel, A: Evaluable & Value, P: Evaluable & Value
](FrameEvaluator):
    """An `Aggregate` over every frame of a window, the machine chosen the way
    `Aggregate.to_operator` chooses one: a fused aggregate folds its frames
    straight from the operand's lanes (`FoldEvaluator`), any other
    `Windowable` one evaluates its operand to a column first
    (`ColumnEvaluator`), and the rest run once per distinct frame
    (`PerFrameEvaluator`).

    One type for the three, because `Aggregate.Evaluator` must name a single
    type whatever the branch; the branch is taken in `run`, at compile time.
    """

    var _input: Self.A
    var _where: Self.P
    var _schema: Schema
    var _bindings: Bindings
    var _frame: WindowFrame

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Self.P,
        var schema: Schema,
        var bindings: Bindings,
        frame: WindowFrame,
    ):
        self._input = input^
        self._where = predicate^
        self._schema = schema^
        self._bindings = bindings^
        self._frame = frame

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        comptime if Aggregate[Self.Agg, Self.A, Self.P].fuses:
            var fold = FoldEvaluator[Self.Agg, Self.A, Self.P](
                self._input.copy(),
                self._where.copy(),
                self._bindings.copy(),
                self._input.dtype(self._schema),
                self._frame,
            )
            return fold.run(sorted, extents, ctx)
        elif conforms_to(Self.Agg, Windowable):
            var predicate: Optional[DynOperator] = None
            comptime if is_filled[Self.P]:
                predicate = Optional(
                    self._where.to_operator(
                        self._schema, False, self._bindings.copy()
                    )
                )
            var column = ColumnEvaluator[Self.Agg, Self.A](
                self._input.copy(),
                predicate^,
                self._bindings.copy(),
                self._frame,
            )
            return column.run(sorted, extents, ctx)
        else:
            var per_frame = PerFrameEvaluator(
                Aggregate[Self.Agg, Self.A, Self.P](
                    self._input.copy(), self._where.copy()
                ),
                self._schema.copy(),
                self._bindings.copy(),
                self._frame,
            )
            return per_frame.run(sorted, extents, ctx)


struct FoldEvaluator[Agg: Foldable, A: PrimitiveValue, P: Evaluable = Nothing](
    FrameEvaluator
):
    """A fused fold over every window frame of one batch: the frame leaves
    come straight from the operand's lanes — `bind` once, `lane[1]` per row,
    `FILTER` as a validity mask — with no intermediate column, and
    `Foldable.fold_frames` answers each frame.

    The window counterpart of `RegisterAggregateOperator`.
    """

    comptime filters = is_filled[Self.P]

    var _input: Self.A
    var _where: Self.P
    var _bindings: Bindings
    var _in_dtype: DynType
    var _frame: WindowFrame

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Self.P,
        var bindings: Bindings,
        var in_dtype: DynType,
        frame: WindowFrame,
    ):
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._in_dtype = in_dtype^
        self._frame = frame

    def run(
        mut self, sorted: StructArray, extents: WindowExtents, ctx: ExecContext
    ) raises -> DynArray:
        comptime Leaf = FoldPartial[Self.Agg.Lane, Self.Agg.Acc]
        var n = len(sorted)
        var bound = self._input.bind(sorted, self._bindings)
        var leaves = List[Leaf](capacity=n)
        for i in range(n):
            leaves.append(
                Leaf(self._input.lane[1](bound, i).cast[Self.Agg.Acc](), 1)
            )
        var valid: Optional[Bitmap[mut=False]]
        comptime if Self.filters:
            valid = _admitted_validity(
                self._input, bound, self._where, sorted, self._bindings
            )
        else:
            valid = self._input.validity(bound)
        if valid:
            var bits = valid.value().view()
            for i in range(n):
                if not bits[i]:
                    leaves[i] = Leaf.identity()
        var agg = Self.Agg(self._in_dtype, True)
        agg.fold_frames(leaves, extents, self._frame)
        return agg.finish().to_dyn()


struct RegisterAggregateOperator[
    Agg: Foldable, A: PrimitiveValue, P: Evaluable = Nothing
](Operator):
    """A fused fold with **one** slot, accumulated in registers — no `GROUP
    BY`, a lane-readable operand and a lane algebra.

    The fastest shape in the tree: a scatter at one group is a million
    serially dependent read-modify-writes through a builder slot, where
    `Foldable.reduce` keeps the accumulator in SIMD registers for the whole
    morsel and hands it to slot 0 once, through `AggState.combine_reserved`.
    The loop is the kernel's; this operator binds the operand and hands over
    its lanes.

    It never touches `morsel.groups` — every row is group 0 by construction, so
    there are no ids to load and none to materialise. That is the same saving
    `Groups.single` makes one level down, and it is why placement is a comptime
    distinction here rather than a field: the two loops are different programs,
    not one program with a branch.

    The register accumulator is **per-batch scratch**, the same status `Bound`
    has. `AggState` stays the only thing crossing a morsel boundary, so
    `K.finalize`, `K.empty_is_null` and the count-is-zero rule stay defined in
    exactly one place.
    """

    var _input: Self.A
    var _where: Self.P
    """The `FILTER (WHERE ...)` predicate, or the empty slot."""

    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype."""

    var _partial: Bool
    """Whether `drain` answers the state rather than the aggregate."""
    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var predicate: Self.P,
        var bindings: Bindings,
        in_dtype: DynType,
        partial: Bool,
    ) raises:
        self._input = input^
        self._where = predicate^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype, False)
        self._partial = partial
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Bind the operand and hand its lanes to `Foldable.reduce`, masked by
        its validity and the `FILTER`: the morsel is folded in registers and
        handed to slot 0 once."""
        _bind_and_fold[scatters=False](
            self._state, self._input, self._where, morsel, self._bindings
        )
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        return emit_fold(self._state, self._partial)
