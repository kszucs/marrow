"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the state binds the subtree itself and reads `lane[W]`
straight into a register. Measured 2026-08-22: 1.17-1.68x over
materialise-then-scatter when grouped, and 14.6x when not.

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**The fold lives here too.** `FusedAggregateOperator` is the executor for the nodes
below it, and it sits beside them for the same reason `Column` sits beside its
own `bind`/`lane`: this lane is organised by **family**, not by
logical-versus-physical. That split is real one level up (`logical.mojo` /
`physical.mojo`); inside a lane the distinction that matters is
lane-agnostic versus lane-specific. `EvalOperator` is generic over any
`Evaluable` and belongs in `physical.mojo`; `FusedAggregateOperator` is parameterised on
`A: NumericValue` and cannot go there without making the lane-agnostic layer
name a lane.

`AggState[K, V]` is the accumulator, it lives
in `kernels`, and it is the only thing here that crosses a batch boundary. The
ungrouped path folds into registers, but those are *per-batch scratch* — the
same status `Bound` has — and hand off through `combine_at` once per morsel. So
`K.finalize`, `K.empty_is_null` and the count-is-zero rule stay defined in
exactly one place.
"""

from ...dtypes import DynType, NumericType
from ...kernels.groupby import Grouping, HashGrouping, ScalarGrouping
from std.sys.info import simd_width_of

from ...arrays import Int32Array, StructArray
from ...kernels.aggregate import (
    FoldKernel,
    AggState,
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    Foldable,
    CountKernel,
    MaxKernel,
    MaxOp,
    MeanKernel,
    MinKernel,
    MinOp,
    ProductKernel,
    StringExtremum,
    SumKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, Value
from ..params import Bindings
from ...execution import ExecContext
from ...kernels.concat import concat
from ...kernels.core import Groups
from ...arrays import DynArray
from ..physical import (
    Datum,
    EvalOperator,
    DynOperator,
    Morsel,
    Operator,
)

from .core import ComptimeValue, NumericValue, PrimitiveValue


struct Aggregate[Agg: AggKernel, A: ComptimeValue](Value):
    """One aggregate over one operand — every aggregate, both ways of running.

    **Whether this fuses is computed, not declared.** `to_operator` asks two
    independent comptime questions, and there is no runtime test:

    - does `Agg` have a lane algebra — `conforms_to(Agg, Foldable)`? A hash
      set and a bytewise scan do not.
    - is the operand lane-readable — `conforms_to(A, NumericValue)`? A lane is
      `SIMD[Type.native, W]`, so it needs a fixed-width element.

    Both yes, and the fold is fused into the operand's loop: `sum(a * 2 + b)`
    never builds `a * 2 + b`, worth 1.17-1.68x grouped and **14.6x** ungrouped.
    Otherwise the operand is evaluated to a column and the aggregate runs over
    it.

    That is why this is one struct rather than two. `col("v", int64).min()` and
    `col("ts", timestamp(us)).min()` are the *same aggregate* — both spell
    `Aggregate[Fold[MinKernel], _]` — and differ only in whether their operand
    can be read as a lane. The previous `FusedAggregate` / `BufferedAggregate`
    pair made that one difference into two node types, so every fluent method
    had to restate the split by hand and get it right.

    **The operand is never erased.** Only the aggregation step can
    materialise; `upper(region)` stays a fused string subtree that compiles to
    one loop even though `min` over it cannot fuse. Boxing the operand into a
    `DynValue` here would throw away the operand's fusion along with the
    aggregate's, which is a loss nothing forces. So `A` stays a type parameter
    and the operand builds its own `EvalOperator[A]`.

    **`A: ComptimeValue` — the looser bound — is load-bearing.** It is a
    concrete trait rather than a projection off `Agg`, and that is what makes
    `_input` storable: a trait-valued associated type (`A: Agg.Operand`) does
    reduce and its members do resolve, but it cannot type a *field*. See
    CLAUDE.md's associated-types section. The tight bound is recovered inside
    `to_operator`, where `comptime if conforms_to` is enough to hand `Self.A`
    to an operator that requires `NumericValue`.

    **An ordinary `Value`.** It conforms to exactly what `x + 1` conforms to
    and is boxed by the same `DynValue`. There is no `AggValue` trait: once
    every logical node answers `to_operator`, an aggregate stopped being a
    different *kind* of node and became one that answers from
    `Operator.finish` rather than from `Operator.push`. That difference is
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

    comptime fuses = conforms_to(Self.Agg, Foldable) and conforms_to(
        Self.A, NumericValue
    )
    """Whether this instantiation folds lanes or evaluates a column.

    Stated once and read by `to_operator`. It is also the honest answer to "is
    this aggregate cheap?": a fused fold keeps one scalar per group, while the
    other path accumulates every morsel's column and ids until `drain` and is
    therefore **O(rows)** in the aggregate's input.
    """

    var _input: Self.A
    var _alias: String
    """What `Value.name()` answers. The aggregate itself is `Agg`, a comptime
    parameter, so `alias` cannot possibly change which kernel runs — the
    two-field split `RuntimeAggregate` needs is structural here."""

    def __init__(out self, var input: Self.A):
        self._input = input^
        self._alias = String(Self.Agg.name)

    def __init__(out self, var input: Self.A, var name: String):
        self._input = input^
        self._alias = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self._input.columns()

    def name(self) -> String:
        return self._alias.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        """Through `Agg.dtype`, from the *operand's* dtype.

        Not from `A.Type()`, even where this fuses: a temporal dtype is not
        constructible from its type — a timestamp carries a unit and a
        timezone — which is exactly why `TemporalColumn.dtype` reads the
        schema. `Agg.dtype` applies the same widening rule either way, so
        `sum(int32)` still answers int64.
        """
        var in_dtypes = List[DynType](capacity=1)
        in_dtypes.append(self._input.dtype(schema))
        return Self.Agg.dtype(in_dtypes)

    comptime shape = Shape.scalar
    """An aggregate yields one value per group, so it is scalar-shaped in the
    same sense a literal is: it does not produce a value per input row."""

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """Pick the machine and the placement, once, when the plan is built.

        Both decisions resolve to comptime *types*: whether the query has keys
        is a property of the plan, and whether this aggregate can fuse is a
        property of `Agg` and `A`. Resolving them here is what keeps them out
        of the inner loop.

        Both operators are parameterised, so nothing in this lane is erased:
        `FusedAggregateOperator` because its body *is* the per-row loop, and
        `BufferedAggregateOperator` because keeping `A` means its operand is an
        `EvalOperator[A]` rather than a `DynOperator` box and `Agg.grouped` is
        a direct call rather than a pointer. The erased shape lives in
        `runtime/` as `RuntimeAggregateOperator`, where the name genuinely is not
        known until run time.
        """
        comptime if Self.fuses:
            if grouped:
                return FusedAggregateOperator[
                    Self.Agg.Lane, Self.A, HashGrouping
                ](self._input.copy(), bindings.copy())
            return FusedAggregateOperator[
                Self.Agg.Lane, Self.A, ScalarGrouping
            ](self._input.copy(), bindings.copy())
        else:
            return ColumnAggregateOperator[Self.Agg, Self.A](
                self._input.copy(), bindings.copy(), grouped
            )

    def alias(self, var name: String) -> Self:
        """Rename this aggregate. `col("x", int64).sum().alias("total")`.

        Returns a copy rather than mutating, so an aggregate stays a pure
        description and the same subtree can be named twice.
        """
        return Self(self._input.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.Agg.name, "(", self._input, ")")


# ---------------------------------------------------------------------------
# The vocabulary — one alias per aggregate, partially applied on its operand.
#
# `Sum[Self]` rather than `Aggregate[Fold[SumKernel], Self]` at every fluent
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
# `col("ts", timestamp(us)).min()` is `Fold[MinKernel, TimestampType]`, not a
# runtime lookup. `StringMin`, `CountDistinct` and friends stay partially
# applied, since their kernels are genuinely dtype-generic.
# ---------------------------------------------------------------------------
comptime Sum[A: PrimitiveValue] = Aggregate[Fold[SumKernel, A.Type], A]
comptime Product[A: PrimitiveValue] = Aggregate[Fold[ProductKernel, A.Type], A]
comptime Min[A: PrimitiveValue] = Aggregate[Fold[MinKernel, A.Type], A]
comptime Max[A: PrimitiveValue] = Aggregate[Fold[MaxKernel, A.Type], A]
comptime Mean[A: PrimitiveValue] = Aggregate[Fold[MeanKernel, A.Type], A]
comptime Count[A: PrimitiveValue] = Aggregate[Fold[CountKernel, A.Type], A]
comptime Variance[ddof: Int, A: ComptimeValue] = Aggregate[
    Dispersion[ddof, False], A
]
comptime StdDev[ddof: Int, A: ComptimeValue] = Aggregate[
    Dispersion[ddof, True], A
]
comptime StringMin = Aggregate[StringExtremum[MinOp], _]
comptime StringMax = Aggregate[StringExtremum[MaxOp], _]
comptime CountDistinct = Aggregate[DistinctCount[True], _]
comptime ApproxCountDistinct = Aggregate[DistinctCount[False], _]


struct FusedAggregateOperator[K: FoldKernel, A: NumericValue, G: Grouping](
    Operator
):
    """The fold in progress: this aggregate's input, plus the kernel's state.

    Three axes, all comptime: the **algebra** (`K`), the whole **input**
    subtree (`A`), and the **placement** (`G`). `FusedAggregateOperator[SumKernel,
    Mul[Column[Int64Type], Literal[Int64Type]], ScalarGrouping]` is one type,
    and therefore one loop with nothing interpreted inside it.

    `G` is a *phantom* parameter — the fold reads `G.scatters` and never holds
    a `G`. The grouping instance itself belongs to the operator above, which
    assigns every row once and shares the result with every aggregate in the
    query; a fold that owned its own grouping would re-hash the keys once per
    aggregate.

    Placement being a **type** rather than a flag is what makes the two loops
    two instantiations of one struct, neither compiling the other's body. It is
    not a runtime branch: which one a query needs is known when the plan is
    built, and running the scattering loop at a single group costs **14.6x**.
    A sorted or partitioned placement arrives as another conformer, not as
    another `Bool`.
    """

    comptime Acc = Self.K.AccType[Self.A.Type]
    comptime acc = Self.Acc.native
    comptime W = simd_width_of[Scalar[Self.acc]]()

    var _input: Self.A
    var _bindings: Bindings
    """This execution's parameter values, held by the operator rather than the
    node — which is what keeps the plan immutable and lets two executions of it
    bind different values. `bind` receives them and a `Param` reads them
    there."""
    var _state: AggState[Self.K, Self.A.Type]
    var _num_groups: Int
    var _emitted: Bool
    """Whether the fold has already been drained.

    `drain` is **repeatable** — the driver calls it until it answers `None` —
    so a fold that answered `Some` every time would make
    `while True: drain()` spin forever."""

    def __init__(out self, var input: Self.A, var bindings: Bindings):
        self._input = input^
        self._bindings = bindings^
        self._emitted = False
        # One implicit slot when this fold does not scatter, including over an
        # input that yields nothing: `sum` of no rows is one NULL, not no rows.
        self._num_groups = 1 if not Self.G.scatters else 0
        # The accumulator's dtype as a value, from `FoldKernel.acc_dtype`.
        # `A.Type()` is constructible today because the fused lane is numeric,
        # and that is precisely the constraint this has to shed: when `A` can
        # be temporal, the input dtype is not known until `bind(batch)` and
        # must come from the bound column rather than from the type.
        self._state = AggState[Self.K, Self.A.Type](
            Self.K.acc_dtype[Self.A.Type](Self.A.Type())
        )

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Fold one morsel in and answer nothing — the result arrives at
        `drain`. The grouping travels *with* the batch, which is what lets this
        be an `Operator` at all."""
        ref batch = morsel.batch
        ref groups = morsel.groups.ids
        comptime if Self.G.scatters:
            self._num_groups = morsel.groups.num_groups
        var num_groups = self._num_groups
        var n = len(batch)
        if n == 0:
            return None
        comptime W = Self.W
        var bound = self._input.bind(batch, self._bindings)
        var v = self._input.validity(bound)

        # The SIMD body stops at the last whole chunk. A `range(0, n, W)` loop
        # reads past the view on the final chunk and **aborts the process**:
        # buffer *size* is rounded up to 64 bytes, and that is not slack.
        var simd_end = (n // W) * W

        comptime if Self.G.scatters:
            var gids = groups.values()
            var i = 0
            if v:
                var vb = v.value()
                var bits = vb.view()
                while i < simd_end:
                    self._state.accumulate[W](
                        gids.load[W](i),
                        self._input.lane[W](bound, i).cast[Self.acc](),
                        bits.load[W](i),
                        num_groups,
                    )
                    i += W
                while i < n:
                    self._state.accumulate[1](
                        gids.load[1](i),
                        self._input.lane[1](bound, i).cast[Self.acc](),
                        bits.load[1](i),
                        num_groups,
                    )
                    i += 1
            else:
                while i < simd_end:
                    self._state.accumulate[W](
                        gids.load[W](i),
                        self._input.lane[W](bound, i).cast[Self.acc](),
                        SIMD[DType.bool, W](fill=True),
                        num_groups,
                    )
                    i += W
                while i < n:
                    self._state.accumulate[1](
                        gids.load[1](i),
                        self._input.lane[1](bound, i).cast[Self.acc](),
                        SIMD[DType.bool, 1](True),
                        num_groups,
                    )
                    i += 1
        else:
            # One group by construction, so `groups` is unused: building a zero
            # vector to say so is exactly the cost this path exists to avoid.
            var ident = Self.K.identity[Self.acc]()
            var vec = SIMD[Self.acc, W](ident)
            var acc = ident
            var count = 0
            var i = 0
            if v:
                var vb = v.value()
                var bits = vb.view()
                # A second accumulator, not a horizontal reduce per chunk: the
                # count is `K.finalize`'s divisor for `mean`, and the only thing
                # distinguishing "sum of nothing" from "sum of zeros" — both 0.
                # int64, not the accumulator type: `mean` accumulates in
                # float64, and a count is a count.
                var cnt = SIMD[DType.int64, W](0)
                while i < simd_end:
                    # `lane[W]` is null-blind: it returns data bits regardless
                    # of validity, so a null must become the identity here.
                    # Without the mask an unmasked `sum` is *silently correct*
                    # whenever the null slot's payload happens to be zero —
                    # only min/max expose it.
                    var mask = bits.load[W](i)
                    vec = Self.K.combine[Self.acc, W](
                        vec,
                        mask.select(
                            self._input.lane[W](bound, i).cast[Self.acc](),
                            SIMD[Self.acc, W](ident),
                        ),
                    )
                    cnt += mask.select(
                        SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
                    )
                    i += W
                while i < n:
                    if bits.load[1](i)[0]:
                        acc = Self.K.combine[Self.acc, 1](
                            acc, self._input.lane[1](bound, i).cast[Self.acc]()
                        )
                        count += 1
                    i += 1
                for j in range(W):
                    count += Int(cnt[j])
            else:
                while i < simd_end:
                    vec = Self.K.combine[Self.acc, W](
                        vec, self._input.lane[W](bound, i).cast[Self.acc]()
                    )
                    i += W
                while i < n:
                    acc = Self.K.combine[Self.acc, 1](
                        acc, self._input.lane[1](bound, i).cast[Self.acc]()
                    )
                    i += 1
                count += n
            for j in range(W):
                acc = Self.K.combine[Self.acc, 1](acc, vec[j])

            # The registers were per-batch scratch; this is the hand-off, once
            # per morsel rather than once per row.
            if count > 0:
                self._state.combine_at(0, acc, count)
            else:
                self._state.combine_at(0, ident, 0)

        return None

    def drain(mut self) raises -> Optional[Datum]:
        """Always an answer: `sum` of no rows is one NULL, not no rows."""
        if self._emitted:
            return None
        self._emitted = True
        return Datum(self._state.finish(self._num_groups).to_dyn())


struct ColumnAggregateOperator[Agg: AggKernel, A: ComptimeValue](Operator):
    """An aggregate that consumes its operand as a **column** rather than as
    lanes, and streams.

    The counterpart of `FusedAggregateOperator`, and the whole difference is
    what the kernel eats. A fold reads `lane[W]` straight out of a fused
    subtree and keeps one scalar per group; this one has an aggregate whose
    state is a hash set, a sketch, a best value or a Welford triple, so its
    input has to *exist* as a column — but only for the morsel being absorbed.

    **It used to be `BufferedAggregateOperator`, and the name was accurate.**
    `AggKernel.grouped` was one-shot over the whole input, so this operator
    kept every morsel's column and ids and concatenated them at `drain` —
    O(rows), and it contained no aggregation logic at all: nine fields whose
    only job was rebuilding one call's arguments. `AggKernel` is a state
    machine now (`open` / `update` / `finish`), so the state lives in the
    kernel and this holds O(groups).

    `Column` is doing work in the name here, where it would not on a kernel:
    every aggregate aggregates a column, but only one of these two *operators*
    hands its kernel one.

    The operand keeps its fusion: `A` stays a type parameter and evaluates
    through its own `EvalOperator[A]`, so `min(upper(s))` still compiles
    `upper(s)` to one loop even though `min` over a string cannot fuse.
    """

    var _input: EvalOperator[Self.A]
    """The operand's own operator, typed. A `DynOperator` here would box a
    subtree whose type this struct already names."""

    var _state: Optional[Self.Agg]
    """`None` until the first morsel: `Agg.open` needs the input's dtype, and
    an operator is built before any schema is in hand."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys — a runtime `Bool`, where
    `FusedAggregateOperator` makes it a `G: Grouping` parameter.

    Measured, not assumed. `Aggregate.to_operator` branches on a *runtime*
    `grouped`, so parameterising this instantiates both arms for every
    aggregate: a keyless binary with three of these linked both `HashGrouping`
    and `ScalarGrouping` and grew **4.6%**. The fused path earns its `G`
    because specialising the loop is worth 14.6x; handing a column to a kernel
    does not."""

    var _emitted: Bool

    def __init__(
        out self, var input: Self.A, var bindings: Bindings, scatters: Bool
    ):
        self._input = EvalOperator[Self.A](input^, bindings^)
        self._state = None
        self._scatters = scatters
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Evaluate the operand and let the kernel absorb it.

        Empty batches are absorbed too rather than skipped: an empty column
        still carries its dtype, and that is the one thing `Agg.open` needs and
        the one thing a later morsel cannot supply retroactively.
        """
        var n = len(morsel.batch)
        var d = self._input.push(morsel)
        var column = d.value().to_array(n)
        if not self._state:
            var in_dtypes = List[DynType](capacity=1)
            in_dtypes.append(column.dtype())
            self._state = Self.Agg.open(in_dtypes)
        var inputs = List[DynArray](capacity=1)
        inputs.append(column^)
        var groups = morsel.groups.copy() if self._scatters else Groups.single(
            n
        )
        self._state.value().update(groups, inputs)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._state:
            # No morsel ever arrived, so there is no dtype to open against. A
            # distinct count still has an answer; an extremum does not, and
            # `GroupByOperator` fills its slot from the output schema.
            var answer = Self.Agg.empty()
            if answer:
                return Datum(answer.value().copy())
            return None
        return Datum(self._state.value().finish())
