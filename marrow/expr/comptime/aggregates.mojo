"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the state binds the subtree itself and reads `lane[W]`
straight into a register. Measured 2026-08-22: 1.17-1.68x over
materialise-then-scatter when grouped, and 14.6x when not.

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**The fold lives here too.** `FusedAccumulator` is the executor for the nodes
below it, and it sits beside them for the same reason `Column` sits beside its
own `bind`/`lane`: this lane is organised by **family**, not by
logical-versus-physical. That split is real one level up (`logical.mojo` /
`physical.mojo`); inside a lane the distinction that matters is
lane-agnostic versus lane-specific. `EvalOperator` is generic over any
`Evaluable` and belongs in `physical.mojo`; `FusedAccumulator` is parameterised on
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
    Accumulable,
    AggregateOperator,
    Datum,
    EvalOperator,
    DynOperator,
    Morsel,
    Operator,
)

from .core import ComptimeValue, NumericValue


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
    """It answers from `AggregateOperator.drain`, never per batch. `Project`
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
        `FusedAccumulator` because its body *is* the per-row loop, and
        `BufferedAccumulator` because keeping `A` means its operand is an
        `EvalOperator[A]` rather than a `DynOperator` box and `Agg.grouped` is
        a direct call rather than a pointer. The erased shape lives in
        `runtime/` as `RuntimeAccumulator`, where the name genuinely is not
        known until run time.
        """
        comptime if Self.fuses:
            if grouped:
                return AggregateOperator(
                    FusedAccumulator[Self.Agg.Lane, Self.A, HashGrouping](
                        self._input.copy(), bindings.copy()
                    )
                )
            return AggregateOperator(
                FusedAccumulator[Self.Agg.Lane, Self.A, ScalarGrouping](
                    self._input.copy(), bindings.copy()
                )
            )
        else:
            return AggregateOperator(
                BufferedAccumulator[Self.Agg, Self.A](
                    self._input.copy(), bindings.copy(), grouped
                )
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
# `Variance` and `StdDev` name **both** parameters where the others leave the
# operand as a `_` hole. That is not a style choice: an alias carrying its own
# parameter cannot also be partially applied — `Variance[ddof]` declares, but
# `Variance[ddof, Self]` at the use site is then "unexpected parameter". So an
# alias with a comptime argument of its own spells the operand out.
# ---------------------------------------------------------------------------
comptime Sum = Aggregate[Fold[SumKernel], _]
comptime Product = Aggregate[Fold[ProductKernel], _]
comptime Min = Aggregate[Fold[MinKernel], _]
comptime Max = Aggregate[Fold[MaxKernel], _]
comptime Mean = Aggregate[Fold[MeanKernel], _]
comptime Count = Aggregate[Fold[CountKernel], _]
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


struct FusedAccumulator[K: FoldKernel, A: NumericValue, G: Grouping](
    Accumulable
):
    """The fold in progress: this aggregate's input, plus the kernel's state.

    Three axes, all comptime: the **algebra** (`K`), the whole **input**
    subtree (`A`), and the **placement** (`G`). `FusedAccumulator[SumKernel,
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

    def __init__(out self, var input: Self.A, var bindings: Bindings):
        self._input = input^
        self._bindings = bindings^
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

    def absorb(mut self, morsel: Morsel) raises:
        """Fold one morsel into the state. The grouping travels *with* the
        batch, which is what lets a fold be an `Accumulable` at all."""
        ref batch = morsel.batch
        ref groups = morsel.groups.ids
        comptime if Self.G.scatters:
            self._num_groups = morsel.groups.num_groups
        var num_groups = self._num_groups
        var n = len(batch)
        if n == 0:
            return
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

    def finish(mut self) raises -> Optional[DynArray]:
        """Always an answer: `sum` of no rows is one NULL, not no rows."""
        return self._state.finish(self._num_groups).to_dyn()


struct BufferedAccumulator[Agg: AggKernel, A: ComptimeValue](Accumulable):
    """An aggregate that must see its whole input as a column, so it **buffers
    every morsel** until `drain`. Fully typed, like its fused sibling.

    The counterpart of `FusedAccumulator`, and named for the cost that
    separates them. Both are *value* operators — two of the things
    `GroupByOperator` holds in its `_folds` list — and both answer a `Datum` of
    one value per slot. They differ in what they can do with a morsel: a fold
    reads `lane[W]` out of a fused subtree and keeps one scalar per group, so
    it buffers nothing; this one has an aggregate whose state is a hash set, a
    sketch, a best-row index or a Welford triple, which `AggregateFn` cannot
    finish incrementally — so its input has to exist as a column first, and the
    accumulated columns and ids are **O(rows)**.

    **Buffering and erasure are different things, and only the first is forced
    here.** This used to be one zero-parameter struct shared with the runtime
    lane, which meant a comptime aggregate boxed its operand into a
    `DynOperator`, reached its kernel through a thin `AggregateFn` pointer, and
    carried an `Optional[RuntimeAggregate]` field it never set. None of that
    was needed: `Agg` and `A` are known where the plan is written. The erased
    shape is `RuntimeAccumulator` in `runtime/`, where the name really does
    arrive at run time.

    `A` earns its parameter the same way `EvalOperator`'s does — the operand
    subtree stays one type through the boundary, so `min(upper(s))` still
    compiles `upper(s)` to one loop.
    """

    var _input: EvalOperator[Self.A]
    """The operand's own operator, typed. A `DynOperator` here would box a
    subtree whose type this struct already names."""

    var _chunks: List[DynArray]
    """Every morsel's evaluated column. The O(rows) this is named for."""

    var _ids: List[Int32Array]
    """Group ids per morsel, **typed**. They are always `Int32Array`; storing
    them as `DynArray` erased a type this lane already knew and paid a narrow
    back at `finish`. The typed `concat` overload is what made it spellable."""

    var _num_groups: Int
    var _rows: Int

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys — a runtime `Bool`, where
    `FusedAccumulator` makes it a `G: Grouping` parameter.

    Measured, not assumed. `Aggregate.to_operator` branches on a *runtime*
    `grouped`, so parameterising this instantiates both arms for every
    aggregate: a keyless binary with three buffered aggregates linked both
    `HashGrouping` and `ScalarGrouping` and grew **4.6%**. The fused path earns
    its `G` because specialising the loop is worth 14.6x; this body is a
    `concat` and a call, where one predictable branch per morsel costs
    nothing."""

    def __init__(
        out self, var input: Self.A, var bindings: Bindings, scatters: Bool
    ):
        self._input = EvalOperator[Self.A](input^, bindings^)
        self._chunks = List[DynArray]()
        self._ids = List[Int32Array]()
        self._scatters = scatters
        self._num_groups = 0 if scatters else 1
        self._rows = 0

    def absorb(mut self, morsel: Morsel) raises:
        """Evaluate the operand against the morsel and keep the column.

        Empty batches are kept too rather than skipped: an empty column still
        carries its dtype, and `Agg.grouped` dispatches on it.
        """
        var n = len(morsel.batch)
        var d = self._input.push(morsel)
        self._chunks.append(d.value().to_array(n))
        self._rows += n
        if self._scatters:
            self._ids.append(morsel.groups.ids.copy())
            self._num_groups = max(self._num_groups, morsel.groups.num_groups)

    def finish(mut self) raises -> Optional[DynArray]:
        if len(self._chunks) == 0:
            # No morsel ever arrived. A distinct count still has an answer; an
            # extremum does not, and `GroupByOperator` fills its slot from the
            # output schema.
            return Self.Agg.empty()
        var columns = List[DynArray](capacity=1)
        columns.append(concat(self._chunks, ExecContext.serial()))
        var groups: Groups
        if self._scatters:
            groups = Groups(
                concat(self._ids, ExecContext.serial()), self._num_groups
            )
        else:
            groups = Groups.single(self._rows)
        # A direct static call, not a pointer: `Agg` is a type here.
        return Self.Agg.grouped(groups, columns)
