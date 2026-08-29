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

from ...arrays import BinaryLikeArray, PrimitiveArray, Int32Array, StructArray
from ...kernels.aggregate import (
    FoldKernel,
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
    Evaluable,
    Datum,
    EvalOperator,
    DynOperator,
    Morsel,
    Operator,
)

from .core import ComptimeValue, NumericValue, PrimitiveValue, StringValue


struct Aggregate[Agg: AggKernel, A: Evaluable & Value](Value):
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
        return Self.Agg.dtype(self._input.dtype(schema))

    comptime shape = Shape.scalar
    """An aggregate yields one value per group, so it is scalar-shaped in the
    same sense a literal is: it does not produce a value per input row."""

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
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
                return AggregateOperator[Self.Agg, Self.A, HashGrouping](
                    self._input.copy(),
                    bindings.copy(),
                    True,
                    self._input.dtype(schema),
                )
            return AggregateOperator[Self.Agg, Self.A, ScalarGrouping](
                self._input.copy(),
                bindings.copy(),
                False,
                self._input.dtype(schema),
            )
        else:
            # `G` is pinned: the non-fusing arm reads `_scatters` at run time,
            # so instantiating both groupings for it would double the code for
            # nothing — measured at +4.6%.
            return AggregateOperator[Self.Agg, Self.A, ScalarGrouping](
                self._input.copy(),
                bindings.copy(),
                grouped,
                self._input.dtype(schema),
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
comptime Variance[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, False, A.Type], A
]
comptime StdDev[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, True, A.Type], A
]
comptime StringMin[A: StringValue] = Aggregate[StringExtremum[MinOp, A.Type], A]
comptime StringMax[A: StringValue] = Aggregate[StringExtremum[MaxOp, A.Type], A]
# The three cardinalities are dtype-generic in what they *compute* — an int64
# whatever was counted — but each still names the array it reads, because a
# validity scan and a hash are both faster typed. In this lane the operand
# knows it: a `PrimitiveValue` evaluates to a `PrimitiveArray[A.Type]` and a
# `StringValue` to a `BinaryLikeArray[A.Type]`, so each family's fluent method
# supplies the array and no `ArrayType` companion is needed on the node.
comptime CountDistinct[A: PrimitiveValue] = Aggregate[
    DistinctCount[True, PrimitiveArray[A.Type]], A
]
comptime ApproxCountDistinct[A: PrimitiveValue] = Aggregate[
    DistinctCount[False, PrimitiveArray[A.Type]], A
]
comptime StringCountDistinct[A: StringValue] = Aggregate[
    DistinctCount[True, BinaryLikeArray[A.Type]], A
]
comptime StringApproxCountDistinct[A: StringValue] = Aggregate[
    DistinctCount[False, BinaryLikeArray[A.Type]], A
]


struct AggregateOperator[Agg: AggKernel, A: Evaluable & Value, G: Grouping](
    Operator
):
    """Any aggregate over a comptime operand — one struct, two ways of feeding
    the same state.

    `comptime fuses` decides which, from two independent facts: whether `Agg`
    has a lane algebra (`Foldable`) and whether the operand is lane-readable
    (`NumericValue`). Both yes and the fold runs in registers over the operand's
    own `lane[W]`, so `sum(a * 2 + b)` never builds `a * 2 + b` — 1.17-1.68x
    grouped, **14.6x** ungrouped. Otherwise the operand is evaluated to a column
    per morsel and the kernel absorbs it.

    **One state field, not a union.** Both paths hold `Optional[Agg]`: the
    fused one reaches it through `Foldable`'s lane entry points (`grow` /
    `scatter` / `combine_at`), the other through `AggKernel.update`. That is
    what makes the merge possible at all — a struct cannot declare a field
    conditionally, and an earlier attempt that kept a separate
    `AggState[K, A.Type]` field had to carry both field sets and paid for it.
    Holding `Agg` instead also stops this layer naming `AggState`, a kernel's
    own state struct, which it had no business reaching into.

    The tight bounds are recovered inside `comptime if Self.fuses`, where
    `conforms_to` prunes the untaken branch **before** name resolution: the
    fused arm calls `A.lane[W]` on a field bound only on `ComptimeValue`, and
    `Foldable`'s methods on a field bound only on `AggKernel`. Both resolve.

    `G` is comptime for the fused arm, where specialising the scatter loop is
    the 14.6x. The other arm ignores it and reads `_scatters`, and
    `Aggregate.to_operator` pins `G = ScalarGrouping` there so a non-fusing
    aggregate is instantiated once rather than twice — instantiating both
    groupings for aggregates that cannot use them measured **+4.6%**.
    """

    comptime fuses = conforms_to(Self.Agg, Foldable) and conforms_to(
        Self.A, NumericValue
    )

    var _input: Self.A
    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """`None` until the first morsel: `Agg.open` needs the input's dtype, and
    an operator is built before any schema is in hand."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys. Read only by the non-fusing arm;
    the fused arm has `G.scatters` at comptime."""

    var _num_groups: Int
    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var bindings: Bindings,
        scatters: Bool,
        in_dtype: DynType,
    ) raises:
        """The kernel is built here, from the operand's dtype, which the plan
        already knows: `to_operator` takes the input schema. That is what let
        `AggKernel.open` disappear — and with it this operator's
        `Optional[Agg]` and the `_opened` latch that guarded it."""
        self._input = input^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype)
        self._scatters = scatters
        self._num_groups = 1 if not scatters else 0
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Absorb one morsel, two ways.

        Both arms are inline rather than delegating to a method each, and that
        is forced: `comptime if` prunes an untaken *branch* before name
        resolution, but a separate method's body is elaborated unconditionally.
        Moving the fused loop out gave `'A' value has no attribute 'bind'`,
        because `A` is only bound on `ComptimeValue` here.
        """
        comptime if Self.fuses:
            ref batch = morsel.batch
            ref groups = morsel.groups.ids
            comptime if Self.G.scatters:
                self._num_groups = morsel.groups.num_groups
            var num_groups = self._num_groups
            var n = len(batch)
            ref agg = self._state
            if n == 0:
                return None
            comptime W = simd_width_of[Scalar[Self.Agg.Acc]]()
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
                        agg.scatter[W](
                            gids.load[W](i),
                            self._input.lane[W](bound, i).cast[Self.Agg.Acc](),
                            bits.load[W](i),
                            num_groups,
                        )
                        i += W
                    while i < n:
                        agg.scatter[1](
                            gids.load[1](i),
                            self._input.lane[1](bound, i).cast[Self.Agg.Acc](),
                            bits.load[1](i),
                            num_groups,
                        )
                        i += 1
                else:
                    while i < simd_end:
                        agg.scatter[W](
                            gids.load[W](i),
                            self._input.lane[W](bound, i).cast[Self.Agg.Acc](),
                            SIMD[DType.bool, W](fill=True),
                            num_groups,
                        )
                        i += W
                    while i < n:
                        agg.scatter[1](
                            gids.load[1](i),
                            self._input.lane[1](bound, i).cast[Self.Agg.Acc](),
                            SIMD[DType.bool, 1](True),
                            num_groups,
                        )
                        i += 1
            else:
                # One group by construction, so `groups` is unused: building a zero
                # vector to say so is exactly the cost this path exists to avoid.
                var ident = Self.Agg.Lane.identity[Self.Agg.Acc]()
                var vec = SIMD[Self.Agg.Acc, W](ident)
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
                        vec = Self.Agg.Lane.combine[Self.Agg.Acc, W](
                            vec,
                            mask.select(
                                self._input.lane[W](bound, i).cast[
                                    Self.Agg.Acc
                                ](),
                                SIMD[Self.Agg.Acc, W](ident),
                            ),
                        )
                        cnt += mask.select(
                            SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
                        )
                        i += W
                    while i < n:
                        if bits.load[1](i)[0]:
                            acc = Self.Agg.Lane.combine[Self.Agg.Acc, 1](
                                acc,
                                self._input.lane[1](bound, i).cast[
                                    Self.Agg.Acc
                                ](),
                            )
                            count += 1
                        i += 1
                    for j in range(W):
                        count += Int(cnt[j])
                else:
                    while i < simd_end:
                        vec = Self.Agg.Lane.combine[Self.Agg.Acc, W](
                            vec,
                            self._input.lane[W](bound, i).cast[Self.Agg.Acc](),
                        )
                        i += W
                    while i < n:
                        acc = Self.Agg.Lane.combine[Self.Agg.Acc, 1](
                            acc,
                            self._input.lane[1](bound, i).cast[Self.Agg.Acc](),
                        )
                        i += 1
                    count += n
                for j in range(W):
                    acc = Self.Agg.Lane.combine[Self.Agg.Acc, 1](acc, vec[j])

                # The registers were per-batch scratch; this is the hand-off, once
                # per morsel rather than once per row.
                if count > 0:
                    agg.combine_at(0, acc, count)
                else:
                    agg.combine_at(0, ident, 0)

            return None
        else:
            var n = len(morsel.batch)
            var column = self._input.evaluate(
                morsel.batch, self._bindings
            ).to_array(n)
            var groups = (
                morsel.groups.copy() if self._scatters else Groups.single(n)
            )
            # The one narrowing in this lane, and it is comptime-resolved:
            # `Agg` is a parameter here, so `InArray` is a concrete type and
            # this is a conversion rather than a dispatch.
            self._state.update(groups, Self.Agg.InArray(column.to_data()))
            return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        # No "did a morsel arrive?" branch: the kernel is built at
        # construction, so an aggregate over zero rows answers from an
        # untouched state — `sum` of nothing is one NULL — exactly as it would
        # after a morsel that folded nothing.
        comptime if Self.fuses:
            # A fused aggregate over zero rows never grew a slot — `push`
            # returns early at `n == 0` — so `drain` seeds them here. That is
            # what makes `sum` of nothing one NULL rather than no rows.
            self._state.grow(self._num_groups)
        return Datum(self._state.finish())
