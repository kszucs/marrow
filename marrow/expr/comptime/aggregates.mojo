"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the state binds the subtree itself and reads `lane[W]`
straight into a register. Measured 2026-08-22: 1.17-1.68x over
materialise-then-scatter when grouped, and 14.6x when not.

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**The three executors live here too.** `ScatteredAggregateOperator`,
`RegisterAggregateOperator` and `BufferedAggregateOperator` are what the nodes
below become, and they sit beside them for the same reason `Column` sits beside
its own `bind`/`lane`: this lane is organised by **family**, not by
logical-versus-physical. That split is real one level up (`logical.mojo` /
`physical.mojo`); inside a lane the distinction that matters is lane-agnostic
versus lane-specific. `EvalOperator` is generic over any `Evaluable` and
belongs in `physical.mojo`; the two fused operators are parameterised on
`A: PrimitiveValue` and cannot go there without making the lane-agnostic layer
name a lane.

`AggState[K, V]` is the accumulator, it lives
in `kernels`, and it is the only thing here that crosses a batch boundary. The
ungrouped path folds into registers, but those are *per-batch scratch* — the
same status `Bound` has — and hand off through `combine_at` once per morsel. So
`K.finalize`, `K.empty_is_null` and the count-is-zero rule stay defined in
exactly one place.
"""

from ...dtypes import DynType, NumericType
from std.sys.info import simd_width_of

from ...arrays import BinaryLikeArray, PrimitiveArray, Int32Array, StructArray
from ...kernels.aggregate import (
    FoldKernel,
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    Foldable,
    CountFold,
    MaxFold,
    MaxOp,
    MeanFold,
    MinFold,
    MinOp,
    ProductFold,
    LexicalExtremum,
    SumFold,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, Value
from ..bindings import Bindings
from ...execution import ExecContext
from ...kernels.concat import concat
from ...kernels.core import Groups
from ...arrays import DynArray
from ..physical import (
    BufferedAggregateOperator,
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
    - is the operand lane-readable — `conforms_to(A, PrimitiveValue)`? A lane
      is `SIMD[Type.native, W]`, so it needs a fixed-width element — which is
      all it needs, temporal columns included.

    Both yes, and the fold is fused into the operand's loop: `sum(a * 2 + b)`
    never builds `a * 2 + b`, worth 1.17-1.68x grouped and **14.6x** ungrouped.
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

        All three operators are parameterised, so nothing in this lane is
        erased: the two fused ones because their bodies *are* the per-row
        loop, and `BufferedAggregateOperator` because keeping `A` means its
        operand is evaluated through a direct call rather than through a
        `DynOperator` box. All three are boxed once, here, in the
        `DynOperator` every operator already pays for, so the return type does
        not depend on the branch.

        **Placement is a comptime choice for a fused fold and a runtime one
        for a buffered fold**, and that asymmetry is the measurement rather
        than an oversight. `ScatteredAggregateOperator` and
        `RegisterAggregateOperator` are different programs — a random write per
        row against a register accumulated across the whole morsel — and the
        gap between them is the 14.6x. `BufferedAggregateOperator` reads its
        placement once per morsel to build a `Groups`, never per row, so
        instantiating it twice would double the code for nothing — measured
        at +4.6%.
        """
        comptime if Self.fuses:
            if grouped:
                return ScatteredAggregateOperator[Self.Agg, Self.A](
                    self._input.copy(),
                    bindings.copy(),
                    self._input.dtype(schema),
                )
            return RegisterAggregateOperator[Self.Agg, Self.A](
                self._input.copy(),
                bindings.copy(),
                self._input.dtype(schema),
            )
        else:
            return BufferedAggregateOperator[Self.Agg, Self.A](
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
comptime Count[A: PrimitiveValue] = Aggregate[Fold[CountFold, A.Type], A]
comptime Variance[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, False, A.Type], A
]
comptime StdDev[ddof: Int, A: NumericValue] = Aggregate[
    Dispersion[ddof, True, A.Type], A
]
comptime StringMin[A: StringValue] = Aggregate[
    LexicalExtremum[MinOp, A.Type], A
]
comptime StringMax[A: StringValue] = Aggregate[
    LexicalExtremum[MaxOp, A.Type], A
]
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
# state in every instantiation and would need the `fuses` predicate written
# twice -- once on the node to pick the operator, once on the operator to pick
# the arm -- with nothing enforcing that the two agreed. Splitting on placement
# costs no instantiations: `scatters` was already a comptime parameter, so the
# same two fused programs are emitted either way.
# ---------------------------------------------------------------------------


def _emit_fold[Agg: AggKernel](mut state: Agg, slots: Int) raises -> Datum:
    """Seed `slots` slots, then finalize — the tail all three operators share.

    **A free function, not a method and not a trait default.** A method body on
    an operator struct elaborates unconditionally, which is the trap that
    forced the fused `push` bodies inline: `comptime if` prunes an untaken
    *branch* before name resolution, but never a method. This takes the state
    as an argument instead, so it is elaborated once per `Agg` and names
    nothing from the operator that calls it.

    `reserve` is what makes an aggregate over **zero** morsels answer at all.
    The fused operators return early at `n == 0` and a buffered one over an
    empty stream never sees `update`, so nothing grew a slot; a keyless query
    still owes exactly one row. `reserve(0)` is a no-op in every conformer, so
    a grouped query that saw no rows correctly answers no rows.
    """
    state.reserve(slots)
    return Datum(state.finish().to_dyn())


struct ScatteredAggregateOperator[Agg: Foldable, A: PrimitiveValue](Operator):
    """A fused fold that scatters into the slot each row's group id names —
    `GROUP BY` with a lane-readable operand and a lane algebra.

    `sum(a * 2 + b)` never builds `a * 2 + b`: the state binds the subtree and
    reads `lane[W]` straight into a register before scattering. Measured
    2026-08-22 at 1.17-1.68x over materialise-then-scatter.

    The scatter stays scalar per lane even at `W > 1`, and that is not an
    oversight — two lanes may carry the same group, and a vector
    read-modify-write would lose one of them without conflict detection. `W`
    still pays because the loads and the arithmetic feeding it vectorise.

    **Separate from `RegisterAggregateOperator` rather than one struct with a
    `scatters` parameter.** The two bodies share no statement: this one issues
    a random write per row through `agg.scatter[W]`, the other accumulates in
    registers and calls `combine_at` once per morsel — and that register path
    is where the **14.6x** came from. A merged struct also had to carry
    `_num_groups` in the instantiation that never reads it, because a struct
    body admits no `comptime if` and therefore no conditional field.

    **The bounds are the honest ones.** `Agg: Foldable` because this calls
    `scatter`; `A: PrimitiveValue` because it calls `bind`, `validity` and
    `lane[W]`. `Aggregate.to_operator` recovers both inside
    `comptime if Self.fuses`, where `conforms_to` has already established them.
    """

    var _input: Self.A
    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype. Not an
    `Optional`: `to_operator` takes the input schema, so the dtype is in hand
    before the first morsel, which is what let `AggKernel.open` and the
    `_opened` latch that guarded it disappear."""

    var _num_groups: Int
    """How many slots the last morsel asked for. `drain` reads it to seed the
    slots an input of zero rows never grew — it stays 0 when no morsel arrived,
    and a grouped query over nothing correctly emits no rows."""

    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var bindings: Bindings,
        in_dtype: DynType,
    ) raises:
        self._input = input^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype)
        self._num_groups = 0
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Scatter this morsel's lanes into their groups.

        Inline rather than delegating to a method per validity case, and that
        is forced: a separate method's body is elaborated unconditionally, and
        moving the loop out gave `'A' value has no attribute 'bind'` because
        `A` is bound only on `PrimitiveValue` at the point of the call.
        """
        ref batch = morsel.batch
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

        var gids = morsel.groups.ids.values()
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
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        # No "did a morsel arrive?" branch: the kernel is built at
        # construction, so this answers from an untouched state exactly as it
        # would after a morsel that folded nothing.
        return _emit_fold(self._state, self._num_groups)


struct RegisterAggregateOperator[Agg: Foldable, A: PrimitiveValue](Operator):
    """A fused fold with **one** slot, accumulated in registers — no `GROUP
    BY`, a lane-readable operand and a lane algebra.

    The fastest shape in the tree, and the one the **14.6x** was measured on: a
    scatter at one group is a million serially dependent read-modify-writes
    through a builder slot, where this keeps the accumulator in a SIMD register
    for the whole morsel and hands it over once through `combine_at`.

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
    var _bindings: Bindings
    """This execution's parameter values, held by the *operator* rather than
    the node — which is what keeps the plan immutable and lets two executions
    of it bind different values."""

    var _state: Self.Agg
    """The accumulator, built at construction from the operand's dtype."""

    var _emitted: Bool

    def __init__(
        out self,
        var input: Self.A,
        var bindings: Bindings,
        in_dtype: DynType,
    ) raises:
        self._input = input^
        self._bindings = bindings^
        self._state = Self.Agg(in_dtype)
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Fold the whole morsel in registers, then hand it to slot 0 once."""
        ref batch = morsel.batch
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
            # int64, not the accumulator type: `mean` accumulates in float64,
            # and a count is a count.
            var cnt = SIMD[DType.int64, W](0)
            while i < simd_end:
                # `lane[W]` is null-blind: it returns data bits regardless of
                # validity, so a null must become the identity here. Without
                # the mask an unmasked `sum` is *silently correct* whenever the
                # null slot's payload happens to be zero — only min/max expose
                # it.
                var mask = bits.load[W](i)
                vec = Self.Agg.Lane.combine[Self.Agg.Acc, W](
                    vec,
                    mask.select(
                        self._input.lane[W](bound, i).cast[Self.Agg.Acc](),
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
                        self._input.lane[1](bound, i).cast[Self.Agg.Acc](),
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

        # The registers were per-batch scratch; this is the hand-off, once per
        # morsel rather than once per row.
        if count > 0:
            agg.combine_at(0, acc, count)
        else:
            agg.combine_at(0, ident, 0)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        # One slot, always: `sum` of nothing is one NULL, not no rows. `push`
        # returns early at `n == 0` and never grew it, so the seed is here.
        return _emit_fold(self._state, 1)
