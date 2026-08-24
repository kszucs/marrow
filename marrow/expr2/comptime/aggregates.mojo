"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the state binds the subtree itself and reads `lane[W]`
straight into a register. Measured 2026-08-22: 1.17-1.68x over
materialise-then-scatter when grouped, and 14.6x when not.

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**The fold lives here too.** `FoldOperator` is the executor for the nodes
below it, and it sits beside them for the same reason `Column` sits beside its
own `bind`/`lane`: this lane is organised by **family**, not by
logical-versus-physical. That split is real one level up (`logical.mojo` /
`physical.mojo`); inside a lane the distinction that matters is
lane-agnostic versus lane-specific. `EvalOperator` is generic over any
`Evaluable` and belongs in `physical.mojo`; `FoldOperator` is parameterised on
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

from ...arrays import Int32Array
from ...kernels.aggregate import (
    AggKernel,
    AggState,
    MaxKernel,
    MeanKernel,
    MinKernel,
    ProductKernel,
    SumKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, Value
from ..params import Bindings
from ..physical import Datum, DynOperator, Evaluable, Morsel, Operator

from .core import NumericValue


struct NumericAggregate[K: AggKernel, A: NumericValue](Evaluable, Value):
    """`sum(x)`, `min(x)`, … — pure, and rewritable because of it.

    **An ordinary `Value`.** It conforms to exactly what `x + 1` conforms to
    and is boxed by the same `DynValue`. There is no `AggValue` trait: once
    every logical node answers `to_operator`, an aggregate stopped being a
    different *kind* of node and became one that answers from
    `Operator.finish` rather than from `Operator.push`.

    That difference is behavioural, not structural, and it is deliberately not
    encoded as a marker trait — a trait constraining nothing documents nothing,
    since this struct satisfies `Value` either way. When a planner needs to
    *check* it (to reject `project([col("a").sum()])` by inspection rather than
    by raising), the mechanism is a comptime `kind` on `Analyzable`, not a
    parallel hierarchy.
    """

    comptime Type = Self.K.AccType[Self.A.Type]
    """The accumulator's type, not the input's: summing `int32` yields `int64`.

    Delegating to `AggKernel.AccType` keeps that widening rule in one place.
    It must never appear **unerased** in a signature — it is a comptime
    conditional type, which reduces inside the struct but fails to unify at a
    return site. That is why `finish` answers `DynArray`.
    """

    var _input: Self.A
    var _name: String

    def __init__(out self, var input: Self.A, var name: String):
        self._input = input^
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self._input.columns()

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    comptime shape = Shape.scalar
    """An aggregate yields one value per group, so it is scalar-shaped in the
    same sense a literal is: it does not produce a value per input row."""

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: RecordBatch, bindings: Bindings) raises -> Datum:
        """An aggregate has no per-batch value, and saying so is the point.

        Folding needs every batch, so there is nothing to answer here — the
        result exists only once the stream ends, which is what `to_operator`
        and `Operator.finish` express. Raising makes
        `project([col("a").sum()])` a **plan-time** error naming the mistake,
        which is what DuckDB, DataFusion and Polars all do. The alternative,
        a compile-time rejection, is what a `Kind` on the trait would buy and
        is not worth a second erased box.
        """
        raise Error(
            "aggregate '",
            self._name,
            (
                "' cannot be evaluated per batch; use .aggregate() rather than"
                " projecting or filtering on it"
            ),
        )

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator[Datum]:
        """Pick the placement, once, when the plan is built.

        A runtime `Bool` in, a comptime *type* out: whether the query has keys
        is a property of the plan, and resolving it here is what keeps it out
        of the inner loop.
        """
        if grouped:
            return FoldOperator[Self.K, Self.A, HashGrouping](
                self._input.copy(), bindings.copy()
            )
        return FoldOperator[Self.K, Self.A, ScalarGrouping](
            self._input.copy(), bindings.copy()
        )

    def alias(self, var name: String) -> Self:
        """Rename this aggregate. `col("x", int64).sum().alias("total")`.

        Returns a copy rather than mutating, so an aggregate stays a pure
        description and the same subtree can be named twice.
        """
        return Self(self._input.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self._input, ")")


comptime Sum = NumericAggregate[SumKernel, _]
comptime Product = NumericAggregate[ProductKernel, _]
comptime Min = NumericAggregate[MinKernel, _]
comptime Max = NumericAggregate[MaxKernel, _]
comptime Mean = NumericAggregate[MeanKernel, _]


struct FoldOperator[K: AggKernel, A: NumericValue, G: Grouping](Operator):
    """The fold in progress: this aggregate's input, plus the kernel's state.

    Three axes, all comptime: the **algebra** (`K`), the whole **input**
    subtree (`A`), and the **placement** (`G`). `FoldOperator[SumKernel,
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

    comptime Out = Datum
    """A `Datum` — one value per slot, carried in the same box an
    elementwise value uses.

    This is what `Operator.Out` was introduced for. A fold and a filter are the
    same shape of thing (`push` until there is nothing left, then `finish`) and
    differ only in what they produce, so parameterising the output is all that
    was ever needed to make them one trait. Before this, an aggregate had its
    own trait *and* its own erased box for exactly that difference.
    """

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
    `while True: drain()` spin forever. It is only reachable through
    `AggregateOperator` today, which calls it once, but the contract is the
    contract: an operator that cannot say "spent" cannot be driven generically.
    """

    def __init__(out self, var input: Self.A, var bindings: Bindings):
        self._input = input^
        self._bindings = bindings^
        # One implicit slot when this fold does not scatter, including over an
        # input that yields nothing: `sum` of no rows is one NULL, not no rows.
        self._num_groups = 1 if not Self.G.scatters else 0
        self._emitted = False
        # The accumulator's dtype as a value. `A.Type()` is constructible
        # today because the fused lane is numeric, and that is precisely the
        # constraint this has to shed: when `A` can be temporal, the dtype is
        # not known until `bind(batch)` and must come from the bound column
        # rather than from the type.
        self._state = AggState[Self.K, Self.A.Type](
            Self.K.AccType[Self.A.Type]()
        )

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """FoldOperator one morsel in and answer nothing — the result arrives at
        `finish`. The grouping travels *with* the batch, which is what lets this
        be an `Operator` at all."""
        ref batch = morsel.batch
        ref groups = morsel.groups.ids
        comptime if Self.G.scatters:
            self._num_groups = morsel.groups.num_groups
        var num_groups = self._num_groups
        var n = batch.num_rows()
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
        if self._emitted:
            return None
        self._emitted = True
        return Datum(self._state.finish(self._num_groups).to_dyn())
