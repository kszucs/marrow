"""Fused aggregates — `sum`, `min`, `max`, `mean`, `product`.

`sum(a * 2 + b)` folds **without materialising `a * 2 + b`**. The input is a
`NumericValue`, so the state binds the subtree itself and reads `lane[W]`
straight into a register. Measured 2026-08-22: 1.17-1.68x over
materialise-then-scatter when grouped, and 14.6x when not.

That is the one thing no other engine can do. DataFusion's `GroupsAccumulator`
takes `&[ArrayRef]`, ClickHouse's `IAggregateFunction::add` takes `IColumn**`,
Polars aggregates a `Series` — all three must compute the intermediate column
first, because none has comptime types.

**This module declares no state.** `AggState[K, V]` is the accumulator, it lives
in `kernels`, and it is the only thing here that crosses a batch boundary. The
ungrouped path folds into registers, but those are *per-batch scratch* — the
same status `Bound` has — and hand off through `combine_at` once per morsel. So
`K.finalize`, `K.empty_is_null` and the count-is-zero rule stay defined in
exactly one place.
"""

from std.sys.info import simd_width_of

from ...arrays import DynArray, Int32Array
from ...dtypes import DynType, NumericType
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
from ..core import AggValue
from ..physical import AggregateState, DynAggregateState
from .core import NumericValue


struct NumericAggregateState[K: AggKernel, A: NumericValue, grouped: Bool](
    AggregateState
):
    """The fold in progress: this aggregate's input, plus the kernel's state.

    `grouped` is a **comptime** parameter, so the two loops are two
    instantiations of one struct rather than two structs — and neither compiles
    the other's body. It is not a runtime branch: which one a query needs is
    known when the plan is built, and running the grouped loop at one group
    costs 14.6x.
    """

    comptime Acc = Self.K.AccType[Self.A.Type]
    comptime acc = Self.Acc.native
    comptime W = simd_width_of[Scalar[Self.acc]]()

    var _input: Self.A
    var _state: AggState[Self.K, Self.A.Type]

    def __init__(out self, var input: Self.A):
        self._input = input^
        self._state = AggState[Self.K, Self.A.Type]()

    def update(
        mut self, batch: RecordBatch, groups: Int32Array, num_groups: Int
    ) raises:
        var n = batch.num_rows()
        if n == 0:
            return
        comptime W = Self.W
        var bound = self._input.bind(batch)
        var v = self._input.validity(bound)

        # The SIMD body stops at the last whole chunk. A `range(0, n, W)` loop
        # reads past the view on the final chunk and **aborts the process**:
        # buffer *size* is rounded up to 64 bytes, and that is not slack.
        var simd_end = (n // W) * W

        comptime if Self.grouped:
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

    def finish(mut self, num_groups: Int) raises -> DynArray:
        return self._state.finish(num_groups).to_dyn()


struct NumericAggregate[K: AggKernel, A: NumericValue](AggValue):
    """`sum(x)`, `min(x)`, … — pure, and rewritable because of it."""

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

    # -- Analyzable ---------------------------------------------------------

    def columns(self) -> List[String]:
        return self._input.columns()

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- AggValue -----------------------------------------------------------

    def to_state(self, grouped: Bool) raises -> DynAggregateState:
        if grouped:
            return NumericAggregateState[Self.K, Self.A, True](
                self._input.copy()
            )
        return NumericAggregateState[Self.K, Self.A, False](self._input.copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self._input, ")")


comptime Sum = NumericAggregate[SumKernel, _]
comptime Product = NumericAggregate[ProductKernel, _]
comptime Min = NumericAggregate[MinKernel, _]
comptime Max = NumericAggregate[MaxKernel, _]
comptime Mean = NumericAggregate[MeanKernel, _]
