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

from ...arrays import DynArray
from ...dtypes import DynType, NumericType
from ...kernels.groupby import HashGrouping, ScalarGrouping
from ...kernels.aggregate import (
    AggKernel,
    MaxKernel,
    MeanKernel,
    MinKernel,
    ProductKernel,
    SumKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..core import AggValue
from ..physical import DynOperator
from .core import NumericValue
from .folds import Fold


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

    def to_processor(self, grouped: Bool) raises -> DynOperator[DynArray]:
        """Pick the placement, once, when the plan is built.

        A runtime `Bool` in, a comptime *type* out: whether the query has keys
        is a property of the plan, and resolving it here is what keeps it out
        of the inner loop.
        """
        if grouped:
            return Fold[Self.K, Self.A, HashGrouping](self._input.copy())
        return Fold[Self.K, Self.A, ScalarGrouping](self._input.copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self._input, ")")


comptime Sum = NumericAggregate[SumKernel, _]
comptime Product = NumericAggregate[ProductKernel, _]
comptime Min = NumericAggregate[MinKernel, _]
comptime Max = NumericAggregate[MaxKernel, _]
comptime Mean = NumericAggregate[MeanKernel, _]
