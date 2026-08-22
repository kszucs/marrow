"""Named reductions over a fused value — `sum`, `min`, `max`, `mean`.

The input is any `NumericValue`, so `sum(a * 2 + b)` folds a fused subtree
without materialising the intermediate: the expression evaluates once per
morsel and the result goes straight into the accumulator.

Nothing here implements folding. `AggState[K, V]` does, and it is already
typed on both the kernel and the input dtype, so these two structs are the
*expression* half only — one pure, one holding the state the kernel owns.
"""

from ...arrays import Int32Array, PrimitiveArray
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
from ...scalars import DynScalar, Int32Scalar
from ...schema import Schema
from ...tabular import RecordBatch
from ..aggregates import Accumulator, DynAccumulator, Reduction
from .core import NumericValue


struct NumericReductionState[K: AggKernel, A: NumericValue](Accumulator):
    """The fold in progress: this reduction's input, plus the kernel's state."""

    comptime Type = Self.K.AccType[Self.A.Type]

    var _input: Self.A
    var _state: AggState[Self.K, Self.A.Type]

    def __init__(out self, var input: Self.A):
        self._input = input^
        self._state = AggState[Self.K, Self.A.Type]()

    def update(mut self, batch: RecordBatch) raises:
        """Evaluate the input over this morsel and scatter-fold it.

        `AggState` is a *grouped* accumulator, so an ungrouped aggregate is the
        one-group case: every row maps to group 0. The zero vector is built per
        morsel rather than cached, which is one allocation per batch against a
        fold over every row in it — and reusing the grouped path is what keeps
        one implementation of identity/finalize instead of two.
        """
        var n = batch.num_rows()
        if n == 0:
            return
        var values = (
            self._input.evaluate(batch)
            .to_array(n)
            .as_primitive[Self.A.Type]()
            .copy()
        )
        var group_ids = Int32Scalar(0).to_dyn().repeat(n).as_int32().copy()
        self._state.update(group_ids, values, 1)

    def finish(mut self) raises -> DynScalar:
        """The single group's value.

        `finish(1)` yields a one-element column; an aggregate is a scalar, so
        it is unwrapped here. An empty input is NULL rather than the kernel's
        identity — `SUM` of no rows is NULL in SQL, and `AggState.finish`
        already encodes that.
        """
        return self._state.finish(1)[0].to_dyn()


struct NumericReduction[K: AggKernel, A: NumericValue](Reduction):
    """`sum(x)`, `min(x)`, … — pure, and rewritable because of it."""

    comptime Type = Self.K.AccType[Self.A.Type]
    """The accumulator's type, not the input's: summing `int32` yields `int64`.

    Delegating to `AggKernel.AccType` is what keeps that widening rule in one
    place; restating it here is how a `sum` overflows in the expression layer
    but not in the kernel.
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
        """The alias, always.

        Unlike a `Value`, an aggregate has no bare-column case — `sum(a)` is
        not `a` — so there is no source `Field` to carry over and nothing to
        infer a name from. `expr/` defaults it to `"<kernel>(<input>)"`; that
        belongs in the builder that has both, not here.
        """
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.Type())

    # -- AggValue -----------------------------------------------------------

    def to_accumulator(self) raises -> DynAccumulator:
        return NumericReductionState[Self.K, Self.A](self._input.copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self._input, ")")


comptime Sum = NumericReduction[SumKernel, _]
comptime Product = NumericReduction[ProductKernel, _]
comptime Min = NumericReduction[MinKernel, _]
comptime Max = NumericReduction[MaxKernel, _]
comptime Mean = NumericReduction[MeanKernel, _]
