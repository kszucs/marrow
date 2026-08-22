"""Aggregates: the pure/state split, one level down from relations.

`Relation` describes and `Processor` runs; `AggValue` describes and
`Accumulator` folds. The shape is deliberately the same, because the problem
is: an aggregate expression is immutable and rewritable, while summing across
morsels is inherently stateful, and welding them together is what makes a plan
un-rewritable.

**Accumulation itself is not implemented here.** `kernels.aggregate.AggState`
already owns it — typed accumulator, identity, `finalize`, and a `merge` for
thread-local partials. This layer answers only *which value, reduced how, named
what*, and hands the folding to the kernel that already does it.
"""

from std.memory import ArcPointer

from .core import Analyzable
from ..scalars import DynScalar
from ..schema import Schema
from ..dtypes import DynType
from ..tabular import RecordBatch


trait Accumulator(Deinitable, Movable):
    """A fold in progress. The aggregate counterpart of `Processor`.

    Move-only for the same reason: it owns mutable state, so copying one would
    fork a fold halfway through and double-count whatever came before.
    """

    def update(mut self, batch: RecordBatch) raises:
        """Fold one morsel in."""
        ...

    def finish(mut self) raises -> DynScalar:
        """The result, once every morsel has been folded."""
        ...


trait Reduction(Analyzable, Copyable, Deinitable, Writable):
    """A value reduced to a single scalar, named. Pure, like `Relation`.

    Extends `Analyzable` rather than `Value`: a reduction answers the same
    three questions about itself — which columns it reads, what it is called,
    what type it produces — but it does **not** `evaluate` to a `Datum` per
    batch, which is the whole of `Evaluable`. Folding is the accumulator's job.

    Named for what it is rather than `Aggregation`, which
    `kernels.aggregate` already uses for the monomorphized kernel-level thing,
    or `Aggregate`, which the relational node wants. It is also ibis's term for
    this exact concept.
    """

    def to_accumulator(self) raises -> DynAccumulator:
        """Begin a fold. Mirrors `Relation.to_processor`, and returns the
        erased form for the same reason: the caller holds a heterogeneous list
        of reductions and needs one type back from all of them."""
        ...


struct DynAccumulator(Movable):
    """An `Accumulator` of any aggregate, erased."""

    var _data: ArcPointer[NoneType]
    var _virt_update: def(ArcPointer[NoneType], RecordBatch) thin raises
    var _virt_finish: def(ArcPointer[NoneType]) thin raises -> DynScalar

    @staticmethod
    def _update_tramp[
        A: Accumulator
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises:
        rebind[ArcPointer[A]](ptr)[].update(batch)

    @staticmethod
    def _finish_tramp[
        A: Accumulator
    ](ptr: ArcPointer[NoneType]) raises -> DynScalar:
        return rebind[ArcPointer[A]](ptr)[].finish()

    @implicit
    def __init__[A: Accumulator](out self, var value: A):
        var ptr = ArcPointer[A](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_update = Self._update_tramp[A]
        self._virt_finish = Self._finish_tramp[A]

    def update(mut self, batch: RecordBatch) raises:
        self._virt_update(self._data, batch)

    def finish(mut self) raises -> DynScalar:
        return self._virt_finish(self._data)


struct DynReduction(Copyable, Movable, Writable):
    """An `AggValue` of any reduction, erased.

    Four slots, matching the four things a reduction is asked: its name, the
    columns it reads, the type it produces, and how to start folding it.
    `write_to` is the fifth and exists for the same reason `DynValue`'s does —
    a plan that cannot print itself cannot be debugged.
    """

    var _data: ArcPointer[NoneType]
    var _virt_columns: def(ArcPointer[NoneType]) thin -> List[String]
    var _virt_name: def(ArcPointer[NoneType]) thin -> String
    var _virt_dtype: def(ArcPointer[NoneType], Schema) thin raises -> DynType
    var _virt_acc: def(ArcPointer[NoneType]) thin raises -> DynAccumulator
    var _virt_write: def(ArcPointer[NoneType]) thin -> String

    @staticmethod
    def _columns_tramp[A: Reduction](ptr: ArcPointer[NoneType]) -> List[String]:
        return rebind[ArcPointer[A]](ptr)[].columns()

    @staticmethod
    def _name_tramp[A: Reduction](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[A]](ptr)[].name()

    @staticmethod
    def _dtype_tramp[
        A: Reduction
    ](ptr: ArcPointer[NoneType], schema: Schema) raises -> DynType:
        return rebind[ArcPointer[A]](ptr)[].dtype(schema)

    @staticmethod
    def _acc_tramp[
        A: Reduction
    ](ptr: ArcPointer[NoneType]) raises -> DynAccumulator:
        return rebind[ArcPointer[A]](ptr)[].to_accumulator()

    @staticmethod
    def _write_tramp[A: Reduction](ptr: ArcPointer[NoneType]) -> String:
        return String(rebind[ArcPointer[A]](ptr)[])

    @implicit
    def __init__[A: Reduction](out self, var value: A):
        var ptr = ArcPointer[A](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_columns = Self._columns_tramp[A]
        self._virt_name = Self._name_tramp[A]
        self._virt_dtype = Self._dtype_tramp[A]
        self._virt_acc = Self._acc_tramp[A]
        self._virt_write = Self._write_tramp[A]

    def columns(self) -> List[String]:
        return self._virt_columns(self._data)

    def name(self) -> String:
        return self._virt_name(self._data)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._virt_dtype(self._data, schema)

    def to_accumulator(self) raises -> DynAccumulator:
        return self._virt_acc(self._data)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write(self._data))
