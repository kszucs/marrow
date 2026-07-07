"""``AnyValue`` — the universal value box — plus single-shot erased relations.

``AnyValue`` is the one type-erased value handle the relational layer holds. It
wraps any concrete value node behind a thin trampoline exposing only
``to_array(batch)``:

- the **fusable** comptime nodes from ``marrow.expr.values`` (``Column`` /
  ``Add`` / ``Gt`` / …) — the AOT path, tiny binary;
- the runtime ``DynValue`` interpreter from ``marrow.expr.runtime`` — what the
  Python bindings build.

The fused-vs-interpreted choice is *which node you box*, not which container:
boxing a ``DynValue`` links the per-op/per-dtype interpreter, while a program
that only boxes fused nodes leaves it dead-code-eliminated and stays ~250 KB
(``benchmarks/binary_size/``). ``Project``/``Filter`` here are the single-shot
self-executing relations over ``List[AnyValue]``; the streaming counterparts
live in ``marrow.expr.streaming``.
"""

from std.memory import ArcPointer

from marrow.arrays import AnyArray
from marrow.dtypes import Field
from marrow.kernels.filter import filter
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr.relations import Column
from marrow.expr.runtime import DynValue
from marrow.expr.values import BoolValue


# ---------------------------------------------------------------------------
# Trampolines — thin, one instantiation per boxed concrete node type
# ---------------------------------------------------------------------------


def _col_to_array_tramp[
    V: Column
](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
    """Delegate to a projected column's own fused ``to_array()``."""
    var typed = rebind[ArcPointer[V]](ptr)
    return typed[].to_array(batch)


def _col_name_tramp[V: Column](ptr: ArcPointer[NoneType]) -> String:
    var typed = rebind[ArcPointer[V]](ptr)
    return typed[].field_name()


def _pred_to_array_tramp[
    V: BoolValue
](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
    """Delegate to a predicate's own fused ``execute()``, erasing the
    bit-packed ``BoolArray`` result to ``AnyArray``."""
    var typed = rebind[ArcPointer[V]](ptr)
    return typed[].execute(batch).to_any()


def _no_name_tramp(ptr: ArcPointer[NoneType]) -> String:
    """Predicates are never projected, so they carry no output name."""
    return String()


def _dyn_to_array_tramp(
    ptr: ArcPointer[NoneType], batch: RecordBatch
) raises -> AnyArray:
    """Delegate to a ``DynValue`` interpreter node's ``to_array()``."""
    var typed = rebind[ArcPointer[DynValue]](ptr)
    return typed[].to_array(batch)


def _dyn_name_tramp(ptr: ArcPointer[NoneType]) -> String:
    var typed = rebind[ArcPointer[DynValue]](ptr)
    return typed[].field_name()


# ---------------------------------------------------------------------------
# AnyValue — the universal value box (no eval() tag switch of its own)
# ---------------------------------------------------------------------------


struct AnyValue(Copyable, Movable):
    """Type-erased handle over a single value node.

    Holds the concrete node in an ``ArcPointer`` and a thin trampoline to its
    own ``to_array()``. There is deliberately no tag and no ``eval()`` switch on
    ``AnyValue`` itself — boxing a fused node never links the interpreter.
    """

    var _boxed: ArcPointer[NoneType]
    var _to_array: def(
        ArcPointer[NoneType], RecordBatch
    ) thin raises -> AnyArray
    var _field_name: def(ArcPointer[NoneType]) thin -> String

    def __init__[V: Column](out self, value: V):
        """Box a projected column (``NumericColumn`` / ``StringColumn``)."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = _col_to_array_tramp[V]
        self._field_name = _col_name_tramp[V]

    def __init__[V: BoolValue](out self, value: V):
        """Box a fused predicate node (``Lt`` / ``Gt`` / ``Eq``)."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = _pred_to_array_tramp[V]
        self._field_name = _no_name_tramp

    def __init__(out self, var value: DynValue):
        """Box the runtime ``DynValue`` interpreter (what the Python bindings
        build). Links its ``to_array`` (the per-op/per-dtype interpreter); a
        program that never constructs one leaves that trampoline unreferenced,
        so it is dead-code-eliminated and the fused-only path stays tiny."""
        var ptr = ArcPointer[DynValue](value^)
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = _dyn_to_array_tramp
        self._field_name = _dyn_name_tramp

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self._to_array(self._boxed, batch)

    def field_name(self) -> String:
        return self._field_name(self._boxed)


# ---------------------------------------------------------------------------
# Project — runtime, self-executing projection over a List[AnyValue]
# ---------------------------------------------------------------------------


struct Project(Copyable, Movable):
    """Self-executing projection: iterate a runtime list of boxed values, each
    of which runs as its own pass, and assemble a ``RecordBatch``.

    The outer loop is O(#columns); the inner loop lives inside each
    ``AnyValue.to_array()``. No ``Planner``, no type pack — a plain walkable
    list you can rewrite before executing.
    """

    var exprs: List[AnyValue]

    def __init__(out self, var exprs: List[AnyValue]):
        self.exprs = exprs^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var cols = List[AnyArray]()
        var fields = List[Field]()
        for ref e in self.exprs:
            var arr = e.to_array(batch)
            fields.append(Field(e.field_name(), arr.dtype()))
            cols.append(arr^)
        return RecordBatch(Schema(fields=fields^), cols^)

    def filter(var self, var predicate: AnyValue) -> Filter:
        return Filter(self^, predicate^)


# ---------------------------------------------------------------------------
# Filter — runtime, self-executing row filter over a Project
# ---------------------------------------------------------------------------


struct Filter(Copyable, Movable):
    """Self-executing row filter: run the boxed predicate into a mask, run the
    input projection, and filter each column by the mask."""

    var input: Project
    var predicate: AnyValue

    def __init__(out self, var input: Project, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var mask = self.predicate.to_array(batch)
        var projected = self.input.execute(batch)
        var cols = List[AnyArray]()
        for i in range(len(projected.columns)):
            cols.append(filter(projected.columns[i].copy(), mask.copy()))
        return RecordBatch(projected.schema.copy(), cols^)
