"""Value-erased, relation-erased AOT relational layer — a **runtime**,
walkable plan tree over **fused-only** boxed values.

This is the "option 1" surface explored in ``docs/aot-relations-design.md``:
the relational operators (``Project``/``Filter``) are plain runtime structs
holding ``List[AnyValue]`` (not a ``*Es`` type pack), so a plan is a mutable
tree you can rewrite (predicate/projection pushdown) — but each *value* stays a
fully-monomorphized fused subtree, boxed behind a thin trampoline.

The crucial difference from ``marrow.expr.runtime.Expr``: ``AnyValue`` exposes
**only** ``to_array(batch)`` via a trampoline into the concrete node's own
fused ``execute()``. It carries **no** ``eval()`` tag-switch, so nothing here
makes the open per-node/per-dtype value interpreter reachable. And the
operators execute themselves single-shot (no ``Planner`` / ``RelationProcessor``
open dispatch), so only the relation kinds actually constructed get linked.

The point of this module is to *measure* whether that combination — fused-only
value box + self-executing runtime tree — keeps the binary near the fully-typed
``Project[*Es]`` layer (~250 KB) rather than the runtime ``dyn`` path (~7.7 MB).
See ``benchmarks/binary_size/query_erased_aot.mojo``.
"""

from std.memory import ArcPointer

from marrow.arrays import AnyArray
from marrow.dtypes import AnyDataType, Field
from marrow.kernels.arithmetic import add, subtract, multiply, divide, neg, abs_
from marrow.kernels.boolean import and_, or_, not_, is_null, select
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from marrow.kernels.filter import filter
from marrow.kernels.string import string_lengths
from marrow.scalars import AnyScalar
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr.relations import Column
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


# ---------------------------------------------------------------------------
# AnyValue — fused-only value box (no eval() tag interpreter)
# ---------------------------------------------------------------------------


struct AnyValue(Copyable, Movable):
    """Type-erased handle over a single fused value node.

    Holds the concrete node in an ``ArcPointer`` and a thin trampoline to its
    own fused ``execute()``/``to_array()``. There is deliberately no tag and no
    ``eval()`` switch — the only entry point is ``to_array(batch)``, so boxing a
    value never links the open per-op/per-dtype interpreter.
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
        """Box a predicate node (``Lt`` / ``Gt`` / ``Eq``)."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = _pred_to_array_tramp[V]
        self._field_name = _no_name_tramp

    def __init__(out self, var value: DynValue):
        """Box the type-erased tag-interpreter node.

        The same box holds fused nodes *and* the interpreter — the fused-vs-
        interpreted choice is which node you box, not which container. Boxing a
        ``DynValue`` links its ``to_array`` (the per-op/per-dtype interpreter);
        a program that never constructs one leaves that trampoline unreferenced,
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
    """Self-executing projection: iterate a runtime list of boxed values,
    each of which runs as its own fused pass, and assemble a ``RecordBatch``.

    The outer loop is O(#columns); the fused inner loop lives inside each
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
    """Self-executing row filter: run the boxed predicate as a fused pass into a
    mask, run the input projection, and filter each column by the mask.

    Mirrors ``marrow.expr.relations.Filter`` but over runtime boxed values
    instead of a ``Pred`` type parameter, and executes itself directly (no
    pull-based processor pipeline)."""

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


# ---------------------------------------------------------------------------
# DynValue — the type-erased tag-interpreter node (seed of expr/runtime.mojo)
# ---------------------------------------------------------------------------
#
# The reworked `marrow.expr.runtime.Expr`, reduced to what it *is* in the unified
# design: one boxable node kind whose own `to_array()` interprets a tag over
# `List[AnyValue]` children — with no `FUSED` slots (fusion is expressed by
# boxing a fused node into `AnyValue` directly, not by a bridge inside the
# interpreter). Ports Expr.eval()'s op set (CAST excepted, as in Expr) node for
# node, recursing through the boxed children — so this is a faithful drop-in for
# the interpreter half of `Expr`, ready to become `expr/runtime.mojo`. What is
# not yet ported is Expr's *plan-manipulation* API (`kind`/`resolve_names`/
# `inputs`), which lands when the relational consumers are repointed.


comptime LOAD = UInt8(0)
comptime LITERAL = UInt8(1)
comptime ADD = UInt8(2)
comptime SUB = UInt8(3)
comptime MUL = UInt8(4)
comptime DIV = UInt8(5)
comptime EQ = UInt8(6)
comptime NE = UInt8(7)
comptime LT = UInt8(8)
comptime LE = UInt8(9)
comptime GT = UInt8(10)
comptime GE = UInt8(11)
comptime AND = UInt8(12)
comptime OR = UInt8(13)
comptime NEG = UInt8(14)
comptime ABS = UInt8(15)
comptime NOT = UInt8(16)
comptime IS_NULL = UInt8(17)
comptime IF_ELSE = UInt8(18)
comptime LENGTH = UInt8(19)


struct DynValue(Copyable, Movable):
    """Type-erased interpreter node — boxes into the same ``AnyValue`` as the
    fused nodes. Its ``to_array()`` is the per-op/per-dtype interpreter,
    recursing through its boxed ``List[AnyValue]`` children.

    ``_index``/``_name`` carry a LOAD's column reference (resolved by name when
    ``_name`` is set, matching the fused columns' name resolution, else by
    ``_index``); ``_value`` carries a LITERAL's scalar."""

    var _tag: UInt8
    var _args: List[AnyValue]
    var _index: Int
    var _name: String
    var _value: Optional[AnyScalar]

    def __init__(
        out self,
        tag: UInt8,
        var args: List[AnyValue],
        index: Int,
        var name: String,
        var value: Optional[AnyScalar],
    ):
        self._tag = tag
        self._args = args^
        self._index = index
        self._name = name^
        self._value = value^

    @staticmethod
    def load(index: Int, var name: String) -> DynValue:
        return DynValue(LOAD, List[AnyValue](), index, name^, None)

    @staticmethod
    def lit(var value: AnyScalar) -> DynValue:
        return DynValue(LITERAL, List[AnyValue](), 0, String(), value^)

    @staticmethod
    def unary(tag: UInt8, var child: AnyValue) -> DynValue:
        var args = List[AnyValue]()
        args.append(child^)
        return DynValue(tag, args^, 0, String(), None)

    @staticmethod
    def binary(tag: UInt8, var left: AnyValue, var right: AnyValue) -> DynValue:
        var args = List[AnyValue]()
        args.append(left^)
        args.append(right^)
        return DynValue(tag, args^, 0, String(), None)

    @staticmethod
    def gt(var left: AnyValue, var right: AnyValue) -> DynValue:
        return DynValue.binary(GT, left^, right^)

    def _a(self, i: Int, batch: RecordBatch) raises -> AnyArray:
        return self._args[i].to_array(batch)

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        if self._tag == LOAD:
            var idx = self._index
            if self._name.byte_length() > 0:
                idx = batch.schema.get_field_index(self._name)
            return batch.columns[idx].copy()
        elif self._tag == LITERAL:
            return self._value.value().repeat(batch.num_rows())
        elif self._tag == ADD:
            return add(self._a(0, batch), self._a(1, batch))
        elif self._tag == SUB:
            return subtract(self._a(0, batch), self._a(1, batch))
        elif self._tag == MUL:
            return multiply(self._a(0, batch), self._a(1, batch))
        elif self._tag == DIV:
            return divide(self._a(0, batch), self._a(1, batch))
        elif self._tag == EQ:
            return equal(self._a(0, batch), self._a(1, batch))
        elif self._tag == NE:
            return not_equal(self._a(0, batch), self._a(1, batch))
        elif self._tag == LT:
            return less(self._a(0, batch), self._a(1, batch))
        elif self._tag == LE:
            return less_equal(self._a(0, batch), self._a(1, batch))
        elif self._tag == GT:
            return greater(self._a(0, batch), self._a(1, batch))
        elif self._tag == GE:
            return greater_equal(self._a(0, batch), self._a(1, batch))
        elif self._tag == AND:
            return and_(self._a(0, batch), self._a(1, batch))
        elif self._tag == OR:
            return or_(self._a(0, batch), self._a(1, batch))
        elif self._tag == NEG:
            return neg(self._a(0, batch))
        elif self._tag == ABS:
            return abs_(self._a(0, batch))
        elif self._tag == NOT:
            return not_(self._a(0, batch))
        elif self._tag == IS_NULL:
            return is_null(self._a(0, batch))
        elif self._tag == LENGTH:
            return string_lengths(self._a(0, batch)).to_any()
        elif self._tag == IF_ELSE:
            return select(
                self._a(0, batch), self._a(1, batch), self._a(2, batch)
            )
        else:
            raise Error("DynValue.to_array: unknown tag ", self._tag)

    def field_name(self) -> String:
        return self._name.copy()

    def dtype(self) -> Optional[AnyDataType]:
        if self._tag == LITERAL:
            return self._value.value().type()
        return None


def _dyn_to_array_tramp(
    ptr: ArcPointer[NoneType], batch: RecordBatch
) raises -> AnyArray:
    var typed = rebind[ArcPointer[DynValue]](ptr)
    return typed[].to_array(batch)


def _dyn_name_tramp(ptr: ArcPointer[NoneType]) -> String:
    var typed = rebind[ArcPointer[DynValue]](ptr)
    return typed[].field_name()
