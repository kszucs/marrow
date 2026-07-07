"""Pull-based streaming relational ops — fat, self-executing nodes.

The Phase-2 shape from ``docs/expr-unification-plan.md``: each relational
operator is *its own processor* (holds its child ``AnySource``, has ``pull()``),
so morsel-at-a-time streaming works **without** a central ``Planner`` — there is
no open per-kind dispatch to link every processor, so only the nodes actually
constructed get compiled in. Values are ``AnyValue`` (fused-capable, via
``marrow.expr.erased``), not the ``Expr`` interpreter, so a fused-value pipeline
stays tiny while an interpreter-value pipeline pays only for what it constructs.

This mirrors ``marrow.expr.executor``'s ``*Processor`` classes exactly (same
``pull()``/``Exhausted`` morsel protocol) minus the ``Planner`` translation
step — the node *is* the processor. It seeds ``expr/relations.mojo``; the full
op set (Aggregate/Join) and ``DynValue``'s plan-manipulation API land as this
grows. ``Exhausted`` is defined here rather than imported from
``marrow.expr.executor`` so streaming does not pull the dyn executor (and its
``Planner``/kernel fanout) into the binary.
"""

from std.memory import ArcPointer

from marrow.arrays import AnyArray, BoolArray, PrimitiveArray, StringArray
from marrow.builders import PrimitiveBuilder, BoolBuilder, StringBuilder
from marrow.dtypes import (
    Field,
    PrimitiveType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)
from marrow.kernels.filter import filter
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr.erased import AnyValue


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by ``pull()`` when a source has no more morsels to yield."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


# ---------------------------------------------------------------------------
# _concat — a CLOSED, flat-only column concat local to the expr layer
# ---------------------------------------------------------------------------
#
# `collect()` merges morsels by concatenating each column. The general
# `marrow.kernels.concat` routes through `AnyBuilder(dtype)` — an open switch
# that instantiates a builder for *every* dtype (incl. nested) whenever it is
# reachable, so it is never DCE'd and inflates the binary ~6x (measured). The
# expr layer's projections produce only flat columns, so it uses this local
# concat instead: typed builders for primitive/bool/string and a `raise` for
# anything else (like `filter`), keeping the streaming layer closed and small.


def _concat_primitive[
    T: PrimitiveType
](arrays: List[AnyArray]) raises -> AnyArray:
    var total = 0
    for ref a in arrays:
        total += a.length()
    var builder = PrimitiveBuilder[T](arrays[0].as_primitive[T]().dtype.copy(), total)
    for ref a in arrays:
        builder.extend(a.as_primitive[T]())
    return builder.finish().to_any()


def _concat(arrays: List[AnyArray]) raises -> AnyArray:
    """Concatenate same-dtype flat columns; raises on nested/other dtypes."""
    var dtype = arrays[0].dtype()
    if dtype == bool_:
        var builder = BoolBuilder(capacity=0)
        for ref a in arrays:
            builder.extend(a.as_bool())
        return builder.finish().to_any()
    elif dtype == int8:
        return _concat_primitive[Int8Type](arrays)
    elif dtype == int16:
        return _concat_primitive[Int16Type](arrays)
    elif dtype == int32:
        return _concat_primitive[Int32Type](arrays)
    elif dtype == int64:
        return _concat_primitive[Int64Type](arrays)
    elif dtype == uint8:
        return _concat_primitive[UInt8Type](arrays)
    elif dtype == uint16:
        return _concat_primitive[UInt16Type](arrays)
    elif dtype == uint32:
        return _concat_primitive[UInt32Type](arrays)
    elif dtype == uint64:
        return _concat_primitive[UInt64Type](arrays)
    elif dtype == float16:
        return _concat_primitive[Float16Type](arrays)
    elif dtype == float32:
        return _concat_primitive[Float32Type](arrays)
    elif dtype == float64:
        return _concat_primitive[Float64Type](arrays)
    elif dtype.is_string():
        var builder = StringBuilder(capacity=0)
        for ref a in arrays:
            builder.extend(a.as_string())
        return builder.finish().to_any()
    else:
        raise Error(
            "streaming collect: unsupported column dtype ",
            dtype,
            " (flat types only)",
        )


# ---------------------------------------------------------------------------
# Source — the fat-node streaming interface (was RelationProcessor)
# ---------------------------------------------------------------------------


trait Source(Movable, ImplicitlyDeletable):
    """Pull-based streaming relation node.

    ``pull()`` yields morsel-sized ``RecordBatch`` values, raising ``Exhausted``
    when done. ``schema()`` is ``mut`` so a node whose output schema is only
    known from data (a ``Project`` over interpreter values) can resolve it
    lazily on first pull."""

    def schema(mut self) raises -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        ...


# ---------------------------------------------------------------------------
# AnySource — type-erased streaming node (was AnyRelationProcessor)
# ---------------------------------------------------------------------------


struct AnySource(Movable):
    """Type-erased streaming node — one trampoline box for the whole pipeline.

    Each boxed node's ``pull``/``schema`` is reached via a per-instance
    trampoline (references only its own methods), so composing a pipeline links
    only the node kinds constructed — no central dispatch, unlike ``Planner``."""

    var _data: ArcPointer[NoneType]
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch
    var _virt_schema: def(ArcPointer[NoneType]) thin raises -> Schema
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_pull[
        T: Source
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

    @staticmethod
    def _tramp_schema[T: Source](ptr: ArcPointer[NoneType]) raises -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_drop[T: Source](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: Source](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_pull = Self._tramp_pull[T]
        self._virt_schema = Self._tramp_schema[T]
        self._virt_drop = Self._tramp_drop[T]

    def schema(mut self) raises -> Schema:
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        return self._virt_pull(self._data)

    def collect(mut self) raises -> RecordBatch:
        """Drain the pipeline into a single ``RecordBatch`` (convenience for the
        single-shot surface: build a pipeline, ``collect()`` it)."""
        var batches = List[RecordBatch]()
        while True:
            try:
                batches.append(self.pull())
            except Exhausted:
                break
        if len(batches) == 0:
            return RecordBatch(schema=self.schema(), columns=List[AnyArray]())
        if len(batches) == 1:
            return RecordBatch(copy=batches[0])
        var schema = batches[0].schema
        var num_cols = batches[0].num_columns()
        var result_cols = List[AnyArray](capacity=num_cols)
        for c in range(num_cols):
            var col_arrays = List[AnyArray](capacity=len(batches))
            for b in range(len(batches)):
                col_arrays.append(batches[b].columns[c].copy())
            result_cols.append(_concat(col_arrays))
        return RecordBatch(schema=Schema(copy=schema), columns=result_cols^)

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Scan — morsel-sized slices of an in-memory RecordBatch
# ---------------------------------------------------------------------------


struct Scan(Source):
    var batch: RecordBatch
    var offset: Int
    var morsel_size: Int

    def __init__(out self, var batch: RecordBatch, morsel_size: Int = 1024):
        self.batch = batch^
        self.offset = 0
        self.morsel_size = morsel_size

    def schema(mut self) raises -> Schema:
        return self.batch.schema.copy()

    def pull(mut self) raises -> RecordBatch:
        if self.offset >= self.batch.num_rows():
            raise Exhausted()
        var length = min(
            self.morsel_size, self.batch.num_rows() - self.offset
        )
        var result = self.batch.slice(self.offset, length)
        self.offset += length
        return result^


# ---------------------------------------------------------------------------
# Filter — pulls child, applies a boolean AnyValue predicate
# ---------------------------------------------------------------------------


struct Filter(Source):
    var child: AnySource
    var predicate: AnyValue

    def __init__(out self, var child: AnySource, var predicate: AnyValue):
        self.child = child^
        self.predicate = predicate^

    def schema(mut self) raises -> Schema:
        # Filter preserves its child's columns.
        return self.child.schema()

    def pull(mut self) raises -> RecordBatch:
        # Skip morsels that filter to 0 rows; Exhausted propagates from child.
        while True:
            var batch = self.child.pull()
            var mask = self.predicate.to_array(batch)
            var cols = List[AnyArray]()
            for i in range(batch.num_columns()):
                cols.append(filter(batch.columns[i].copy(), mask.copy()))
            var result = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if result.num_rows() > 0:
                return result^


# ---------------------------------------------------------------------------
# Project — pulls child, evaluates a List[AnyValue] into output columns
# ---------------------------------------------------------------------------


struct Project(Source):
    var child: AnySource
    var values: List[AnyValue]
    var _buffered: Optional[RecordBatch]
    var _schema: Optional[Schema]
    var _started: Bool

    def __init__(out self, var child: AnySource, var values: List[AnyValue]):
        self.child = child^
        self.values = values^
        self._buffered = None
        self._schema = None
        self._started = False

    def _project(self, batch: RecordBatch) raises -> RecordBatch:
        var cols = List[AnyArray]()
        var fields = List[Field]()
        for ref v in self.values:
            var arr = v.to_array(batch)
            fields.append(Field(v.field_name(), arr.dtype()))
            cols.append(arr^)
        return RecordBatch(schema=Schema(fields=fields^), columns=cols^)

    def _start(mut self) raises:
        # Output dtypes come from the produced arrays, so schema is resolved by
        # projecting the first morsel and buffering it for the first pull.
        if self._started:
            return
        self._started = True
        try:
            var out = self._project(self.child.pull())
            self._schema = out.schema.copy()
            self._buffered = out^
        except Exhausted:
            # Empty input: dtypes are unknown without data (a known gap the
            # Planner used to cover; resolved when values carry type inference).
            self._schema = Schema(fields=List[Field]())

    def schema(mut self) raises -> Schema:
        self._start()
        return self._schema.value().copy()

    def pull(mut self) raises -> RecordBatch:
        self._start()
        if self._buffered:
            var b = self._buffered.value().copy()
            self._buffered = None
            return b^
        return self._project(self.child.pull())
