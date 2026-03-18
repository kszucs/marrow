"""Logical relational plan nodes.

``Relation``    — the trait every relational plan node must implement.
``AnyRelation`` — the type-erased, ArcPointer-backed container.

Concrete plan nodes
-------------------
``Scan``           — reads from a named source (leaf node).
``Filter``         — applies a boolean predicate to its input.
``Project``        — evaluates a list of expressions to produce output columns.
``InMemoryTable``  — leaf node backed by a RecordBatch.

Plan-building API
-----------------
``AnyRelation.select(*names)``  — project columns by name.
``AnyRelation.filter(pred)``    — filter rows by predicate.
``in_memory_table(batch)``      — create an in-memory relation.

Example
-------
    var plan = in_memory_table(batch).filter(col("x") > lit[int64](0)).select("x")
    var result = PipelineExecutor().execute(plan)
"""

from std.memory import ArcPointer
from marrow.dtypes import Field
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr.values import AnyValue, col, resolve_columns


# ---------------------------------------------------------------------------
# Relation node kind constants
# ---------------------------------------------------------------------------

comptime SCAN_NODE: UInt8 = 0
comptime FILTER_NODE: UInt8 = 1
comptime PROJECT_NODE: UInt8 = 2
comptime IN_MEMORY_TABLE_NODE: UInt8 = 3


# ---------------------------------------------------------------------------
# Relation trait — interface every relational plan node must implement
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDestructible, Movable):
    """Interface for immutable relational plan nodes."""

    fn kind(self) -> UInt8:
        """Return the node-kind constant."""
        ...

    fn schema(self) -> Schema:
        """Return the output schema produced by this plan node."""
        ...

    fn inputs(self) -> List[AnyRelation]:
        """Return child plan nodes (empty for leaf nodes such as Scan)."""
        ...

    fn exprs(self) -> List[AnyValue]:
        """Return scalar expressions attached to this node."""
        ...

    fn write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...


# ---------------------------------------------------------------------------
# AnyRelation — type-erased, ArcPointer-backed relational plan container
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased relational plan node.

    Wraps any ``Relation``-conforming type on the heap behind an
    ``ArcPointer`` so copies are O(1) ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_kind: fn(ArcPointer[NoneType]) -> UInt8
    var _virt_schema: fn(ArcPointer[NoneType]) -> Schema
    var _virt_inputs: fn(ArcPointer[NoneType]) -> List[AnyRelation]
    var _virt_exprs: fn(ArcPointer[NoneType]) -> List[AnyValue]
    var _virt_write_to_string: fn(ArcPointer[NoneType]) -> String
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_kind[T: Relation](ptr: ArcPointer[NoneType]) -> UInt8:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    fn _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    fn _tramp_inputs[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> List[AnyRelation]:
        return rebind[ArcPointer[T]](ptr)[].inputs()

    @staticmethod
    fn _tramp_exprs[T: Relation](ptr: ArcPointer[NoneType]) -> List[AnyValue]:
        return rebind[ArcPointer[T]](ptr)[].exprs()

    @staticmethod
    fn _tramp_write_to_string[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    fn _tramp_drop[T: Relation](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    fn __init__[T: Relation](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_kind = Self._tramp_kind[T]
        self._virt_schema = Self._tramp_schema[T]
        self._virt_inputs = Self._tramp_inputs[T]
        self._virt_exprs = Self._tramp_exprs[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_kind = copy._virt_kind
        self._virt_schema = copy._virt_schema
        self._virt_inputs = copy._virt_inputs
        self._virt_exprs = copy._virt_exprs
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop

    # --- public API ---

    fn kind(self) -> UInt8:
        return self._virt_kind(self._data)

    fn schema(self) -> Schema:
        return self._virt_schema(self._data)

    fn inputs(self) -> List[AnyRelation]:
        return self._virt_inputs(self._data)

    fn exprs(self) -> List[AnyValue]:
        return self._virt_exprs(self._data)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    # --- plan-building API ---

    fn select(self, *names: String) raises -> AnyRelation:
        """Project columns by name, returning a new plan node."""
        var schema = self.schema()
        var col_names = List[String]()
        var exprs = List[AnyValue]()
        var fields = List[Field]()
        for i in range(len(names)):
            var name = names[i]
            var idx = schema.get_field_index(name)
            if idx == -1:
                raise Error("select: column '" + name + "' not found")
            col_names.append(name)
            exprs.append(col(idx))
            fields.append(schema.fields[idx])
        var out_schema = Schema(fields=fields^)
        var proj = Project(
            input=self,
            names=col_names^,
            exprs_=exprs^,
            schema_=out_schema,
        )
        return AnyRelation(proj^)

    fn filter(self, predicate: AnyValue) raises -> AnyRelation:
        """Filter rows by a boolean predicate, returning a new plan node.

        Column references using ``col("name")`` are resolved to positional
        indices against this node's output schema.
        """
        var schema = self.schema()
        var resolved = resolve_columns(predicate, schema)
        var filt = Filter(input=self, predicate=resolved^)
        return AnyRelation(filt^)

    # --- downcast ---

    fn downcast[T: Relation](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())


    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Concrete relation nodes
# ---------------------------------------------------------------------------


struct Scan(Relation):
    """Table / array scan — leaf node with no child plans.

    ``name``    — identifier for the data source.
    ``schema_`` — output schema of this scan.
    """

    var name: String
    var schema_: Schema

    fn __init__(out self, *, var name: String, var schema_: Schema):
        self.name = name^
        self.schema_ = schema_^

    fn kind(self) -> UInt8:
        return SCAN_NODE

    fn schema(self) -> Schema:
        return Schema(copy=self.schema_)

    fn inputs(self) -> List[AnyRelation]:
        return List[AnyRelation]()

    fn exprs(self) -> List[AnyValue]:
        return List[AnyValue]()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Scan({self.name})")


struct Filter(Relation):
    """Filter — apply a boolean predicate to the input relation.

    ``input``     — child relation.
    ``predicate`` — boolean expression; rows where True are kept.

    Output schema equals the input schema.
    """

    var input: AnyRelation
    var predicate: AnyValue

    fn __init__(out self, *, var input: AnyRelation, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    fn kind(self) -> UInt8:
        return FILTER_NODE

    fn schema(self) -> Schema:
        return self.input.schema()

    fn inputs(self) -> List[AnyRelation]:
        var result = List[AnyRelation](capacity=1)
        result.append(self.input)
        return result^

    fn exprs(self) -> List[AnyValue]:
        var result = List[AnyValue](capacity=1)
        result.append(self.predicate)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Projection — evaluate a list of named expressions.

    ``input``   — child relation.
    ``names``   — output column names (parallel to ``exprs_``).
    ``exprs_``  — scalar expressions to evaluate.
    ``schema_`` — pre-computed output schema.
    """

    var input: AnyRelation
    var names: List[String]
    var exprs_: List[AnyValue]
    var schema_: Schema

    fn __init__(
        out self,
        *,
        var input: AnyRelation,
        var names: List[String],
        var exprs_: List[AnyValue],
        var schema_: Schema,
    ):
        self.input = input^
        self.names = names^
        self.exprs_ = exprs_^
        self.schema_ = schema_^

    fn kind(self) -> UInt8:
        return PROJECT_NODE

    fn schema(self) -> Schema:
        return Schema(copy=self.schema_)

    fn inputs(self) -> List[AnyRelation]:
        var result = List[AnyRelation](capacity=1)
        result.append(self.input)
        return result^

    fn exprs(self) -> List[AnyValue]:
        var result = List[AnyValue](capacity=len(self.exprs_))
        for ref e in self.exprs_:
            result.append(e)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Project([")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(t", ")
            writer.write(self.names[i])
            writer.write(t"=")
            self.exprs_[i].write_to(writer)
        writer.write(t"])")


struct InMemoryTable(Relation):
    """In-memory table — leaf node backed by a RecordBatch.

    Holds actual data in the plan tree. Created via ``in_memory_table(batch)``.
    """

    var batch: RecordBatch

    fn __init__(out self, *, batch: RecordBatch):
        self.batch = RecordBatch(copy=batch)

    fn kind(self) -> UInt8:
        return IN_MEMORY_TABLE_NODE

    fn schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    fn inputs(self) -> List[AnyRelation]:
        return List[AnyRelation]()

    fn exprs(self) -> List[AnyValue]:
        return List[AnyValue]()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"InMemoryTable(num_rows={self.batch.num_rows()}, schema={self.batch.schema})"
        )


# ---------------------------------------------------------------------------
# Free-standing factory
# ---------------------------------------------------------------------------


fn in_memory_table(batch: RecordBatch) -> AnyRelation:
    """Create a relation backed by an in-memory RecordBatch."""
    return InMemoryTable(batch=batch)
