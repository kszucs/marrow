"""Relational plans: a descriptive IR that opens into pull-based operators.

Two layers, cleanly separated:

- ``Relation`` nodes are **pure, immutable descriptions** (``kind``/``schema``/
  ``open``): they hold only their parameters and child relations, no execution
  state. ``AnyRelation`` erases them behind an ``ArcPointer``, so copying a plan
  is an O(1) share and the plan is a reusable, inspectable, rewritable template.
- ``Processor`` (``schema``/``pull``) is the executing layer, built by
  ``Relation.open(ctx)``; it owns *all* mutable state (scan offset, built hash
  index, grouper, child operators). ``AnyProcessor`` erases it and drives the
  pull loop (``collect``). Operators are single-use and move-only.

``execute(plan, ctx)`` opens the plan into a fresh operator tree and drains it,
so the plan itself is never mutated — run it repeatedly or concurrently.

Concrete nodes / operators: ``InMemoryTable``/``InMemoryTableProcessor``,
``Filter``/``FilterProcessor``, ``Project``/``ProjectProcessor``, ``Aggregate``/``AggregateProcessor``,
``Join``/``JoinProcessor``, ``ParquetScan``/``ParquetScanProcessor``, ``Scan`` (unbound leaf).

Plan-building API
-----------------
``AnyRelation.select(*names)``                   — project columns by name.
``AnyRelation.filter(pred)``                     — filter rows by predicate.
``AnyRelation.aggregate(keys, values, funcs)``   — grouped aggregation.
``AnyRelation.join(right, left_on, right_on)``   — hash join.
``in_memory_table(batch)`` / ``parquet_scan(path)`` — leaf sources.
``execute(plan)``                                — drain to a single RecordBatch.

Example
-------
    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = execute(plan)
"""

from std.memory import ArcPointer

from ..dtypes import AnyDataType, Field, int64, float64
from ..schema import Schema
from ..tabular import RecordBatch
from .values import AnyValue
from .dynamic import DynValue, col, LOAD
from .execution import (
    ExecutionContext,
    DEFAULT_MORSEL_SIZE,
    AnyProcessor,
    InMemoryTableProcessor,
    ParquetScanProcessor,
    FilterProcessor,
    ProjectProcessor,
    AggregateProcessor,
    JoinProcessor,
)
from ..kernels.join import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)


def _value_dtype(expr: DynValue, input_schema: Schema) -> Optional[AnyDataType]:
    """Best-effort dtype of an aggregated value expression: its static dtype, or
    the input column's dtype when it is a (bound) column reference; else None.
    """
    var dt = expr.dtype()
    if dt:
        return dt^
    if expr.kind() == LOAD:
        return Optional[AnyDataType](
            input_schema.fields[Int(expr.kind_data())].dtype.copy()
        )
    return None


# ---------------------------------------------------------------------------
# Relation trait — the descriptive IR node (pure data; no execution state)
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDeletable, Movable):
    """A relational plan node: a pure, immutable description of an operation.

    Nodes hold only their parameters and child relations — no execution state.
    ``open(ctx)`` builds the stateful ``Processor`` that runs (opening children
    recursively), so a plan is a reusable template you can inspect, copy cheaply
    (O(1) — nodes are immutable and shared), and rewrite."""

    def schema(self) -> Schema:
        ...

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        """Build the physical operator for this node (opening its children)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        ...


# ---------------------------------------------------------------------------
# AnyRelation — type-erased IR container + plan-building API
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased plan node behind an ``ArcPointer``.

    Nodes are immutable descriptions, so copying an ``AnyRelation`` is an O(1)
    ``ArcPointer`` share — no deep clone, no reset. ``open(ctx)`` builds the
    operator tree that executes; the plan is never mutated, so it is a reusable
    template and copies never share execution state. Carries the plan-building
    API (``select``/``filter``/``aggregate``/``join``)."""

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_open: def(
        ArcPointer[NoneType], ExecutionContext
    ) thin raises -> AnyProcessor
    var _virt_write_to_string: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_open[
        T: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecutionContext) raises -> AnyProcessor:
        return rebind[ArcPointer[T]](ptr)[].open(ctx)

    @staticmethod
    def _tramp_write_to_string[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    def _tramp_drop[T: Relation](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: Relation](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._tramp_schema[T]
        self._virt_open = Self._tramp_open[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        # O(1) share — nodes are immutable, so aliasing is safe.
        self._data = copy._data
        self._virt_schema = copy._virt_schema
        self._virt_open = copy._virt_open
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop

    def __del__(deinit self):
        self._virt_drop(self._data^)

    # --- introspection ---

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    def downcast[T: Relation](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    # --- execution ---

    def open(
        self, ctx: ExecutionContext = ExecutionContext()
    ) raises -> AnyProcessor:
        """Build the operator tree for this plan; the plan is left untouched."""
        return self._virt_open(self._data, ctx)

    # --- plan-building API ---

    def select(self, *names: String) raises -> AnyRelation:
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
            exprs.append(AnyValue(col(idx)))
            fields.append(schema.fields[idx].copy())
        return AnyRelation(
            Project(
                input=self,
                names=col_names^,
                values=exprs^,
                schema=Schema(fields=fields^),
            )
        )

    def filter(self, var predicate: AnyValue) raises -> AnyRelation:
        """Filter rows by a boolean predicate. Column references resolve by name
        against the batch schema when the boxed value executes."""
        return AnyRelation(Filter(input=self, predicate=predicate^))

    def aggregate(
        self,
        keys: List[DynValue],
        values: List[DynValue],
        funcs: List[String],
    ) raises -> AnyRelation:
        """Grouped aggregation: key columns + aggregated value columns."""
        from marrow.dtypes import int64

        var input_schema = self.schema()

        # Bind key/value expressions to positional form once (names -> indices),
        # so per-morsel eval and the dtype lookups below use positions directly.
        var resolved_keys = List[DynValue]()
        for ref k in keys:
            resolved_keys.append(k.resolve_names(input_schema))
        var resolved_values = List[DynValue]()
        for ref v in values:
            resolved_values.append(v.resolve_names(input_schema))

        # Output schema: one field per key (named after its source column, with
        # that column's dtype) then one field per aggregate (named after its
        # function).
        var fields = List[Field]()
        for i in range(len(resolved_keys)):
            ref k = resolved_keys[i]
            if k.kind() == LOAD:
                ref src = input_schema.fields[Int(k.kind_data())]
                fields.append(Field(src.name, src.dtype.copy()))
            else:
                var kdt = k.dtype()
                if not kdt:
                    raise Error(
                        "aggregate: cannot infer dtype for computed key "
                        + String(i)
                    )
                fields.append(Field("key" + String(i), kdt.value().copy()))
        for i in range(len(funcs)):
            if funcs[i] == "count":
                fields.append(Field(funcs[i], AnyDataType(int64)))
            elif funcs[i] == "mean":
                fields.append(Field(funcs[i], AnyDataType(float64)))
            else:
                var maybe_dt = _value_dtype(resolved_values[i], input_schema)
                if maybe_dt and maybe_dt.value().is_integer():
                    fields.append(Field(funcs[i], AnyDataType(int64)))
                else:
                    fields.append(Field(funcs[i], AnyDataType(float64)))
        var out_schema = Schema(fields=fields^)

        # Key struct fields (first len(keys) output fields) + value accumulator
        # dtypes (the input dtype of each aggregated value expression).
        var key_fields = List[Field]()
        for i in range(len(resolved_keys)):
            key_fields.append(out_schema.fields[i].copy())
        var value_dtypes = List[AnyDataType]()
        for i in range(len(resolved_values)):
            var dt = _value_dtype(resolved_values[i], input_schema)
            if dt:
                value_dtypes.append(dt.value().copy())
            else:
                value_dtypes.append(AnyDataType(float64))

        var key_exprs = List[AnyValue]()
        for ref k in resolved_keys:
            key_exprs.append(AnyValue(k.copy()))
        var val_exprs = List[AnyValue]()
        for ref v in resolved_values:
            val_exprs.append(AnyValue(v.copy()))

        return AnyRelation(
            Aggregate(
                input=self,
                keys=key_exprs^,
                agg_exprs=val_exprs^,
                funcs=funcs.copy(),
                value_dtypes=value_dtypes^,
                key_fields=key_fields^,
                schema=out_schema,
            )
        )

    def join(
        self,
        right: AnyRelation,
        left_on: List[DynValue],
        right_on: List[DynValue],
        how: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> AnyRelation:
        """Hash join on equijoin key expressions."""
        if len(left_on) != len(right_on):
            raise Error("join: len(left_on) != len(right_on)")

        var left_schema = self.schema()
        var right_schema = right.schema()

        # Resolve key expressions to positional column indices (each key must be
        # a bare column reference — column_index raises otherwise).
        var left_indices = List[Int]()
        for ref k in left_on:
            left_indices.append(k.column_index(left_schema))
        var right_indices = List[Int]()
        for ref k in right_on:
            right_indices.append(k.column_index(right_schema))

        # Output schema: left columns + (suffixed) right columns.
        var fields = List[Field]()
        for ref f in left_schema.fields:
            fields.append(f.copy())
        if how != JOIN_SEMI and how != JOIN_ANTI:
            var left_names = List[String]()
            for ref f in left_schema.fields:
                left_names.append(f.name)
            for ref f in right_schema.fields:
                var name = f.name
                var collides = False
                for ref ln in left_names:
                    if ln == name:
                        collides = True
                        break
                if collides:
                    name = name + "_right"
                fields.append(Field(name, f.dtype.copy()))

        return AnyRelation(
            Join(
                left=self,
                right=right,
                left_key_indices=left_indices^,
                right_key_indices=right_indices^,
                join_kind=how,
                strictness=strictness,
                schema=Schema(fields=fields^),
            )
        )


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------


struct Scan(Relation):
    """Unbound named scan — a leaf with no data (cannot execute directly)."""

    var name: String
    var _schema: Schema

    def __init__(out self, *, var name: String, var schema: Schema):
        self.name = name^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        raise Error("Scan requires external data source binding")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Scan({self.name})")


struct InMemoryTable(Relation):
    """Leaf backed by a RecordBatch; opens into a morsel-slicing operator."""

    var batch: RecordBatch
    var morsel_size: Int

    def __init__(
        out self, *, batch: RecordBatch, morsel_size: Int = DEFAULT_MORSEL_SIZE
    ):
        self.batch = RecordBatch(copy=batch)
        self.morsel_size = morsel_size

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return InMemoryTableProcessor(
            batch=RecordBatch(copy=self.batch),  # shares buffers (O(1))
            morsel_size=self.morsel_size,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"InMemoryTable(num_rows={self.batch.num_rows()},"
            t" schema={self.batch.schema})"
        )


def in_memory_table(batch: RecordBatch) -> AnyRelation:
    """Create a relation backed by an in-memory RecordBatch."""
    return InMemoryTable(batch=batch)


struct ParquetScan(Relation):
    """Leaf describing a Parquet file scan with a known schema."""

    var path: String
    var _schema: Schema
    var morsel_size: Int

    def __init__(
        out self,
        *,
        var path: String,
        var schema: Schema,
        morsel_size: Int = DEFAULT_MORSEL_SIZE,
    ):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return ParquetScanProcessor(
            path=self.path.copy(),
            schema=Schema(copy=self._schema),
            morsel_size=self.morsel_size,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"ParquetScan({self.path})")


def parquet_scan(path: String, schema: Schema) -> AnyRelation:
    """Create a Parquet file scan with a known schema."""
    return ParquetScan(path=path, schema=schema)


# ---------------------------------------------------------------------------
# Streaming operators
# ---------------------------------------------------------------------------


struct Filter(Relation):
    """Apply a boolean predicate; keep rows where True (schema unchanged)."""

    var input: AnyRelation
    var predicate: AnyValue

    def __init__(out self, *, var input: AnyRelation, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return FilterProcessor(
            input=self.input.open(ctx), predicate=self.predicate.copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Evaluate a list of named expressions into output columns."""

    var input: AnyRelation
    var names: List[String]
    var values: List[AnyValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var names: List[String],
        var values: List[AnyValue],
        var schema: Schema,
    ):
        self.input = input^
        self.names = names^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return ProjectProcessor(
            input=self.input.open(ctx),
            values=self.values.copy(),
            schema=Schema(copy=self._schema),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Project([")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(t", ")
            writer.write(self.names[i])
            writer.write(t"=")
            self.values[i].write_to(writer)
        writer.write(t"])")


# ---------------------------------------------------------------------------
# Blocking operators
# ---------------------------------------------------------------------------


struct Aggregate(Relation):
    """Grouped aggregation — the descriptive node (keys, aggregates, schema)."""

    var input: AnyRelation
    var keys: List[AnyValue]
    var agg_exprs: List[AnyValue]
    var funcs: List[String]
    var value_dtypes: List[AnyDataType]
    var key_fields: List[Field]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var keys: List[AnyValue],
        var agg_exprs: List[AnyValue],
        var funcs: List[String],
        var value_dtypes: List[AnyDataType],
        var key_fields: List[Field],
        var schema: Schema,
    ):
        self.input = input^
        self.keys = keys^
        self.agg_exprs = agg_exprs^
        self.funcs = funcs^
        self.value_dtypes = value_dtypes^
        self.key_fields = key_fields^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return AggregateProcessor(
            input=self.input.open(ctx),
            keys=self.keys.copy(),
            agg_exprs=self.agg_exprs.copy(),
            funcs=self.funcs.copy(),
            value_dtypes=self.value_dtypes.copy(),
            key_fields=self.key_fields.copy(),
            schema=Schema(copy=self._schema),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Aggregate(keys=[")
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(", ")
            self.keys[i].write_to(writer)
        writer.write("])")


struct Join(Relation):
    """Equijoin — the descriptive node (key indices, kind, output schema)."""

    var left: AnyRelation
    var right: AnyRelation
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: UInt8
    var strictness: UInt8
    var _schema: Schema

    def __init__(
        out self,
        *,
        var left: AnyRelation,
        var right: AnyRelation,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: UInt8,
        strictness: UInt8,
        var schema: Schema,
    ):
        self.left = left^
        self.right = right^
        self.left_key_indices = left_key_indices^
        self.right_key_indices = right_key_indices^
        self.join_kind = join_kind
        self.strictness = strictness
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyProcessor:
        return JoinProcessor(
            left=self.left.open(ctx),
            right=self.right.open(ctx),
            left_key_indices=self.left_key_indices.copy(),
            right_key_indices=self.right_key_indices.copy(),
            join_kind=self.join_kind,
            strictness=self.strictness,
            schema=Schema(copy=self._schema),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Join(kind={self.join_kind})")


# ---------------------------------------------------------------------------
# execute — drain a plan into a single RecordBatch
# ---------------------------------------------------------------------------


def execute(
    plan: AnyRelation, ctx: ExecutionContext = ExecutionContext()
) raises -> RecordBatch:
    """Execute a plan: open it into a fresh operator tree and drain it. The plan
    (a pure description) is never mutated, so it can be executed repeatedly and
    concurrently."""
    var op = plan.open(ctx)
    return op.collect()
