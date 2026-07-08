"""Relational plans: a descriptive IR that opens into pull-based operators.

Two layers, cleanly separated:

- ``Relation`` nodes are **pure, immutable descriptions** (``kind``/``schema``/
  ``open``): they hold only their parameters and child relations, no execution
  state. ``AnyRelation`` erases them behind an ``ArcPointer``, so copying a plan
  is an O(1) share and the plan is a reusable, inspectable, rewritable template.
- ``Operator`` (``schema``/``pull``) is the executing layer, built by
  ``Relation.open(ctx)``; it owns *all* mutable state (scan offset, built hash
  index, grouper, child operators). ``AnyOperator`` erases it and drives the
  pull loop (``collect``). Operators are single-use and move-only.

``execute(plan, ctx)`` opens the plan into a fresh operator tree and drains it,
so the plan itself is never mutated — run it repeatedly or concurrently.

Concrete nodes / operators: ``InMemoryTable``/``InMemoryTableOp``,
``Filter``/``FilterOp``, ``Project``/``ProjectOp``, ``Aggregate``/``AggregateOp``,
``Join``/``JoinOp``, ``ParquetScan``/``ParquetScanOp``, ``Scan`` (unbound leaf).

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
from std.gpu.host import DeviceContext
from std.reflection import reflect

import marrow.dtypes as dt
from marrow.arrays import AnyArray, StringArray, StructArray
from marrow.builders import PrimitiveBuilder, BoolBuilder, StringBuilder
from marrow.dtypes import (
    AnyDataType,
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
    struct_,
)
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.kernels.concat import concat
from marrow.kernels.filter import filter
from marrow.kernels.groupby import HashGrouper
from marrow.kernels.join import HashJoin
from marrow.kernels.hashing import rapidhash
from marrow.parquet import read_table
from marrow.expr.values import AnyValue
from marrow.expr.dynamic import DynValue, col, LOAD
from marrow.expr.values import NumericValue, StringValue
from marrow.kernels.join import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)


# ---------------------------------------------------------------------------
# Relation node kind constants
# ---------------------------------------------------------------------------

comptime SCAN_NODE: UInt8 = 0
comptime FILTER_NODE: UInt8 = 1
comptime PROJECT_NODE: UInt8 = 2
comptime IN_MEMORY_TABLE_NODE: UInt8 = 3
comptime PARQUET_SCAN_NODE: UInt8 = 4
comptime AGGREGATE_NODE: UInt8 = 5
comptime JOIN_NODE: UInt8 = 6

comptime DEFAULT_MORSEL_SIZE: Int = 65_536


# ---------------------------------------------------------------------------
# Exhausted — signals a node has no more morsels
# ---------------------------------------------------------------------------


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by ``pull()`` when a node has no more batches to yield."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


# ---------------------------------------------------------------------------
# ExecutionContext — runtime configuration (morsel size, optional GPU)
# ---------------------------------------------------------------------------


struct ExecutionContext(Copyable, ImplicitlyCopyable, Movable):
    """Runtime configuration for query execution."""

    var device_ctx: Optional[DeviceContext]
    var num_cpu_workers: Int
    var morsel_size: Int
    var gpu_threshold: Int

    def __init__(out self):
        self.device_ctx = None
        self.num_cpu_workers = 0
        self.morsel_size = DEFAULT_MORSEL_SIZE
        self.gpu_threshold = 1_000_000

    def __init__(out self, ctx: DeviceContext, gpu_threshold: Int = 1_000_000):
        self.device_ctx = ctx
        self.num_cpu_workers = 0
        self.morsel_size = DEFAULT_MORSEL_SIZE
        self.gpu_threshold = gpu_threshold


# ---------------------------------------------------------------------------
# _concat — a CLOSED, flat-only column concat for collect()
# ---------------------------------------------------------------------------
#
# `collect()` merges morsels by concatenating each column. The general
# `marrow.kernels.concat` routes through `AnyBuilder(dtype)` — an open switch
# that instantiates a builder for *every* dtype (incl. nested) whenever it is
# reachable, so it is never DCE'd and inflates the binary ~6x (measured). The
# relational layer's projections produce only flat columns, so `collect()` uses
# this local concat instead: typed builders for primitive/bool/string and a
# `raise` otherwise (like `filter`), keeping the fused path closed and small.


def _concat_primitive[
    T: PrimitiveType
](arrays: List[AnyArray]) raises -> AnyArray:
    var total = 0
    for ref a in arrays:
        total += a.length()
    var builder = PrimitiveBuilder[T](
        arrays[0].as_primitive[T]().dtype.copy(), total
    )
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
        raise Error("collect: unsupported column dtype ", dtype)


def _value_dtype(
    expr: DynValue, input_schema: Schema
) -> Optional[AnyDataType]:
    """Best-effort dtype of an aggregated value expression: its static dtype, or
    the input column's dtype when it is a (bound) column reference; else None."""
    var dt = expr.dtype()
    if dt:
        return dt^
    if expr.kind() == LOAD:
        return Optional[AnyDataType](
            input_schema.fields[Int(expr.kind_data())].dtype.copy()
        )
    return None


# ---------------------------------------------------------------------------
# Operators — the execution layer (own all mutable state; built from IR nodes)
# ---------------------------------------------------------------------------


trait Operator(ImplicitlyDeletable, Movable):
    """The executing counterpart of a ``Relation`` node.

    Created by ``Relation.open(ctx)``; owns all mutable execution state — scan
    offset, built hash index, grouper, child operators. ``pull()`` yields the
    next morsel or raises ``Exhausted``. Operators are single-use and move-only;
    the descriptive plan they were opened from is never touched."""

    def schema(self) -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        """Yield the next morsel, or raise ``Exhausted`` when done."""
        ...


struct AnyOperator(Movable):
    """Type-erased, move-only operator behind an ``ArcPointer``; also the pull
    driver (``collect()`` drains it into one ``RecordBatch``)."""

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_schema[T: Operator](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_pull[
        T: Operator
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

    @staticmethod
    def _tramp_drop[T: Operator](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: Operator](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._tramp_schema[T]
        self._virt_pull = Self._tramp_pull[T]
        self._virt_drop = Self._tramp_drop[T]

    def __del__(deinit self):
        self._virt_drop(self._data^)

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        return self._virt_pull(self._data)

    def collect(mut self) raises -> RecordBatch:
        """Drain all morsels into a single RecordBatch."""
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


# ---------------------------------------------------------------------------
# Relation trait — the descriptive IR node (pure data; no execution state)
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDeletable, Movable):
    """A relational plan node: a pure, immutable description of an operation.

    Nodes hold only their parameters and child relations — no execution state.
    ``open(ctx)`` builds the stateful ``Operator`` that runs (opening children
    recursively), so a plan is a reusable template you can inspect, copy cheaply
    (O(1) — nodes are immutable and shared), and rewrite."""

    def kind(self) -> UInt8:
        ...

    def schema(self) -> Schema:
        ...

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        """Build the physical operator for this node (opening its children)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        ...


# ---------------------------------------------------------------------------
# AnyRelation — type-erased container + streaming driver + plan-building API
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased plan node behind an ``ArcPointer``.

    Nodes are immutable descriptions, so copying an ``AnyRelation`` is an O(1)
    ``ArcPointer`` share — no deep clone, no reset. ``open(ctx)`` builds the
    operator tree that executes; the plan is never mutated, so it is a reusable
    template and copies never share execution state. Carries the plan-building
    API (``select``/``filter``/``aggregate``/``join``)."""

    var _data: ArcPointer[NoneType]
    var _virt_kind: def(ArcPointer[NoneType]) thin -> UInt8
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_open: def(
        ArcPointer[NoneType], ExecutionContext
    ) thin raises -> AnyOperator
    var _virt_write_to_string: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_kind[T: Relation](ptr: ArcPointer[NoneType]) -> UInt8:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    def _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_open[
        T: Relation
    ](ptr: ArcPointer[NoneType], ctx: ExecutionContext) raises -> AnyOperator:
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
        self._virt_kind = Self._tramp_kind[T]
        self._virt_schema = Self._tramp_schema[T]
        self._virt_open = Self._tramp_open[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        # O(1) share — nodes are immutable, so aliasing is safe.
        self._data = copy._data
        self._virt_kind = copy._virt_kind
        self._virt_schema = copy._virt_schema
        self._virt_open = copy._virt_open
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop

    def __del__(deinit self):
        self._virt_drop(self._data^)

    # --- introspection ---

    def kind(self) -> UInt8:
        return self._virt_kind(self._data)

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    def downcast[T: Relation](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    # --- execution ---

    def open(
        self, ctx: ExecutionContext = ExecutionContext()
    ) raises -> AnyOperator:
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

    def kind(self) -> UInt8:
        return SCAN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
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

    def kind(self) -> UInt8:
        return IN_MEMORY_TABLE_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return InMemoryTableOp(
            batch=RecordBatch(copy=self.batch),  # shares buffers (O(1))
            morsel_size=self.morsel_size,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"InMemoryTable(num_rows={self.batch.num_rows()},"
            t" schema={self.batch.schema})"
        )


struct InMemoryTableOp(Operator):
    """Yields morsel-sized slices of an in-memory batch."""

    var batch: RecordBatch
    var morsel_size: Int
    var _offset: Int

    def __init__(out self, *, var batch: RecordBatch, morsel_size: Int):
        self.batch = batch^
        self.morsel_size = morsel_size
        self._offset = 0

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def pull(mut self) raises -> RecordBatch:
        if self._offset >= self.batch.num_rows():
            raise Exhausted()
        var remaining = self.batch.num_rows() - self._offset
        var length = self.morsel_size if self.morsel_size < remaining else remaining
        var result = self.batch.slice(self._offset, length)
        self._offset += length
        return result^


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

    def kind(self) -> UInt8:
        return PARQUET_SCAN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return ParquetScanOp(
            path=self.path.copy(),
            schema=Schema(copy=self._schema),
            morsel_size=self.morsel_size,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"ParquetScan({self.path})")


struct ParquetScanOp(Operator):
    """Reads the Parquet file on first pull, then yields morsels."""

    var path: String
    var _schema: Schema
    var morsel_size: Int
    var _batch: Optional[RecordBatch]
    var _offset: Int

    def __init__(out self, *, var path: String, var schema: Schema, morsel_size: Int):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size
        self._batch = None
        self._offset = 0

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        if not self._batch:
            var table = read_table(self.path)
            var batches = table.to_batches()
            if len(batches) == 0:
                self._batch = RecordBatch(
                    schema=table.schema, columns=List[AnyArray]()
                )
            elif len(batches) == 1:
                self._batch = RecordBatch(copy=batches[0])
            else:
                var num_cols = batches[0].num_columns()
                var cols = List[AnyArray](capacity=num_cols)
                for c in range(num_cols):
                    var col_arrays = List[AnyArray](capacity=len(batches))
                    for b in range(len(batches)):
                        col_arrays.append(batches[b].columns[c].copy())
                    cols.append(concat(col_arrays))
                self._batch = RecordBatch(
                    schema=Schema(copy=batches[0].schema), columns=cols^
                )
        ref batch = self._batch.value()
        if self._offset >= batch.num_rows():
            raise Exhausted()
        var remaining = batch.num_rows() - self._offset
        var length = self.morsel_size if self.morsel_size < remaining else remaining
        var result = batch.slice(self._offset, length)
        self._offset += length
        return result^


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

    def kind(self) -> UInt8:
        return FILTER_NODE

    def schema(self) -> Schema:
        return self.input.schema()

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return FilterOp(
            input=self.input.open(ctx), predicate=self.predicate.copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct FilterOp(Operator):
    """Keeps rows where the predicate is True; skips empty morsels."""

    var input: AnyOperator
    var predicate: AnyValue

    def __init__(out self, *, var input: AnyOperator, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def pull(mut self) raises -> RecordBatch:
        # Skip morsels that filter to 0 rows; Exhausted propagates from input.
        while True:
            var batch = self.input.pull()
            var mask = self.predicate.to_array(batch)
            var cols = List[AnyArray]()
            for i in range(batch.num_columns()):
                cols.append(filter(batch.columns[i].copy(), mask.copy()))
            var result = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if result.num_rows() > 0:
                return result^


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

    def kind(self) -> UInt8:
        return PROJECT_NODE

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return ProjectOp(
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


struct ProjectOp(Operator):
    """Evaluates each projected value against every input morsel."""

    var input: AnyOperator
    var values: List[AnyValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyOperator,
        var values: List[AnyValue],
        var schema: Schema,
    ):
        self.input = input^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        var batch = self.input.pull()  # raises Exhausted when done
        var cols = List[AnyArray]()
        for ref v in self.values:
            cols.append(v.to_array(batch))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


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

    def kind(self) -> UInt8:
        return AGGREGATE_NODE

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return AggregateOp(
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


struct AggregateOp(Operator):
    """Blocking: consume all input into a grouper, then emit once."""

    var input: AnyOperator
    var keys: List[AnyValue]
    var agg_exprs: List[AnyValue]
    var key_fields: List[Field]
    var _schema: Schema
    var _grouper: HashGrouper
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: AnyOperator,
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
        self.key_fields = key_fields^
        self._schema = schema^
        self._grouper = HashGrouper(funcs^, value_dtypes^)
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        while True:
            try:
                var batch = self.input.pull()
                var key_children = List[AnyArray]()
                var key_struct_fields = List[Field]()
                for i in range(len(self.keys)):
                    key_children.append(self.keys[i].to_array(batch))
                    key_struct_fields.append(self.key_fields[i].copy())
                var key_struct = StructArray(
                    dtype=struct_(key_struct_fields^),
                    length=batch.num_rows(),
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    children=key_children^,
                )
                var gids = self._grouper.consume_keys(key_struct)
                var val_arrays = List[AnyArray]()
                for i in range(len(self.agg_exprs)):
                    val_arrays.append(self.agg_exprs[i].to_array(batch))
                self._grouper.consume_values(gids, val_arrays)
            except Exhausted:
                break
        self._emitted = True
        # The grouper generates its own aggregate column names (col{i}_{func}).
        # Re-label with the plan's declared schema so plan.schema() matches the
        # executed output exactly (columns are in the same key-then-agg order).
        var raw = self._grouper.finish(self.key_fields)
        var cols = List[AnyArray]()
        for ref c in raw.columns:
            cols.append(c.copy())
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


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

    def kind(self) -> UInt8:
        return JOIN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def open(self, ctx: ExecutionContext) raises -> AnyOperator:
        return JoinOp(
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


struct JoinOp(Operator):
    """Builds the left side fully on first pull, then streams the right side."""

    var left: AnyOperator
    var right: AnyOperator
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: UInt8
    var strictness: UInt8
    var _schema: Schema
    var _index: Optional[HashJoin[rapidhash]]
    var _exhausted: Bool

    def __init__(
        out self,
        *,
        var left: AnyOperator,
        var right: AnyOperator,
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
        self._index = None
        self._exhausted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        if self._exhausted:
            raise Exhausted()
        if not self._index:
            var left_struct = self.left.collect().to_struct_array()
            var index = HashJoin[rapidhash]()
            index.build(left_struct, self.left_key_indices)
            self._index = index^
        try:
            var right_morsel = self.right.pull()
            var result = self._index.value().probe(
                right_morsel.to_struct_array(),
                self.right_key_indices,
                self.join_kind,
                self.strictness,
            )
            return RecordBatch(
                schema=self._schema.copy(), columns=result.children.copy()
            )
        except Exhausted:
            self._exhausted = True
        raise Exhausted()


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


# ===========================================================================
# Name-resolved column handles — Table[T]() / col() produce these fused leaves
# ===========================================================================


struct NumericColumn[T: dt.NumericType](NumericValue):
    """Named typed numeric column reference — carries only its ``name`` (runtime
    field); the type parameter is just the dtype that drives the SIMD ``core``.
    The position is resolved by name against ``batch.schema`` at execution. Built
    by ``Table[Tbl]()`` and ``col(name, dtype)``, never directly.

    ``to_array`` comes from ``NumericValue``'s default; only ``field_name`` is
    overridden here (columns are named, unlike anonymous computed values)."""

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def field_name(self) -> String:
        return self.name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", self.name, "]")


struct StringColumn(StringValue):
    """Named typed string column reference — the string counterpart of
    ``NumericColumn[T]`` (one type across all string columns; position resolved
    by name). ``to_array`` comes from ``StringValue``'s default."""

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    def resolve(self, batch: RecordBatch) -> StringArray:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_string()
            .copy()
        )

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def field_name(self) -> String:
        return self.name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", self.name, "]")


struct Table[T: AnyType](Copyable, Movable):
    """Column-access handle over a plain schema struct — ``Table[Orders]()``.

    ``T`` is any struct of plain dtype-tag fields (``var a: Int64Type``).
    ``t.a`` reflects field ``a``'s dtype on ``T`` at compile time
    (``reflect[T].field[name].T``) to pick the column type; the position is
    resolved by name at execution. A companion handle is required because
    ``T``'s own fields shadow ``__getattr_param__``; ``T`` is never instantiated
    (only reflected). The two overloads route numeric/string fields to
    ``NumericColumn``/``StringColumn`` via a ``where`` clause the constraint
    solver can prove (the reflection query folds to a builtin KGEN attribute)."""

    comptime _dtype[name: StringLiteral] = reflect[Self.T].field[name].T

    def __init__(out self):
        pass

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> NumericColumn[Self._dtype[name]] where conforms_to(
        Self._dtype[name], dt.NumericType
    ):
        return NumericColumn[Self._dtype[name]](String(name))

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> StringColumn where conforms_to(
        Self._dtype[name], dt.StringLikeType
    ):
        return StringColumn(String(name))
