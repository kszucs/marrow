"""Relational plan = execution: self-executing (fat) relation nodes.

Each node is both the logical plan node *and* its own pull-based executor —
there is no separate ``Planner``/``*Processor`` hierarchy. ``pull()`` yields
morsel-sized ``RecordBatch`` values (raising ``Exhausted`` when done); a node
drives its children by pulling from them. This folds the former
``marrow.expr.executor`` processors onto the nodes.

``Relation``    — the trait every node implements (``kind``/``schema``/``pull``).
``AnyRelation`` — the type-erased, ArcPointer-backed container; also the
                  streaming driver (``pull``/``collect``) and plan-building API.

Concrete nodes: ``InMemoryTable``, ``Filter``, ``Project``, ``Aggregate``,
``Join``, ``ParquetScan``, ``Scan`` (unbound leaf).

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

Nodes carry execution state (scan offset, built hash index, ...), so a plan is
single-use per ``execute()`` — build it, run it once.
"""

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from marrow.arrays import AnyArray, StructArray
from marrow.dtypes import AnyDataType, Field, float64, struct_
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.kernels.concat import concat
from marrow.kernels.filter import filter
from marrow.kernels.groupby import HashGrouper
from marrow.kernels.join import HashJoin
from marrow.kernels.hashing import rapidhash
from marrow.parquet import read_table
from marrow.expr.erased import AnyValue
from marrow.expr.runtime import DynValue, col, LOAD
from marrow.kernels.join import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_MARK,
    JOIN_SINGLE,
    JOIN_ALL,
    JOIN_ANY,
    JOIN_ASOF,
    JOIN_ALGO_AUTO,
    JOIN_ALGO_HASH,
    JOIN_ALGO_SORT_MERGE,
    JOIN_ALGO_PIECEWISE,
    JOIN_ALGO_GRACE_HASH,
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
# Relation trait — node = structure + self-execution
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDeletable, Movable):
    """A relational node that describes itself and executes itself."""

    def kind(self) -> UInt8:
        ...

    def schema(self) -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        """Yield the next morsel, or raise ``Exhausted`` when done."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        ...


# ---------------------------------------------------------------------------
# AnyRelation — type-erased container + streaming driver + plan-building API
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased relational node behind an ``ArcPointer`` (O(1) copies).

    Also the pull driver: ``pull()``/``collect()`` fold in the former
    ``AnyRelationProcessor``."""

    var _data: ArcPointer[NoneType]
    var _virt_kind: def(ArcPointer[NoneType]) thin -> UInt8
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch
    var _virt_write_to_string: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_kind[T: Relation](ptr: ArcPointer[NoneType]) -> UInt8:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    def _tramp_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_pull[
        T: Relation
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

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
        self._virt_pull = Self._tramp_pull[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_kind = copy._virt_kind
        self._virt_schema = copy._virt_schema
        self._virt_pull = copy._virt_pull
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

    # --- streaming driver ---

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
            result_cols.append(concat(col_arrays))
        return RecordBatch(schema=Schema(copy=schema), columns=result_cols^)

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
                exprs_=exprs^,
                schema_=Schema(fields=fields^),
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

        # Output schema: key fields + agg result fields.
        var fields = List[Field]()
        for ref k in keys:
            var kdt = k.dtype()
            if kdt:
                fields.append(Field("key", kdt.value().copy()))
            else:
                fields.append(Field("key", input_schema.fields[0].dtype.copy()))
        for i in range(len(funcs)):
            if funcs[i] == "count":
                fields.append(Field(funcs[i], AnyDataType(int64)))
            elif funcs[i] == "mean":
                fields.append(Field(funcs[i], AnyDataType(float64)))
            else:
                var maybe_dt = values[i].dtype()
                if maybe_dt and maybe_dt.value().is_integer():
                    fields.append(Field(funcs[i], AnyDataType(int64)))
                else:
                    fields.append(Field(funcs[i], AnyDataType(float64)))
        var out_schema = Schema(fields=fields^)

        # Key struct fields (first len(keys) output fields) + value accumulator
        # dtypes (resolve LOAD nodes against the input schema).
        var key_fields = List[Field]()
        for i in range(len(keys)):
            key_fields.append(out_schema.fields[i].copy())
        var value_dtypes = List[AnyDataType]()
        for i in range(len(values)):
            ref agg_expr = values[i]
            var dt = agg_expr.dtype()
            if not dt and agg_expr.kind() == LOAD:
                dt = Optional[AnyDataType](
                    input_schema.fields[Int(agg_expr.kind_data())].dtype.copy()
                )
            if dt:
                value_dtypes.append(dt.value().copy())
            else:
                value_dtypes.append(AnyDataType(float64))

        var key_exprs = List[AnyValue]()
        for ref k in keys:
            key_exprs.append(AnyValue(k.copy()))
        var val_exprs = List[AnyValue]()
        for ref v in values:
            val_exprs.append(AnyValue(v.copy()))

        return AnyRelation(
            Aggregate(
                input=self,
                keys=key_exprs^,
                agg_exprs=val_exprs^,
                grouper=HashGrouper(funcs.copy(), value_dtypes^),
                key_fields=key_fields^,
                schema_=out_schema,
            )
        )

    def join(
        self,
        right: AnyRelation,
        left_on: List[DynValue],
        right_on: List[DynValue],
        how: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
        algorithm: UInt8 = JOIN_ALGO_AUTO,
    ) raises -> AnyRelation:
        """Hash join on equijoin key expressions."""
        if len(left_on) != len(right_on):
            raise Error("join: len(left_on) != len(right_on)")

        var left_schema = self.schema()
        var right_schema = right.schema()

        # Resolve key expressions to positional column indices.
        var left_indices = List[Int]()
        for ref k in left_on:
            left_indices.append(Int(k.resolve_names(left_schema)._kind_data))
        var right_indices = List[Int]()
        for ref k in right_on:
            right_indices.append(Int(k.resolve_names(right_schema)._kind_data))

        # Output schema: left columns + (suffixed) right columns.
        var fields = List[Field]()
        for ref f in left_schema.fields:
            fields.append(f.copy())
        if how != JOIN_SEMI and how != JOIN_ANTI and how != JOIN_MARK:
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
                schema_=Schema(fields=fields^),
            )
        )


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------


struct Scan(Relation):
    """Unbound named scan — a leaf with no data (cannot execute directly)."""

    var name: String
    var schema_: Schema

    def __init__(out self, *, var name: String, var schema_: Schema):
        self.name = name^
        self.schema_ = schema_^

    def kind(self) -> UInt8:
        return SCAN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.schema_)

    def pull(mut self) raises -> RecordBatch:
        raise Error("Scan requires external data source binding")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Scan({self.name})")


struct InMemoryTable(Relation):
    """Leaf backed by a RecordBatch; yields morsel-sized slices."""

    var batch: RecordBatch
    var morsel_size: Int
    var _offset: Int

    def __init__(
        out self, *, batch: RecordBatch, morsel_size: Int = DEFAULT_MORSEL_SIZE
    ):
        self.batch = RecordBatch(copy=batch)
        self.morsel_size = morsel_size
        self._offset = 0

    def kind(self) -> UInt8:
        return IN_MEMORY_TABLE_NODE

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

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            t"InMemoryTable(num_rows={self.batch.num_rows()},"
            t" schema={self.batch.schema})"
        )


def in_memory_table(batch: RecordBatch) -> AnyRelation:
    """Create a relation backed by an in-memory RecordBatch."""
    return InMemoryTable(batch=batch)


struct ParquetScan(Relation):
    """Leaf that reads a Parquet file on first pull, then yields morsels."""

    var path: String
    var schema_: Schema
    var morsel_size: Int
    var _batch: Optional[RecordBatch]
    var _offset: Int

    def __init__(
        out self,
        *,
        var path: String,
        var schema_: Schema,
        morsel_size: Int = DEFAULT_MORSEL_SIZE,
    ):
        self.path = path^
        self.schema_ = schema_^
        self.morsel_size = morsel_size
        self._batch = None
        self._offset = 0

    def kind(self) -> UInt8:
        return PARQUET_SCAN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.schema_)

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

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"ParquetScan({self.path})")


def parquet_scan(path: String, schema: Schema) -> AnyRelation:
    """Create a Parquet file scan with a known schema."""
    return ParquetScan(path=path, schema_=schema)


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

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Evaluate a list of named expressions into output columns."""

    var input: AnyRelation
    var names: List[String]
    var exprs_: List[AnyValue]
    var schema_: Schema

    def __init__(
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

    def kind(self) -> UInt8:
        return PROJECT_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.schema_)

    def pull(mut self) raises -> RecordBatch:
        var batch = self.input.pull()  # raises Exhausted when done
        var cols = List[AnyArray]()
        for ref v in self.exprs_:
            cols.append(v.to_array(batch))
        return RecordBatch(schema=self.schema_.copy(), columns=cols^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Project([")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(t", ")
            writer.write(self.names[i])
            writer.write(t"=")
            self.exprs_[i].write_to(writer)
        writer.write(t"])")


# ---------------------------------------------------------------------------
# Blocking operators
# ---------------------------------------------------------------------------


struct Aggregate(Relation):
    """Grouped aggregation — blocking: consume all input, then emit once."""

    var input: AnyRelation
    var keys: List[AnyValue]
    var agg_exprs: List[AnyValue]
    var grouper: HashGrouper
    var key_fields: List[Field]
    var schema_: Schema
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: AnyRelation,
        var keys: List[AnyValue],
        var agg_exprs: List[AnyValue],
        var grouper: HashGrouper,
        var key_fields: List[Field],
        var schema_: Schema,
    ):
        self.input = input^
        self.keys = keys^
        self.agg_exprs = agg_exprs^
        self.grouper = grouper^
        self.key_fields = key_fields^
        self.schema_ = schema_^
        self._emitted = False

    def kind(self) -> UInt8:
        return AGGREGATE_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.schema_)

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
                var gids = self.grouper.consume_keys(key_struct)
                var val_arrays = List[AnyArray]()
                for i in range(len(self.agg_exprs)):
                    val_arrays.append(self.agg_exprs[i].to_array(batch))
                self.grouper.consume_values(gids, val_arrays)
            except Exhausted:
                break
        self._emitted = True
        return self.grouper.finish(self.key_fields)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Aggregate(keys=[")
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(", ")
            self.keys[i].write_to(writer)
        writer.write("])")


struct Join(Relation):
    """Equijoin — build the left side fully, then stream the right side."""

    var left: AnyRelation
    var right: AnyRelation
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: UInt8
    var strictness: UInt8
    var schema_: Schema
    var _index: Optional[HashJoin[rapidhash]]
    var _exhausted: Bool

    def __init__(
        out self,
        *,
        var left: AnyRelation,
        var right: AnyRelation,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: UInt8,
        strictness: UInt8,
        var schema_: Schema,
    ):
        self.left = left^
        self.right = right^
        self.left_key_indices = left_key_indices^
        self.right_key_indices = right_key_indices^
        self.join_kind = join_kind
        self.strictness = strictness
        self.schema_ = schema_^
        self._index = None
        self._exhausted = False

    def kind(self) -> UInt8:
        return JOIN_NODE

    def schema(self) -> Schema:
        return Schema(copy=self.schema_)

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
                schema=self.schema_.copy(), columns=result.children.copy()
            )
        except Exhausted:
            self._exhausted = True
        raise Exhausted()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Join(kind={self.join_kind})")


# ---------------------------------------------------------------------------
# execute — drain a plan into a single RecordBatch
# ---------------------------------------------------------------------------


def execute(
    plan: AnyRelation, ctx: ExecutionContext = ExecutionContext()
) raises -> RecordBatch:
    """Execute a relational plan, materialising the full result."""
    var p = plan
    return p.collect()
