"""Processor-based streaming query executor.

Execution model
---------------
Each logical plan node maps to a physical **processor** that implements a
pull-based pipeline.  There are two processor hierarchies mirroring the two
expression hierarchies:

**Relation processors** (yield ``RecordBatch`` via ``pull()``):
    ``ScanProcessor``, ``FilterProcessor``, ``ProjectProcessor``

**Value processors** (evaluate ``AnyArray`` via ``eval(inputs)``):
    *Leaf* — ``ColumnProcessor``, ``LiteralProcessor`` (hold data)
    *Operations* — ``BinaryProcessor``, ``UnaryProcessor``,
    ``IsNullProcessor``, ``IfElseProcessor`` (compose nested value processors)

``Planner.build(expr)`` walks the expression tree bottom-up and builds both
hierarchies.  Consuming the top-level relation processor pulls morsel-sized
``RecordBatch`` values through the pipeline.

Convenience wrapper ``execute(relation)`` materialises the full result as a
``RecordBatch``.
"""

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
import std.math as math

from marrow.arrays import PrimitiveArray, AnyArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import numeric_types, bool_ as bool_dt
from marrow.kernels.arithmetic import add, sub, mul, div, neg, abs_
from marrow.kernels.boolean import and_, or_, not_, is_null, select
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from marrow.kernels.concat import concat
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.kernels.filter import filter_
from marrow.expr.values import (
    AnyValue,
    Column,
    Literal,
    Binary,
    Unary,
    IsNull,
    IfElse,
    Cast,
    LOAD,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ABS,
    LITERAL,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND,
    OR,
    NOT,
    IS_NULL,
    IF_ELSE,
    CAST,
    DISPATCH_CPU,
    DISPATCH_GPU,
)
from marrow.arrays import StructArray
from marrow.dtypes import Field, struct_
from marrow.kernels.groupby import HashGrouper
from marrow.kernels.join import HashJoin
from marrow.kernels.hashing import rapidhash
from marrow.expr.relations import (
    AnyRelation,
    Scan,
    Filter as PlanFilter,
    Project,
    InMemoryTable,
    ParquetScan,
    Aggregate as PlanAggregate,
    Join as PlanJoin,
    SCAN_NODE,
    FILTER_NODE,
    PROJECT_NODE,
    IN_MEMORY_TABLE_NODE,
    PARQUET_SCAN_NODE,
    AGGREGATE_NODE,
    JOIN_NODE,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_ALL,
    JOIN_ANY,
)
from marrow.parquet import read_table


# ---------------------------------------------------------------------------
# Exhausted — signals that a processor has no more batches
# ---------------------------------------------------------------------------


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by ``pull()`` when a processor has no more batches to yield."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------


struct ExecutionContext(Copyable, ImplicitlyCopyable, Movable):
    """Runtime configuration for query execution."""

    var device_ctx: Optional[DeviceContext]
    """GPU device context.  None = CPU-only execution."""

    var num_cpu_workers: Int
    """Number of CPU worker threads.  0 = ``parallelism_level()`` at execute time."""

    var morsel_size: Int
    """Number of rows per batch yielded by ``ScanProcessor``.  Default 65 536."""

    var gpu_threshold: Int
    """Minimum array length to auto-dispatch to GPU when ``DISPATCH_AUTO``."""

    def __init__(out self):
        """Default: CPU-only, parallelism_level() workers, morsel 65 536."""
        self.device_ctx = None
        self.num_cpu_workers = 0
        self.morsel_size = 65_536
        self.gpu_threshold = 1_000_000

    def __init__(out self, ctx: DeviceContext, gpu_threshold: Int = 1_000_000):
        """GPU-enabled context.

        Args:
            ctx: GPU device context.
            gpu_threshold: Minimum length to auto-route to GPU.
        """
        self.device_ctx = ctx
        self.num_cpu_workers = 0
        self.morsel_size = 65_536
        self.gpu_threshold = gpu_threshold


# ===================================================================
# Value processor hierarchy — evaluate scalar expressions
# ===================================================================


# ---------------------------------------------------------------------------
# ValueProcessor trait
# ---------------------------------------------------------------------------


trait ValueProcessor(ImplicitlyDestructible, Movable):
    """Evaluates a scalar expression against a morsel batch.

    Leaf processors (``ColumnProcessor``, ``LiteralProcessor``) hold data.
    Operation processors (``BinaryProcessor``, ``UnaryProcessor``, etc.)
    compose nested ``AnyValueProcessor`` children.
    """

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate the expression against the given batch."""
        ...


# ---------------------------------------------------------------------------
# AnyValueProcessor — type-erased value processor
# ---------------------------------------------------------------------------


struct AnyValueProcessor(ImplicitlyCopyable, Movable):
    """Type-erased value processor.

    Wraps any ``ValueProcessor``-conforming type on the heap behind an
    ``ArcPointer`` so the value processor tree can be composed at runtime.
    Copies are O(1) ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_eval: def(ArcPointer[NoneType], RecordBatch) raises -> AnyArray
    var _virt_drop: def(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    def _tramp_eval[
        T: ValueProcessor
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
        return rebind[ArcPointer[T]](ptr)[].eval(batch)

    @staticmethod
    def _tramp_drop[T: ValueProcessor](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    def __init__[T: ValueProcessor](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_eval = Self._tramp_eval[T]
        self._virt_drop = Self._tramp_drop[T]

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_eval = copy._virt_eval
        self._virt_drop = copy._virt_drop

    # --- public API ---

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate the expression against the given batch."""
        return self._virt_eval(self._data, batch)

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Leaf value processors — hold data
# ---------------------------------------------------------------------------


struct ColumnProcessor(ValueProcessor):
    """Returns the input array at the given column index."""

    var index: Int

    def __init__(out self, index: Int):
        self.index = index

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        return batch.columns[self.index].copy()


struct LiteralProcessor(ValueProcessor):
    """Broadcasts a scalar value to match the input length."""

    var value: AnyArray

    def __init__(out self, value: AnyArray):
        self.value = value.copy()

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        return _broadcast_literal(batch.num_rows(), self.value)


# ---------------------------------------------------------------------------
# Operation value processors — compose nested value processors
# ---------------------------------------------------------------------------


struct BinaryProcessor(ValueProcessor):
    """Evaluates two children and applies a binary kernel."""

    var left: AnyValueProcessor
    var right: AnyValueProcessor
    var op: UInt8

    def __init__(
        out self,
        var left: AnyValueProcessor,
        var right: AnyValueProcessor,
        op: UInt8,
    ):
        self.left = left^
        self.right = right^
        self.op = op

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        var l = self.left.eval(batch)
        var r = self.right.eval(batch)
        if self.op == ADD:
            return add(l, r)
        elif self.op == SUB:
            return sub(l, r)
        elif self.op == MUL:
            return mul(l, r)
        elif self.op == DIV:
            return div(l, r)
        elif self.op == EQ:
            return equal(l, r)
        elif self.op == NE:
            return not_equal(l, r)
        elif self.op == LT:
            return less(l, r)
        elif self.op == LE:
            return less_equal(l, r)
        elif self.op == GT:
            return greater(l, r)
        elif self.op == GE:
            return greater_equal(l, r)
        elif self.op == AND:
            return and_(l, r)
        elif self.op == OR:
            return or_(l, r)
        else:
            raise Error("BinaryProcessor: unknown op ", self.op)


struct UnaryProcessor(ValueProcessor):
    """Evaluates one child and applies a unary kernel."""

    var child: AnyValueProcessor
    var op: UInt8

    def __init__(out self, var child: AnyValueProcessor, op: UInt8):
        self.child = child^
        self.op = op

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        var c = self.child.eval(batch)
        if self.op == NEG:
            return neg(c)
        elif self.op == ABS:
            return abs_(c)
        elif self.op == NOT:
            return not_(c)
        else:
            raise Error("UnaryProcessor: unknown op ", self.op)


struct IsNullProcessor(ValueProcessor):
    """Evaluates one child and returns a boolean null-check array."""

    var child: AnyValueProcessor

    def __init__(out self, var child: AnyValueProcessor):
        self.child = child^

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        return is_null(self.child.eval(batch))


struct IfElseProcessor(ValueProcessor):
    """Evaluates condition, then, and else branches and selects."""

    var cond: AnyValueProcessor
    var then_: AnyValueProcessor
    var else_: AnyValueProcessor

    def __init__(
        out self,
        var cond: AnyValueProcessor,
        var then_: AnyValueProcessor,
        var else_: AnyValueProcessor,
    ):
        self.cond = cond^
        self.then_ = then_^
        self.else_ = else_^

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        var c = self.cond.eval(batch)
        var t = self.then_.eval(batch)
        var e = self.else_.eval(batch)
        return select(c, t, e)


# ===================================================================
# Relation processor hierarchy — stream RecordBatch values
# ===================================================================


# ---------------------------------------------------------------------------
# RelationProcessor trait
# ---------------------------------------------------------------------------


trait RelationProcessor(ImplicitlyDestructible, Movable):
    """Pull-based relation processor.

    Concrete processors implement ``pull()`` to yield morsel-sized
    ``RecordBatch`` values, raising ``Exhausted`` when done.
    """

    def schema(self) -> Schema:
        """Return the output schema of this processor."""
        ...

    def pull(mut self) raises -> RecordBatch:
        """Return the next batch, or raise ``Exhausted`` when done."""
        ...


# ---------------------------------------------------------------------------
# AnyRelationProcessor — type-erased relation processor
# ---------------------------------------------------------------------------


struct AnyRelationProcessor(ImplicitlyCopyable, Movable):
    """Type-erased relation processor.

    Wraps any ``RelationProcessor``-conforming type on the heap behind an
    ``ArcPointer`` so the processor hierarchy can be composed at runtime.
    Copies are O(1) ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_pull: def(ArcPointer[NoneType]) raises -> RecordBatch
    var _virt_schema: def(ArcPointer[NoneType]) -> Schema
    var _virt_drop: def(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    def _tramp_pull[
        T: RelationProcessor
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

    @staticmethod
    def _tramp_schema[
        T: RelationProcessor
    ](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_drop[T: RelationProcessor](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- copy ---

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_pull = copy._virt_pull
        self._virt_schema = copy._virt_schema
        self._virt_drop = copy._virt_drop

    # --- construction ---

    @implicit
    def __init__[T: RelationProcessor](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_pull = Self._tramp_pull[T]
        self._virt_schema = Self._tramp_schema[T]
        self._virt_drop = Self._tramp_drop[T]

    # --- public API ---

    def schema(self) -> Schema:
        """Return the output schema of this processor."""
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        """Return the next batch, or raise ``Exhausted`` when done."""
        return self._virt_pull(self._data)

    def read_all(mut self) raises -> RecordBatch:
        """Consume all remaining batches and concatenate into a single RecordBatch.
        """
        var batches = self.to_batches()
        if len(batches) == 0:
            return RecordBatch(schema=self.schema(), columns=List[AnyArray]())
        if len(batches) == 1:
            return RecordBatch(copy=batches[0])
        # Concat each column across batches.
        var schema = batches[0].schema
        var num_cols = batches[0].num_columns()
        var result_cols = List[AnyArray](capacity=num_cols)
        for c in range(num_cols):
            var col_arrays = List[AnyArray](capacity=len(batches))
            for b in range(len(batches)):
                col_arrays.append(batches[b].columns[c].copy())
            result_cols.append(concat(col_arrays))
        return RecordBatch(schema=Schema(copy=schema), columns=result_cols^)

    def to_batches(mut self) raises -> List[RecordBatch]:
        """Consume all remaining batches into a list."""
        var result = List[RecordBatch]()
        while True:
            try:
                result.append(self.pull())
            except Exhausted:
                break
        return result^

    def __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# ScanProcessor — yields morsel-sized slices of an in-memory RecordBatch
# ---------------------------------------------------------------------------


struct ScanProcessor(RelationProcessor):
    """Yields morsel-sized slices of an in-memory ``RecordBatch``.

    Each ``__next__()`` call returns a morsel-sized batch with offset-0 arrays,
    advancing the internal offset.
    """

    var batch: RecordBatch
    var offset: Int
    var morsel_size: Int

    def __init__(out self, batch: RecordBatch, morsel_size: Int):
        self.batch = RecordBatch(copy=batch)
        self.offset = 0
        self.morsel_size = morsel_size

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def pull(mut self) raises -> RecordBatch:
        if self.offset >= self.batch.num_rows():
            raise Exhausted()
        var length = min(self.morsel_size, self.batch.num_rows() - self.offset)
        var result = self.batch.slice(self.offset, length)
        self.offset += length
        return result^


# ---------------------------------------------------------------------------
# ParquetScanProcessor — reads a Parquet file and yields morsel-sized slices
# ---------------------------------------------------------------------------


struct ParquetScanProcessor(RelationProcessor):
    """Reads a Parquet file at construction time and yields morsel-sized slices.

    Uses ``marrow.parquet.read_table`` (backed by PyArrow) for I/O, then
    streams the result the same way as ``ScanProcessor``.
    """

    var batch: RecordBatch
    var offset: Int
    var morsel_size: Int

    # TODO: make the reading lazy as well instead of materializing the whole table upfront
    # TODO: add support for projection pushdown to avoid reading unnecessary columns
    def __init__(out self, path: String, morsel_size: Int) raises:
        var table = read_table(path)
        var batches = table.to_batches()
        if len(batches) == 0:
            self.batch = RecordBatch(
                schema=table.schema, columns=List[AnyArray]()
            )
        elif len(batches) == 1:
            self.batch = RecordBatch(copy=batches[0])
        else:
            var schema = batches[0].schema
            var num_cols = batches[0].num_columns()
            var result_cols = List[AnyArray](capacity=num_cols)
            for c in range(num_cols):
                var col_arrays = List[AnyArray](capacity=len(batches))
                for b in range(len(batches)):
                    col_arrays.append(batches[b].columns[c].copy())
                result_cols.append(concat(col_arrays))
            self.batch = RecordBatch(
                schema=Schema(copy=schema), columns=result_cols^
            )
        self.offset = 0
        self.morsel_size = morsel_size

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def pull(mut self) raises -> RecordBatch:
        if self.offset >= self.batch.num_rows():
            raise Exhausted()
        var length = min(self.morsel_size, self.batch.num_rows() - self.offset)
        var result = self.batch.slice(self.offset, length)
        self.offset += length
        return result^


# ---------------------------------------------------------------------------
# FilterProcessor — pulls from child, applies boolean predicate
# ---------------------------------------------------------------------------


struct FilterProcessor(RelationProcessor):
    """Pulls batches from a child processor, evaluates a boolean predicate,
    and yields only the rows where the predicate is true.

    Batches where all rows are filtered out are silently skipped.
    """

    var child: AnyRelationProcessor
    var predicate: AnyValueProcessor
    var schema_: Schema

    def __init__(
        out self,
        var child: AnyRelationProcessor,
        var predicate: AnyValueProcessor,
        schema_: Schema,
    ):
        self.child = child^
        self.predicate = predicate^
        self.schema_ = schema_

    def schema(self) -> Schema:
        return self.schema_.copy()

    def pull(mut self) raises -> RecordBatch:
        # Skips batches that filter to 0 rows. Exhausted propagates from child.
        while True:
            var batch = self.child.pull()
            var mask = self.predicate.eval(batch)
            var result_cols = List[AnyArray]()
            for i in range(batch.num_columns()):
                result_cols.append(
                    filter_(batch.columns[i].copy(), mask.copy())
                )
            var result = RecordBatch(
                schema=self.schema_.copy(), columns=result_cols^
            )
            if result.num_rows() > 0:
                return result^


# ---------------------------------------------------------------------------
# ProjectProcessor — pulls from child, evaluates projection expressions
# ---------------------------------------------------------------------------


struct ProjectProcessor(RelationProcessor):
    """Pulls batches from a child processor and evaluates a list of
    value processors to produce new output columns.
    """

    var child: AnyRelationProcessor
    var values: List[AnyValueProcessor]
    var schema_: Schema

    def __init__(
        out self,
        var child: AnyRelationProcessor,
        var values: List[AnyValueProcessor],
        schema_: Schema,
    ):
        self.child = child^
        self.values = values^
        self.schema_ = schema_

    def schema(self) -> Schema:
        return self.schema_.copy()

    def pull(mut self) raises -> RecordBatch:
        var batch = self.child.pull()  # raises Exhausted when done
        var result_cols = List[AnyArray]()
        for ref v in self.values:
            result_cols.append(v.eval(batch))
        return RecordBatch(schema=self.schema_.copy(), columns=result_cols^)


# ---------------------------------------------------------------------------
# AggregateProcessor — blocking grouped aggregation
# ---------------------------------------------------------------------------


# TODO: have this implemented on RecordBatch.to_struct_array()
def _batch_to_struct(batch: RecordBatch) raises -> StructArray:
    """Convert a RecordBatch to a StructArray (key columns)."""
    var fields = List[Field]()
    var children = List[AnyArray]()
    for i in range(batch.num_columns()):
        fields.append(batch.schema.fields[i].copy())
        children.append(batch.columns[i].copy())
    return StructArray(
        dtype=struct_(fields),
        length=batch.num_rows(),
        nulls=0,
        offset=0,
        bitmap=None,
        children=children^,
    )


struct AggregateProcessor(RelationProcessor):
    """Blocking grouped aggregation processor.

    Single child input. Key and value expressions are evaluated per morsel.

    Phase 1: Pull all morsels. For each: evaluate key expressions → build
             StructArray → ``consume_keys()`` → store group_ids. Evaluate
             value expressions → ``consume_values()`` with stored group_ids.
             Values are scatter-updated and discarded — never stored.
    Phase 2: Emit the aggregated result once via ``finish()``.
    """

    var child: AnyRelationProcessor
    var key_exprs: List[AnyValueProcessor]
    var value_exprs: List[AnyValueProcessor]
    var grouper: HashGrouper
    var schema_: Schema
    var key_fields: List[Field]
    var _emitted: Bool

    def __init__(
        out self,
        var child: AnyRelationProcessor,
        var key_exprs: List[AnyValueProcessor],
        var value_exprs: List[AnyValueProcessor],
        var grouper: HashGrouper,
        schema_: Schema,
        var key_fields: List[Field],
    ):
        self.child = child^
        self.key_exprs = key_exprs^
        self.value_exprs = value_exprs^
        self.grouper = grouper^
        self.schema_ = schema_
        self.key_fields = key_fields^
        self._emitted = False

    def schema(self) -> Schema:
        return self.schema_.copy()

    def pull(mut self) raises -> RecordBatch:
        # Phase 1: consume all input morsels.
        if not self._emitted:
            while True:
                try:
                    var batch = self.child.pull()

                    # Evaluate key expressions → StructArray.
                    var key_arrays = List[AnyArray]()
                    for i in range(len(self.key_exprs)):
                        key_arrays.append(self.key_exprs[i].eval(batch).copy())
                    var key_children = List[AnyArray]()
                    var key_struct_fields = List[Field]()
                    for i in range(len(key_arrays)):
                        key_children.append(key_arrays[i].copy())
                        key_struct_fields.append(self.key_fields[i].copy())
                    var key_struct = StructArray(
                        dtype=struct_(key_struct_fields),
                        length=batch.num_rows(),
                        nulls=0,
                        offset=0,
                        bitmap=None,
                        children=key_children^,
                    )

                    # Consume keys → get group_ids.
                    var gids = self.grouper.consume_keys(key_struct)

                    # Evaluate value expressions → scatter-update.
                    var val_arrays = List[AnyArray]()
                    for i in range(len(self.value_exprs)):
                        val_arrays.append(
                            self.value_exprs[i].eval(batch).copy()
                        )
                    self.grouper.consume_values(gids, val_arrays)
                except Exhausted:
                    break

            self._emitted = True
            return self.grouper.finish(self.key_fields)

        raise Exhausted()


# ---------------------------------------------------------------------------
# JoinProcessor — blocking hash join (build left, probe right)
# ---------------------------------------------------------------------------


struct JoinProcessor(RelationProcessor):
    """Hash join processor: build left side entirely, then stream right.

    Phase 1 (first ``pull()`` call):
        Consume the entire left input into a single RecordBatch (blocking).

    Phase 2 (subsequent ``pull()`` calls):
        Pull one morsel from the right input, call ``hash_join``, emit result.

    When the right input is exhausted, raise ``Exhausted``.

    This is the "CollectLeft" strategy from DataFusion — simplest correct
    approach, O(|left|) memory for the build side.
    """

    var left_proc: AnyRelationProcessor
    var right_proc: AnyRelationProcessor
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var kind: UInt8
    var strictness: UInt8
    var schema_: Schema
    var ctx: ExecutionContext
    var _index: Optional[HashJoin[rapidhash]]
    var _exhausted: Bool

    def __init__(
        out self,
        var left_proc: AnyRelationProcessor,
        var right_proc: AnyRelationProcessor,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
        schema_: Schema,
        ctx: ExecutionContext,
    ):
        self.left_proc = left_proc^
        self.right_proc = right_proc^
        self.left_key_indices = left_key_indices^
        self.right_key_indices = right_key_indices^
        self.kind = kind
        self.strictness = strictness
        self.schema_ = schema_
        self.ctx = ctx
        self._index = None
        self._exhausted = False

    def schema(self) -> Schema:
        return self.schema_.copy()

    def pull(mut self) raises -> RecordBatch:
        if self._exhausted:
            raise Exhausted()

        # Phase 1: consume entire left side and build hash index (once).
        if not self._index:
            var left_struct = self.left_proc.read_all().to_struct_array()
            var index = HashJoin[rapidhash]()
            index.build(left_struct, self.left_key_indices)
            self._index = index^

        # Phase 2: pull one right morsel and probe the pre-built index.
        try:
            var right_morsel = self.right_proc.pull()
            var result = self._index.value().probe(
                right_morsel.to_struct_array(),
                self.right_key_indices,
                self.kind,
                self.strictness,
            )
            return RecordBatch(
                schema=self.schema_.copy(), columns=result.children.copy()
            )
        except Exhausted:
            self._exhausted = True
        # Right side exhausted — propagate to caller.
        raise Exhausted()


# ---------------------------------------------------------------------------
# Planner — builds processor hierarchies from expression trees
# ---------------------------------------------------------------------------


struct Planner:
    """Builds physical processor pipelines from logical expression trees.

    Walks expression trees bottom-up and creates the corresponding processor
    hierarchies.  Holds the ``ExecutionContext`` used during planning.
    """

    var ctx: ExecutionContext

    def __init__(out self, ctx: ExecutionContext = ExecutionContext()):
        self.ctx = ctx

    def build(self, expr: AnyValue) raises -> AnyValueProcessor:
        """Build a value processor tree from a scalar expression tree."""
        var k = expr.kind()
        if k == LOAD:
            var arc = expr.downcast[Column]()
            return ColumnProcessor(arc[].index)
        elif k == LITERAL:
            var arc = expr.downcast[Literal]()
            return LiteralProcessor(arc[].value)
        elif k >= ADD and k <= OR:
            var arc = expr.downcast[Binary]()
            var left = self.build(arc[].left)
            var right = self.build(arc[].right)
            return BinaryProcessor(left^, right^, arc[].op)
        elif k >= NEG and k <= NOT:
            var arc = expr.downcast[Unary]()
            var child = self.build(arc[].child)
            return UnaryProcessor(child^, arc[].op)
        elif k == IS_NULL:
            var arc = expr.downcast[IsNull]()
            var child = self.build(arc[].child)
            return IsNullProcessor(child^)
        elif k == IF_ELSE:
            var arc = expr.downcast[IfElse]()
            var c = self.build(arc[].cond)
            var t = self.build(arc[].then_)
            var e = self.build(arc[].else_)
            return IfElseProcessor(c^, t^, e^)
        elif k == CAST:
            raise Error("Planner.build: cast not implemented")
        else:
            raise Error("Planner.build: unknown expression kind ", k)

    def build(self, expr: AnyRelation) raises -> AnyRelationProcessor:
        """Build a relation processor pipeline from a relational expression tree.
        """
        var k = expr.kind()
        if k == IN_MEMORY_TABLE_NODE:
            var arc = expr.downcast[InMemoryTable]()
            return ScanProcessor(
                batch=arc[].batch, morsel_size=self.ctx.morsel_size
            )
        if k == FILTER_NODE:
            var arc = expr.downcast[PlanFilter]()
            var child = self.build(arc[].input)
            var predicate = self.build(arc[].predicate)
            return FilterProcessor(
                child=child^,
                predicate=predicate^,
                schema_=arc[].input.schema(),
            )
        if k == PROJECT_NODE:
            var arc = expr.downcast[Project]()
            var child = self.build(arc[].input)
            var values = List[AnyValueProcessor]()
            for ref e in arc[].exprs_:
                values.append(self.build(e))
            return ProjectProcessor(
                child=child^,
                values=values^,
                schema_=arc[].schema(),
            )
        if k == PARQUET_SCAN_NODE:
            var arc = expr.downcast[ParquetScan]()
            return ParquetScanProcessor(
                path=arc[].path, morsel_size=self.ctx.morsel_size
            )
        if k == AGGREGATE_NODE:
            var arc = expr.downcast[PlanAggregate]()
            var child = self.build(arc[].input)

            # Build key and value expression processors.
            var key_exprs = List[AnyValueProcessor]()
            for ref e in arc[].keys:
                key_exprs.append(self.build(e))
            var value_exprs = List[AnyValueProcessor]()
            for ref e in arc[].agg_exprs:
                value_exprs.append(self.build(e))

            var key_fields = List[Field]()
            for i in range(len(arc[].keys)):
                key_fields.append(arc[].schema().fields[i].copy())

            return AggregateProcessor(
                child=child^,
                key_exprs=key_exprs^,
                value_exprs=value_exprs^,
                grouper=HashGrouper(arc[].agg_funcs),
                schema_=arc[].schema(),
                key_fields=key_fields^,
            )
        if k == JOIN_NODE:
            var arc = expr.downcast[PlanJoin]()
            var left_proc = self.build(arc[].left)
            var right_proc = self.build(arc[].right)

            # Extract positional key indices from pre-resolved Column exprs.
            var left_key_indices = List[Int]()
            for ref e in arc[].left_keys:
                var col_arc = e.downcast[Column]()
                left_key_indices.append(col_arc[].index)
            var right_key_indices = List[Int]()
            for ref e in arc[].right_keys:
                var col_arc = e.downcast[Column]()
                right_key_indices.append(col_arc[].index)

            return JoinProcessor(
                left_proc=left_proc^,
                right_proc=right_proc^,
                left_key_indices=left_key_indices^,
                right_key_indices=right_key_indices^,
                kind=arc[].join_kind,
                strictness=arc[].strictness,
                schema_=arc[].schema(),
                ctx=self.ctx,
            )
        if k == SCAN_NODE:
            raise Error(
                "Planner.build: Scan requires external data source binding"
            )
        raise Error("Planner.build: unknown relation kind ", k)


# ---------------------------------------------------------------------------
# Convenience execute() functions
# ---------------------------------------------------------------------------


def execute(
    expr: AnyRelation, ctx: ExecutionContext = ExecutionContext()
) raises -> RecordBatch:
    """Execute a relational expression tree, materialising the full result.

    Args:
        expr: The logical relational expression tree.
        ctx: Execution context (morsel size, device, etc.).

    Returns:
        A single ``RecordBatch`` containing all result rows.
    """
    var proc = Planner(ctx).build(expr)
    return proc.read_all()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _broadcast_literal(length: Int, scalar_array: AnyArray) raises -> AnyArray:
    """Broadcast a length-1 scalar array to the given length."""
    comptime for dt in numeric_types:
        if scalar_array.dtype() == dt():
            var val = scalar_array.as_primitive[dt]().unsafe_get(0)
            var builder = PrimitiveBuilder[dt](length)
            for _ in range(length):
                builder.unsafe_append(val)
            return builder.finish().to_any()
    raise Error(t"_broadcast_literal: unsupported dtype {scalar_array.dtype()}")
