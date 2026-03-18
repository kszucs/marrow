"""Processor-based streaming query executor.

Execution model
---------------
Each logical plan node maps to a physical **processor** that implements a
pull-based pipeline.  There are two processor hierarchies mirroring the two
expression hierarchies:

**Relation processors** (yield ``RecordBatch`` via ``pull()``):
    ``ScanProcessor``, ``FilterProcessor``, ``ProjectProcessor``

**Value processors** (evaluate ``Array`` via ``eval(inputs)``):
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

from marrow.arrays import PrimitiveArray, Array
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import numeric_dtypes, bool_ as bool_dt
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
from marrow.expr.relations import (
    AnyRelation,
    Scan,
    Filter as PlanFilter,
    Project,
    InMemoryTable,
    SCAN_NODE,
    FILTER_NODE,
    PROJECT_NODE,
    IN_MEMORY_TABLE_NODE,
)


# ---------------------------------------------------------------------------
# Exhausted — signals that a processor has no more batches
# ---------------------------------------------------------------------------


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by ``pull()`` when a processor has no more batches to yield."""

    fn __init__(out self):
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

    fn __init__(out self):
        """Default: CPU-only, parallelism_level() workers, morsel 65 536."""
        self.device_ctx = None
        self.num_cpu_workers = 0
        self.morsel_size = 65_536
        self.gpu_threshold = 1_000_000

    fn __init__(out self, ctx: DeviceContext, gpu_threshold: Int = 1_000_000):
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

    fn eval(self, batch: RecordBatch) raises -> Array:
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
    var _virt_eval: fn(ArcPointer[NoneType], RecordBatch) raises -> Array
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_eval[
        T: ValueProcessor
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> Array:
        return rebind[ArcPointer[T]](ptr)[].eval(batch)

    @staticmethod
    fn _tramp_drop[T: ValueProcessor](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    fn __init__[T: ValueProcessor](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_eval = Self._tramp_eval[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_eval = copy._virt_eval
        self._virt_drop = copy._virt_drop

    # --- public API ---

    fn eval(self, batch: RecordBatch) raises -> Array:
        """Evaluate the expression against the given batch."""
        return self._virt_eval(self._data, batch)

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Leaf value processors — hold data
# ---------------------------------------------------------------------------


struct ColumnProcessor(ValueProcessor):
    """Returns the input array at the given column index."""

    var index: Int

    fn __init__(out self, index: Int):
        self.index = index

    fn eval(self, batch: RecordBatch) raises -> Array:
        return batch.columns[self.index].copy()


struct LiteralProcessor(ValueProcessor):
    """Broadcasts a scalar value to match the input length."""

    var value: Array

    fn __init__(out self, value: Array):
        self.value = value.copy()

    fn eval(self, batch: RecordBatch) raises -> Array:
        return _broadcast_literal(batch.num_rows(), self.value)


# ---------------------------------------------------------------------------
# Operation value processors — compose nested value processors
# ---------------------------------------------------------------------------


struct BinaryProcessor(ValueProcessor):
    """Evaluates two children and applies a binary kernel."""

    var left: AnyValueProcessor
    var right: AnyValueProcessor
    var op: UInt8

    fn __init__(
        out self,
        var left: AnyValueProcessor,
        var right: AnyValueProcessor,
        op: UInt8,
    ):
        self.left = left^
        self.right = right^
        self.op = op

    fn eval(self, batch: RecordBatch) raises -> Array:
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
            return Array(
                and_(
                    PrimitiveArray[bool_dt](data=l),
                    PrimitiveArray[bool_dt](data=r),
                )
            )
        elif self.op == OR:
            return Array(
                or_(
                    PrimitiveArray[bool_dt](data=l),
                    PrimitiveArray[bool_dt](data=r),
                )
            )
        else:
            raise Error("BinaryProcessor: unknown op ", self.op)


struct UnaryProcessor(ValueProcessor):
    """Evaluates one child and applies a unary kernel."""

    var child: AnyValueProcessor
    var op: UInt8

    fn __init__(out self, var child: AnyValueProcessor, op: UInt8):
        self.child = child^
        self.op = op

    fn eval(self, batch: RecordBatch) raises -> Array:
        var c = self.child.eval(batch)
        if self.op == NEG:
            return neg(c)
        elif self.op == ABS:
            return abs_(c)
        elif self.op == NOT:
            return Array(not_(PrimitiveArray[bool_dt](data=c)))
        else:
            raise Error("UnaryProcessor: unknown op ", self.op)


struct IsNullProcessor(ValueProcessor):
    """Evaluates one child and returns a boolean null-check array."""

    var child: AnyValueProcessor

    fn __init__(out self, var child: AnyValueProcessor):
        self.child = child^

    fn eval(self, batch: RecordBatch) raises -> Array:
        return is_null(self.child.eval(batch))


struct IfElseProcessor(ValueProcessor):
    """Evaluates condition, then, and else branches and selects."""

    var cond: AnyValueProcessor
    var then_: AnyValueProcessor
    var else_: AnyValueProcessor

    fn __init__(
        out self,
        var cond: AnyValueProcessor,
        var then_: AnyValueProcessor,
        var else_: AnyValueProcessor,
    ):
        self.cond = cond^
        self.then_ = then_^
        self.else_ = else_^

    fn eval(self, batch: RecordBatch) raises -> Array:
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

    fn schema(self) -> Schema:
        """Return the output schema of this processor."""
        ...

    fn pull(mut self) raises -> RecordBatch:
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
    var _virt_pull: fn(ArcPointer[NoneType]) raises -> RecordBatch
    var _virt_schema: fn(ArcPointer[NoneType]) -> Schema
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_pull[
        T: RelationProcessor
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

    @staticmethod
    fn _tramp_schema[T: RelationProcessor](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    fn _tramp_drop[T: RelationProcessor](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- copy ---

    fn __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_pull = copy._virt_pull
        self._virt_schema = copy._virt_schema
        self._virt_drop = copy._virt_drop

    # --- construction ---

    @implicit
    fn __init__[T: RelationProcessor](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_pull = Self._tramp_pull[T]
        self._virt_schema = Self._tramp_schema[T]
        self._virt_drop = Self._tramp_drop[T]

    # --- public API ---

    fn schema(self) -> Schema:
        """Return the output schema of this processor."""
        return self._virt_schema(self._data)

    fn pull(mut self) raises -> RecordBatch:
        """Return the next batch, or raise ``Exhausted`` when done."""
        return self._virt_pull(self._data)

    fn read_all(mut self) raises -> RecordBatch:
        """Consume all remaining batches and concatenate into a single RecordBatch.
        """
        var batches = self.to_batches()
        if len(batches) == 0:
            return RecordBatch(schema=self.schema(), columns=List[Array]())
        if len(batches) == 1:
            return RecordBatch(copy=batches[0])
        # Concat each column across batches.
        var schema = batches[0].schema
        var num_cols = batches[0].num_columns()
        var result_cols = List[Array](capacity=num_cols)
        for c in range(num_cols):
            var col_arrays = List[Array](capacity=len(batches))
            for b in range(len(batches)):
                col_arrays.append(batches[b].columns[c].copy())
            result_cols.append(concat(col_arrays))
        return RecordBatch(schema=Schema(copy=schema), columns=result_cols^)

    fn to_batches(mut self) raises -> List[RecordBatch]:
        """Consume all remaining batches into a list."""
        var result = List[RecordBatch]()
        while True:
            try:
                result.append(self.pull())
            except Exhausted:
                break
        return result^

    fn __del__(deinit self):
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

    fn __init__(out self, batch: RecordBatch, morsel_size: Int):
        self.batch = RecordBatch(copy=batch)
        self.offset = 0
        self.morsel_size = morsel_size

    fn schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    fn pull(mut self) raises -> RecordBatch:
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

    fn __init__(
        out self,
        var child: AnyRelationProcessor,
        var predicate: AnyValueProcessor,
        schema_: Schema,
    ):
        self.child = child^
        self.predicate = predicate^
        self.schema_ = schema_

    fn schema(self) -> Schema:
        return self.schema_.copy()

    fn pull(mut self) raises -> RecordBatch:
        # Skips batches that filter to 0 rows. Exhausted propagates from child.
        while True:
            var batch = self.child.pull()
            var mask = self.predicate.eval(batch)
            var result_cols = List[Array]()
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

    fn __init__(
        out self,
        var child: AnyRelationProcessor,
        var values: List[AnyValueProcessor],
        schema_: Schema,
    ):
        self.child = child^
        self.values = values^
        self.schema_ = schema_

    fn schema(self) -> Schema:
        return self.schema_.copy()

    fn pull(mut self) raises -> RecordBatch:
        var batch = self.child.pull()  # raises Exhausted when done
        var result_cols = List[Array]()
        for ref v in self.values:
            result_cols.append(v.eval(batch))
        return RecordBatch(schema=self.schema_.copy(), columns=result_cols^)


# ---------------------------------------------------------------------------
# Planner — builds processor hierarchies from expression trees
# ---------------------------------------------------------------------------


struct Planner:
    """Builds physical processor pipelines from logical expression trees.

    Walks expression trees bottom-up and creates the corresponding processor
    hierarchies.  Holds the ``ExecutionContext`` used during planning.
    """

    var ctx: ExecutionContext

    fn __init__(out self, ctx: ExecutionContext = ExecutionContext()):
        self.ctx = ctx

    fn build(self, expr: AnyValue) raises -> AnyValueProcessor:
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

    fn build(self, expr: AnyRelation) raises -> AnyRelationProcessor:
        """Build a relation processor pipeline from a relational expression tree."""
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
        if k == SCAN_NODE:
            raise Error("Planner.build: Scan requires external data source binding")
        raise Error("Planner.build: unknown relation kind ", k)


# ---------------------------------------------------------------------------
# Convenience execute() functions
# ---------------------------------------------------------------------------


fn execute(
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


fn _broadcast_literal(length: Int, scalar_array: Array) raises -> Array:
    """Broadcast a length-1 scalar array to the given length."""
    comptime for dt in numeric_dtypes:
        if scalar_array.dtype == dt:
            var val = PrimitiveArray[dt](data=scalar_array).unsafe_get(0)
            var builder = PrimitiveBuilder[dt](length)
            for i in range(length):
                builder._buffer.unsafe_set[dt.native](i, val)
            builder._length = length
            return Array(builder.finish_typed())
    raise Error(t"_broadcast_literal: unsupported dtype {scalar_array.dtype}")


