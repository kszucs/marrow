"""Morsel-driven parallel expression executor.

Execution model
---------------
The executor splits input arrays into equal-sized **morsels** and distributes
them across CPU worker threads using ``sync_parallelize`` from
``std.algorithm``.  Worker threads that finish early are assigned additional
morsels by the stdlib's coalescing logic — this provides natural load
balancing similar to DuckDB/Umbra morsel-driven scheduling, without an
explicit work-stealing scheduler.

CPU path (default)::

    Input arrays [length N]
        │
        ├── morsel 0 [0, M)      → thread 0 → result[0]
        ├── morsel 1 [M, 2M)     → thread 1 → result[1]
        ├── morsel 2 [2M, 3M)    → thread 0 → result[2]   ← stolen!
        └── morsel 3 [3M, N)     → thread 1 → result[3]
                                              │
                                         concat(results)
                                              │
                                           Array

NOTE: Exceptions raised inside ``sync_parallelize`` tasks are caught
internally and cause ``abort()`` rather than propagating to the caller.  This
is a known stdlib limitation.  Use the sequential fast path (set
``morsel_size`` larger than your array length) when error propagation matters.
"""

import std.math as math
from std.algorithm import sync_parallelize
from std.runtime.asyncrt import parallelism_level
from std.gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, numeric_dtypes, bool_ as bool_dt
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
    Column, Literal, Binary, Unary, IsNull, IfElse, Cast,
    LOAD, ADD, SUB, MUL, DIV, NEG, ABS, LITERAL,
    EQ, NE, LT, LE, GT, GE, AND, OR, NOT, IS_NULL, IF_ELSE, CAST,
    DISPATCH_CPU,
    DISPATCH_GPU,
)
from marrow.expr.relations import (
    AnyRelation,
    Scan, Filter as PlanFilter, Project, InMemoryTable,
    SCAN_NODE, FILTER_NODE, PROJECT_NODE, IN_MEMORY_TABLE_NODE,
)


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------


struct ExecutionContext(Copyable, ImplicitlyCopyable, Movable):
    """Runtime dispatch configuration for the ``PipelineExecutor``."""

    var device_ctx: Optional[DeviceContext]
    """GPU device context.  None = CPU-only execution."""

    var num_cpu_workers: Int
    """Number of CPU worker threads.  0 = ``parallelism_level()`` at execute time."""

    var morsel_size: Int
    """Number of elements per CPU work chunk.  Default 65 536."""

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


# ---------------------------------------------------------------------------
# PipelineExecutor
# ---------------------------------------------------------------------------


struct PipelineExecutor(Copyable, Movable):
    """Morsel-driven parallel executor with CPU / GPU dispatch.

    All public methods accept and return type-erased ``Array`` values.
    Runtime dtype dispatch to typed kernels happens internally.
    """

    var ctx: ExecutionContext

    fn __init__(out self, var ctx: ExecutionContext = ExecutionContext()):
        self.ctx = ctx^

    fn execute(self, plan: AnyRelation) raises -> RecordBatch:
        """Execute a relational plan tree, returning a RecordBatch."""
        return RelationsExecutor().execute(plan)

    fn execute(
        self,
        expr: AnyValue,
        inputs: List[Array],
    ) raises -> Array:
        """Evaluate an expression tree over the given input arrays.

        Handles both arithmetic (ADD, SUB, …) and predicate (EQ, LT, …)
        nodes.  Returns a type-erased ``Array``.

        Args:
            expr: The expression tree to evaluate.
            inputs: Input arrays referenced by ``col(idx)`` nodes.

        Returns:
            A new ``Array`` with the expression result.
        """
        if len(inputs) == 0:
            raise Error("PipelineExecutor.execute: inputs must be non-empty")
        var length = inputs[0].length

        var morsel_size = self.ctx.morsel_size
        var n_morsels = math.ceildiv(length, morsel_size)
        var effective_workers = (
            parallelism_level()
            if self.ctx.num_cpu_workers == 0
            else self.ctx.num_cpu_workers
        )
        var num_workers = min(effective_workers, n_morsels)

        if num_workers <= 1 or n_morsels == 1:
            var input_copy = _copy_inputs(inputs)
            var evaluator = ValuesExecutor(input_copy^)
            return evaluator.execute(expr)

        var results = List[Array](capacity=n_morsels)
        for _ in range(n_morsels):
            results.append(_empty_like(inputs[0]))

        @parameter
        fn process_morsel(i: Int) raises:
            var start = i * morsel_size
            var end = min(start + morsel_size, length)
            var chunk_inputs = _slice_inputs(inputs, start, end)
            var evaluator = ValuesExecutor(chunk_inputs^)
            results[i] = evaluator.execute(expr)

        sync_parallelize[process_morsel](n_morsels)
        return concat(results)


fn execute(
    expr: AnyValue,
    inputs: List[Array],
    morsel_size: Int = 65_536,
    num_cpu_workers: Int = 0,
) raises -> Array:
    """Convenience wrapper around ``PipelineExecutor`` for expression evaluation.

    Args:
        expr: The expression tree to evaluate.
        inputs: Input arrays referenced by ``col(idx)`` nodes.
        morsel_size: Number of elements per parallel work chunk.
        num_cpu_workers: Number of CPU threads (0 = ``parallelism_level()``).

    Returns:
        A new ``Array`` with the expression result.
    """
    var ctx = ExecutionContext()
    ctx.morsel_size = morsel_size
    ctx.num_cpu_workers = num_cpu_workers
    return PipelineExecutor(ctx).execute(expr, inputs)


# ---------------------------------------------------------------------------
# Expression tree executor (overloaded eval per node type)
# ---------------------------------------------------------------------------


struct ValuesExecutor:
    """Evaluates an expression tree against a list of input arrays.

    Dispatches to overloaded ``execute`` methods per concrete node type,
    mirroring the RelationsExecutor pattern for relation nodes.
    """

    var inputs: List[Array]

    fn __init__(out self, var inputs: List[Array]):
        self.inputs = inputs^

    fn execute(mut self, expr: AnyValue) raises -> Array:
        """Dispatch to the typed overload matching the runtime kind."""
        var k = expr.kind()
        if k == LOAD:
            return self.execute(expr.downcast[Column]()[])
        elif k == LITERAL:
            return self.execute(expr.downcast[Literal]()[])
        elif k >= ADD and k <= OR:
            return self.execute(expr.downcast[Binary]()[])
        elif k >= NEG and k <= NOT:
            return self.execute(expr.downcast[Unary]()[])
        elif k == IS_NULL:
            return self.execute(expr.downcast[IsNull]()[])
        elif k == IF_ELSE:
            return self.execute(expr.downcast[IfElse]()[])
        elif k == CAST:
            return self.execute(expr.downcast[Cast]()[])
        else:
            raise Error("ValuesExecutor: unknown kind ", k)

    fn execute(mut self, col: Column) raises -> Array:
        return self.inputs[col.index].copy()

    fn execute(mut self, lit: Literal) raises -> Array:
        return _broadcast_literal(self.inputs[0].length, lit.value)

    fn execute(mut self, bin: Binary) raises -> Array:
        var left = self.execute(bin.left)
        var right = self.execute(bin.right)
        if bin.op == ADD:
            return add(left, right)
        elif bin.op == SUB:
            return sub(left, right)
        elif bin.op == MUL:
            return mul(left, right)
        elif bin.op == DIV:
            return div(left, right)
        elif bin.op == EQ:
            return equal(left, right)
        elif bin.op == NE:
            return not_equal(left, right)
        elif bin.op == LT:
            return less(left, right)
        elif bin.op == LE:
            return less_equal(left, right)
        elif bin.op == GT:
            return greater(left, right)
        elif bin.op == GE:
            return greater_equal(left, right)
        elif bin.op == AND:
            return Array(
                and_(
                    PrimitiveArray[bool_dt](data=left),
                    PrimitiveArray[bool_dt](data=right),
                )
            )
        elif bin.op == OR:
            return Array(
                or_(
                    PrimitiveArray[bool_dt](data=left),
                    PrimitiveArray[bool_dt](data=right),
                )
            )
        else:
            raise Error("ValuesExecutor: unknown binary op ", bin.op)

    fn execute(mut self, un: Unary) raises -> Array:
        var child = self.execute(un.child)
        if un.op == NEG:
            return neg(child)
        elif un.op == ABS:
            return abs_(child)
        elif un.op == NOT:
            return Array(not_(PrimitiveArray[bool_dt](data=child)))
        else:
            raise Error("ValuesExecutor: unknown unary op ", un.op)

    fn execute(mut self, node: IsNull) raises -> Array:
        return is_null(self.execute(node.child))

    fn execute(mut self, node: IfElse) raises -> Array:
        var cond = self.execute(node.cond)
        var then_ = self.execute(node.then_)
        var else_ = self.execute(node.else_)
        return select(cond, then_, else_)

    fn execute(mut self, node: Cast) raises -> Array:
        raise Error("ValuesExecutor: cast not implemented")


# ---------------------------------------------------------------------------
# Relational plan executor (overloaded execute per node type)
# ---------------------------------------------------------------------------


struct RelationsExecutor:
    """Executes a relational plan tree, returning a RecordBatch.

    Dispatches to overloaded ``execute`` methods based on ``plan.kind()``,
    mirroring the ValuesExecutor pattern for expression nodes.
    """

    fn __init__(out self):
        pass

    fn execute(self, plan: AnyRelation) raises -> RecordBatch:
        var k = plan.kind()
        if k == IN_MEMORY_TABLE_NODE:
            return self.execute(plan.downcast[InMemoryTable]()[])
        if k == SCAN_NODE:
            return self.execute(plan.downcast[Scan]()[])
        if k == FILTER_NODE:
            return self.execute(plan.downcast[PlanFilter]()[])
        if k == PROJECT_NODE:
            return self.execute(plan.downcast[Project]()[])
        raise Error("RelationsExecutor: unknown relation kind ", k)

    fn execute(self, node: InMemoryTable) raises -> RecordBatch:
        return RecordBatch(copy=node.batch)

    fn execute(self, node: Scan) raises -> RecordBatch:
        raise Error("RelationsExecutor: Scan requires external data source binding")

    fn execute(self, node: PlanFilter) raises -> RecordBatch:
        var child_batch = self.execute(node.input)
        var input_arrays = _batch_to_inputs(child_batch)
        var evaluator = ValuesExecutor(input_arrays^)
        var mask = evaluator.execute(node.predicate)
        var result_cols = List[Array]()
        for i in range(child_batch.num_columns()):
            result_cols.append(filter_(child_batch.columns[i].copy(), mask.copy()))
        return RecordBatch(
            schema=Schema(copy=child_batch.schema), columns=result_cols^
        )

    fn execute(self, node: Project) raises -> RecordBatch:
        var child_batch = self.execute(node.input)
        var result_cols = List[Array]()
        for ref e in node.exprs_:
            var input_arrays = _batch_to_inputs(child_batch)
            var evaluator = ValuesExecutor(input_arrays^)
            var result = evaluator.execute(e)
            result_cols.append(result.copy())
        return RecordBatch(schema=node.schema(), columns=result_cols^)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


fn _copy_inputs(inputs: List[Array]) -> List[Array]:
    """Copy a list of arrays (O(1) per array via ref-counting)."""
    var result = List[Array](capacity=len(inputs))
    for ref inp in inputs:
        result.append(inp.copy())
    return result^


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


fn _empty_like(arr: Array) raises -> Array:
    """Create a zero-length array with the same dtype."""
    comptime for dt in numeric_dtypes:
        if arr.dtype == dt:
            var b = PrimitiveBuilder[dt](0)
            return Array(b.finish_typed())
    raise Error(t"_empty_like: unsupported dtype {arr.dtype}")


fn _slice_inputs(
    inputs: List[Array],
    start: Int,
    end: Int,
) raises -> List[Array]:
    """Return zero-copy slices of each input array for [start, end)."""
    var slices = List[Array](capacity=len(inputs))
    for ref inp in inputs:
        slices.append(inp.slice(start, end - start))
    return slices^


fn _batch_to_inputs(batch: RecordBatch) -> List[Array]:
    """Convert a RecordBatch's columns to a List[Array] for the evaluator."""
    var inputs = List[Array](capacity=batch.num_columns())
    for i in range(batch.num_columns()):
        inputs.append(batch.columns[i].copy())
    return inputs^
