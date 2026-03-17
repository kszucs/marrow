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
from marrow.expr import (
    Expr,
    LOAD, ADD, SUB, MUL, DIV, NEG, ABS, LITERAL,
    EQ, NE, LT, LE, GT, GE, AND, OR, NOT, IS_NULL, IF_ELSE,
    DISPATCH_CPU,
    DISPATCH_GPU,
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

    fn execute(
        self,
        expr: Expr,
        inputs: List[Array],
    ) raises -> Array:
        """Evaluate an expression tree over the given input arrays.

        Handles both arithmetic (ADD, SUB, …) and predicate (EQ, LT, …)
        nodes.  Returns a type-erased ``Array``.

        Args:
            expr: The expression tree to evaluate.
            inputs: Input arrays referenced by ``Expr.input(idx)`` nodes.

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
            return _eval(expr, inputs)

        var results = List[Array](capacity=n_morsels)
        for _ in range(n_morsels):
            results.append(_empty_like(inputs[0]))

        @parameter
        fn process_morsel(i: Int) raises:
            var start = i * morsel_size
            var end = min(start + morsel_size, length)
            var chunk_inputs = _slice_inputs(inputs, start, end)
            results[i] = _eval(expr, chunk_inputs)

        sync_parallelize[process_morsel](n_morsels)
        return concat(results)


# ---------------------------------------------------------------------------
# Expression tree interpreter (type-erased)
# ---------------------------------------------------------------------------


fn _eval(expr: Expr, inputs: List[Array]) raises -> Array:
    """Walk the expression tree using type-erased kernel APIs."""
    var k = expr.kind()
    if k == LOAD:
        return inputs[expr.as_column()[].index].copy()
    if k == LITERAL:
        return _broadcast_literal(
            inputs[0].dtype, inputs[0].length, expr.as_literal()[].value
        )
    if k == ADD:
        var bin = expr.as_binary()
        return add(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == SUB:
        var bin = expr.as_binary()
        return sub(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == MUL:
        var bin = expr.as_binary()
        return mul(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == DIV:
        var bin = expr.as_binary()
        return div(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == NEG:
        return neg(_eval(expr.as_unary()[].child, inputs))
    if k == ABS:
        return abs_(_eval(expr.as_unary()[].child, inputs))
    if k == EQ:
        var bin = expr.as_binary()
        return equal(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == NE:
        var bin = expr.as_binary()
        return not_equal(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == LT:
        var bin = expr.as_binary()
        return less(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == LE:
        var bin = expr.as_binary()
        return less_equal(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == GT:
        var bin = expr.as_binary()
        return greater(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == GE:
        var bin = expr.as_binary()
        return greater_equal(_eval(bin[].left, inputs), _eval(bin[].right, inputs))
    if k == AND:
        var bin = expr.as_binary()
        return Array(
            and_(
                PrimitiveArray[bool_dt](data=_eval(bin[].left, inputs)),
                PrimitiveArray[bool_dt](data=_eval(bin[].right, inputs)),
            )
        )
    if k == OR:
        var bin = expr.as_binary()
        return Array(
            or_(
                PrimitiveArray[bool_dt](data=_eval(bin[].left, inputs)),
                PrimitiveArray[bool_dt](data=_eval(bin[].right, inputs)),
            )
        )
    if k == NOT:
        return Array(
            not_(
                PrimitiveArray[bool_dt](
                    data=_eval(expr.as_unary()[].child, inputs)
                )
            )
        )
    if k == IS_NULL:
        return is_null(_eval(expr.as_is_null()[].child, inputs))
    if k == IF_ELSE:
        var ie = expr.as_if_else()
        return select(
            _eval(ie[].cond, inputs),
            _eval(ie[].then_, inputs),
            _eval(ie[].else_, inputs),
        )
    raise Error(t"_eval: unknown Expr kind {k}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


fn _broadcast_literal(
    dtype: DataType, length: Int, value: Float64
) raises -> Array:
    """Create an array filled with a scalar literal, matching the given dtype."""
    comptime for dt in numeric_dtypes:
        if dtype == dt:
            var scalar = Scalar[dt.native](value)
            var builder = PrimitiveBuilder[dt](length)
            for i in range(length):
                builder._buffer.unsafe_set[dt.native](i, scalar)
            builder._length = length
            return Array(builder.finish_typed())
    raise Error(t"_broadcast_literal: unsupported dtype {dtype}")


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
