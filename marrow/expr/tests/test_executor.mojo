"""Tests for PipelineExecutor: correctness, parallel paths, edge cases."""

from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import int64, float64, bool_ as bool_dt
from marrow.expr import Expr, DISPATCH_CPU
from marrow.builders import PrimitiveBuilder
from marrow.expr.executor import ExecutionContext, PipelineExecutor


fn _make(a: PrimitiveArray[int64]) -> List[Array]:
    var inputs = List[Array]()
    inputs.append(Array(a))
    return inputs^


fn _make(
    a: PrimitiveArray[int64], b: PrimitiveArray[int64]
) -> List[Array]:
    var inputs = List[Array]()
    inputs.append(Array(a))
    inputs.append(Array(b))
    return inputs^


fn _arrays_equal(a: PrimitiveArray[int64], b: PrimitiveArray[int64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a.unsafe_get(i) != b.unsafe_get(i):
            return False
    return True


# ---------------------------------------------------------------------------
# Sequential fallback (morsel_size > array length)
# ---------------------------------------------------------------------------


def test_sequential_fallback() raises:
    """When morsel_size > array length, uses single-thread path."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([10, 20, 30, 40, 50])
    var expr = Expr.add(Expr.input(0), Expr.input(1))

    var ctx = ExecutionContext()
    ctx.morsel_size = 1_000_000
    var exec = PipelineExecutor(ctx)
    var result = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))

    var sequential = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a, b))
    )
    assert_equal(_arrays_equal(result, sequential), True)


# ---------------------------------------------------------------------------
# Parallel path — output matches sequential
# ---------------------------------------------------------------------------


def test_parallel_matches_sequential_add() raises:
    """Parallel execute produces the same result as sequential for add."""
    var n = 1000
    var ab = PrimitiveBuilder[int64](n)
    var bb = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i))
        bb.unsafe_append(Scalar[int64.native](n - i))

    var a = ab.finish_typed()
    var b = bb.finish_typed()
    var expr = Expr.add(Expr.input(0), Expr.input(1))

    var sequential = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a, b))
    )

    var ctx = ExecutionContext()
    ctx.morsel_size = 64
    ctx.num_cpu_workers = 4
    var exec = PipelineExecutor(ctx)
    var parallel = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))

    assert_equal(_arrays_equal(parallel, sequential), True)


def test_parallel_matches_sequential_mul() raises:
    """Parallel execute produces the same result as sequential for mul."""
    var n = 500
    var ab = PrimitiveBuilder[int64](n)
    var bb = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i + 1))
        bb.unsafe_append(Scalar[int64.native](2))

    var a = ab.finish_typed()
    var b = bb.finish_typed()
    var expr = Expr.mul(Expr.input(0), Expr.input(1))

    var sequential = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a, b))
    )

    var ctx = ExecutionContext()
    ctx.morsel_size = 50
    ctx.num_cpu_workers = 2
    var exec = PipelineExecutor(ctx)
    var parallel = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))

    assert_equal(_arrays_equal(parallel, sequential), True)


# ---------------------------------------------------------------------------
# Chunk boundary correctness
# ---------------------------------------------------------------------------


def test_chunk_boundary_values() raises:
    """Values at chunk boundaries (first/last element of each morsel) are correct."""
    var n = 128
    var ab = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i * 10))

    var a = ab.finish_typed()
    var expr = Expr.add(Expr.input(0), Expr.literal[int64](1))

    var ctx = ExecutionContext()
    ctx.morsel_size = 32
    ctx.num_cpu_workers = 2
    var exec = PipelineExecutor(ctx)

    var result = PrimitiveArray[int64](data=exec.execute(expr, _make(a)))
    var expected = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a))
    )

    assert_equal(_arrays_equal(result, expected), True)


# ---------------------------------------------------------------------------
# Non-SIMD-aligned lengths
# ---------------------------------------------------------------------------


def test_non_aligned_length_parallel() raises:
    """Parallel execute handles lengths not divisible by morsel_size."""
    var n = 100
    var ab = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i))

    var a = ab.finish_typed()
    var expr = Expr.neg(Expr.input(0))

    var ctx = ExecutionContext()
    ctx.morsel_size = 32
    ctx.num_cpu_workers = 2
    var exec = PipelineExecutor(ctx)

    var result = PrimitiveArray[int64](data=exec.execute(expr, _make(a)))
    var expected = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a))
    )

    assert_equal(_arrays_equal(result, expected), True)


# ---------------------------------------------------------------------------
# Single-element arrays
# ---------------------------------------------------------------------------


def test_single_element_parallel() raises:
    """Single-element array falls through to sequential path gracefully."""
    var a = array[int64]([42])
    var b = array[int64]([8])
    var expr = Expr.add(Expr.input(0), Expr.input(1))

    var exec = PipelineExecutor()
    var result = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))
    assert_equal(result.unsafe_get(0), 50)


# ---------------------------------------------------------------------------
# Predicate execution
# ---------------------------------------------------------------------------


def test_execute_pred_parallel() raises:
    """Parallel predicate produces same result as sequential."""
    var n = 200
    var ab = PrimitiveBuilder[int64](n)
    var bb = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i))
        bb.unsafe_append(Scalar[int64.native](100))

    var a = ab.finish_typed()
    var b = bb.finish_typed()
    var expr = Expr.less(Expr.input(0), Expr.input(1))

    var sequential = PrimitiveArray[bool_dt](
        data=PipelineExecutor().execute(expr, _make(a, b))
    )

    var ctx = ExecutionContext()
    ctx.morsel_size = 40
    ctx.num_cpu_workers = 3
    var exec = PipelineExecutor(ctx)
    var parallel = PrimitiveArray[bool_dt](
        data=exec.execute(expr, _make(a, b))
    )

    assert_equal(len(parallel), n)
    for i in range(n):
        assert_equal(
            parallel.unsafe_get(i), sequential.unsafe_get(i)
        )


# ---------------------------------------------------------------------------
# Chained expression
# ---------------------------------------------------------------------------


def test_parallel_chained_expression() raises:
    """Parallel executor handles chained expressions correctly."""
    var n = 256
    var ab = PrimitiveBuilder[int64](n)
    var bb = PrimitiveBuilder[int64](n)
    for i in range(n):
        ab.unsafe_append(Scalar[int64.native](i))
        bb.unsafe_append(Scalar[int64.native](i + 1))

    var a = ab.finish_typed()
    var b = bb.finish_typed()

    # (a + b) * (a - b)
    var expr = Expr.mul(
        Expr.add(Expr.input(0), Expr.input(1)),
        Expr.sub(Expr.input(0), Expr.input(1)),
    )
    var sequential = PrimitiveArray[int64](
        data=PipelineExecutor().execute(expr, _make(a, b))
    )

    var ctx = ExecutionContext()
    ctx.morsel_size = 64
    ctx.num_cpu_workers = 2
    var exec = PipelineExecutor(ctx)
    var parallel = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))

    assert_equal(_arrays_equal(parallel, sequential), True)


# ---------------------------------------------------------------------------
# DISPATCH_CPU hint bypasses GPU auto-dispatch
# ---------------------------------------------------------------------------


def test_dispatch_cpu_hint() raises:
    """DISPATCH_CPU hint keeps execution on CPU even if ctx has device."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([5, 4, 3, 2, 1])

    var expr = Expr.add(Expr.input(0), Expr.input(1)).with_dispatch(
        DISPATCH_CPU
    )

    var exec = PipelineExecutor()
    var result = PrimitiveArray[int64](data=exec.execute(expr, _make(a, b)))

    for i in range(5):
        assert_equal(result.unsafe_get(i), Scalar[int64.native](6))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
