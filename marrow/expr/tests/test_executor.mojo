"""Tests for expression execution and streaming query execution."""

from std.testing import assert_equal, assert_true, TestSuite

from marrow.arrays import array, Array
from marrow.dtypes import int64, float64, bool_ as bool_dt
from marrow.tabular import record_batch
from marrow.expr import (
    Planner,
    col,
    lit,
    in_memory_table,
    execute,
    DISPATCH_CPU,
)
from marrow.expr.executor import ExecutionContext
from marrow.builders import arange


# ---------------------------------------------------------------------------
# Value expression evaluation — morsel boundary correctness
# ---------------------------------------------------------------------------


def test_sequential_fallback() raises:
    """When batch fits in a single morsel, evaluation works correctly."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([10, 20, 30, 40, 50])
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(col(0) + col(1)).eval(batch)
    assert_true(result == Array(array[int64]([11, 22, 33, 44, 55])))


def test_large_add() raises:
    """Add kernel produces correct results on large arrays."""
    var n = 1000
    var a = arange[int64](0, n)
    var b = arange[int64](0, n)
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(col(0) + col(1)).eval(batch).as_primitive[int64]()
    for i in range(n):
        assert_equal(result[i], Scalar[int64.native](i * 2))


def test_large_mul() raises:
    """Mul kernel produces correct results on large arrays."""
    var n = 500
    var a = arange[int64](1, n + 1)
    var b = arange[int64](0, n)
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(col(0) * col(1)).eval(batch).as_primitive[int64]()
    for i in range(n):
        assert_equal(result[i], Scalar[int64.native]((i + 1) * i))


def test_chunk_boundary_values() raises:
    """Values at boundaries are correct."""
    var a = arange[int64](0, 128)
    var batch = record_batch([a], names=["c0"])
    var result = Planner().build(col(0) + lit[int64](1)).eval(batch).as_primitive[int64]()
    for i in range(128):
        assert_equal(result[i], Scalar[int64.native](i + 1))


def test_non_aligned_length() raises:
    """Handles lengths not divisible by SIMD width."""
    var a = arange[int64](0, 100)
    var batch = record_batch([a], names=["c0"])
    var result = Planner().build(-col(0)).eval(batch).as_primitive[int64]()
    for i in range(100):
        assert_equal(result[i], Scalar[int64.native](-i))


def test_single_element() raises:
    """Single-element array works correctly."""
    var a = array[int64]([42])
    var b = array[int64]([8])
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(col(0) + col(1)).eval(batch)
    assert_true(result == Array(array[int64]([50])))


def test_predicate() raises:
    """Predicate produces correct boolean results."""
    var n = 200
    var a = arange[int64](0, n)
    var b = arange[int64](0, n)
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(col(0) < col(1)).eval(batch).as_primitive[bool_dt]()
    # a == b everywhere, so all False
    for i in range(n):
        assert_equal(result[i], 0)


def test_chained_expression() raises:
    """Chained expressions produce correct results."""
    var a = arange[int64](0, 256)
    var b = arange[int64](1, 257)
    var batch = record_batch([a, b], names=["c0", "c1"])
    # (a + b) * (a - b)
    var result = Planner().build(
        (col(0) + col(1)) * (col(0) - col(1))
    ).eval(batch).as_primitive[int64]()
    for i in range(256):
        var expected = (i + (i + 1)) * (i - (i + 1))
        assert_equal(result[i], Scalar[int64.native](expected))


def test_dispatch_cpu_hint() raises:
    """DISPATCH_CPU hint keeps execution on CPU."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([5, 4, 3, 2, 1])
    var batch = record_batch([a, b], names=["c0", "c1"])
    var result = Planner().build(
        (col(0) + col(1)).with_dispatch(DISPATCH_CPU)
    ).eval(batch)
    assert_true(result == Array(array[int64]([6, 6, 6, 6, 6])))


# ---------------------------------------------------------------------------
# in_memory_table
# ---------------------------------------------------------------------------


def test_in_memory_table_identity() raises:
    """Executing without any operations returns the original batch."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(in_memory_table(record_batch([x, y], names=["x", "y"])))
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 2)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_single_column() raises:
    """Selecting a single column returns a 1-column batch."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(in_memory_table(record_batch([x, y], names=["x", "y"])).select("x"))
    assert_equal(result.num_columns(), 1)
    assert_equal(result.num_rows(), 5)
    assert_equal(result.schema.fields[0].name, "x")
    var col_x = result.columns[0].as_primitive[int64]()
    assert_equal(col_x[0], 1)
    assert_equal(col_x[4], 5)


def test_select_multiple_columns() raises:
    """Selecting multiple columns preserves order."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(in_memory_table(record_batch([x, y], names=["x", "y"])).select("y", "x"))
    assert_equal(result.num_columns(), 2)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.schema.fields[1].name, "x")
    var col_y = result.columns[0].as_primitive[int64]()
    assert_equal(col_y[0], 10)
    var col_x = result.columns[1].as_primitive[int64]()
    assert_equal(col_x[0], 1)


def test_select_preserves_values() raises:
    """All values are preserved through select."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(in_memory_table(record_batch([x, y], names=["x", "y"])).select("x"))
    var col_x = result.columns[0].as_primitive[int64]()
    for i in range(5):
        assert_equal(col_x[i], Scalar[int64.native](i + 1))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_greater_than() raises:
    """Filter col("x") > lit(3) keeps rows [4, 5]."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") > lit[int64](3))
    )
    assert_equal(result.num_rows(), 2)
    var col_x = result.columns[0].as_primitive[int64]()
    assert_equal(col_x[0], 4)
    assert_equal(col_x[1], 5)


def test_filter_equality() raises:
    """Filter col("x") == lit(3) keeps one row."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") == lit[int64](3))
    )
    assert_equal(result.num_rows(), 1)
    var col_x = result.columns[0].as_primitive[int64]()
    assert_equal(col_x[0], 3)
    var col_y = result.columns[1].as_primitive[int64]()
    assert_equal(col_y[0], 30)


def test_filter_no_match() raises:
    """Filter that matches no rows returns empty batch."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") > lit[int64](100))
    )
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# Chained operations
# ---------------------------------------------------------------------------


def test_select_then_filter() raises:
    """Select followed by filter works correctly."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .select("x", "y")
        .filter(col("x") > lit[int64](2))
    )
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)
    var col_x = result.columns[0].as_primitive[int64]()
    assert_equal(col_x[0], 3)
    assert_equal(col_x[1], 4)
    assert_equal(col_x[2], 5)


def test_filter_then_select() raises:
    """Filter followed by select works correctly."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var result = execute(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") > lit[int64](3))
        .select("y")
    )
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    var col_y = result.columns[0].as_primitive[int64]()
    assert_equal(col_y[0], 40)
    assert_equal(col_y[1], 50)


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------


def test_streaming_morsel_boundaries() raises:
    """Small morsel_size produces multiple batches that together contain all rows."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(in_memory_table(record_batch([x, y], names=["x", "y"])))
    var batches = proc.to_batches()
    # 5 rows, morsel_size=2 → 3 batches (2+2+1)
    assert_equal(len(batches), 3)
    assert_equal(batches[0].num_rows(), 2)
    assert_equal(batches[1].num_rows(), 2)
    assert_equal(batches[2].num_rows(), 1)


def test_streaming_read_all_matches_execute() raises:
    """``read_all()`` produces the same result as execute()."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var rel = in_memory_table(record_batch([x, y], names=["x", "y"])).filter(col("x") > lit[int64](2))
    var result_exec = execute(rel)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(rel)
    var result_stream = proc.read_all()
    assert_equal(result_stream.num_rows(), result_exec.num_rows())
    assert_equal(result_stream.num_columns(), result_exec.num_columns())


def test_streaming_filter_skips_empty() raises:
    """Filter that eliminates entire morsels yields no empty batches."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") > lit[int64](4))
    )
    var batches = proc.to_batches()
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 1)
    var col_x = batches[0].columns[0].as_primitive[int64]()
    assert_equal(col_x[0], 5)


def test_streaming_chained_filter_project() raises:
    """Chained filter + project via streaming produces correct values."""
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(
        in_memory_table(record_batch([x, y], names=["x", "y"]))
        .filter(col("x") > lit[int64](2))
        .select("y")
    )
    var result = proc.read_all()
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    var col_y = result.columns[0].as_primitive[int64]()
    assert_equal(col_y[0], 30)
    assert_equal(col_y[1], 40)
    assert_equal(col_y[2], 50)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
