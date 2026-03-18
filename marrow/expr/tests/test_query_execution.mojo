"""Tests for in_memory_table, select/filter on AnyRelation, and plan execution."""

from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import Field, int64, float64
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr import col, lit, in_memory_table, AnyRelation, execute
from marrow.expr.executor import ExecutionContext, plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn _batch() raises -> RecordBatch:
    """Create a test batch with columns x=[1,2,3,4,5] and y=[10,20,30,40,50]."""
    var schema = Schema()
    schema.append(Field("x", int64))
    schema.append(Field("y", int64))
    var x = array[int64]([1, 2, 3, 4, 5])
    var y = array[int64]([10, 20, 30, 40, 50])
    var cols = List[Array]()
    cols.append(Array(x))
    cols.append(Array(y))
    return RecordBatch(schema=schema, columns=cols^)


# ---------------------------------------------------------------------------
# in_memory_table
# ---------------------------------------------------------------------------


def test_in_memory_table_identity() raises:
    """Executing without any operations returns the original batch."""
    var result = execute(in_memory_table(_batch()))
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 2)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_single_column() raises:
    """Selecting a single column returns a 1-column batch."""
    var rel = in_memory_table(_batch()).select("x")
    var result = execute(rel)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.num_rows(), 5)
    assert_equal(result.schema.fields[0].name, "x")
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 1)
    assert_equal(x.unsafe_get(4), 5)


def test_select_multiple_columns() raises:
    """Selecting multiple columns preserves order."""
    var rel = in_memory_table(_batch()).select("y", "x")
    var result = execute(rel)
    assert_equal(result.num_columns(), 2)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.schema.fields[1].name, "x")
    var y = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(y.unsafe_get(0), 10)
    var x = PrimitiveArray[int64](data=result.columns[1].copy())
    assert_equal(x.unsafe_get(0), 1)


def test_select_preserves_values() raises:
    """All values are preserved through select."""
    var rel = in_memory_table(_batch()).select("x")
    var result = execute(rel)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    for i in range(5):
        assert_equal(x.unsafe_get(i), Scalar[int64.native](i + 1))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_greater_than() raises:
    """Filter col("x") > lit(3) keeps rows [4, 5]."""
    var rel = in_memory_table(_batch()).filter(col("x") > lit[int64](3))
    var result = execute(rel)
    assert_equal(result.num_rows(), 2)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 4)
    assert_equal(x.unsafe_get(1), 5)


def test_filter_equality() raises:
    """Filter col("x") == lit(3) keeps one row."""
    var rel = in_memory_table(_batch()).filter(col("x") == lit[int64](3))
    var result = execute(rel)
    assert_equal(result.num_rows(), 1)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 3)
    var y = PrimitiveArray[int64](data=result.columns[1].copy())
    assert_equal(y.unsafe_get(0), 30)


def test_filter_no_match() raises:
    """Filter that matches no rows returns empty batch."""
    var rel = in_memory_table(_batch()).filter(col("x") > lit[int64](100))
    var result = execute(rel)
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# Chained operations
# ---------------------------------------------------------------------------


def test_select_then_filter() raises:
    """Select followed by filter works correctly."""
    var rel = (
        in_memory_table(_batch())
        .select("x", "y")
        .filter(col("x") > lit[int64](2))
    )
    var result = execute(rel)
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 3)
    assert_equal(x.unsafe_get(1), 4)
    assert_equal(x.unsafe_get(2), 5)


def test_filter_then_select() raises:
    """Filter followed by select works correctly."""
    var rel = (
        in_memory_table(_batch()).filter(col("x") > lit[int64](3)).select("y")
    )
    var result = execute(rel)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    var y = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(y.unsafe_get(0), 40)
    assert_equal(y.unsafe_get(1), 50)


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------


def test_streaming_morsel_boundaries() raises:
    """Small morsel_size produces multiple batches that together contain all rows.
    """
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = plan(in_memory_table(_batch()), ctx)
    var batches = proc.to_batches()
    # 5 rows, morsel_size=2 → 3 batches (2+2+1)
    assert_equal(len(batches), 3)
    assert_equal(batches[0].num_rows(), 2)
    assert_equal(batches[1].num_rows(), 2)
    assert_equal(batches[2].num_rows(), 1)


def test_streaming_read_all_matches_execute() raises:
    """read_all() produces the same result as execute()."""
    var rel = in_memory_table(_batch()).filter(col("x") > lit[int64](2))
    var result_exec = execute(rel)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = plan(rel, ctx)
    var result_stream = proc.read_all()
    assert_equal(result_stream.num_rows(), result_exec.num_rows())
    assert_equal(result_stream.num_columns(), result_exec.num_columns())


def test_streaming_filter_skips_empty() raises:
    """Filter that eliminates entire morsels yields no empty batches."""
    # x=[1,2,3,4,5], filter x > 4 → only row 5 passes
    # With morsel_size=2: batches are [1,2], [3,4], [5]
    # First two morsels fully filtered → skipped
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var rel = in_memory_table(_batch()).filter(col("x") > lit[int64](4))
    var proc = plan(rel, ctx)
    var batches = proc.to_batches()
    # Only the last morsel has a matching row
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 1)
    var x = PrimitiveArray[int64](data=batches[0].columns[0].copy())
    assert_equal(x.unsafe_get(0), 5)


def test_streaming_chained_filter_project() raises:
    """Chained filter + project via streaming produces correct values."""
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var rel = (
        in_memory_table(_batch()).filter(col("x") > lit[int64](2)).select("y")
    )
    var proc = plan(rel, ctx)
    var result = proc.read_all()
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    var y = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(y.unsafe_get(0), 30)
    assert_equal(y.unsafe_get(1), 40)
    assert_equal(y.unsafe_get(2), 50)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
