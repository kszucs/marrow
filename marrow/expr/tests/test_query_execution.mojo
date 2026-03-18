"""Tests for in_memory_table, select/filter on AnyRelation, and plan execution."""

from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import Field, int64, float64
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr import col, lit, in_memory_table, AnyRelation, PipelineExecutor


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
    var result = PipelineExecutor().execute(in_memory_table(_batch()))
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 2)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_single_column() raises:
    """Selecting a single column returns a 1-column batch."""
    var plan = in_memory_table(_batch()).select("x")
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.num_rows(), 5)
    assert_equal(result.schema.fields[0].name, "x")
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 1)
    assert_equal(x.unsafe_get(4), 5)


def test_select_multiple_columns() raises:
    """Selecting multiple columns preserves order."""
    var plan = in_memory_table(_batch()).select("y", "x")
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_columns(), 2)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.schema.fields[1].name, "x")
    var y = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(y.unsafe_get(0), 10)
    var x = PrimitiveArray[int64](data=result.columns[1].copy())
    assert_equal(x.unsafe_get(0), 1)


def test_select_preserves_values() raises:
    """All values are preserved through select."""
    var plan = in_memory_table(_batch()).select("x")
    var result = PipelineExecutor().execute(plan)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    for i in range(5):
        assert_equal(x.unsafe_get(i), Scalar[int64.native](i + 1))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_greater_than() raises:
    """Filter col("x") > lit(3) keeps rows [4, 5]."""
    var plan = in_memory_table(_batch()).filter(col("x") > lit[int64](3))
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_rows(), 2)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 4)
    assert_equal(x.unsafe_get(1), 5)


def test_filter_equality() raises:
    """Filter col("x") == lit(3) keeps one row."""
    var plan = in_memory_table(_batch()).filter(col("x") == lit[int64](3))
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_rows(), 1)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 3)
    var y = PrimitiveArray[int64](data=result.columns[1].copy())
    assert_equal(y.unsafe_get(0), 30)


def test_filter_no_match() raises:
    """Filter that matches no rows returns empty batch."""
    var plan = in_memory_table(_batch()).filter(col("x") > lit[int64](100))
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# Chained operations
# ---------------------------------------------------------------------------


def test_select_then_filter() raises:
    """Select followed by filter works correctly."""
    var plan = in_memory_table(_batch()).select(
        "x", "y"
    ).filter(col("x") > lit[int64](2))
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)
    var x = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(x.unsafe_get(0), 3)
    assert_equal(x.unsafe_get(1), 4)
    assert_equal(x.unsafe_get(2), 5)


def test_filter_then_select() raises:
    """Filter followed by select works correctly."""
    var plan = in_memory_table(_batch()).filter(
        col("x") > lit[int64](3)
    ).select("y")
    var result = PipelineExecutor().execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    var y = PrimitiveArray[int64](data=result.columns[0].copy())
    assert_equal(y.unsafe_get(0), 40)
    assert_equal(y.unsafe_get(1), 50)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
