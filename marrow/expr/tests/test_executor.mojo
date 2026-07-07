"""Tests for expression execution and streaming query execution."""

from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite
from std.python import Python
from std.os import remove

from marrow.arrays import AnyArray, BoolArray
from marrow.builders import array
from marrow.dtypes import (
    int64,
    float64,
    bool_ as bool_dt,
    Int64Type,
    AnyDataType,
)
from marrow.tabular import record_batch
from marrow.expr import (
    DynValue,
    Planner,
    col,
    lit,
    in_memory_table,
    parquet_scan,
    execute,
)
from marrow.expr.executor import ExecutionContext
from marrow.builders import arange


# ---------------------------------------------------------------------------
# Value expression evaluation — morsel boundary correctness
# ---------------------------------------------------------------------------


def test_sequential_fallback() raises:
    """When batch fits in a single morsel, evaluation works correctly."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = (col(0) + col(1)).eval(batch)
    assert_true(result == array([11, 22, 33, 44, 55], int64).to_any())


def test_large_add() raises:
    """Add kernel produces correct results on large arrays."""
    var n = 1000
    var a = arange[Int64Type](0, n)
    var b = arange[Int64Type](0, n)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var tmp_large_add = (col(0) + col(1)).eval(batch)
    ref result = tmp_large_add.as_int64()
    for i in range(n):
        assert_equal(result[i].value(), Scalar[int64.native](i * 2))


def test_large_mul() raises:
    """Mul kernel produces correct results on large arrays."""
    var n = 500
    var a = arange[Int64Type](1, n + 1)
    var b = arange[Int64Type](0, n)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var tmp_large_mul = (col(0) * col(1)).eval(batch)
    ref result = tmp_large_mul.as_int64()
    for i in range(n):
        assert_equal(result[i].value(), Int64((i + 1) * i))


def test_chunk_boundary_values() raises:
    """Values at boundaries are correct."""
    var a = arange[Int64Type](0, 128)
    var batch = record_batch([a^], names=["c0"])
    var tmp_chunk_boundary = (col(0) + lit[Int64Type](1)).eval(batch)
    ref result = tmp_chunk_boundary.as_int64()
    for i in range(128):
        assert_equal(result[i].value(), Scalar[int64.native](i + 1))


def test_non_aligned_length() raises:
    """Handles lengths not divisible by SIMD width."""
    var a = arange[Int64Type](0, 100)
    var batch = record_batch([a^], names=["c0"])
    var tmp_non_aligned = (-col(0)).eval(batch)
    ref result = tmp_non_aligned.as_int64()
    for i in range(100):
        assert_equal(result[i].value(), Scalar[int64.native](-i))


def test_single_element() raises:
    """Single-element array works correctly."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = (col(0) + col(1)).eval(batch)
    assert_true(result == array([50], int64).to_any())


def test_predicate() raises:
    """Predicate produces correct boolean results."""
    var n = 200
    var a = arange[Int64Type](0, n)
    var b = arange[Int64Type](0, n)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var tmp_pred = (col(0) < col(1)).eval(batch)
    ref result = tmp_pred.as_bool()
    # a == b everywhere, so all False
    for i in range(n):
        assert_equal(result[i], False)


def test_chained_expression() raises:
    """Chained expressions produce correct results."""
    var a = arange[Int64Type](0, 256)
    var b = arange[Int64Type](1, 257)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    # (a + b) * (a - b)
    var tmp_chained = ((col(0) + col(1)) * (col(0) - col(1))).eval(batch)
    ref result = tmp_chained.as_int64()
    for i in range(256):
        var expected = (i + (i + 1)) * (i - (i + 1))
        assert_equal(result[i].value(), Scalar[int64.native](expected))


def test_dispatch_cpu_hint() raises:
    """Expression evaluation works without dispatch hints."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([5, 4, 3, 2, 1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = (col(0) + col(1)).eval(batch)
    assert_true(result == array([6, 6, 6, 6, 6], int64).to_any())


# ---------------------------------------------------------------------------
# in_memory_table
# ---------------------------------------------------------------------------


def test_in_memory_table_identity() raises:
    """Executing without any operations returns the original batch."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"]))
    )
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 2)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_single_column() raises:
    """Selecting a single column returns a 1-column batch."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).select("x")
    )
    assert_equal(result.num_columns(), 1)
    assert_equal(result.num_rows(), 5)
    assert_equal(result.schema.fields[0].name, "x")
    ref col_x = result.columns[0].as_int64()
    assert_equal(col_x[0].value(), 1)
    assert_equal(col_x[4].value(), 5)


def test_select_multiple_columns() raises:
    """Selecting multiple columns preserves order."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).select(
            "y", "x"
        )
    )
    assert_equal(result.num_columns(), 2)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.schema.fields[1].name, "x")
    ref col_y = result.columns[0].as_int64()
    assert_equal(col_y[0].value(), 10)
    ref col_x = result.columns[1].as_int64()
    assert_equal(col_x[0].value(), 1)


def test_select_preserves_values() raises:
    """All values are preserved through select."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).select("x")
    )
    ref col_x = result.columns[0].as_int64()
    for i in range(5):
        assert_equal(col_x[i].value(), Scalar[int64.native](i + 1))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_greater_than() raises:
    """Filter col("x") > lit(3) keeps rows [4, 5]."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).filter(
            col(0) > lit[Int64Type](3)
        )
    )
    assert_equal(result.num_rows(), 2)
    ref col_x = result.columns[0].as_int64()
    assert_equal(col_x[0].value(), 4)
    assert_equal(col_x[1].value(), 5)


def test_filter_equality() raises:
    """Filter col(0) == lit(3) keeps one row."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).filter(
            col(0) == lit[Int64Type](3)
        )
    )
    assert_equal(result.num_rows(), 1)
    ref col_x = result.columns[0].as_int64()
    assert_equal(col_x[0].value(), 3)
    ref col_y = result.columns[1].as_int64()
    assert_equal(col_y[0].value(), 30)


def test_filter_no_match() raises:
    """Filter that matches no rows returns empty batch."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).filter(
            col(0) > lit[Int64Type](100)
        )
    )
    assert_equal(result.num_rows(), 0)


# ---------------------------------------------------------------------------
# Chained operations
# ---------------------------------------------------------------------------


def test_select_then_filter() raises:
    """Select followed by filter works correctly."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"]))
        .select("x", "y")
        .filter(col(0) > lit[Int64Type](2))
    )
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 2)
    ref col_x = result.columns[0].as_int64()
    assert_equal(col_x[0].value(), 3)
    assert_equal(col_x[1].value(), 4)
    assert_equal(col_x[2].value(), 5)


def test_filter_then_select() raises:
    """Filter followed by select works correctly."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var result = execute(
        in_memory_table(record_batch([x^, y^], names=["x", "y"]))
        .filter(col(0) > lit[Int64Type](3))
        .select("y")
    )
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    ref col_y = result.columns[0].as_int64()
    assert_equal(col_y[0].value(), 40)
    assert_equal(col_y[1].value(), 50)


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------


def test_streaming_morsel_boundaries() raises:
    """Small morsel_size produces multiple batches that together contain all rows.
    """
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(
        in_memory_table(record_batch([x^, y^], names=["x", "y"]))
    )
    var batches = proc.to_batches()
    # 5 rows, morsel_size=2 → 3 batches (2+2+1)
    assert_equal(len(batches), 3)
    assert_equal(batches[0].num_rows(), 2)
    assert_equal(batches[1].num_rows(), 2)
    assert_equal(batches[2].num_rows(), 1)


def test_streaming_read_all_matches_execute() raises:
    """``read_all()`` produces the same result as execute()."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var rel = in_memory_table(record_batch([x^, y^], names=["x", "y"])).filter(
        col("x") > lit[Int64Type](2)
    )
    var result_exec = execute(rel)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(rel)
    var result_stream = proc.read_all()
    assert_equal(result_stream.num_rows(), result_exec.num_rows())
    assert_equal(result_stream.num_columns(), result_exec.num_columns())


def test_streaming_filter_skips_empty() raises:
    """Filter that eliminates entire morsels yields no empty batches."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(
        in_memory_table(record_batch([x^, y^], names=["x", "y"])).filter(
            col(0) > lit[Int64Type](4)
        )
    )
    var batches = proc.to_batches()
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 1)
    ref col_x = batches[0].columns[0].as_int64()
    assert_equal(col_x[0].value(), 5)


def test_streaming_chained_filter_project() raises:
    """Chained filter + project via streaming produces correct values."""
    var x = array([1, 2, 3, 4, 5], int64)
    var y = array([10, 20, 30, 40, 50], int64)
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var proc = Planner(ctx).build(
        in_memory_table(record_batch([x^, y^], names=["x", "y"]))
        .filter(col(0) > lit[Int64Type](2))
        .select("y")
    )
    var result = proc.read_all()
    assert_equal(result.num_rows(), 3)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    ref col_y = result.columns[0].as_int64()
    assert_equal(col_y[0].value(), 30)
    assert_equal(col_y[1].value(), 40)
    assert_equal(col_y[2].value(), 50)


# ---------------------------------------------------------------------------
# ParquetScan execution
# ---------------------------------------------------------------------------


def _write_test_parquet(path: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = pa.table(
        Python.dict(
            id=pa.array(Python.list(1, 2, 3, 4, 5), type=pa.int64()),
            val=pa.array(
                Python.list(1.0, 2.0, 3.0, 4.0, 5.0), type=pa.float64()
            ),
        )
    )
    pq.write_table(t, path)


def test_parquet_scan_execute() raises:
    """Full scan reads all rows and columns from a Parquet file."""
    var path = "/tmp/marrow_test_parquet_scan.parquet"
    _write_test_parquet(path)
    var result = execute(parquet_scan(path))
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 2)
    ref ids = result.columns[0].as_int64()
    assert_equal(ids[0].value(), 1)
    assert_equal(ids[4].value(), 5)
    remove(path)


def test_parquet_scan_filter() raises:
    """Filter over a ParquetScan keeps only matching rows."""
    var path = "/tmp/marrow_test_parquet_scan_filter.parquet"
    _write_test_parquet(path)
    var result = execute(parquet_scan(path).filter(col(0) > lit[Int64Type](3)))
    assert_equal(result.num_rows(), 2)
    ref ids = result.columns[0].as_int64()
    assert_equal(ids[0].value(), 4)
    assert_equal(ids[1].value(), 5)
    remove(path)


def test_parquet_scan_select() raises:
    """Select over a ParquetScan projects to a single column."""
    var path = "/tmp/marrow_test_parquet_scan_select.parquet"
    _write_test_parquet(path)
    var result = execute(parquet_scan(path).select("id"))
    assert_equal(result.num_rows(), 5)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "id")
    remove(path)


def test_parquet_scan_filter_select() raises:
    """Chained filter + select over a ParquetScan."""
    var path = "/tmp/marrow_test_parquet_scan_filter_select.parquet"
    _write_test_parquet(path)
    var result = execute(
        parquet_scan(path).filter(col(0) > lit[Int64Type](3)).select("id")
    )
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "id")
    ref ids = result.columns[0].as_int64()
    assert_equal(ids[0].value(), 4)
    assert_equal(ids[1].value(), 5)
    remove(path)


# ---------------------------------------------------------------------------
# Aggregate processor
# ---------------------------------------------------------------------------


def test_aggregate_sum() raises:
    """Grouped sum via the expression system."""
    var cols = List[AnyArray]()
    cols.append(array([1, 2, 1, 2, 1], int64).to_any())
    cols.append(array([10, 20, 30, 40, 50], int64).to_any())
    var batch = record_batch(cols^, names=["key", "val"])

    var plan = in_memory_table(batch).aggregate([col(0)], [col(1)], ["sum"])
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.num_columns(), 2)  # key + sum

    # Key column.
    ref k = result.columns[0].as_int64()
    assert_equal(k[0].value(), 1)
    assert_equal(k[1].value(), 2)

    # Sum column (int64 — integer input produces integer output).
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 90)  # 10 + 30 + 50
    assert_equal(s[1].value(), 60)  # 20 + 40


def test_aggregate_count() raises:
    """Grouped count via the expression system."""
    var cols = List[AnyArray]()
    cols.append(array([1, 2, 1, 2, 1], int64).to_any())
    cols.append(array([10, 20, 30, 40, 50], int64).to_any())
    var batch = record_batch(cols^, names=["key", "val"])

    var plan = in_memory_table(batch).aggregate([col(0)], [col(1)], ["count"])
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    ref c = result.columns[1].as_int64()
    assert_equal(c[0].value(), 3)  # key=1: 3 rows
    assert_equal(c[1].value(), 2)  # key=2: 2 rows


def test_aggregate_sum_int64_precision() raises:
    """Grouped int64 sum via the expression system must stay exact above 2**53.
    """
    var cols = List[AnyArray]()
    cols.append(array([1, 1], int64).to_any())
    cols.append(array([9_007_199_254_740_993, 1], int64).to_any())
    var batch = record_batch(cols^, names=["key", "val"])

    var plan = in_memory_table(batch).aggregate([col(0)], [col(1)], ["sum"])
    var result = execute(plan)
    assert_equal(result.num_rows(), 1)
    assert_true(result.schema.fields[1].dtype == AnyDataType(int64))
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 9_007_199_254_740_994)


def test_aggregate_small_morsel() raises:
    """Aggregate with small morsel size forces multiple pulls."""
    var cols = List[AnyArray]()
    cols.append(array([1, 2, 1, 2, 1, 2], int64).to_any())
    cols.append(array([10, 20, 30, 40, 50, 60], int64).to_any())
    var batch = record_batch(cols^, names=["key", "val"])

    var plan = in_memory_table(batch).aggregate([col(0)], [col(1)], ["sum"])
    var ctx = ExecutionContext()
    ctx.morsel_size = 2
    var result = execute(plan, ctx)
    assert_equal(result.num_rows(), 2)
    ref s = result.columns[1].as_int64()
    assert_equal(s[0].value(), 90)  # key=1: 10+30+50
    assert_equal(s[1].value(), 120)  # key=2: 20+40+60


def main() raises:
    TestSuite.run[__functions_in_module()]()
