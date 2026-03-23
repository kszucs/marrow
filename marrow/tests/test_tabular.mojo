"""Tests for RecordBatch and Table abstractions."""
from std.testing import assert_equal, assert_true, TestSuite
from marrow.tabular import RecordBatch, Table
from marrow.arrays import AnyArray
from marrow.schema import Schema
from marrow.dtypes import int32, int64, float64, Field
from marrow.builders import array, PrimitiveBuilder


def test_record_batch_construction() raises:
    """Test basic RecordBatch construction and property accessors."""
    var schema = Schema(fields=[Field("x", int32), Field("y", float64)])
    var col_x: AnyArray = array[int32]([1, 2, 3])
    var by = PrimitiveBuilder[float64](3)
    by.append(1.0)
    by.append(2.0)
    by.append(3.0)
    var col_y: AnyArray = by.finish()
    var columns = List[AnyArray]()
    columns.append(col_x^)
    columns.append(col_y^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.num_rows(), 3)
    assert_equal(batch.num_columns(), 2)
    assert_equal(batch.column_names(), List[String](["x", "y"]))


def test_record_batch_column_access_by_index() raises:
    """Test column access by index."""
    var schema = Schema(fields=[Field("a", int32), Field("b", int64)])
    var col_a: AnyArray = array[int32]([10, 20, 30])
    var col_b: AnyArray = array[int64]([100, 200, 300])
    var columns = List[AnyArray]()
    columns.append(col_a^)
    columns.append(col_b^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.column(0).length(), 3)
    assert_equal(batch.column(1).length(), 3)
    assert_equal(batch.column(0).as_int32().unsafe_get(0), 10)
    assert_equal(batch.column(1).as_int64().unsafe_get(2), 300)


def test_record_batch_column_access_by_name() raises:
    """Test column access by name."""
    var schema = Schema(fields=[Field("a", int32), Field("b", int64)])
    var col_a: AnyArray = array[int32]([7, 8, 9])
    var col_b: AnyArray = array[int64]([70, 80, 90])
    var columns = List[AnyArray]()
    columns.append(col_a^)
    columns.append(col_b^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.column("a").as_int32().unsafe_get(0), 7)
    assert_equal(batch.column("b").as_int64().unsafe_get(1), 80)


def test_record_batch_slice() raises:
    """Test zero-copy RecordBatch slice."""
    var schema = Schema(fields=[Field("v", int32)])
    var col: AnyArray = array[int32]([10, 20, 30, 40, 50])
    var columns = List[AnyArray]()
    columns.append(col^)
    var batch = RecordBatch(schema=schema, columns=columns^)

    var sliced = batch.slice(1, 3)
    assert_equal(sliced.num_rows(), 3)
    assert_equal(sliced.column(0).as_int32().unsafe_get(0), 20)
    assert_equal(sliced.column(0).as_int32().unsafe_get(2), 40)

    var tail = batch.slice(3)
    assert_equal(tail.num_rows(), 2)
    assert_equal(tail.column(0).as_int32().unsafe_get(0), 40)


def test_table_from_batches() raises:
    """Test Table.from_batches with multiple record batches."""
    var schema = Schema(fields=[Field("v", int32)])
    var cols1 = List[AnyArray]()
    cols1.append(array[int32]([1, 2, 3]))
    var b1 = RecordBatch(schema=schema, columns=cols1^)
    var cols2 = List[AnyArray]()
    cols2.append(array[int32]([4, 5]))
    var b2 = RecordBatch(schema=schema, columns=cols2^)
    var batches = List[RecordBatch]()
    batches.append(b1^)
    batches.append(b2^)
    var table = Table.from_batches(schema, batches^)
    assert_equal(table.num_rows(), 5)
    assert_equal(table.num_columns(), 1)
    assert_equal(len(table.column(0).chunks), 2)
    assert_equal(table.column_names(), List[String](["v"]))


def test_table_column_access() raises:
    """Test Table column access by index and name."""
    var schema = Schema(fields=[Field("a", int32), Field("b", float64)])
    var bf = PrimitiveBuilder[float64](2)
    bf.append(3.0)
    bf.append(4.0)
    var columns = List[AnyArray]()
    columns.append(array[int32]([1, 2]))
    columns.append(bf.finish())
    var b = RecordBatch(schema=schema, columns=columns^)
    var batches = List[RecordBatch]()
    batches.append(b^)
    var table = Table.from_batches(schema, batches^)
    assert_equal(table.column(0).length, 2)
    assert_equal(table.column("b").length, 2)


def test_record_batch_eq() raises:
    """Two RecordBatches with the same schema and column values are equal."""
    var schema = Schema(fields=[Field("x", int32), Field("y", int32)])
    var a = RecordBatch(
        schema,
        [AnyArray(array[int32]([1, 2, 3])), AnyArray(array[int32]([4, 5, 6]))],
    )
    var b = RecordBatch(
        schema,
        [AnyArray(array[int32]([1, 2, 3])), AnyArray(array[int32]([4, 5, 6]))],
    )
    assert_true(a == b)


def test_record_batch_eq_unequal() raises:
    """RecordBatches with different column values are not equal."""
    var schema = Schema(fields=[Field("x", int32)])
    var a = RecordBatch(schema, [AnyArray(array[int32]([1, 2, 3]))])
    var b = RecordBatch(schema, [AnyArray(array[int32]([1, 2, 99]))])
    assert_true(not (a == b))


def test_record_batch_eq_sliced() raises:
    """Regression: sliced RecordBatches (non-zero column offset) compare correctly.

    The old _arrays_equal implementation compared raw buffer bytes without
    respecting array offsets, so logically-equal slices of different buffers
    could compare unequal.
    """
    var schema = Schema(fields=[Field("x", int32)])
    # Build two independent arrays with the same values.
    var full_a = array[int32]([10, 20, 30, 40, 50])
    var full_b = array[int32]([10, 20, 30, 40, 50])
    # Slice both at the same range: [20, 30, 40]
    var batch_a = RecordBatch(schema, [AnyArray(full_a.slice(1, 3))])
    var batch_b = RecordBatch(schema, [AnyArray(full_b.slice(1, 3))])
    assert_true(batch_a == batch_b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
