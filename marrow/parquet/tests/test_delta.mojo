"""DELTA_BINARY_PACKED integer decoding."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def _delta_roundtrip(compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # 1000 rows > 128 (default block size) spans multiple delta blocks;
    # int64 monotone + int32 with negatives and every-7th null.
    var t = pa.table(
        Python.dict(
            a=pa.array(np.arange(1000) * 3 - 500, type=pa.int64()),
            b=pa.array(
                Python.evaluate(
                    "[None if i % 7 == 0 else i * i - 100000 for i in"
                    " range(1000)]"
                ),
                type=pa.int32(),
            ),
        )
    )
    var path = String("/tmp/marrow_delta.parquet")
    pq.write_table(
        t,
        path,
        use_dictionary=False,
        column_encoding="DELTA_BINARY_PACKED",
        compression=compression,
    )
    # confirm PyArrow actually used DELTA_BINARY_PACKED
    var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    assert_true(Bool(Python.evaluate("'DELTA_BINARY_PACKED'") in enc))

    var back = read_table(path)
    assert_equal(back.num_rows(), 1000)
    var bat = back.to_batches()[0].copy()

    var ca = bat.columns[0].copy()
    ref a = ca.as_int64()
    assert_equal(a[0].value(), -500)
    assert_equal(a[1].value(), -497)
    assert_equal(a[999].value(), 2497)

    var cb = bat.columns[1].copy()
    ref b = cb.as_int32()
    assert_equal(b.null_count(), 143)  # ceil(1000/7)
    assert_false(b.is_valid(0))
    assert_true(b.is_valid(1))
    assert_equal(b[1].value(), -99999)
    assert_equal(b[999].value(), 898001)
    remove(path)


def test_read_delta_binary_packed() raises:
    _delta_roundtrip("none")


def test_read_delta_binary_packed_snappy() raises:
    _delta_roundtrip("snappy")


def main() raises:
    TestSuite.run[__functions_in_module()]()
