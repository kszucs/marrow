"""DELTA_BYTE_ARRAY and DELTA_LENGTH_BYTE_ARRAY string decoding."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def _roundtrip(encoding: String, compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # varying lengths + shared prefixes exercise the incremental reconstruction;
    # every 9th null; 600 rows spans multiple delta blocks.
    var t = pa.table(
        Python.dict(
            s=pa.array(
                Python.evaluate(
                    "[None if i % 9 == 0 else 'item-%04d-%s' % (i, 'x' * (i %"
                    " 13)) for i in range(600)]"
                ),
                type=pa.string(),
            ),
        )
    )
    var path = String("/tmp/marrow_dba.parquet")
    pq.write_table(
        t,
        path,
        use_dictionary=False,
        column_encoding=encoding,
        compression=compression,
    )
    var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    assert_true(Bool(Python.evaluate("'" + encoding + "'") in enc))

    var back = read_table(path)
    assert_equal(back.num_rows(), 600)
    var bat = back.to_batches()[0].copy()
    var cs = bat.columns[0].copy()
    ref s = cs.as_string()
    assert_equal(s.null_count(), 67)  # ceil(600/9)
    assert_false(s.is_valid(0))
    assert_equal(String(s[1]), "item-0001-x")
    assert_equal(String(s[12]), "item-0012-xxxxxxxxxxxx")  # 12 % 13 = 12
    assert_equal(String(s[599]), "item-0599-x")  # 599 % 13 = 1
    remove(path)


def test_delta_byte_array() raises:
    _roundtrip("DELTA_BYTE_ARRAY", "none")


def test_delta_byte_array_snappy() raises:
    _roundtrip("DELTA_BYTE_ARRAY", "snappy")


def test_delta_length_byte_array() raises:
    _roundtrip("DELTA_LENGTH_BYTE_ARRAY", "none")


def main() raises:
    TestSuite.run[__functions_in_module()]()
