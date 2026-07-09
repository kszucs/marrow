"""BYTE_STREAM_SPLIT float decoding."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def _bss_roundtrip(compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var t = pa.table(
        Python.dict(
            f=pa.array(np.arange(500, dtype="float64") * 0.25 - 3.0),
            g=pa.array(
                Python.evaluate(
                    "[None if i % 5 == 0 else float(i) * 1.5 for i in"
                    " range(500)]"
                ),
                type=pa.float32(),
            ),
        )
    )
    var path = String("/tmp/marrow_bss.parquet")
    pq.write_table(
        t,
        path,
        use_byte_stream_split=True,
        use_dictionary=False,
        compression=compression,
    )
    var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    assert_true(Bool(Python.evaluate("'BYTE_STREAM_SPLIT'") in enc))

    var back = read_table(path)
    assert_equal(back.num_rows(), 500)
    var bat = back.to_batches()[0].copy()

    var cf = bat.columns[0].copy()
    ref f = cf.as_float64()
    assert_true(f[0].value() == -3.0)
    assert_true(f[499].value() == 121.75)

    var cg = bat.columns[1].copy()
    ref g = cg.as_float32()
    assert_equal(g.null_count(), 100)  # every 5th of 500
    assert_false(g.is_valid(0))
    assert_true(g[1].value() == 1.5)
    remove(path)


def test_read_byte_stream_split() raises:
    _bss_roundtrip("none")


def test_read_byte_stream_split_zstd() raises:
    _bss_roundtrip("zstd")


def main() raises:
    TestSuite.run[__functions_in_module()]()
