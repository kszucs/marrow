"""Compression codecs: the `Compression` compress/decompress roundtrip plus
reading PyArrow-written files across the codecs marrow supports on read."""

from std.testing import assert_equal, assert_true
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.parquet.codecs import Compression
from marrow.parquet.utils import CompressionLibs


def _sample() -> List[UInt8]:
    var data = List[UInt8]()
    for i in range(4096):
        data.append(UInt8((i * 7 + (i // 13)) & 0xFF))
    return data^


def _roundtrip(codec: Compression) raises:
    var libs = CompressionLibs()
    var data = _sample()
    var packed = codec.compress(libs, Span(data))
    var restored = codec.decompress(libs, Span(packed), len(data))
    assert_equal(len(restored), len(data))
    for i in range(len(data)):
        assert_equal(restored[i], data[i])


def test_uncompressed_roundtrip() raises:
    _roundtrip(Compression.UNCOMPRESSED)


def test_snappy_roundtrip() raises:
    _roundtrip(Compression.SNAPPY)


def test_zstd_roundtrip() raises:
    _roundtrip(Compression.ZSTD)


def _roundtrip_read(compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3, 4, 5), type=pa.int64()),
            s=pa.array(Python.list("a", "bb", "ccc", "d", "ee")),
        )
    )
    var path = String("/tmp/marrow_codec_" + compression + ".parquet")
    pq.write_table(tbl, path, compression=compression)

    var t = read_table(path)
    assert_equal(t.num_rows(), 5)
    var b = t.to_batches()[0].copy()
    var ci = b.columns[0].copy()
    assert_equal(ci.as_int64()[0].value(), 1)
    assert_equal(ci.as_int64()[4].value(), 5)
    var cs = b.columns[1].copy()
    assert_equal(String(cs.as_string()[2]), "ccc")
    remove(path)


def test_read_gzip() raises:
    _roundtrip_read("gzip")


def test_read_lz4() raises:
    _roundtrip_read("lz4")


def main() raises:
    TestSuite.run[__functions_in_module()]()
