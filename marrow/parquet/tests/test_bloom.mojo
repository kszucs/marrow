"""Bloom filter tests: the XXH64 value hash, the split-block filter membership,
and the write -> read round-trip through `write_bloom_filter=True`.

`test_bloom_reference_vector` embeds a spec-compliant 2-block filter built by an
independent reference (matching arrow-rs / parquet-mr, salt `0x2df1424b`) holding
{Hello, World, marrow, parquet} — it pins the hash + salt + block layout so a
regression in any of them fails here rather than silently producing a filter
only marrow can read.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table, write_table, SplitBlockBloomFilter, xxh64
from marrow.parquet.reader import ParquetFile
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _to_marrow(py: PythonObject) raises -> Table:
    return CArrowArrayStream.from_pycapsule(
        py.__arrow_c_stream__(Python.none())
    ).to_table()


def _contains(bf: SplitBlockBloomFilter, s: String) -> Bool:
    var b = List[UInt8](s.as_bytes())
    return bf.might_contain(Span(b))


def test_xxh64_vectors() raises:
    # canonical XXH64 (seed 0) test vectors
    var empty = List[UInt8]()
    assert_equal(xxh64(Span(empty)), 0xEF46DB3751D8E999)
    var hello = List[UInt8](String("Hello").as_bytes())
    assert_equal(xxh64(Span(hello)), 0x0A75A91375B27D44)


def test_bloom_reference_vector() raises:
    # a reference-built (arrow-rs/parquet-mr-compatible) filter; marrow must read
    # every member as present and clear non-members
    var bytes: List[UInt8] = [
        4,
        4,
        65,
        0,
        130,
        2,
        64,
        0,
        64,
        0,
        64,
        36,
        16,
        8,
        24,
        0,
        132,
        64,
        0,
        4,
        0,
        176,
        1,
        0,
        16,
        0,
        1,
        6,
        3,
        4,
        64,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    var bf = SplitBlockBloomFilter.from_bytes(Span(bytes))
    assert_equal(bf.num_blocks, 2)
    assert_true(_contains(bf, "Hello"))
    assert_true(_contains(bf, "World"))
    assert_true(_contains(bf, "marrow"))
    assert_true(_contains(bf, "parquet"))
    assert_false(_contains(bf, "Hello_Not_Exists"))
    assert_false(_contains(bf, "absent_value_xyz"))


def test_bloom_no_false_negatives() raises:
    # every inserted value must test positive (the split-block invariant)
    var bf = SplitBlockBloomFilter.with_ndv(500)
    for i in range(500):
        var s = List[UInt8](String("v" + String(i)).as_bytes())
        bf.insert(Span(s))
    for i in range(500):
        var s = List[UInt8](String("v" + String(i)).as_bytes())
        assert_true(bf.might_contain(Span(s)))


def test_bloom_serialize_roundtrip() raises:
    var bf = SplitBlockBloomFilter.with_ndv(100)
    for i in range(100):
        var s = List[UInt8](String("x" + String(i)).as_bytes())
        bf.insert(Span(s))
    var bytes = bf.to_bytes()
    var bf2 = SplitBlockBloomFilter.from_bytes(Span(bytes))
    assert_equal(bf2.num_blocks, bf.num_blocks)
    for i in range(100):
        var s = List[UInt8](String("x" + String(i)).as_bytes())
        assert_true(bf2.might_contain(Span(s)))


def test_write_bloom_filter_string() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var strs = Python.list()
    for i in range(300):
        strs.append(Python.str("key_") + Python.str(i % 40))
    var want = pa.table(Python.dict(s=pa.array(strs)))
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_bloom_str.parquet")
    write_table(t, path, use_dictionary=False, write_bloom_filter=True)

    # PyArrow still reads the file (bloom filter is out of the way)
    var back = pq.read_table(path)
    assert_true(Bool(back.column(0).equals(want.column(0))))

    # marrow reads the bloom filter: all present, no false negatives
    var pf = ParquetFile(path)
    var sbf = pf.bloom_filter(0, 0)
    assert_true(Bool(sbf))
    ref f = sbf.value()
    for i in range(40):
        assert_true(_contains(f, "key_" + String(i)))
    # a clearly-absent value should be pruned (allowing a rare false positive)
    assert_false(_contains(f, "definitely_absent_key_99999"))
    remove(path)


def test_write_bloom_filter_int() raises:
    var pa = Python.import_module("pyarrow")
    var ints = Python.list()
    for i in range(300):
        ints.append(i % 50)
    var want = pa.table(Python.dict(n=pa.array(ints, type=pa.int64())))
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_bloom_int.parquet")
    write_table(t, path, use_dictionary=False, write_bloom_filter=True)

    var pf = ParquetFile(path)
    var nbf = pf.bloom_filter(0, 0)
    assert_true(Bool(nbf))
    ref f = nbf.value()
    # int64 values are hashed over their 8 little-endian bytes
    for i in range(50):
        var b = List[UInt8]()
        var v = Int64(i)
        for k in range(8):
            b.append(UInt8((v >> Int64(k * 8)) & 0xFF))
        assert_true(f.might_contain(Span(b)))
    remove(path)


def test_no_bloom_filter_by_default() raises:
    var pa = Python.import_module("pyarrow")
    var want = pa.table(Python.dict(n=pa.array([1, 2, 3], type=pa.int64())))
    var t = _to_marrow(want)
    var path = String("/tmp/marrow_no_bloom.parquet")
    write_table(t, path)  # write_bloom_filter defaults to False
    var pf = ParquetFile(path)
    assert_false(Bool(pf.bloom_filter(0, 0)))
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
