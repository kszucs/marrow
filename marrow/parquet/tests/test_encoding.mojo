"""Value/level encodings: the RLE/bit-packed hybrid primitives plus the read
paths for the non-dictionary encodings (DELTA_BINARY_PACKED, BYTE_STREAM_SPLIT,
DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY) against PyArrow-written files."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.parquet.encoding import bit_width, rle_decode, rle_encode


# ---------------------------------------------------------------------------
# RLE / bit-packed hybrid primitives
# ---------------------------------------------------------------------------


def test_bit_width() raises:
    assert_equal(bit_width(0), 0)
    assert_equal(bit_width(1), 1)
    assert_equal(bit_width(2), 2)
    assert_equal(bit_width(7), 3)
    assert_equal(bit_width(8), 4)
    assert_equal(bit_width(255), 8)


def _check(values: List[Int32], width: Int) raises:
    var encoded = rle_encode(values, width)
    var decoded = rle_decode(Span(encoded), width, len(values))
    assert_equal(len(decoded), len(values))
    for i in range(len(values)):
        assert_equal(decoded[i], values[i])


def test_rle_roundtrip_levels() raises:
    # all-ones definition levels (bit width 1)
    var ones = List[Int32]()
    for _ in range(37):
        ones.append(1)
    _check(ones, 1)


def test_rle_roundtrip_mixed() raises:
    var vals = List[Int32]()
    for i in range(100):
        vals.append(Int32(i % 4))
    _check(vals, 2)


def test_rle_roundtrip_wide() raises:
    var vals = List[Int32]()
    for i in range(50):
        vals.append(Int32((i * 17) % 1000))
    _check(vals, 10)


def test_rle_bitpacked_decode() raises:
    # Hand-build a bit-packed run of 8 values, width=3: values 0..7.
    # header = (1 group << 1) | 1 = 3
    # packed LSB-first: 0,1,2,3,4,5,6,7 -> bytes
    var data = List[UInt8]()
    data.append(3)  # header: 1 group, bit-packed
    # bits: 000 001 010 011 100 101 110 111 (LSB first per value)
    # byte0 = v0(000) | v1(001)<<3 | v2(010)<<6 low2 = 0b10_001_000 = 0x88
    data.append(0x88)
    data.append(0xC6)  # continue packing
    data.append(0xFA)
    var decoded = rle_decode(Span(data), 3, 8)
    assert_equal(len(decoded), 8)
    for i in range(8):
        assert_equal(decoded[i], Int32(i))


# ---------------------------------------------------------------------------
# DELTA_BINARY_PACKED integers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# BYTE_STREAM_SPLIT floats
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY strings
# ---------------------------------------------------------------------------


def _dba_roundtrip(encoding: String, compression: String) raises:
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
    _dba_roundtrip("DELTA_BYTE_ARRAY", "none")


def test_delta_byte_array_snappy() raises:
    _dba_roundtrip("DELTA_BYTE_ARRAY", "snappy")


def test_delta_length_byte_array() raises:
    _dba_roundtrip("DELTA_LENGTH_BYTE_ARRAY", "none")


def main() raises:
    TestSuite.run[__functions_in_module()]()
