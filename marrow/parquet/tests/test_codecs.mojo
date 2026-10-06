# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Value/level encodings: the read paths for the non-dictionary encodings
(DELTA_BINARY_PACKED, BYTE_STREAM_SPLIT, DELTA_BYTE_ARRAY /
DELTA_LENGTH_BYTE_ARRAY) against PyArrow-written files. Also
covers the `Compression` compress/decompress roundtrip and reading
PyArrow-written files across the compression codecs marrow supports on read."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os.path import join
from ...utils.testing import ScratchDir
from ...parquet import read_table
from ...parquet.codecs import Compression
from ...errors import CorruptError, DynError
from ...utils import CompressionLibs


def _delta_roundtrip(compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # 1000 rows > 128 (default block size) spans multiple delta blocks;
    # int64 monotone + int32 with negatives and every-7th null.
    var idx = np.arange(1000)
    var t = pa.table(
        Python.dict(
            a=pa.array(idx * 3 - 500, type=pa.int64()),
            b=pa.array(
                idx * idx - 100000, mask=(idx % 7 == 0), type=pa.int32()
            ),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_delta.parquet")
        pq.write_table(
            t,
            path,
            use_dictionary=False,
            column_encoding="DELTA_BINARY_PACKED",
            compression=compression,
        )
        # confirm PyArrow actually used DELTA_BINARY_PACKED
        var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
        assert_true(Bool(Python.str("DELTA_BINARY_PACKED") in enc))

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
    var idx = np.arange(500)
    var t = pa.table(
        Python.dict(
            f=pa.array(idx.astype("float64") * 0.25 - 3.0),
            g=pa.array(idx * 1.5, mask=(idx % 5 == 0), type=pa.float32()),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_bss.parquet")
        pq.write_table(
            t,
            path,
            use_byte_stream_split=True,
            use_dictionary=False,
            compression=compression,
        )
        var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
        assert_true(Bool(Python.str("BYTE_STREAM_SPLIT") in enc))

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


def test_read_byte_stream_split() raises:
    _bss_roundtrip("none")


def test_read_byte_stream_split_zstd() raises:
    _bss_roundtrip("zstd")


def _bss_int_roundtrip(dtype: String) raises:
    # BYTE_STREAM_SPLIT for integers (Parquet 2.8+): the width comes from the
    # physical type, so int32 and int64 split into 4 / 8 byte-planes.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var ty = pa.int32() if dtype == "int32" else pa.int64()
    var idx = np.arange(300)
    var t = pa.table(
        Python.dict(
            a=pa.array(idx * 3 - 500, type=ty),
            b=pa.array(idx * idx, mask=(idx % 6 == 0), type=ty),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_bss_int.parquet")
        pq.write_table(
            t,
            path,
            use_byte_stream_split=True,
            use_dictionary=False,
            compression="none",
        )
        var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
        assert_true(Bool(Python.str("BYTE_STREAM_SPLIT") in enc))

        var back = read_table(path)
        assert_equal(back.num_rows(), 300)
        var bat = back.to_batches()[0].copy()

        if dtype == "int32":
            ref a = bat.columns[0].copy().as_int32()
            assert_equal(a[0].value(), -500)
            assert_equal(a[299].value(), 397)
            ref b = bat.columns[1].copy().as_int32()
            assert_equal(b.null_count(), 50)  # every 6th of 300
            assert_false(b.is_valid(0))
            assert_equal(b[1].value(), 1)
            assert_equal(b[299].value(), 299 * 299)
        else:
            ref a = bat.columns[0].copy().as_int64()
            assert_equal(a[0].value(), -500)
            assert_equal(a[299].value(), 397)
            ref b = bat.columns[1].copy().as_int64()
            assert_equal(b.null_count(), 50)
            assert_equal(b[299].value(), 299 * 299)


def test_read_byte_stream_split_int32() raises:
    _bss_int_roundtrip("int32")


def test_read_byte_stream_split_int64() raises:
    _bss_int_roundtrip("int64")


# ---------------------------------------------------------------------------
# DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY strings
# ---------------------------------------------------------------------------


def _dba_roundtrip(encoding: String, compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # varying lengths + shared prefixes exercise the incremental reconstruction;
    # every 9th null; 600 rows spans multiple delta blocks. Value i is
    # "item-<i:04d>-<'x' repeated i%13>".
    var vals = Python.list()
    for i in range(600):
        if i % 9 == 0:
            vals.append(Python.none())
        else:
            vals.append(
                Python.str("item-")
                + Python.str(String(i)).zfill(4)
                + Python.str("-")
                + Python.str("x") * (i % 13)
            )
    var t = pa.table(Python.dict(s=pa.array(vals, type=pa.string())))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_dba.parquet")
        pq.write_table(
            t,
            path,
            use_dictionary=False,
            column_encoding=encoding,
            compression=compression,
        )
        var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
        assert_true(Bool(Python.str(encoding) in enc))

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


def test_delta_byte_array() raises:
    _dba_roundtrip("DELTA_BYTE_ARRAY", "none")


def test_delta_byte_array_snappy() raises:
    _dba_roundtrip("DELTA_BYTE_ARRAY", "snappy")


def test_delta_length_byte_array() raises:
    _dba_roundtrip("DELTA_LENGTH_BYTE_ARRAY", "none")


# ---------------------------------------------------------------------------
# Compression codecs: compress/decompress roundtrip + reading PyArrow files
# ---------------------------------------------------------------------------


def _sample() -> List[UInt8]:
    var data = List[UInt8]()
    for i in range(4096):
        data.append(UInt8((i * 7 + (i // 13)) & 0xFF))
    return data^


def _roundtrip(codec: Compression) raises:
    var libs = CompressionLibs()
    var data = _sample()
    var packed = codec.compress(libs, Span(data))
    var restored: List[UInt8] = [7]
    codec.decompress_into(libs, Span(packed), len(data), restored)
    assert_true(restored[1:] == data[:], "appended after what was there")
    assert_equal(restored[0], 7)


def test_codecs_library_implementations() raises:
    """LZ4, LZ4_RAW and ZSTD in Mojo and through liblz4 and libzstd: each
    decodes what the other wrote."""
    var data = _sample()
    for codec in [Compression.ZSTD, Compression.LZ4, Compression.LZ4_RAW]:
        for written in [True, False]:
            var writer = CompressionLibs(native=written)
            var packed = codec.compress(writer, Span(data))
            for read_native in [True, False]:
                var reader = CompressionLibs(native=read_native)
                var restored = List[UInt8]()
                codec.decompress_into(reader, Span(packed), len(data), restored)
                assert_true(
                    restored == data,
                    String(
                        t"codec {codec.code}: native writer {written}, native"
                        t" reader {read_native}"
                    ),
                )


def test_codecs_needs_libs() raises:
    """Only a codec a library runs opens the libraries: LZ4, LZ4_RAW and
    ZSTD when they are not run in Mojo; Snappy, GZIP and Brotli always."""
    assert_false(Compression.UNCOMPRESSED.needs_libs(False))
    for codec in [Compression.ZSTD, Compression.LZ4, Compression.LZ4_RAW]:
        assert_false(codec.needs_libs(True))
        assert_true(codec.needs_libs(False))
    for codec in [Compression.SNAPPY, Compression.GZIP, Compression.BROTLI]:
        assert_true(codec.needs_libs(True))


def test_decompress_refuses_sizes_the_page_cannot_hold() raises:
    """A page header's uncompressed size is checked against what the
    compressed bytes can decode to before it is allocated: as is for
    UNCOMPRESSED, and LZ4's and ZSTD's own bounds."""
    var libs = CompressionLibs()
    var data = _sample()
    for codec in [
        Compression.UNCOMPRESSED,
        Compression.ZSTD,
        Compression.LZ4,
        Compression.LZ4_RAW,
    ]:
        var packed = codec.compress(libs, Span(data))
        var most = codec.max_decompressed_length(len(packed))
        for size in [most + 1, 1 << 60]:
            var refused = False
            try:
                var dst = List[UInt8]()
                codec.decompress_into(libs, Span(packed), size, dst)
            except e:
                refused = DynError(e).isa[CorruptError]()
            assert_true(refused, String(t"codec {codec.code}: {size} accepted"))


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_codec_" + compression + ".parquet")
        pq.write_table(tbl, path, compression=compression)

        var t = read_table(path)
        assert_equal(t.num_rows(), 5)
        var b = t.to_batches()[0].copy()
        var ci = b.columns[0].copy()
        assert_equal(ci.as_int64()[0].value(), 1)
        assert_equal(ci.as_int64()[4].value(), 5)
        var cs = b.columns[1].copy()
        assert_equal(String(cs.as_string()[2]), "ccc")


def test_read_gzip() raises:
    _roundtrip_read("gzip")


def test_read_lz4() raises:
    _roundtrip_read("lz4")
