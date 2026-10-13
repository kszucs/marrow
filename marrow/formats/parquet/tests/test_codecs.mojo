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
from ....utils.testing import ScratchDir
from ...parquet import read_table
from ...parquet.compression import Compression
from ....errors import CorruptError, DynError
from ....compress import Hadoop, LibLz4, Lz4
from ....utils.byteorder import BigEndian
from ....compress.tests.codec_data import (
    assert_bytes,
    assert_canary,
    canary,
    shapes,
    words,
)


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
    var data = _sample()
    var packed = List[UInt8]()
    codec.compress(True, Span(data), packed)
    var restored: List[UInt8] = [7]
    codec.decompress_into(True, Span(packed), len(data), restored)
    assert_true(restored[1:] == data[:], "appended after what was there")
    assert_equal(restored[0], 7)


def test_codecs_library_implementations() raises:
    """LZ4, LZ4_RAW, SNAPPY and ZSTD in Mojo and through their libraries:
    each decodes what the other wrote. BROTLI decodes both ways too."""
    var data = _sample()
    for codec in [
        Compression.ZSTD,
        Compression.LZ4,
        Compression.LZ4_RAW,
        Compression.SNAPPY,
        Compression.BROTLI,
    ]:
        for written in [True, False]:
            var packed = List[UInt8]()
            codec.compress(written, Span(data), packed)
            for read_native in [True, False]:
                var restored = List[UInt8]()
                codec.decompress_into(
                    read_native, Span(packed), len(data), restored
                )
                assert_true(
                    restored == data,
                    String(
                        t"codec {codec.code}: native writer {written}, native"
                        t" reader {read_native}"
                    ),
                )


def test_codecs_needs_libs() raises:
    """Only a codec a library decodes opens the libraries: BROTLI, LZ4,
    LZ4_RAW, SNAPPY and ZSTD when they are not run in Mojo; GZIP always."""
    assert_false(Compression.UNCOMPRESSED.needs_libs(False))
    for codec in [
        Compression.ZSTD,
        Compression.LZ4,
        Compression.LZ4_RAW,
        Compression.SNAPPY,
        Compression.BROTLI,
    ]:
        assert_false(codec.needs_libs(True))
        assert_true(codec.needs_libs(False))
    assert_true(Compression.GZIP.needs_libs(True))
    assert_true(Compression(3).needs_libs(True))  # LZO: no method


def test_decompress_refuses_sizes_the_page_cannot_hold() raises:
    """A page header's uncompressed size is checked against what the
    compressed bytes can decode to before it is allocated: as is for
    UNCOMPRESSED, and each codec's own bound for the rest."""
    var data = _sample()
    for codec in [
        Compression.UNCOMPRESSED,
        Compression.ZSTD,
        Compression.LZ4,
        Compression.LZ4_RAW,
        Compression.SNAPPY,
        Compression.GZIP,
        Compression.BROTLI,
    ]:
        var packed = List[UInt8]()
        codec.compress(True, Span(data), packed)
        var most = codec.method().max_decompressed_length(len(packed))
        for size in [most + 1, 1 << 60]:
            var refused = False
            try:
                var dst = List[UInt8]()
                codec.decompress_into(True, Span(packed), size, dst)
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


# ---------------------------------------------------------------------------
# Hadoop framing, Parquet's deprecated LZ4
# ---------------------------------------------------------------------------


def _hadoop[
    native: Bool = True
](data: List[UInt8], cuts: List[Int] = []) raises -> List[UInt8]:
    """`data` as Hadoop frames, a new one at each of `cuts` -- their blocks
    liblz4's without `native`."""
    var out = List[UInt8]()
    var bounds = cuts.copy()
    bounds.append(len(data))
    var pos = 0
    for end in bounds:
        comptime if native:
            Hadoop[Lz4].compress(Span(data)[pos:end], out)
        else:
            Hadoop[LibLz4].compress(Span(data)[pos:end], out)
        pos = end
    return out^


def _block(data: List[UInt8]) raises -> List[UInt8]:
    """`data` as a plain block, as Parquet C++ wrote before it framed one."""
    var out = List[UInt8]()
    Lz4.compress(Span(data), out)
    return out^


def _decode_hadoop[
    native: Bool
](src: List[UInt8], mut buf: List[UInt8], n: Int) raises:
    comptime if native:
        Hadoop[Lz4].decompress(Span(src), Span(buf)[:n])
    else:
        Hadoop[LibLz4].decompress(Span(src), Span(buf)[:n])


def _expect_hadoop(src: List[UInt8], want: List[UInt8], what: String) raises:
    """`src` decodes to exactly `want`, by `Lz4` and by liblz4, writing
    nothing past it."""
    var n = len(want)
    var buf = canary(n)
    _decode_hadoop[True](src, buf, n)
    assert_bytes(List[UInt8](Span(buf)[:n]), want, what)
    assert_canary(buf, n, what)
    buf = canary(n)
    _decode_hadoop[False](src, buf, n)
    assert_bytes(List[UInt8](Span(buf)[:n]), want, what + " (liblz4)")
    assert_canary(buf, n, what + " (liblz4)")


def _rejects_hadoop(src: List[UInt8], n: Int, what: String) raises:
    """`src` is refused into `n` bytes, by `Lz4` and by liblz4, and nothing
    past them is written."""
    comptime for native in [True, False]:
        var buf = canary(n)
        var accepted = True
        try:
            _decode_hadoop[native](src, buf, n)
        except:
            accepted = False
        assert_true(not accepted, what + ": accepted")
        assert_canary(buf, n, what)


def test_parquet_lz4_hadoop_roundtrip() raises:
    """One frame per input: its decompressed and compressed sizes, big-endian,
    then the block."""
    var all = shapes()
    for n in range(0, 301, 13):
        all.append(words(n, seed=UInt64(n)))
    for i in range(len(all)):
        var src = _hadoop(all[i])
        assert_equal(
            Int(BigEndian.fixed[DType.uint32](Span(src), 0)), len(all[i])
        )
        assert_equal(
            Int(BigEndian.fixed[DType.uint32](Span(src), 4)), len(src) - 8
        )
        _expect_hadoop(src, all[i], String(t"input {i}"))
        _expect_hadoop(_hadoop[False](all[i]), all[i], String(t"liblz4 {i}"))


def test_parquet_lz4_hadoop_reads_what_writers_wrote() raises:
    """A run of frames, as parquet-mr's codec writes a page over its buffer
    size; a plain block, as Parquet C++ wrote before it framed them; and an
    empty page either way."""
    var data = words(300_000, seed=53)
    _expect_hadoop(_hadoop(data, [100_000, 250_000]), data, "three frames")
    _expect_hadoop(_hadoop(data, [0, 300_000]), data, "empty frames")
    _expect_hadoop(_block(data), data, "plain block")
    _expect_hadoop(_hadoop([]), [], "empty frame")
    _expect_hadoop(_block([]), [], "empty block")


def test_parquet_lz4_hadoop_rejects() raises:
    var data = words(10_000, seed=59)
    var n = len(data)
    var src = _hadoop(data, [4_000])
    _rejects_hadoop(src, n - 1, "short destination")
    _rejects_hadoop(src, n + 1, "long destination")
    var sizes = src.copy()
    sizes[3] ^= 1  # the first frame's decompressed size
    _rejects_hadoop(sizes, n, "frame size")
    for cut in range(1, len(src), 61):
        _rejects_hadoop(List[UInt8](Span(src)[:cut]), n, String(t"cut {cut}"))
