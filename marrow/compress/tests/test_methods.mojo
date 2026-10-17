# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Method`: every method round-trips through marrow's codec and the
system library's, and a method outside the caller's list is refused."""

from std.testing import assert_true

from ...errors import DynError, NotImplementedError
from ..brotli import Brotli, LibBrotli
from ..methods import Method
from .codec_data import words


def _compress_all(
    m: Method, native: Bool, data: List[UInt8]
) raises -> List[UInt8]:
    var out = List[UInt8]()
    m.compress[
        Method.NONE,
        Method.SNAPPY,
        Method.GZIP,
        Method.DEFLATE,
        Method.BROTLI,
        Method.LZ4,
        Method.LZ4_FRAME,
        Method.LZ4_HADOOP,
        Method.ZSTD,
    ](native, Span(data), out)
    return out^


def test_utils_method_roundtrip() raises:
    var data = words(10_000)
    for m in [
        Method.NONE,
        Method.SNAPPY,
        Method.GZIP,
        Method.DEFLATE,
        Method.BROTLI,
        Method.LZ4,
        Method.LZ4_FRAME,
        Method.LZ4_HADOOP,
        Method.ZSTD,
    ]:
        for written in [True, False]:
            var packed = _compress_all(m, written, data)
            assert_true(len(data) <= m.max_decompressed_length(len(packed)))
            for reader in [True, False]:
                var out = List[UInt8](unsafe_uninit_length=len(data))
                m.decompress[
                    Method.NONE,
                    Method.SNAPPY,
                    Method.GZIP,
                    Method.DEFLATE,
                    Method.BROTLI,
                    Method.LZ4,
                    Method.LZ4_FRAME,
                    Method.LZ4_HADOOP,
                    Method.ZSTD,
                ](reader, Span(packed), Span(out))
                assert_true(
                    out == data,
                    String(m, ": native writer ", written, ", reader ", reader),
                )


def test_utils_method_brotli_writer() raises:
    """BROTLI is written by marrow's encoder when `native`, else by
    libbrotli, and either stream decodes back."""
    var data = words(10_000)
    var mine = List[UInt8]()
    Brotli.compress(Span(data), mine)
    var theirs = List[UInt8]()
    LibBrotli.compress(Span(data), theirs)
    var native = _compress_all(Method.BROTLI, True, data)
    var library = _compress_all(Method.BROTLI, False, data)
    assert_true(native == mine)
    assert_true(library == theirs)
    for by_marrow in [True, False]:
        var packed = native.copy() if by_marrow else library.copy()
        var out = List[UInt8](unsafe_uninit_length=len(data))
        Method.BROTLI.decompress[Method.BROTLI](True, Span(packed), Span(out))
        assert_true(out == data)


def test_utils_method_decompress_unsized() raises:
    """NONE, SNAPPY, GZIP, DEFLATE and ZSTD decode without being told the
    length; the rest are refused."""
    var data = words(10_000)
    for m in [
        Method.NONE,
        Method.SNAPPY,
        Method.GZIP,
        Method.DEFLATE,
        Method.ZSTD,
    ]:
        var packed = _compress_all(m, True, data)
        for reader in [True, False]:
            var out = m.decompress_unsized[
                Method.NONE,
                Method.SNAPPY,
                Method.GZIP,
                Method.DEFLATE,
                Method.ZSTD,
                Method.LZ4,
            ](reader, Span(packed))
            assert_true(out == data, String(m, ": native reader ", reader))
    var packed = _compress_all(Method.LZ4, True, data)
    var refused = False
    try:
        _ = Method.LZ4.decompress_unsized[Method.LZ4](True, Span(packed))
    except e:
        refused = DynError(e).isa[NotImplementedError]()
    assert_true(refused)


def test_utils_method_decompress_unsized_zstd_without_size() raises:
    """A Zstandard frame that leaves its size out decodes through libzstd's
    streaming decoder, `native` or not."""
    var data = words(200)
    var frame = _compress_all(Method.ZSTD, True, data)
    # One segment with a 1-byte content size: valid only under 256 bytes.
    frame[4] = 0x00  # no content size, not one segment
    frame[5] = 0x00  # window descriptor: 1 KiB
    for native in [True, False]:
        var out = Method.ZSTD.decompress_unsized[Method.ZSTD](
            native, Span(frame)
        )
        assert_true(out == data, String("native ", native))


def test_utils_method_names() raises:
    assert_true(String(Method.ZSTD) == "zstd")
    assert_true(String(Method.LZ4_HADOOP) == "lz4_hadoop")
    assert_true(String(Method.NONE) == "none")
    assert_true(String(Method(9)) == "unknown(9)")
    assert_true(String(Method(-1)) == "unknown(-1)")


def test_utils_method_outside_the_list_is_refused() raises:
    var data = words(1000)
    var packed = _compress_all(Method.SNAPPY, True, data)
    var message = String()
    try:
        var out = List[UInt8]()
        Method.SNAPPY.compress[Method.LZ4_FRAME, Method.ZSTD](
            True, Span(data), out
        )
    except e:
        assert_true(DynError(e).isa[NotImplementedError]())
        message = String(e)
    assert_true("snappy" in message, message)
    var refused = False
    try:
        var out = List[UInt8](unsafe_uninit_length=len(data))
        Method.SNAPPY.decompress[Method.ZSTD](True, Span(packed), Span(out))
    except e:
        refused = DynError(e).isa[NotImplementedError]()
    assert_true(refused)
