# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Method`: every method round-trips through marrow's codec and the
system library's, and a method outside the caller's list is refused."""

from std.testing import assert_true

from ...errors import DynError, NotImplementedError
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
