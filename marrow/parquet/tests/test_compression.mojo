from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite
from marrow.parquet.compression import (
    Codecs,
    CODEC_UNCOMPRESSED,
    CODEC_SNAPPY,
    CODEC_ZSTD,
)


def _sample() -> List[UInt8]:
    var data = List[UInt8]()
    for i in range(4096):
        data.append(UInt8((i * 7 + (i // 13)) & 0xFF))
    return data^


def _roundtrip(codec: Int) raises:
    var codecs = Codecs()
    var data = _sample()
    var packed = codecs.compress(codec, Span(data))
    var restored = codecs.decompress(codec, Span(packed), len(data))
    assert_equal(len(restored), len(data))
    for i in range(len(data)):
        assert_equal(restored[i], data[i])


def test_uncompressed_roundtrip() raises:
    _roundtrip(CODEC_UNCOMPRESSED)


def test_snappy_roundtrip() raises:
    _roundtrip(CODEC_SNAPPY)


def test_zstd_roundtrip() raises:
    _roundtrip(CODEC_ZSTD)


def main() raises:
    TestSuite.run[__functions_in_module()]()
