from std.testing import assert_equal
from marrow.testing import TestSuite
from marrow.parquet.encoding import bit_width, rle_decode, rle_encode


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


def main() raises:
    TestSuite.run[__functions_in_module()]()
