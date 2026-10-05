# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro's value encoding: zigzag varints, IEEE floats, length-prefixed bytes
and block headers."""

from std.testing import assert_equal, assert_raises, assert_true

from ...avro import AvroBytes, AvroCursor


def _encoded(v: Int64) -> List[UInt8]:
    var out = AvroBytes()
    out.long(v)
    return List[UInt8](out.written())


def test_avro_long_encoding_matches_spec() raises:
    # The table in the Avro specification's "Binary Encoding" section.
    assert_true(_encoded(0) == [UInt8(0x00)])
    assert_true(_encoded(-1) == [UInt8(0x01)])
    assert_true(_encoded(1) == [UInt8(0x02)])
    assert_true(_encoded(-2) == [UInt8(0x03)])
    assert_true(_encoded(2) == [UInt8(0x04)])
    assert_true(_encoded(-64) == [UInt8(0x7F)])
    assert_true(_encoded(64) == [UInt8(0x80), UInt8(0x01)])


def test_avro_long_roundtrip_extremes() raises:
    var values: List[Int64] = [
        0,
        -1,
        1,
        Int64(Int32.MIN),
        Int64(Int32.MAX),
        Int64.MIN,
        Int64.MAX,
    ]
    var out = AvroBytes()
    for v in values:
        out.long(v)
    var cur = AvroCursor(out.written())
    for v in values:
        assert_equal(cur.long(), v)
    assert_equal(cur.remaining(), 0)


def test_avro_int_out_of_range() raises:
    var out = AvroBytes()
    out.long(Int64(Int32.MAX) + 1)
    var cur = AvroCursor(out.written())
    with assert_raises(contains="int out of range"):
        _ = cur.int()


def test_avro_scalars_roundtrip() raises:
    var out = AvroBytes()
    out.boolean(True)
    out.boolean(False)
    out.float(1.5)
    out.double(-2.25)
    out.bytes(String("héllo").as_bytes())
    out.fixed(String("abc").as_bytes())
    var cur = AvroCursor(out.written())
    assert_true(cur.boolean())
    assert_true(not cur.boolean())
    assert_equal(cur.float(), 1.5)
    assert_equal(cur.double(), -2.25)
    assert_equal(String(StringSlice(unsafe_from_utf8=cur.bytes())), "héllo")
    assert_equal(String(StringSlice(unsafe_from_utf8=cur.fixed(3))), "abc")
    assert_equal(cur.remaining(), 0)


def test_avro_block_headers() raises:
    var out = AvroBytes()
    out.long(3)  # three items, no byte size
    out.long(-2)  # two items in 5 bytes
    out.long(5)
    out.long(0)  # end
    var cur = AvroCursor(out.written())
    var a = cur.block()
    assert_equal(a[0], 3)
    assert_equal(a[1], -1)
    var b = cur.block()
    assert_equal(b[0], 2)
    assert_equal(b[1], 5)
    assert_equal(cur.block()[0], 0)


def test_avro_truncated_input_is_corrupt() raises:
    var out = AvroBytes()
    out.long(5)  # a 5-byte string with only 2 bytes behind it
    out.fixed(String("ab").as_bytes())
    var cur = AvroCursor(out.written())
    with assert_raises(contains="CorruptError"):
        _ = cur.bytes()
    var empty = List[UInt8]()
    var cur2 = AvroCursor(Span(empty))
    with assert_raises(contains="CorruptError"):
        _ = cur2.double()


def test_avro_invalid_boolean_is_corrupt() raises:
    var data: List[UInt8] = [2]
    var cur = AvroCursor(Span(data))
    with assert_raises(contains="invalid boolean"):
        _ = cur.boolean()
