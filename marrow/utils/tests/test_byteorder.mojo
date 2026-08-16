"""`LittleEndian` — the byte, bit and LEB128-varint primitives.

Byte order is asserted against explicit byte sequences rather than round-trips,
so the tests would fail on a big-endian host if the reads ever became
host-dependent. `SIMD.from_bytes[big_endian=False]` (which `fixed` is built on)
is covered by the standard library's own `test_simd.mojo`; what is tested here
is everything marrow adds on top — offsets, bounds checking, variable widths,
LEB128 and bit-level reads, none of which exist in std.
"""

from std.testing import assert_equal, assert_true, assert_false, assert_raises

from ..byteorder import LittleEndian


def _bytes(*vals: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for v in vals:
        out.append(UInt8(v))
    return out^


# ---------------------------------------------------------------------------
# fixed / checked
# ---------------------------------------------------------------------------


def test_le_fixed_reads_least_significant_byte_first() raises:
    var b = _bytes(0x78, 0x56, 0x34, 0x12, 0xAA)
    assert_equal(LittleEndian.fixed[DType.uint8](Span(b), 0), UInt8(0x78))
    assert_equal(LittleEndian.fixed[DType.uint16](Span(b), 0), UInt16(0x5678))
    assert_equal(
        LittleEndian.fixed[DType.uint32](Span(b), 0), UInt32(0x12345678)
    )


def test_le_fixed_honours_the_offset() raises:
    var b = _bytes(0xFF, 0xFF, 0x78, 0x56, 0x34, 0x12)
    assert_equal(
        LittleEndian.fixed[DType.uint32](Span(b), 2), UInt32(0x12345678)
    )


def test_le_fixed_reads_signed_and_wide() raises:
    var neg = _bytes(0xFF, 0xFF, 0xFF, 0xFF)
    assert_equal(LittleEndian.fixed[DType.int32](Span(neg), 0), Int32(-1))
    var wide = _bytes(1, 0, 0, 0, 0, 0, 0, 0x80)
    assert_equal(
        LittleEndian.fixed[DType.uint64](Span(wide), 0),
        UInt64(0x8000000000000001),
    )


def test_le_checked_matches_fixed_when_in_bounds() raises:
    var b = _bytes(0x78, 0x56, 0x34, 0x12)
    assert_equal(
        LittleEndian.checked[DType.uint32](Span(b), 0),
        LittleEndian.fixed[DType.uint32](Span(b), 0),
    )


def test_le_checked_raises_past_the_end() raises:
    """A format parser reads untrusted offsets; `checked` is the one that must
    refuse rather than read past the buffer."""
    var b = _bytes(0x01, 0x02, 0x03)
    with assert_raises():
        var _ = LittleEndian.checked[DType.uint32](Span(b), 0)
    with assert_raises():
        var _ = LittleEndian.checked[DType.uint16](Span(b), 2)
    with assert_raises():
        var _ = LittleEndian.checked[DType.uint8](Span(b), -1)


# ---------------------------------------------------------------------------
# write / append / put_*
# ---------------------------------------------------------------------------


def test_le_append_emits_little_endian_bytes() raises:
    var out = List[UInt8]()
    LittleEndian.append[DType.uint32](out, UInt32(0x12345678))
    assert_equal(len(out), 4)
    assert_equal(out[0], UInt8(0x78))
    assert_equal(out[1], UInt8(0x56))
    assert_equal(out[2], UInt8(0x34))
    assert_equal(out[3], UInt8(0x12))


def test_le_write_fills_existing_slots() raises:
    var out = _bytes(0, 0, 0, 0, 0, 0)
    LittleEndian.write[DType.uint16](out, 2, UInt16(0xBEEF))
    assert_equal(out[0], UInt8(0))
    assert_equal(out[2], UInt8(0xEF))
    assert_equal(out[3], UInt8(0xBE))
    assert_equal(out[4], UInt8(0))


def test_le_write_append_roundtrip_through_fixed() raises:
    var out = List[UInt8]()
    LittleEndian.append[DType.uint64](out, UInt64(0xDEADBEEFCAFEBABE))
    assert_equal(
        LittleEndian.fixed[DType.uint64](Span(out), 0),
        UInt64(0xDEADBEEFCAFEBABE),
    )


def test_le_u32_helpers_agree() raises:
    var out = List[UInt8]()
    LittleEndian.put_u32(out, 0x01020304)
    assert_equal(LittleEndian.u32(Span(out), 0), 0x01020304)


def test_le_put_le_writes_a_variable_width() raises:
    """`put_le` takes a *byte count*, not a scalar type — the Parquet RLE run
    header needs 3-byte values, which no fixed-width write covers."""
    var out = List[UInt8]()
    LittleEndian.put_le(out, UInt64(0x123456), 3)
    assert_equal(len(out), 3)
    assert_equal(out[0], UInt8(0x56))
    assert_equal(out[1], UInt8(0x34))
    assert_equal(out[2], UInt8(0x12))

    var one = List[UInt8]()
    LittleEndian.put_le(one, UInt64(0xFF), 1)
    assert_equal(len(one), 1)
    assert_equal(one[0], UInt8(0xFF))


# ---------------------------------------------------------------------------
# LEB128 varints
# ---------------------------------------------------------------------------


def test_le_varint_single_byte() raises:
    var b = _bytes(0x00, 0x01, 0x7F)
    var v0 = LittleEndian.varint(Span(b), 0)
    assert_equal(v0[0], UInt64(0))
    assert_equal(v0[1], 1)
    var v2 = LittleEndian.varint(Span(b), 2)
    assert_equal(v2[0], UInt64(127))
    assert_equal(v2[1], 3)


def test_le_varint_multi_byte() raises:
    # 300 = 0b100101100 -> 0xAC 0x02
    var b = _bytes(0xAC, 0x02)
    var v = LittleEndian.varint(Span(b), 0)
    assert_equal(v[0], UInt64(300))
    assert_equal(v[1], 2)


def test_le_varint_roundtrips_every_boundary() raises:
    var values = List[UInt64]()
    values.append(0)
    values.append(1)
    values.append(127)
    values.append(128)
    values.append(16383)
    values.append(16384)
    values.append(UInt64(0xFFFFFFFF))
    values.append(UInt64(0xFFFFFFFFFFFFFFFF))
    for v in values:
        var out = List[UInt8]()
        LittleEndian.put_varint(out, v)
        var got = LittleEndian.varint(Span(out), 0)
        assert_equal(got[0], v)
        assert_equal(got[1], len(out))


def test_le_varint_raises_when_truncated() raises:
    """Every byte has the continuation bit set and the buffer ends — a
    corrupt/truncated stream must raise, not spin or read past the end."""
    var b = _bytes(0x80, 0x80, 0x80)
    with assert_raises():
        var _ = LittleEndian.varint(Span(b), 0)


def test_le_varint_raises_when_too_long() raises:
    var b = List[UInt8]()
    for _ in range(12):
        b.append(UInt8(0x80))
    b.append(UInt8(0x01))
    with assert_raises():
        var _ = LittleEndian.varint(Span(b), 0)


# ---------------------------------------------------------------------------
# bit reads and byte ordering
# ---------------------------------------------------------------------------


def test_le_bits_reads_least_significant_first() raises:
    # 0b1011_0010 = 0xB2
    var b = _bytes(0xB2)
    assert_equal(LittleEndian.bits(Span(b), 0, 1), UInt64(0))
    assert_equal(LittleEndian.bits(Span(b), 1, 1), UInt64(1))
    assert_equal(LittleEndian.bits(Span(b), 0, 4), UInt64(0x2))
    assert_equal(LittleEndian.bits(Span(b), 4, 4), UInt64(0xB))
    assert_equal(LittleEndian.bits(Span(b), 0, 8), UInt64(0xB2))


def test_le_bits_spans_a_byte_boundary() raises:
    var b = _bytes(0xFF, 0x01)
    assert_equal(LittleEndian.bits(Span(b), 4, 8), UInt64(0x1F))
    assert_equal(LittleEndian.bits(Span(b), 0, 0), UInt64(0))


def test_le_bytes_less_is_unsigned_lexicographic() raises:
    """BYTE_ARRAY ordering: unsigned bytes, and a prefix sorts first."""
    var a = _bytes(1, 2, 3)
    var b = _bytes(1, 2, 4)
    var a_again = _bytes(1, 2, 3)
    assert_true(LittleEndian.bytes_less(Span(a), Span(b)))
    assert_false(LittleEndian.bytes_less(Span(b), Span(a)))
    assert_false(LittleEndian.bytes_less(Span(a), Span(a_again)))

    var short = _bytes(1, 2)
    assert_true(LittleEndian.bytes_less(Span(short), Span(a)))
    assert_false(LittleEndian.bytes_less(Span(a), Span(short)))

    # 0x80 must compare above 0x7F, i.e. unsigned rather than signed
    var high = _bytes(0x80)
    var low = _bytes(0x7F)
    assert_true(LittleEndian.bytes_less(Span(low), Span(high)))
    assert_false(LittleEndian.bytes_less(Span(high), Span(low)))

    var empty = List[UInt8]()
    var empty_again = List[UInt8]()
    assert_true(LittleEndian.bytes_less(Span(empty), Span(low)))
    assert_false(LittleEndian.bytes_less(Span(empty), Span(empty_again)))
