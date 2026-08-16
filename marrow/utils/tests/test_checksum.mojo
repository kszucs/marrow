"""`Crc32` against zlib's own output.

Every expected value here came from CPython's `zlib.crc32`, which is the same
ISO-3309 checksum Parquet specifies for its optional per-page CRC — so these
pin marrow's implementation to the reference rather than to itself.
"""

from std.testing import assert_equal, assert_true

from ..checksum import Crc32


def _bytes(s: String) -> List[UInt8]:
    return List[UInt8](s.as_bytes())


def test_crc32_reference_vectors() raises:
    var empty = List[UInt8]()
    assert_equal(Crc32.compute(Span(empty)), UInt32(0x00000000))
    assert_equal(Crc32.compute(Span(_bytes("a"))), UInt32(0xE8B7BE43))
    assert_equal(Crc32.compute(Span(_bytes("abc"))), UInt32(0x352441C2))
    # the check value every CRC-32 implementation is expected to reproduce
    assert_equal(Crc32.compute(Span(_bytes("123456789"))), UInt32(0xCBF43926))


def test_crc32_covers_every_byte_value() raises:
    """0x00-0xFF exercises all 256 table entries, so a wrong polynomial or a
    reflection mistake cannot slip through on ASCII alone."""
    var all_bytes = List[UInt8]()
    for i in range(256):
        all_bytes.append(UInt8(i))
    assert_equal(Crc32.compute(Span(all_bytes)), UInt32(0x29058C73))


def test_crc32_incremental_matches_one_shot() raises:
    """Parquet v2 checksums the levels then the compressed values, so the
    incremental form has to equal hashing the concatenation."""
    var whole = _bytes("marrowmarrowmarrow")
    var head = _bytes("marrow")
    var tail = _bytes("marrowmarrow")

    var c = Crc32()
    c.update(Span(head))
    c.update(Span(tail))
    assert_equal(c.value(), Crc32.compute(Span(whole)))


def test_crc32_empty_update_is_identity() raises:
    var empty = List[UInt8]()
    var data = _bytes("abc")
    var c = Crc32()
    c.update(Span(empty))
    c.update(Span(data))
    c.update(Span(empty))
    assert_equal(c.value(), Crc32.compute(Span(data)))


def test_crc32_value_is_repeatable() raises:
    """`value()` finalizes without consuming the state, so reading it twice
    gives the same answer."""
    var data = _bytes("123456789")
    var c = Crc32()
    c.update(Span(data))
    assert_equal(c.value(), UInt32(0xCBF43926))
    assert_equal(c.value(), UInt32(0xCBF43926))


def test_crc32_detects_a_single_bit_flip() raises:
    var a = _bytes("marrow")
    var b = List[UInt8]()
    for i in range(len(a)):
        b.append(a[i])
    b[0] = b[0] ^ 1
    assert_true(Crc32.compute(Span(a)) != Crc32.compute(Span(b)))
