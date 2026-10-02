# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The edge cases below follow google/snappy's `snappy_unittest.cc` (Copyright
# 2011 Google Inc., BSD 3-Clause): `AppendSelfPatternExtensionEdgeCases`,
# `FourByteOffset`, `ZeroOffsetCopy`, `TruncatedVarint`, `UnterminatedVarint`,
# `OverflowingVarint` and `VerifyCorrupted`. The licence is in NOTICE.txt.

"""`Snappy` -- the pure-Mojo raw-block codec, checked against libsnappy.

libsnappy is the oracle in both directions: what it compresses, `Snappy`
decodes, and what `Snappy` compresses, libsnappy decodes. The compressor is a
port of libsnappy's level-1 `CompressFragment`, so the *bytes* must match too
-- the strongest test there is of a compressor, and the one that catches a
skipped hash-table insert that every round trip would miss.

Hand-built streams cover what libsnappy never emits: copy-4, copies shorter
than 4, and the over-long literal-length forms. Every overlapping copy
(offset 1..20 x length 1..64) is decoded both as the last tag of a buffer --
the careful tail -- and followed by enough data that the fast loop owns it.

Canaries check the exact-size contract: `decompress_into` writes into the head
of a larger buffer filled with 0xA5, and the bytes past `n` must survive.
"""

from std.sys import stderr
from std.testing import assert_equal, assert_raises, assert_true

from ..byteorder import LittleEndian
from ..compression import CompressionLibs
from ..snappy import Snappy
from ..testing import Rng


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _random(n: Int, seed: UInt64 = 1) -> List[UInt8]:
    var rng = Rng(seed)
    var out = List[UInt8](capacity=n)
    for _ in range(n):
        out.append(UInt8(rng.next() >> 56))
    return out^


def _text(n: Int, seed: UInt64 = 2) -> List[UInt8]:
    """Words from a small vocabulary: short literals, short copies."""
    var words: List[String] = [
        "the ",
        "quick ",
        "brown ",
        "fox ",
        "jumps ",
        "over ",
        "lazy ",
        "dog ",
        "parquet ",
        "arrow ",
        "column ",
        "page ",
        "snappy ",
        "marrow ",
        "mojo ",
        "vector ",
        "of ",
        "and ",
        "a ",
        "in ",
    ]
    var rng = Rng(seed)
    var out = List[UInt8](capacity=n + 16)
    while len(out) < n:
        for b in words[rng.below(len(words))].as_bytes():
            out.append(b)
        if rng.below(7) == 0:
            out.append(UInt8(48 + rng.below(10)))
    out.shrink(n)
    return out^


def _ints(n: Int, seed: UInt64 = 3) -> List[UInt8]:
    """PLAIN int64 with a small range: offset-8 patterns and zero runs."""
    var rng = Rng(seed)
    var out = List[UInt8](capacity=n + 8)
    while len(out) < n:
        LittleEndian.put_le(out, UInt64(rng.below(1000)), 8)
    out.shrink(n)
    return out^


def _shapes() -> List[List[UInt8]]:
    var out = List[List[UInt8]]()
    out.append(List[UInt8]())
    out.append([UInt8(7)])
    out.append(List[UInt8](length=100_000, fill=0))
    out.append(_random(200_000))
    out.append(_text(300_000))
    out.append(_ints(1 << 20))
    out.append(_text(65_535, seed=5))
    out.append(_text(65_536, seed=6))
    out.append(_text(65_537, seed=7))
    out.append(_ints(200_003, seed=8))
    return out^


# ---------------------------------------------------------------------------
# libsnappy, the oracle
# ---------------------------------------------------------------------------


def _lib_compress(data: List[UInt8]) raises -> List[UInt8]:
    var libs = CompressionLibs()
    return libs.snappy_compress(Span(data))


def _lib_decompress(comp: List[UInt8], n: Int) raises -> List[UInt8]:
    var libs = CompressionLibs()
    var out = List[UInt8](length=max(n, 1), fill=0)
    libs.snappy_decompress(Span(comp), out.unsafe_ptr(), n)
    out.shrink(n)
    return out^


def _compress(data: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Snappy.compress(Span(data), out)
    return out^


def _compress_hashing[crc32c: Bool](data: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Snappy.compress[crc32c](Span(data), out)
    return out^


def _decompress(comp: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Snappy.decompress(Span(comp), out)
    return out^


def _assert_bytes(got: List[UInt8], want: List[UInt8], what: String) raises:
    assert_equal(len(got), len(want), what + ": length")
    for i in range(len(want)):
        if got[i] != want[i]:
            assert_equal(Int(got[i]), Int(want[i]), String(t"{what}: byte {i}"))


def _canary(n: Int) -> List[UInt8]:
    """An `n`-byte destination followed by 256 canary bytes."""
    return List[UInt8](length=n + 256, fill=0xA5)


def _assert_canary(buf: List[UInt8], n: Int, what: String) raises:
    """Nothing was written past `n`."""
    for i in range(n, len(buf)):
        if buf[i] != 0xA5:
            assert_equal(Int(buf[i]), 0xA5, String(t"{what}: canary byte {i}"))


def _roundtrip(data: List[UInt8], what: String) raises:
    var comp = _compress(data)
    assert_true(
        len(comp) <= Snappy.max_compressed_length(len(data)),
        what + ": exceeds max_compressed_length",
    )
    assert_equal(Snappy.uncompressed_length(Span(comp)), len(data), what)
    _expect(comp, data, what)


# ---------------------------------------------------------------------------
# round trips and cross-compatibility
# ---------------------------------------------------------------------------


def test_utils_snappy_roundtrip_every_small_length() raises:
    """0..300 bytes: below and across the 15-byte input margin, the 16-byte
    literal fast path, and the 60-byte literal-length boundary."""
    for n in range(301):
        _roundtrip(_random(n, seed=UInt64(n)), String(t"random {n}"))
        _roundtrip(_text(n, seed=UInt64(n)), String(t"text {n}"))


def test_utils_snappy_roundtrip_shapes() raises:
    var shapes = _shapes()
    for i in range(len(shapes)):
        _roundtrip(shapes[i], String(t"shape {i}"))


def test_utils_snappy_compress_matches_libsnappy() raises:
    """Byte-identical to libsnappy's level-1 output: the same fragments, table
    size, skip heuristic and emitters, under one of libsnappy's two hashes.
    Which one it uses is a build flag, so every input must match the same one
    -- and it is logged."""
    var shapes = _shapes()
    shapes.append(_text(3 * 65_536 + 5, seed=9))
    var n = len(shapes)
    var crc = 0
    var multiply = 0
    for i in range(n):
        var lib = _lib_compress(shapes[i])
        crc += Int(_compress_hashing[True](shapes[i]) == lib)
        multiply += Int(_compress_hashing[False](shapes[i]) == lib)
    assert_true(crc < n or multiply < n, "no input tells the hashes apart")
    assert_true(
        crc == n or multiply == n,
        String(
            t"of {n} inputs, {crc} match the CRC32C hash and {multiply} the"
            t" multiply"
        ),
    )
    print(
        "libsnappy hashes with",
        "CRC32C" if crc == n else "the multiply",
        file=stderr,
    )


def test_utils_snappy_decodes_libsnappy() raises:
    var shapes = _shapes()
    for i in range(len(shapes)):
        var comp = _lib_compress(shapes[i])
        _assert_bytes(_decompress(comp), shapes[i], String(t"shape {i}"))


def test_utils_snappy_libsnappy_decodes_ours() raises:
    var shapes = _shapes()
    for i in range(len(shapes)):
        var comp = _compress(shapes[i])
        _assert_bytes(
            _lib_decompress(comp, len(shapes[i])),
            shapes[i],
            String(t"shape {i}"),
        )


def test_utils_snappy_appends() raises:
    """`compress` and `decompress` append; a copy never reaches into what the
    list held before."""
    var data = _text(5000)
    var comp: List[UInt8] = [1, 2, 3]
    Snappy.compress(Span(data), comp)
    var body = List[UInt8](Span(comp)[3:])
    _assert_bytes(_decompress(body), data, "compressed body")
    var out: List[UInt8] = [9, 9]
    Snappy.decompress(Span(body), out)
    assert_equal(len(out), 2 + len(data))
    assert_equal(out[0], 9)
    _assert_bytes(List[UInt8](Span(out)[2:]), data, "appended")


def test_utils_snappy_max_compressed_length() raises:
    assert_equal(Snappy.max_compressed_length(0), 32)
    assert_equal(Snappy.max_compressed_length(6), 39)
    assert_equal(Snappy.max_compressed_length(65536), 32 + 65536 + 10922)


# ---------------------------------------------------------------------------
# hand-built streams
# ---------------------------------------------------------------------------


def _stream(n: Int) -> List[UInt8]:
    """A stream header declaring `n` bytes, for tags to follow."""
    var s = List[UInt8]()
    LittleEndian.put_varint(s, UInt64(n))
    return s^


def _literal(mut out: List[UInt8], data: Span[UInt8, _], extra: Int = -1):
    """A literal tag; `extra` forces a 1..4-byte length form (-1 = shortest)."""
    var n = len(data) - 1
    var nb = extra
    if nb < 0:
        nb = 0 if n < 60 else (1 if n < 256 else (2 if n < 65536 else 3))
    if nb == 0:
        out.append(UInt8(n << 2))
    else:
        out.append(UInt8((59 + nb) << 2))
        LittleEndian.put_le(out, UInt64(n), nb)
    for b in data:
        out.append(b)


def _copy1(mut out: List[UInt8], offset: Int, length: Int):
    out.append(UInt8(1 | ((length - 4) << 2) | ((offset >> 8) << 5)))
    out.append(UInt8(offset & 0xFF))


def _copy2(mut out: List[UInt8], offset: Int, length: Int):
    out.append(UInt8(2 | ((length - 1) << 2)))
    LittleEndian.put_le(out, UInt64(offset), 2)


def _copy4(mut out: List[UInt8], offset: Int, length: Int):
    out.append(UInt8(3 | ((length - 1) << 2)))
    LittleEndian.put_le(out, UInt64(offset), 4)


def _expect(stream: List[UInt8], want: List[UInt8], what: String) raises:
    _assert_bytes(_decompress(stream), want, what)
    var exact = List[UInt8](length=len(want), fill=0)
    Snappy.decompress_into(Span(stream), Span(exact))
    _assert_bytes(exact, want, what + " (decompress_into)")


def test_utils_snappy_literal_length_forms() raises:
    """Tags 60..63 carry len-1 in 1..4 bytes; an over-long form is legal."""
    var data = _text(100)
    for nb in range(1, 5):
        var s = _stream(100)
        _literal(s, Span(data), nb)
        _expect(s, data, String(t"{nb} length bytes"))
    var big = _random(70_000)
    var s = _stream(len(big))
    _literal(s, Span(big))
    _expect(s, big, "3-byte literal length")


def test_utils_snappy_copy_forms() raises:
    """The same back-reference as copy-1, copy-2 and copy-4, and copy-2 and
    copy-4 shorter than the 4 bytes libsnappy ever emits."""
    var head = _text(300)
    for length in range(1, 65):
        var want = head.copy()
        for i in range(length):
            want.append(want[len(head) - 200 + i])
        for kind in range(3):
            if kind == 0 and (length < 4 or length > 11):
                continue
            var s = _stream(len(want))
            _literal(s, Span(head))
            if kind == 0:
                _copy1(s, 200, length)
            elif kind == 1:
                _copy2(s, 200, length)
            else:
                _copy4(s, 200, length)
            _expect(s, want, String(t"kind {kind} length {length}"))


def test_utils_snappy_four_byte_offset() raises:
    """A copy-4 reaching back past 64 KiB -- legal, never emitted by
    libsnappy's 64 KiB fragments."""
    var head = _random(100_000)
    var want = head.copy()
    for i in range(64):
        want.append(head[i + 1])
    var s = _stream(len(want))
    _literal(s, Span(head))
    _copy4(s, 100_000 - 1, 64)
    _expect(s, want, "copy-4")


def test_utils_snappy_pattern_extension() raises:
    """Every overlapping copy (offset < length) for offsets 1..20, as the
    last tag of the buffer and followed by a 300-byte literal."""
    var pre = _random(300, seed=11)
    var post = _random(300, seed=12)
    for offset in range(1, 21):
        for length in range(1, 65):
            for tail in range(2):
                var want = pre.copy()
                for i in range(length):
                    want.append(want[len(pre) - offset + i])
                var s = _stream(len(want) + (len(post) if tail == 1 else 0))
                _literal(s, Span(pre))
                _copy2(s, offset, length)
                if tail == 1:
                    _literal(s, Span(post))
                    want.extend(Span(post))
                _expect(
                    s,
                    want,
                    String(t"offset {offset} length {length} tail {tail}"),
                )


# ---------------------------------------------------------------------------
# corrupt input
# ---------------------------------------------------------------------------


def _rejects(stream: List[UInt8], what: String) raises:
    """`decompress` and `decompress_into` both refuse `stream`."""
    var accepted = String()
    try:
        _ = _decompress(stream)
        accepted = "decompress"
    except:
        pass
    var n = 64
    try:
        n = Snappy.uncompressed_length(Span(stream))
    except:
        pass
    var exact = List[UInt8](length=n, fill=0)
    try:
        Snappy.decompress_into(Span(stream), Span(exact))
        accepted = "decompress_into"
    except:
        pass
    assert_true(accepted == "", String(t"{what}: {accepted} accepted it"))


def test_utils_snappy_rejects_bad_varints() raises:
    _rejects(List[UInt8](), "empty")
    _rejects([UInt8(0x80)], "truncated varint")
    _rejects([0x80, 0x80, 0x80, 0x80, 0x80, 0x0A], "unterminated varint")
    # 5th byte >= 16 overflows 32 bits.
    _rejects([0xFB, 0xFF, 0xFF, 0xFF, 0x7F], "overflowing varint")
    with assert_raises():
        _ = Snappy.uncompressed_length(Span([UInt8(0x80), 0x80]))


def test_utils_snappy_rejects_bad_copies() raises:
    var s = _stream(10)
    _copy1(s, 1, 4)  # a copy before any output
    _rejects(s, "copy first")

    s = _stream(8)
    _literal(s, Span(_text(4)))
    _copy2(s, 0, 4)  # offset 0
    _rejects(s, "zero offset")

    s = _stream(8)
    _literal(s, Span(_text(4)))
    _copy2(s, 5, 4)  # past the start
    _rejects(s, "offset past start")

    s = _stream(6)
    _literal(s, Span(_text(4)))
    _copy1(s, 4, 4)  # overflows the declared 6
    _rejects(s, "copy overflow")


def test_utils_snappy_rejects_bad_literals() raises:
    var s = _stream(10)
    s.append(UInt8(9 << 2))  # 10-byte literal ...
    s.extend(Span(_text(3)))  # ... with 3 bytes present
    _rejects(s, "truncated literal")

    s = _stream(5)
    _literal(s, Span(_text(10)))  # more than declared
    _rejects(s, "literal overflow")

    s = _stream(10)
    _literal(s, Span(_text(5)))  # less than declared
    _rejects(s, "short output")

    s = _stream(100)
    s.append(UInt8(62 << 2))  # 3 length bytes, 1 present
    s.append(UInt8(99))
    _rejects(s, "truncated literal length")


def test_utils_snappy_rejects_wrong_destination() raises:
    var data = _text(1000)
    var comp = _compress(data)
    var short = List[UInt8](length=999, fill=0)
    with assert_raises():
        Snappy.decompress_into(Span(comp), Span(short))
    var long = List[UInt8](length=1001, fill=0)
    with assert_raises():
        Snappy.decompress_into(Span(comp), Span(long))


def test_utils_snappy_decompress_refuses_an_impossible_length() raises:
    """A copy-2 tag turns 3 bytes into 64, the most any tag does, so a stream
    declaring more than 22x its own size is refused before `decompress`
    allocates the length -- and the densest valid stream still decodes."""
    var bomb = _stream(1 << 30)
    _literal(bomb, Span(_text(1)))
    var dst = List[UInt8]()
    with assert_raises():
        Snappy.decompress(Span(bomb), dst)
    assert_true(dst.capacity() < 1 << 20, "allocated the declared length")

    var copies = 1000
    var dense = _stream(1 + 64 * copies)
    var byte = _text(1)
    _literal(dense, Span(byte))
    for _ in range(copies):
        _copy2(dense, 1, 64)
    var want = List[UInt8](length=1 + 64 * copies, fill=byte[0])
    _expect(dense, want, "densest stream")


def test_utils_snappy_survives_corruption() raises:
    """Truncate a valid stream at every position and flip bytes at random:
    each decode either raises or yields exactly `n` bytes -- and never writes
    past them (the canary)."""
    var data = _text(3000)
    var comp = _compress(data)
    for cut in range(len(comp)):
        var s = List[UInt8](Span(comp)[:cut])
        with assert_raises():
            _ = _decompress(s)
    var rng = Rng(99)
    for trial in range(2000):
        var s = comp.copy()
        for _ in range(1 + rng.below(4)):
            s[rng.below(len(s))] = UInt8(rng.below(256))
        var n = len(data) + rng.below(3) - 1
        var buf = _canary(n)
        try:
            Snappy.decompress_into(Span(s), Span(buf)[:n])
        except:
            pass
        _assert_canary(buf, n, String(t"trial {trial}"))


# ---------------------------------------------------------------------------
# the exact-size contract
# ---------------------------------------------------------------------------


def test_utils_snappy_decompress_into_writes_nothing_past_n() raises:
    var cases = List[List[UInt8]]()
    for n in range(0, 400, 7):
        cases.append(_text(n, seed=UInt64(n)))
        cases.append(_ints(n, seed=UInt64(n)))
        cases.append(List[UInt8](length=n, fill=3))
    cases.append(_text(100_000))
    cases.append(_ints(100_000))
    for i in range(len(cases)):
        var comp = _compress(cases[i])
        var n = len(cases[i])
        var buf = _canary(n)
        Snappy.decompress_into(Span(comp), Span(buf)[:n])
        _assert_bytes(List[UInt8](Span(buf)[:n]), cases[i], String(t"case {i}"))
        _assert_canary(buf, n, String(t"case {i}"))


# ---------------------------------------------------------------------------
# two streams at once
# ---------------------------------------------------------------------------


def test_utils_snappy_pair_decodes_every_shape_pairing() raises:
    """Every pairing of shapes -- equal and unequal lengths, one stream far
    shorter, long literals in either -- so each stream leaves the shared loop
    at a different point and finishes alone."""
    var shapes = _shapes()
    shapes.append(_text(5000))
    shapes.append(_random(300))
    var comps = List[List[UInt8]]()
    for i in range(len(shapes)):
        comps.append(_compress(shapes[i]))
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            var na = len(shapes[i])
            var nb = len(shapes[j])
            var buf_a = _canary(na)
            var buf_b = _canary(nb)
            Snappy.decompress_pair_into(
                Span(comps[i]),
                Span(buf_a)[:na],
                Span(comps[j]),
                Span(buf_b)[:nb],
            )
            _assert_bytes(
                List[UInt8](Span(buf_a)[:na]), shapes[i], String(t"a {i} {j}")
            )
            _assert_bytes(
                List[UInt8](Span(buf_b)[:nb]), shapes[j], String(t"b {i} {j}")
            )
            _assert_canary(buf_a, na, String(t"a {i} {j}"))
            _assert_canary(buf_b, nb, String(t"b {i} {j}"))


def _copy_tag_after(stream: List[UInt8], pos: Int) raises -> Int:
    """Where the first copy-1 or copy-2 tag at or past `pos` sits, found by
    walking the tags from the preamble."""
    var ip = LittleEndian.varint(Span(stream), 0)[1]
    while ip < len(stream):
        var c = Int(stream[ip])
        var kind = c & 3
        if kind == 0:
            var length = (c >> 2) + 1
            var nb = max(length - 60, 0)
            if nb > 0:
                var field = Span(stream)[ip + 1 : ip + 1 + nb]
                length = Int(LittleEndian.partial[DType.uint32](field, 0)) + 1
            ip += 1 + nb + length
        elif ip >= pos and kind != 3:
            return ip
        else:
            ip += 1 + (1 << (kind - 1))
    raise Error(t"no copy-1 or copy-2 tag at or past {pos}")


def _pair_rejects(bad: List[UInt8], good: List[UInt8], n: Int) raises:
    """`bad` is refused as either stream of a pair, beside `good`."""
    var a = List[UInt8](length=n, fill=0)
    var b = List[UInt8](length=n, fill=0)
    with assert_raises():
        Snappy.decompress_pair_into(Span(bad), Span(a), Span(good), Span(b))
    with assert_raises():
        Snappy.decompress_pair_into(Span(good), Span(a), Span(bad), Span(b))


def test_utils_snappy_pair_rejects_either_corrupt_stream() raises:
    """A zero copy offset past the middle of a stream -- inside the shared
    loop -- and a stream cut at 60%, each in either position."""
    var n = 50_000
    var good = _compress(_text(n))
    var zero = good.copy()
    var t = _copy_tag_after(good, len(good) // 2)
    if good[t] & 3 == 1:
        zero[t] = good[t] & 0x1F  # copy-1: the offset's high 3 bits ...
        zero[t + 1] = 0  # ... and its low byte
    else:
        zero[t + 1] = 0  # copy-2: both offset bytes
        zero[t + 2] = 0
    _pair_rejects(zero, good, n)
    _pair_rejects(List[UInt8](Span(good)[: len(good) * 3 // 5]), good, n)
    var short = List[UInt8](length=n - 1, fill=0)
    var b = List[UInt8](length=n, fill=0)
    with assert_raises():
        Snappy.decompress_pair_into(
            Span(good), Span(short), Span(good), Span(b)
        )
