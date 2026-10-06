# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The frames in `test_utils_zstd_golden_frames` and
# `test_utils_zstd_golden_errors` are from libzstd's own test corpus,
# Copyright (c) Meta Platforms, Inc. (BSD 3-Clause); see NOTICE.txt.

"""`Zstd` -- the native Zstandard decoder and encoder, checked against
libzstd, the `zstd` tool and pyarrow.

libzstd compresses the shared shapes and the Parquet-like corpora at every
level and with the parameters that change the frame: no content size (so a
window descriptor), a checksum, small windows, superblocks (treeless
literals and repeated tables), forced raw or Huffman literals. A census over
everything decoded asserts each literals type and each sequence mode was
met, so a passing run is known to have exercised them.

Hand-made frames come from libzstd's own decoder regression corpus
(`tests/golden-decompression{,-errors}`), and a compressed block of exactly
128 KiB, which the spec allows and libzstd itself rejected until 1.5.3.

The encoder's frames are held byte for byte to libzstd 1.5.7's level 1, a
census over them asserts each literals type, table mode and block type was
written, and libzstd, pyarrow and the `zstd` tool decode them.
"""

from std.collections import Dict
from std.os.path import join
from std.pathlib import Path
from std.sys import stderr
from std.testing import assert_true

from ...errors import CorruptError, NotImplementedError
from ...views import BufferView
from ..compression import Codecs
from ..testing import Rng, ScratchDir
from ..zstd import Zstd
from ..zstd.huffman import HuffmanTable
from .codec_data import (
    LibZstd,
    assert_bytes,
    assert_canary,
    canary,
    corpus,
    py_list,
    python_oracles,
    random_bytes,
    shapes,
    small_ints,
    words,
    write_file,
)
from ...codecs.byteorder import LittleEndian


# ---------------------------------------------------------------------------
# libzstd
# ---------------------------------------------------------------------------

comptime _LEVEL = 100
comptime _WINDOW_LOG = 101
comptime _STRATEGY = 107
comptime _TARGET_BLOCK = 130
comptime _CONTENT_SIZE = 200
comptime _CHECKSUM = 201
comptime _LITERALS = 1002
"""`ZSTD_c_literalCompressionMode`: 1 always Huffman, 2 always raw."""


def _lib_compress(
    data: List[UInt8], params: List[Tuple[Int, Int]]
) raises -> List[UInt8]:
    """`ZSTD_compress2` with `params` set on a fresh context."""
    var z = Codecs.handle["zstd"]()
    var cctx = z.call["ZSTD_createCCtx", Int]()
    for p in params:
        var rc = z.call["ZSTD_CCtx_setParameter", Int](
            cctx, Int32(p[0]), Int32(p[1])
        )
        assert_true(
            z.call["ZSTD_isError", UInt32](rc) == 0,
            String(t"parameter {p[0]} = {p[1]} refused"),
        )
    var bound = z.call["ZSTD_compressBound", Int](len(data))
    var out = List[UInt8](length=bound, fill=0)
    var src = data.copy()
    src.append(0)  # never a null pointer, even for empty input
    var n = z.call["ZSTD_compress2", Int](
        cctx, out.unsafe_ptr(), bound, src.unsafe_ptr(), len(data)
    )
    _ = z.call["ZSTD_freeCCtx", Int](cctx)
    assert_true(z.call["ZSTD_isError", UInt32](n) == 0, "ZSTD_compress2 failed")
    out.shrink(n)
    return out^


def _lib_rejects(frame: List[UInt8], n: Int) raises -> Bool:
    var out = List[UInt8](length=n + 1, fill=0)
    var z = Codecs.handle["zstd"]()
    var got = z.call["ZSTD_decompress", Int](
        out.unsafe_ptr(), n, frame.unsafe_ptr(), len(frame)
    )
    return z.call["ZSTD_isError", UInt32](got) != 0


def _lib_decompress(frame: List[UInt8], n: Int) raises -> List[UInt8]:
    var out = List[UInt8](length=n, fill=0)
    LibZstd().decompress(Span(frame), Span(out))
    return out^


# ---------------------------------------------------------------------------
# the census: which literals types and sequence modes a frame uses
# ---------------------------------------------------------------------------


def _features() -> List[String]:
    return [
        "raw block",
        "RLE block",
        "compressed block",
        "raw literals",
        "Huffman literals, 1 stream",
        "Huffman literals, 4 streams",
        "treeless literals, 1 stream",
        "treeless literals, 4 streams",
        "no sequences",
        "predefined table",
        "RLE table",
        "FSE table",
        "repeated table",
    ]


def _assert_census(
    seen: Dict[String, Int], what: String, skip: List[String]
) raises:
    """Log the census, and assert every feature but `skip` was met."""
    for entry in seen.items():
        print(t"zstd {what}: {entry.key}: {entry.value}", file=stderr)
    var missing = List[String]()
    for name in _features():
        if seen.get(name, 0) == 0 and name not in skip:
            missing.append(name)
    assert_true(len(missing) == 0, String(t"never {what}: {missing}"))


def _mark(name: String, mut seen: Dict[String, Int]) raises:
    seen[name] = seen.get(name, 0) + 1


def _census(frame: List[UInt8], mut seen: Dict[String, Int]) raises:
    """Count the features of every block in `frame`'s frames."""
    var tables: List[String] = [
        "predefined table",
        "RLE table",
        "FSE table",
        "repeated table",
    ]
    var ip = 0
    while ip < len(frame):
        var magic = LittleEndian.fixed[DType.uint32](Span(frame), ip)
        ip += 4
        if magic != Zstd.MAGIC:
            ip += 4 + Int(LittleEndian.fixed[DType.uint32](Span(frame), ip))
            continue
        var fhd = Int(frame[ip])
        var single = fhd & 0x20 != 0
        var fcs = 1 << (fhd >> 6) if fhd >> 6 > 0 else Int(single)
        ip += 1 + Int(not single) + ((1 << (fhd & 3)) >> 1) + fcs
        while True:
            var header = Int(
                LittleEndian.partial[DType.uint32](Span(frame)[ip : ip + 3], 0)
            )
            ip += 3
            var kind = (header >> 1) & 3
            var size = header >> 3
            if kind == 0:
                _mark("raw block", seen)
                ip += size
            elif kind == 1:
                _mark("RLE block", seen)
                ip += 1
            else:
                _mark("compressed block", seen)
                var b = Span(frame)[ip : ip + size]
                var lt = Int(b[0]) & 3
                var sf = (Int(b[0]) >> 2) & 3
                var section: Int
                if lt <= 1:
                    var h = 1 if sf & 1 == 0 else 2 if sf == 1 else 3
                    var n = Int(
                        LittleEndian.partial[DType.uint32](b[:h], 0)
                    ) >> (3 if sf & 1 == 0 else 4)
                    section = h + (n if lt == 0 else 1)
                    _mark("raw literals" if lt == 0 else "RLE literals", seen)
                else:
                    var h = 3 if sf <= 1 else sf + 2
                    var w = 10 if sf <= 1 else 14 if sf == 2 else 18
                    var v = Int(LittleEndian.partial[DType.uint64](b[:h], 0))
                    section = h + ((v >> (4 + w)) & ((1 << w) - 1))
                    _mark(
                        String(
                            "Huffman" if lt == 2 else "treeless",
                            " literals, ",
                            "1 stream" if sf == 0 else "4 streams",
                        ),
                        seen,
                    )
                var s = b[section:]
                if s[0] == 0:
                    _mark("no sequences", seen)
                else:
                    var at = 1 if s[0] < 128 else 2 if s[0] < 255 else 3
                    var modes = Int(s[at])
                    for shift in [6, 4, 2]:
                        _mark(tables[(modes >> shift) & 3], seen)
                ip += size
            if header & 1 != 0:
                break
        if fhd & 0x04 != 0:
            ip += 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _expect(frame: List[UInt8], want: List[UInt8], what: String) raises:
    """`frame` decodes to exactly `want`, writing nothing past it."""
    var n = len(want)
    var buf = canary(n)
    Zstd.decompress_into(Span(frame), Span(buf)[:n])
    assert_bytes(List[UInt8](Span(buf)[:n]), want, what)
    assert_canary(buf, n, what)


def _rejects(frame: List[UInt8], n: Int, what: String) raises:
    """`frame` is refused as a corrupt one when decoded into `n` bytes, and
    nothing past them is written."""
    var buf = canary(n)
    var accepted = True
    try:
        Zstd.decompress_into(Span(frame), Span(buf)[:n])
    except e:
        accepted = False
        assert_true(e.isa[CorruptError](), what + ": " + String(e))
    assert_true(not accepted, what + ": accepted")
    assert_canary(buf, n, what)


def _hex(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()

    def nibble(c: UInt8) -> UInt8:
        return c - 48 if c <= 57 else c - 87

    for i in range(0, len(b), 2):
        out.append(nibble(b[i]) << 4 | nibble(b[i + 1]))
    return out^


def _inputs() -> List[List[UInt8]]:
    """The shapes, short inputs, and Parquet-like pages."""
    var all = shapes()
    for n in range(0, 300, 37):
        all.append(words(n, seed=UInt64(n)))
    for kind in ["strings", "ints", "floats"]:
        all.append(corpus(kind, 1 << 20))
    return all^


def _records(n: Int) -> List[UInt8]:
    """`id=00000;id=00001;...`: every sequence one literal digit and the same
    match, so literal and match lengths each take a one-symbol table."""
    var out = List[UInt8](capacity=9 * n)
    for i in range(n):
        out.extend(Span("id=".as_bytes()))
        var d = 10000
        while d > 0:
            out.append(UInt8(48 + (i // d) % 10))
            d //= 10
        out.append(59)  # ';'
    return out^


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_utils_zstd_decodes_libzstd() raises:
    """Frames libzstd wrote at every level and with every frame-shaping
    parameter, and a census proving each literals type and sequence mode was
    decoded."""
    var all = _inputs()
    var settings: List[List[Tuple[Int, Int]]] = [
        [(_LEVEL, -5)],
        [(_LEVEL, 1)],
        [(_LEVEL, 3)],
        [(_LEVEL, 9)],
        [(_LEVEL, 19)],
        [(_LEVEL, 3), (_CHECKSUM, 1), (_CONTENT_SIZE, 0)],
        [(_LEVEL, 3), (_WINDOW_LOG, 10)],
        [(_LEVEL, 6), (_WINDOW_LOG, 15)],
        [(_LEVEL, 3), (_TARGET_BLOCK, 1340)],
        [(_LEVEL, 3), (_LITERALS, 1)],
        [(_LEVEL, 3), (_LITERALS, 2)],
        [(_LEVEL, 1), (_STRATEGY, 1)],
        [(_LEVEL, 3), (_STRATEGY, 9)],
    ]
    var seen = Dict[String, Int]()
    for s in range(len(settings)):
        for i in range(len(all)):
            var frame = _lib_compress(all[i], settings[s])
            _census(frame, seen)
            _expect(frame, all[i], String(t"setting {s}, input {i}"))
    _assert_census(seen, "decoded", [])


def test_utils_zstd_ultra_levels() raises:
    """Levels 20-22: long matches and 8 MiB windows."""
    var data = words(400_000, seed=11)
    data.extend(Span(small_ints(400_000, seed=13)))
    for level in [20, 22]:
        _expect(_lib_compress(data, [(_LEVEL, level)]), data, String(level))


def test_utils_zstd_cli_and_pyarrow() raises:
    """The `zstd` tool's frames -- multithreaded, long-distance matching,
    without a checksum or content size -- and pyarrow's at its levels.
    The other direction is `test_utils_zstd_others_decode_ours`."""
    var py = python_oracles()
    var data = corpus("strings", 1 << 20)
    data.extend(Span(random_bytes(50_000, seed=17)))
    data.extend(Span(corpus("floats", 300_000)))
    var variants: List[List[String]] = [
        ["-1"],
        ["-19"],
        ["--ultra", "-21"],
        ["-3", "--long=24"],
        ["-3", "-T2"],
        ["-5", "--no-check", "--no-content-size"],
    ]
    with ScratchDir() as dir:
        var raw = join(dir, "raw")
        var packed = join(dir, "packed")
        write_file(raw, data)
        for v in variants:
            _ = py["cli"]("zstd", py_list(v), raw, packed)
            _expect(Path(packed).read_bytes(), data, String(t"{len(v)} args"))
        for level in [1, 3, 9, 19]:
            _ = py["pa_compress"](raw, packed, "zstd", level)
            _expect(Path(packed).read_bytes(), data, String(t"pa {level}"))


def test_utils_zstd_golden_frames() raises:
    """The decoder regressions libzstd keeps: a compressed block holding
    nothing, an RLE first block, zero sequences in the 2-byte count, and a
    compressed block of exactly 128 KiB."""
    _expect(_hex("28b52ffd00001500000000"), [], "empty block")
    var rle = _hex(
        "28b52ffda4000010000200100002001000020010000200100002001000020010"
        "000200100003001000f13e16e1"
    )
    _expect(rle, _lib_decompress(rle, 1 << 20), "RLE first block")
    var hello = "Hello World!\n".as_bytes()
    _expect(
        _hex("28b52ffd00008500006848656c6c6f20576f726c64210a8000"),
        List[UInt8](hello),
        "zero sequences, 2-byte count",
    )
    # A frame with a 128 KiB window and one compressed block of 128 KiB:
    # raw literals with the 3-byte header, then no sequences.
    var n = (1 << 17) - 4
    var data = words(n, seed=19)
    var frame = _hex("28b52ffd0038")
    var header = 1 | 2 << 1 | (1 << 17) << 3
    LittleEndian.put_le(frame, UInt64(header), 3)
    frame.append(UInt8(0 | 3 << 2 | (n & 15) << 4))
    frame.append(UInt8((n >> 4) & 0xFF))
    frame.append(UInt8(n >> 12))
    frame.extend(Span(data))
    frame.append(0)
    _expect(frame, data, "128 KiB block")
    var over = frame.copy()
    over[6] = 0x30  # a 64 KiB window: the block is now too large for it
    _rejects(over, n, "block over the window")


def test_utils_zstd_rle_literals() raises:
    """RLE literals, which libzstd emits only for a block whose literals are
    one repeated byte: in each of the three header sizes, alone and before a
    sequence."""
    for n in [20, 4000, 100_000]:
        var want = List[UInt8](length=n, fill=0x7A)
        var lits: List[UInt8]
        if n < 32:
            lits = [UInt8(1 | n << 3)]
        elif n < 4096:
            lits = [UInt8(1 | 1 << 2 | (n & 15) << 4), UInt8(n >> 4)]
        else:
            lits = [
                UInt8(1 | 3 << 2 | (n & 15) << 4),
                UInt8((n >> 4) & 0xFF),
                UInt8(n >> 12),
            ]
        lits.append(0x7A)
        lits.append(0)  # no sequences
        var frame = _hex("28b52ffd0040")  # a 1 MiB window
        var header = 1 | 2 << 1 | len(lits) << 3
        LittleEndian.put_le(frame, UInt64(header), 3)
        frame.extend(Span(lits))
        _expect(frame, want, String(t"{n} RLE literals"))


def test_utils_zstd_checks_only_it_makes() raises:
    """Frames built by hand from the predefined tables (the spec's Appendix
    A states), each wrong in a way only one check can see: a byte the
    sequences bitstream never reaches, and a repeat offset reaching back
    across a frame boundary into the output before it. libzstd agrees on
    every one."""
    # Raw literals "abcd", then one sequence: 4 literals, offset value 2 (a
    # repeat: 4, the second initial offset) and a match of 3.
    var good = _hex("28b52ffd000055000020616263640100804b04")
    var want = List[UInt8]("abcdabc".as_bytes())
    assert_bytes(_lib_decompress(good, 7), want, "libzstd")
    _expect(good, want, "one sequence")
    # The same frame with a byte before the bitstream: every sequence
    # decodes as before, but 8 bits are never read.
    var slack = _hex("28b52ffd00005d000020616263640100ff804b04")
    assert_true(_lib_rejects(slack, 7), "libzstd accepts the slack")
    _rejects(slack, 7, "slack")
    # A frame whose sequence has no literals, so offset value 1 means the
    # second initial offset, 4 -- before its own start, though not before
    # the 4 bytes of the frame ahead of it.
    var ahead = _hex(
        "28b52ffd000021000061626364" + "28b52ffd0000350000000100000002"
    )
    assert_true(_lib_rejects(ahead, 7), "libzstd accepts the offset")
    _rejects(ahead, 7, "offset across frames")


def test_utils_zstd_huffman_stream_consumed() raises:
    """A Huffman stream is decoded symbol by symbol to exactly its first
    bit: two symbols of one bit each, three coded, read as three and not
    as two."""
    var table = HuffmanTable()
    _ = table.read(Span(_hex("8010")))  # symbol 0 weight 1; 1 is implied
    var stream = _hex("0d")  # the end mark, then 1, 0, 1
    var out = List[UInt8](length=3, fill=0)
    table.decode(Span(stream), BufferView(Span(out)), 3, 1, len(stream))
    assert_bytes(out, [1, 0, 1], "three symbols")
    var refused = False
    try:
        table.decode(Span(stream), BufferView(Span(out)), 2, 1, len(stream))
    except e:
        refused = True
    assert_true(refused, "a stream with a symbol left over was accepted")


def test_utils_zstd_golden_errors() raises:
    """The corrupt regressions libzstd keeps, refused into any output size: an
    offset of 0, truncated Huffman weight states, and bytes after an empty
    sequences section."""
    for hex in [
        "28b52ffd0000450000080002002f430bae",
        "28b52ffd000055000072800104207e1f02aa00",
        "28b52ffd00009500006848656c6c6f20576f726c64210a80000000",
    ]:
        var frame = _hex(hex)
        for n in range(48):
            _rejects(frame, n, String(t"{hex} into {n}"))


def test_utils_zstd_frame_sequences() raises:
    """Concatenated frames, skippable frames among them, and a dictionary id
    of 0, which means none."""
    var a = words(70_000, seed=23)
    var b = small_ints(90_000, seed=29)
    var want = a.copy()
    want.extend(Span(b))
    var frames = _lib_compress(a, [(_LEVEL, 3)])
    frames.extend(Span(_hex("5a2a4d1805000000aabbccddee")))  # skippable
    frames.extend(Span(_lib_compress(b, [(_LEVEL, 3), (_CHECKSUM, 1)])))
    _expect(frames, want, "two frames and a skippable one")
    _expect(_hex("502a4d1800000000"), [], "a skippable frame alone")
    # FHD 0x21: single segment, a 1-byte dictionary id and a 1-byte size,
    # then one raw block holding "abc".
    _expect(
        _hex("28b52ffd210003190000616263"),
        List[UInt8]("abc".as_bytes()),
        "dictionary 0",
    )


def test_utils_zstd_rejects() raises:
    var data = words(300_000, seed=31)
    var frame = _lib_compress(data, [(_LEVEL, 3), (_CHECKSUM, 1)])
    var n = len(data)
    _rejects(frame, n - 1, "short destination")
    _rejects(frame, n + 1, "long destination")
    _expect([], [], "no frames")
    _rejects([], 1, "no frames for a byte")
    var bad_magic = frame.copy()
    bad_magic[0] ^= 1
    _rejects(bad_magic, n, "magic")
    var reserved = frame.copy()
    reserved[4] |= 0x08
    _rejects(reserved, n, "reserved bit")
    var trailing = frame.copy()
    trailing.append(0)
    _rejects(trailing, n, "trailing byte")
    var dictionary = _hex("28b52ffd210703190000616263")  # dictionary 7
    try:
        var out = List[UInt8](length=3, fill=0)
        Zstd.decompress_into(Span(dictionary), Span(out))
        assert_true(False, "dictionary frame accepted")
    except e:
        assert_true(e.isa[NotImplementedError](), String(e))
    for cut in range(0, len(frame), 1 + len(frame) // 200):
        _rejects(List[UInt8](Span(frame)[:cut]), n, String(t"cut {cut}"))


def test_utils_zstd_survives_corruption() raises:
    """Every bit flip in a checksummed frame is refused or harmless -- the
    output is still the input -- and none writes past the destination."""
    for level in [1, 19]:
        var data = corpus("strings", 40_000)
        data.extend(Span(small_ints(20_000, seed=37)))
        var frame = _lib_compress(data, [(_LEVEL, level), (_CHECKSUM, 1)])
        var n = len(data)
        for pos in range(4, len(frame)):
            for bit in [0, 3, 7]:
                var bad = frame.copy()
                bad[pos] ^= UInt8(1 << bit)
                var buf = canary(n)
                try:
                    Zstd.decompress_into(Span(bad), Span(buf)[:n])
                    assert_bytes(
                        List[UInt8](Span(buf)[:n]),
                        data,
                        String(t"flip {pos}.{bit} accepted"),
                    )
                except e:
                    # A flipped dictionary flag asks for a dictionary.
                    assert_true(
                        e.isa[CorruptError]() or e.isa[NotImplementedError](),
                        String(e),
                    )
                assert_canary(buf, n, String(t"flip {pos}.{bit}"))


# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------


def _compress(data: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Zstd.compress(Span(data), out)
    assert_true(len(out) <= Zstd.max_compressed_length(len(data)))
    return out^


def _encoder_inputs() -> List[List[UInt8]]:
    """`_inputs`, the sizes around the 128 KiB block, and a full block of
    text before a short one of random letters -- literals its code covers,
    too few to pay for a code of their own."""
    var all = _inputs()
    for n in [(1 << 17) - 1, 1 << 17, (1 << 17) + 1, 3 << 17]:
        all.append(words(n, seed=UInt64(n)))
    for letters in [200, 600]:
        var data = words(1 << 17)
        data.resize(1 << 17, 0)
        var rng = Rng(UInt64(letters))
        for _ in range(letters):
            data.append(UInt8(97 + rng.below(26)))
        all.append(data^)
    return all^


def _flat_ends(top: Int) -> List[UInt8]:
    """A 128 KiB block that finds no matches: bytes below 64, which Huffman
    coding would shrink, between a first and a last 4 KiB that are nearly
    flat, shuffled -- the first holding every byte 16 times, the last byte
    0 `top` times and no other byte more than 16. libzstd samples only
    those two ends of literals this unmatched, and stores them when their
    largest counts come to 68 or less: `top` 52 is stored, 53 coded."""
    var rng = Rng(5)
    var out = List[UInt8](capacity=1 << 17)
    for part in range(3):
        var at = len(out)
        if part == 1:
            for _ in range((1 << 17) - 2 * 4096):
                out.append(UInt8(rng.below(64)))
        else:
            var zeros = 16 if part == 0 else top
            for i in range(4096):
                out.append(0 if i < zeros else UInt8(1 + (i - zeros) % 255))
            for i in range(4095, 0, -1):
                out.swap_elements(at + i, at + rng.below(i + 1))
    return out^


def test_utils_zstd_compress_roundtrip() raises:
    """Every input decodes back with our decoder and with libzstd's."""
    var all = _encoder_inputs()
    for i in range(len(all)):
        var frame = _compress(all[i])
        _expect(frame, all[i], String(t"input {i}"))
        assert_bytes(
            _lib_decompress(frame, len(all[i])), all[i], String(t"lib {i}")
        )


def test_utils_zstd_compress_uses_every_feature() raises:
    """The census over our own frames: everything but a block without
    sequences and a repeated table, which level 1 never writes."""
    var all = _encoder_inputs()
    # One byte throughout: RLE blocks after the first.
    all.append(List[UInt8](length=300_000, fill=7))
    all.append(_records(5000))
    var seen = Dict[String, Int]()
    for i in range(len(all)):
        _census(_compress(all[i]), seen)
    _assert_census(seen, "written", ["no sequences", "repeated table"])


def test_utils_zstd_compress_matches_libzstd() raises:
    """Byte for byte libzstd 1.5.7's one-shot level 1 -- its fast match
    finder and window, block splitting, Huffman code reuse and tie-breaking,
    and frame header -- on the shapes, the corpora at sizes across the
    level's parameter rows and past its window, and text. Another libzstd
    chooses differently in places, so against one the frames are held
    within 3% of its size instead, and the version logged."""
    var version = LibZstd().version()
    var all = shapes()
    for kind in ["strings", "ints", "floats"]:
        for n in [1000, 20_000, 200_000, 1 << 18, (1 << 19) + 1, 5 << 20]:
            all.append(corpus(kind, n))
    for n in [37, 4000, (1 << 17) + 1, 3 << 17]:
        all.append(words(n, seed=UInt64(n)))
    # One repeated byte after a block of text, last and between: whether
    # it is an RLE block is decided after matching, whose table keeps it.
    all.append(List[UInt8](length=300_000, fill=7))
    for run in [2, 6, 7, 30, 300_000]:
        var data = words(1 << 17)
        data.resize(1 << 17, 0)
        for _ in range(run):
            data.append(7)
        all.append(data.copy())
        data.extend(Span(words(4000)))
        all.append(data^)
    all.append(_flat_ends(52))
    all.append(_flat_ends(53))
    if version != 10507:
        print(t"zstd: libzstd {version}, so sizes only", file=stderr)
    for i in range(len(all)):
        var ours = _compress(all[i])
        var theirs = _lib_compress(all[i], [(_LEVEL, 1)])
        var what = String(
            t"input {i}: {len(all[i])} bytes in {len(ours)}, level 1"
            t" {len(theirs)}"
        )
        if version == 10507:
            assert_true(ours == theirs, what)
        else:
            assert_true(len(ours) * 100 <= len(theirs) * 103, what)


def test_utils_zstd_others_decode_ours() raises:
    """Pyarrow and the `zstd` tool decode our frames, a 5 MB one among
    them."""
    var py = python_oracles()
    var all = shapes()
    all.append(corpus("strings", 5 << 20))
    with ScratchDir() as dir:
        var packed = join(dir, "packed")
        var back = join(dir, "back")
        for i in range(len(all)):
            write_file(packed, _compress(all[i]))
            _ = py["pa_decompress"](packed, back, "zstd", len(all[i]))
            assert_bytes(Path(back).read_bytes(), all[i], String(t"pa {i}"))
            _ = py["cli"]("zstd", py_list(["-d"]), packed, back)
            assert_bytes(Path(back).read_bytes(), all[i], String(t"cli {i}"))
