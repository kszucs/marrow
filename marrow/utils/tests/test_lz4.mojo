# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Lz4` -- the native LZ4 block and frame codecs, checked against liblz4,
pyarrow and the `lz4` command-line tool.

liblz4 is the oracle in both directions: `LZ4_decompress_safe` decodes what
`Lz4.compress_block` produces -- and enforces the end-of-block rules while
doing so -- and `Lz4.decompress_block_into` decodes what liblz4's default,
fast and high-compression compressors produce. A walker over our own output
checks those rules directly too.

Hand-built blocks cover what a compressor rarely emits: every length-extension
form, overlapping matches at every offset up to 20 both mid-block (the fast
path) and at the end (the checked path), and the corruptions a decoder must
refuse.

Frames are checked against pyarrow's `lz4` codec, which is the frame format
(`LZ4F_compressFrame`, linked 64 KiB blocks for inputs over 64 KiB), and
against the `lz4` tool for everything else a frame can carry: block sizes up
to 4 MiB, linked blocks, block and content checksums, the content size.
Data crosses to Python through files.
"""

from std.bit import byte_swap
from std.os.path import join
from std.pathlib import Path
from std.sys import stderr
from std.testing import assert_equal, assert_true

from ..byteorder import LittleEndian
from ..compression import Codecs
from ...errors import CorruptError, NotImplementedError
from ..hashing import XxHash32
from ..lz4 import Lz4
from ..testing import Rng, ScratchDir
from .codec_data import (
    LibLz4,
    py_list,
    python_oracles,
    write_file,
    assert_bytes,
    assert_canary,
    canary,
    corpus,
    random_bytes,
    shapes,
    small_ints,
    words,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compress(data: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Lz4.compress_block(Span(data), out)
    return out^


def _lib_decompress(block: List[UInt8], n: Int) raises -> List[UInt8]:
    var out = List[UInt8](length=max(n, 1), fill=0)
    LibLz4().decompress(Span(block), Span(out)[:n])
    out.shrink(n)
    return out^


def _lib_compress[
    symbol: StaticString
](data: List[UInt8], parameter: Int) raises -> List[UInt8]:
    """liblz4's `symbol` -- `LZ4_compress_fast` (acceleration) or
    `LZ4_compress_HC` (level) -- which share one signature."""
    var bound = Lz4.max_block_length(len(data))
    var out = List[UInt8](length=bound, fill=0)
    var src = data.copy()
    if len(src) == 0:
        src.append(0)
    var n = Codecs.handle["lz4"]().call[symbol, Int32](
        src.unsafe_ptr(),
        out.unsafe_ptr(),
        Int32(len(data)),
        Int32(bound),
        Int32(parameter),
    )
    assert_true(n > 0, String(symbol) + " failed")
    out.shrink(Int(n))
    return out^


comptime _BLOCK = 0
comptime _FRAME = 1
comptime _HADOOP = 2


def _decode[form: Int](src: List[UInt8], mut buf: List[UInt8], n: Int) raises:
    """`src`, one of the three forms, into the first `n` bytes of `buf`."""
    comptime if form == _BLOCK:
        Lz4.decompress_block_into(Span(src), Span(buf)[:n])
    elif form == _FRAME:
        Lz4.decompress_frame_into(Span(src), Span(buf)[:n])
    else:
        Lz4.decompress_hadoop_into(Span(src), Span(buf)[:n])


def _expect[
    form: Int = _BLOCK
](src: List[UInt8], want: List[UInt8], what: String) raises:
    """`src` decodes to exactly `want`, writing nothing past it."""
    var n = len(want)
    var buf = canary(n)
    _decode[form](src, buf, n)
    assert_bytes(List[UInt8](Span(buf)[:n]), want, what)
    assert_canary(buf, n, what)


def _rejects[form: Int = _BLOCK](src: List[UInt8], n: Int, what: String) raises:
    """`src` is refused when decoded into `n` bytes, and nothing past them
    is written."""
    var buf = canary(n)
    var accepted = True
    try:
        _decode[form](src, buf, n)
    except:
        accepted = False
    assert_true(not accepted, what + ": accepted")
    assert_canary(buf, n, what)


def _roundtrip(data: List[UInt8], what: String) raises:
    var block = _compress(data)
    assert_true(
        len(block) <= Lz4.max_block_length(len(data)),
        what + ": exceeds max_block_length",
    )
    _check_end_rules(block, len(data), what)
    _expect(block, data, what)


def _check_end_rules(block: List[UInt8], n: Int, what: String) raises:
    """Walk `block` and check LZ4's end-of-block rules: the last sequence is
    literals only, no match starts within 12 bytes of the end, and none ends
    within 5."""
    var ip = 0
    var op = 0
    while True:
        var token = Int(block[ip])
        ip += 1
        var lit = token >> 4
        if lit == 15:
            while True:
                var b = Int(block[ip])
                ip += 1
                lit += b
                if b != 255:
                    break
        ip += lit
        op += lit
        if ip == len(block):
            break
        ip += 2
        var ml = token & 15
        if ml == 15:
            while True:
                var b = Int(block[ip])
                ip += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        assert_true(op <= n - 12, String(t"{what}: match starts at {op}"))
        assert_true(
            op + ml <= n - 5, String(t"{what}: match ends at {op + ml}")
        )
        op += ml
    assert_equal(op, n, what + ": walked length")


# --- hand-built blocks -------------------------------------------------------


def _lengths(mut out: List[UInt8], var rest: Int):
    while rest >= 255:
        out.append(255)
        rest -= 255
    out.append(UInt8(rest))


def _sequence(
    mut out: List[UInt8], lit: Span[UInt8, _], offset: Int = 0, ml: Int = -1
):
    """One sequence: a token, its literals and -- unless `ml` is -1, the last
    sequence -- a match of `ml` bytes from `offset` back."""
    var t_ml = 0 if ml < 0 else min(ml - 4, 15)
    out.append(UInt8((min(len(lit), 15) << 4) | t_ml))
    if len(lit) >= 15:
        _lengths(out, len(lit) - 15)
    out.extend(lit)
    if ml >= 0:
        LittleEndian.put_le(out, UInt64(offset), 2)
        if ml - 4 >= 15:
            _lengths(out, ml - 4 - 15)


def _repeat(data: List[UInt8], offset: Int, ml: Int) -> List[UInt8]:
    """`data` with an LZ4-style match of `ml` bytes from `offset` back
    appended."""
    var out = data.copy()
    for _ in range(ml):
        out.append(out[len(out) - offset])
    return out^


# ---------------------------------------------------------------------------
# round trips and liblz4
# ---------------------------------------------------------------------------


def test_utils_lz4_roundtrip_every_small_length() raises:
    """0..300 bytes: below and across the 12-byte match limit, the 15-byte
    input margin and the 15-byte literal-length extension."""
    for n in range(301):
        _roundtrip(random_bytes(n, seed=UInt64(n)), String(t"random {n}"))
        _roundtrip(words(n, seed=UInt64(n)), String(t"words {n}"))
        _roundtrip(small_ints(n, seed=UInt64(n)), String(t"ints {n}"))


def test_utils_lz4_roundtrip_shapes() raises:
    var all = shapes()
    for i in range(len(all)):
        _roundtrip(all[i], String(t"shape {i}"))


def test_utils_lz4_liblz4_decodes_ours() raises:
    """`LZ4_decompress_safe` -- which enforces the end-of-block rules --
    decodes every shape; the compression ratio against liblz4's is logged."""
    var all = shapes()
    var lib = LibLz4()
    var ours = 0
    var theirs = 0
    for i in range(len(all)):
        var block = _compress(all[i])
        assert_bytes(
            _lib_decompress(block, len(all[i])), all[i], String(t"shape {i}")
        )
        ours += len(block)
        theirs += len(lib.compress(Span(all[i])))
    print(
        t"lz4: shapes compress to {ours} bytes, liblz4's to {theirs}",
        file=stderr,
    )


def test_utils_lz4_decodes_liblz4() raises:
    """Every shape from liblz4's default, fast (accelerations 1..64) and
    high-compression (levels 1..12) compressors -- the last finds longer and
    farther matches than ours ever does."""
    var all = shapes()
    var lib = LibLz4()
    for i in range(len(all)):
        var data = all[i].copy()
        _expect(
            lib.compress(Span(data)),
            data,
            String(t"default {i}"),
        )
        for acceleration in [1, 2, 8, 64]:
            _expect(
                _lib_compress["LZ4_compress_fast"](data, acceleration),
                data,
                String(t"fast {acceleration} {i}"),
            )
        for level in [1, 4, 9, 12]:
            _expect(
                _lib_compress["LZ4_compress_HC"](data, level),
                data,
                String(t"hc {level} {i}"),
            )


def test_utils_lz4_appends() raises:
    """`compress_block` appends, leaving what `dst` held."""
    var data = words(5000)
    var out: List[UInt8] = [1, 2, 3]
    Lz4.compress_block(Span(data), out)
    assert_equal(Int(out[0]), 1)
    assert_equal(Int(out[2]), 3)
    _expect(List[UInt8](Span(out)[3:]), data, "appended")


# ---------------------------------------------------------------------------
# hand-built blocks
# ---------------------------------------------------------------------------


def test_utils_lz4_length_forms() raises:
    """Literal and match lengths at and across each extension boundary:
    14 (in the token), 15 (a 0 extension byte), 270 (255 + 0) and 525."""
    for lit_len in [0, 1, 14, 15, 16, 269, 270, 271, 525]:
        for ml in [4, 18, 19, 20, 273, 274, 275, 529]:
            var lit = words(max(lit_len, 1), seed=UInt64(lit_len))
            lit.shrink(lit_len)
            var block = List[UInt8]()
            # 8 bytes of history, then the sequence under test, then 5 more
            # literals so the match never ends in the last 5 bytes.
            var history = words(8, seed=5)
            _sequence(block, Span(history), 8, 4)
            var want = _repeat(history, 8, 4)
            _sequence(block, Span(lit), 7, ml)
            want.extend(Span(lit))
            want = _repeat(want, 7, ml)
            var tail = words(5, seed=7)
            _sequence(block, Span(tail))
            want.extend(Span(tail))
            _expect(block, want, String(t"lit {lit_len} ml {ml}"))


def test_utils_lz4_overlapping_matches() raises:
    """Every offset 1..20 with every length 4..80: an overlapping match
    repeats its pattern. A 14-literal sequence first builds 24 bytes of
    history; the match under test then carries no literals, so the fast loop
    takes it -- up to length 18 with `pattern64` or 16-byte blocks, beyond
    with `match_long`. Followed by only 5 literals instead, the checked path's
    exact copy takes it."""
    var history = words(14, seed=11)
    var empty = List[UInt8]()
    for offset in range(1, 21):
        for ml in range(4, 81):
            for followed in [True, False]:
                var block = List[UInt8]()
                _sequence(block, Span(history), 14, 10)
                var want = _repeat(history, 14, 10)
                _sequence(block, Span(empty), offset, ml)
                want = _repeat(want, offset, ml)
                var tail = words(205 if followed else 5, seed=13)
                _sequence(block, Span(tail))
                want.extend(Span(tail))
                _expect(
                    block,
                    want,
                    String(t"offset {offset} ml {ml} followed {followed}"),
                )


def test_utils_lz4_empty() raises:
    """An empty input is one zero token; an empty block is corrupt."""
    var empty = List[UInt8]()
    var block = _compress(empty)
    assert_equal(len(block), 1)
    assert_equal(Int(block[0]), 0)
    _expect(block, empty, "empty")
    _rejects(empty, 0, "no token")


# ---------------------------------------------------------------------------
# corrupt input
# ---------------------------------------------------------------------------


def test_utils_lz4_rejects_bad_offsets() raises:
    var history = words(8, seed=17)
    var block = List[UInt8]()
    _sequence(block, Span(history), 0, 4)  # offset 0
    _sequence(block, Span(words(5)))
    _rejects(block, 17, "zero offset")

    block.clear()
    _sequence(block, Span(history), 9, 4)  # before the start
    _sequence(block, Span(words(5)))
    _rejects(block, 17, "offset past start")


def test_utils_lz4_rejects_bad_lengths() raises:
    var data = words(1000, seed=19)
    var block = _compress(data)
    _rejects(block, len(data) - 1, "short destination")
    _rejects(block, len(data) + 1, "long destination")
    var truncated: List[UInt8] = [0xF0, 255, 255]  # literal length runs out
    _rejects(truncated, 600, "truncated literal length")
    var no_tail = List[UInt8]()
    _sequence(no_tail, Span(words(8)), 8, 4)  # ends on a match
    _rejects(no_tail, 12, "ends on a match")


def test_utils_lz4_survives_corruption() raises:
    """Truncate a valid block at every position and flip bytes at random:
    each decode either raises or yields exactly `n` bytes -- and never writes
    past them (the canary)."""
    var data = words(3000, seed=23)
    var block = _compress(data)
    for cut in range(len(block)):
        _rejects(
            List[UInt8](Span(block)[:cut]), len(data), String(t"cut {cut}")
        )
    var rng = Rng(29)
    for trial in range(2000):
        var s = block.copy()
        for _ in range(1 + rng.below(4)):
            s[rng.below(len(s))] = UInt8(rng.below(256))
        var n = len(data) + rng.below(3) - 1
        var buf = canary(n)
        try:
            Lz4.decompress_block_into(Span(s), Span(buf)[:n])
        except:
            pass
        assert_canary(buf, n, String(t"trial {trial}"))
    # liblz4's block of a larger input: long literals and matches, which
    # the fast loop takes while it has room.
    var large = words(60_000, seed=31)
    large.extend(Span(random_bytes(2_000, seed=37)))
    large.extend(Span(words(60_000, seed=41)))
    var lib_block = LibLz4().compress(Span(large))
    for trial in range(500):
        var s = lib_block.copy()
        for _ in range(1 + rng.below(4)):
            s[rng.below(len(s))] = UInt8(rng.below(256))
        var buf = canary(len(large))
        try:
            Lz4.decompress_block_into(Span(s), Span(buf)[: len(large)])
        except:
            pass
        assert_canary(buf, len(large), String(t"large trial {trial}"))


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def _with_header_field(
    frame: List[UInt8], flag: Int, var field: List[UInt8]
) -> List[UInt8]:
    """`frame`, an `Lz4.compress_frame` output, with FLG bit `flag` set and
    `field` inserted after BD -- a valid header checksum over both."""
    var out = List[UInt8](Span(frame)[:6])
    out[4] |= UInt8(flag)
    out.extend(Span(field))
    out.append(UInt8((XxHash32.hash(Span(out)[4:]) >> 8) & 0xFF))
    out.extend(Span(frame)[7:])
    return out^


def _frame(data: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    Lz4.compress_frame(Span(data), out)
    return out^


def test_utils_lz4_compress_matches_liblz4() raises:
    """Byte for byte liblz4 1.10.0's `LZ4_compress_default` blocks and
    `LZ4F_compressFrame` frames with default preferences: the shapes, the
    corpora at sizes around its two-byte-slot limit and its 64 KiB frame
    block, and text. Another liblz4 may choose differently, so against one
    only the version is logged."""
    var lib = LibLz4()
    var version = lib.version()
    if version != 11000:
        print(t"lz4: liblz4 {version}, so not compared", file=stderr)
        return
    var all = shapes()
    for kind in ["strings", "ints", "floats"]:
        for n in [
            1000,
            (1 << 16) - 1,
            1 << 16,
            (1 << 16) + 10,
            (1 << 16) + 11,
            200_000,
            1 << 20,
        ]:
            all.append(corpus(kind, n))
    for n in [0, 1, 12, 13, 300, 70_000]:
        all.append(words(n, seed=UInt64(n + 1)))
    for i in range(len(all)):
        var block = List[UInt8]()
        Lz4.compress_block(Span(all[i]), block)
        var theirs = lib.compress(Span(all[i]))
        assert_true(
            block == theirs,
            String(
                t"block {i}: {len(all[i])} bytes in {len(block)}, liblz4"
                t" {len(theirs)}"
            ),
        )
        var frame = _frame(all[i])
        var lib_frame = lib.compress_frame(Span(all[i]))
        assert_true(
            frame == lib_frame,
            String(
                t"frame {i}: {len(all[i])} bytes in {len(frame)}, liblz4"
                t" {len(lib_frame)}"
            ),
        )


def test_utils_lz4_frame_roundtrip() raises:
    """Every length 0..300 and the shapes -- those over 64 KiB span several
    blocks, and random bytes are stored blocks."""
    for n in range(301):
        var data = words(n, seed=UInt64(n))
        _expect[_FRAME](_frame(data), data, String(t"words {n}"))
    var all = shapes()
    for i in range(len(all)):
        var frame = _frame(all[i])
        assert_true(len(frame) <= Lz4.max_frame_length(len(all[i])))
        _expect[_FRAME](frame, all[i], String(t"shape {i}"))


def test_utils_lz4_frame_is_arrows() raises:
    """The header Arrow C++ writes for an IPC buffer: magic, FLG 0x60, BD
    0x40 and the checksum byte LZ4F computes for them."""
    var frame = _frame(words(100))
    assert_equal(Int(frame[4]), 0x60)
    assert_equal(Int(frame[5]), 0x40)
    assert_equal(Int(frame[6]), 0x82)


def test_utils_lz4_frame_pyarrow_both_ways() raises:
    """Pyarrow decodes our frames and we decode its, whose blocks are
    linked for inputs over 64 KiB."""
    var py = python_oracles()
    var all = shapes()
    with ScratchDir() as dir:
        var raw = join(dir, "raw")
        var packed = join(dir, "packed")
        var back = join(dir, "back")
        for i in range(len(all)):
            var n = len(all[i])
            write_file(raw, all[i])
            _ = py["pa_compress"](raw, packed, "lz4", -1)
            _expect[_FRAME](
                Path(packed).read_bytes(), all[i], String(t"pa {i}")
            )
            write_file(packed, _frame(all[i]))
            _ = py["pa_decompress"](packed, back, "lz4", n)
            assert_bytes(Path(back).read_bytes(), all[i], String(t"ours {i}"))


def test_utils_lz4_frame_cli_variants() raises:
    """The `lz4` tool's frames: every block size, linked blocks, block
    checksums, the content size, with and without the content checksum, and
    its high-compression levels."""
    var py = python_oracles()
    var data = words(300_000, seed=31)
    data.extend(Span(random_bytes(100_000, seed=37)))
    data.extend(Span(small_ints(300_000, seed=41)))
    var variants: List[List[String]] = [
        ["-B4"],
        ["-B5"],
        ["-B6"],
        ["-B7"],
        ["-BD"],
        ["-B4", "-BD"],
        ["-BX"],
        ["--content-size"],
        ["--no-frame-crc"],
        ["-B4", "-BD", "-BX", "--content-size", "--no-frame-crc"],
        ["-9"],
        ["-12", "-BD"],
    ]
    with ScratchDir() as dir:
        var raw = join(dir, "raw")
        var packed = join(dir, "packed")
        write_file(raw, data)
        for v in variants:
            _ = py["cli"]("lz4", py_list(v), raw, packed)
            _expect[_FRAME](
                Path(packed).read_bytes(), data, String(t"{len(v)}")
            )


def test_utils_lz4_frame_rejects() raises:
    var data = words(200_000, seed=43)
    var frame = _frame(data)
    var n = len(data)

    var bad_hc = frame.copy()
    bad_hc[6] ^= 1
    _rejects[_FRAME](bad_hc, n, "header checksum")

    var bad_version = frame.copy()
    bad_version[4] = 0xA0
    _rejects[_FRAME](bad_version, n, "version")

    var trailing = frame.copy()
    trailing.append(0)
    _rejects[_FRAME](trailing, n, "trailing byte")

    _rejects[_FRAME](frame, n - 1, "short destination")
    _rejects[_FRAME](frame, n + 1, "long destination")

    var dictionary = _with_header_field(frame, 0x01, [1, 2, 3, 4])
    try:
        Lz4.decompress_frame_into(Span(dictionary), Span(data))
        assert_true(False, "dictionary frame accepted")
    except e:
        assert_true(e.isa[NotImplementedError](), String(e))
    var flipped = frame.copy()
    flipped[4] |= 0x01  # a dictionary flag the header checksum does not cover
    try:
        Lz4.decompress_frame_into(Span(flipped), Span(data))
        assert_true(False, "flipped flag accepted")
    except e:
        assert_true(e.isa[CorruptError](), String(e))

    for cut in range(0, len(frame), 97):
        _rejects[_FRAME](
            List[UInt8](Span(frame)[:cut]), n, String(t"cut {cut}")
        )


def test_utils_lz4_frame_checks_what_it_declares() raises:
    """Each check is the only one that can catch its corruption: a stored
    byte under a block checksum alone, the same under a content checksum
    alone, and a declared content size the destination does not match."""
    var py = python_oracles()
    var data = random_bytes(200_000, seed=47)
    var n = len(data)
    with ScratchDir() as dir:
        var raw = join(dir, "raw")
        var packed = join(dir, "packed")
        write_file(raw, data)
        var by_block: List[String] = ["-BX", "--no-frame-crc"]
        var by_content: List[String] = []
        var sized: List[String] = ["--content-size", "--no-frame-crc"]
        for args in [by_block^, by_content^]:
            _ = py["cli"]("lz4", py_list(args), raw, packed)
            var frame = Path(packed).read_bytes()
            _expect[_FRAME](frame, data, "intact")
            frame[100] ^= 1  # a byte inside the first block, which is stored
            _rejects[_FRAME](frame, n, String(t"checksum {len(args)}"))
        _ = py["cli"]("lz4", py_list(sized), raw, packed)
        var frame = Path(packed).read_bytes()
        _expect[_FRAME](frame, data, "sized")
        _rejects[_FRAME](frame, n - 1, "content size, short destination")
        _rejects[_FRAME](frame, n + 1, "content size, long destination")


# ---------------------------------------------------------------------------
# Hadoop framing
# ---------------------------------------------------------------------------


def _hadoop(data: List[UInt8], cuts: List[Int] = []) raises -> List[UInt8]:
    """`data` as Hadoop frames, a new one at each of `cuts`."""
    var out = List[UInt8]()
    var bounds = cuts.copy()
    bounds.append(len(data))
    var pos = 0
    for end in bounds:
        Lz4.compress_hadoop(Span(data)[pos:end], out)
        pos = end
    return out^


def _be32(data: List[UInt8], pos: Int) -> Int:
    return Int(byte_swap(LittleEndian.fixed[DType.uint32](Span(data), pos)))


def test_utils_lz4_hadoop_roundtrip() raises:
    """One frame per input: its decompressed and compressed sizes, big-endian,
    then the block."""
    var all = shapes()
    for n in range(0, 301, 13):
        all.append(words(n, seed=UInt64(n)))
    for i in range(len(all)):
        var src = _hadoop(all[i])
        assert_equal(_be32(src, 0), len(all[i]))
        assert_equal(_be32(src, 4), len(src) - 8)
        _expect[_HADOOP](src, all[i], String(t"input {i}"))


def test_utils_lz4_hadoop_reads_what_writers_wrote() raises:
    """A run of frames, as parquet-mr's codec writes a page over its buffer
    size; a plain block, as Parquet C++ wrote before it framed them; and an
    empty page either way."""
    var data = words(300_000, seed=53)
    _expect[_HADOOP](_hadoop(data, [100_000, 250_000]), data, "three frames")
    _expect[_HADOOP](_hadoop(data, [0, 300_000]), data, "empty frames")
    _expect[_HADOOP](_compress(data), data, "plain block")
    _expect[_HADOOP](_hadoop([]), [], "empty frame")
    _expect[_HADOOP](_compress([]), [], "empty block")


def test_utils_lz4_hadoop_rejects() raises:
    var data = words(10_000, seed=59)
    var n = len(data)
    var src = _hadoop(data, [4_000])
    _rejects[_HADOOP](src, n - 1, "short destination")
    _rejects[_HADOOP](src, n + 1, "long destination")
    var sizes = src.copy()
    sizes[3] ^= 1  # the first frame's decompressed size
    _rejects[_HADOOP](sizes, n, "frame size")
    for cut in range(1, len(src), 61):
        _rejects[_HADOOP](List[UInt8](Span(src)[:cut]), n, String(t"cut {cut}"))
