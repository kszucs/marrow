# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND MIT

# The streams in `test_brotli_reference_vectors` are from google/brotli
# 1.2.0's `tests/testdata`, Copyright 2013 Google Inc. (MIT); see NOTICE.txt.

"""`Brotli` -- the native Brotli decoder and encoder, checked against
libbrotli.

libbrotli compresses the shared shapes and the Parquet-like corpora at every
quality, at windows from 10 to 24 bits, in each of its modes, and with the
parameters that change the stream: distance postfix bits and direct codes,
small blocks (many block switches), no literal context modeling, flushes
(empty metadata blocks and stored blocks) and metadata. Hand-made streams come
from libbrotli's own test data. Every stream is decoded natively and must
match; truncated and bit-flipped streams must be refused wherever libbrotli
refuses them, and never written past the output.

The encoder's streams are held byte for byte to libbrotli 1.2.0's quality 1,
and decode back with both decoders.
"""

from std.sys import stderr
from std.testing import assert_equal, assert_true

from ...errors import CorruptError
from ...utils.testing import Rng
from ..brotli import Brotli
from ..brotli_dictionary import (
    OFFSETS_BY_LENGTH,
    SIZE_BITS_BY_LENGTH,
    copy_word,
)
from ..core import Codecs
from .codec_data import (
    ResolvedBrotli,
    assert_bytes,
    assert_canary,
    canary,
    corpus,
    random_bytes,
    shapes,
    small_ints,
    words,
)


def _decode(stream: List[UInt8], n: Int) raises -> List[UInt8]:
    """Decode natively into an `n`-byte output with a canary after it."""
    var out = canary(n)
    Brotli.decompress(Span(stream), Span(out)[:n])
    assert_canary(out, n, "brotli")
    out.shrink(n)
    return out^


def _check(
    mut lib: ResolvedBrotli,
    data: List[UInt8],
    stream: List[UInt8],
    what: String,
) raises:
    """The stream decodes natively to `data`, as it does with libbrotli."""
    assert_true(
        len(data) <= Brotli.max_decompressed_length(len(stream)),
        what + ": over the decompressed bound",
    )
    var want = List[UInt8](length=len(data), fill=0)
    lib.decompress(Span(stream), Span(want))
    assert_bytes(want, data, what + ": libbrotli")
    assert_bytes(_decode(stream, len(data)), data, what)


def _hex(s: StaticString) raises -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(0, len(b), 2):
        var v = 0
        for k in range(2):
            var c = Int(b[i + k])
            v = 16 * v + (c - 48 if c < 58 else c - 87)
        out.append(UInt8(v))
    return out^


def _slice(data: List[UInt8], start: Int, end: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=end - start)
    for i in range(start, end):
        out.append(data[i])
    return out^


def _repeat(s: StaticString, n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    var b = s.as_bytes()
    while len(out) < n:
        out.append(b[len(out) % len(b)])
    return out^


# ---------------------------------------------------------------------------
# decoding libbrotli's streams
# ---------------------------------------------------------------------------


def test_brotli_dictionary_matches_libbrotli() raises:
    """Every byte of the static dictionary, and the tables that cut it into
    words, are libbrotli's: `BrotliGetDictionary()` hands out its
    `BrotliDictionary` -- `size_bits_by_length[32]`, `offsets_by_length[32]`,
    `data_size` and `data`."""
    var address = Codecs.handle["brotlidec"]().call[
        "BrotliGetDictionary", Int
    ]()
    var lib = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var size_bits = materialize[SIZE_BITS_BY_LENGTH]()
    var offsets = materialize[OFFSETS_BY_LENGTH]()
    for l in range(25):
        assert_equal(Int(size_bits[l]), Int(lib[unsafe_offset=l]))
        var offset = lib.unsafe_offset(32 + 4 * l).unsafe_bitcast[UInt32]()
        assert_equal(Int(offsets[l]), Int(offset[unsafe_offset=0]))
    var size = Int(offsets[24]) + (1 << Int(size_bits[24])) * 24
    var fields = lib.unsafe_offset(160).unsafe_bitcast[Int]()
    assert_equal(size, fields[unsafe_offset=0])
    var data = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=fields[unsafe_offset=1]
    )
    var buf = Array[UInt8, 64](fill=0)
    var start = 0
    while start < size:
        var n = min(24, size - start)
        copy_word(start, n, buf, 3)
        for k in range(n):
            if buf[3 + k] != data[unsafe_offset=start + k]:
                assert_equal(
                    Int(buf[3 + k]),
                    Int(data[unsafe_offset=start + k]),
                    String(t"dictionary byte {start + k}"),
                )
        start += n


def test_brotli_decodes_every_quality() raises:
    var lib = ResolvedBrotli()
    print(t"libbrotli {hex(lib.version())}", file=stderr)
    var inputs = shapes()
    inputs.append(corpus("strings", 1 << 18))
    inputs.append(corpus("ints", 1 << 18))
    inputs.append(corpus("floats", 1 << 18))
    for i in range(len(inputs)):
        for q in range(12):
            var stream = lib.compress(Span(inputs[i]), q)
            _check(lib, inputs[i], stream, String(t"shape {i} at quality {q}"))


def test_brotli_decodes_every_window() raises:
    var lib = ResolvedBrotli()
    var data = words(300_000)
    var ints = small_ints(300_000)
    for lgwin in range(10, 25):
        for q in [1, 5, 9, 11]:
            var a = lib.compress(Span(data), q, lgwin)
            _check(lib, data, a, String(t"words, lgwin {lgwin}, q {q}"))
            var b = lib.compress(Span(ints), q, lgwin)
            _check(lib, ints, b, String(t"ints, lgwin {lgwin}, q {q}"))


def test_brotli_decodes_every_mode() raises:
    var lib = ResolvedBrotli()
    var inputs = [
        words(200_000),
        corpus("strings", 200_000),
        small_ints(200_000),
    ]
    for i in range(len(inputs)):
        for mode in range(3):
            for q in [2, 6, 11]:
                var stream = lib.compress(Span(inputs[i]), q, 22, mode)
                _check(
                    lib,
                    inputs[i],
                    stream,
                    String(t"input {i}, mode {mode}, q {q}"),
                )


def test_brotli_decodes_stream_parameters() raises:
    """Distance postfix bits and direct codes, small blocks, no literal
    context modeling: what one-shot compression never writes."""
    var lib = ResolvedBrotli()
    var data = small_ints(300_000)
    var text = words(300_000)
    var sets = List[List[Tuple[Int, Int]]]()
    for npostfix in range(4):
        for ndirect in [0, 1, 4, 15]:
            var direct = ndirect << npostfix
            if direct <= 120:
                sets.append(
                    [
                        (ResolvedBrotli.QUALITY, 11),
                        (ResolvedBrotli.NPOSTFIX, npostfix),
                        (ResolvedBrotli.NDIRECT, direct),
                    ]
                )
    sets.append([(ResolvedBrotli.QUALITY, 9), (ResolvedBrotli.LGBLOCK, 16)])
    sets.append([(ResolvedBrotli.QUALITY, 11), (ResolvedBrotli.LGBLOCK, 16)])
    sets.append(
        [
            (ResolvedBrotli.QUALITY, 11),
            (ResolvedBrotli.DISABLE_LITERAL_CONTEXT_MODELING, 1),
        ]
    )
    sets.append([(ResolvedBrotli.QUALITY, 5), (ResolvedBrotli.LGWIN, 10)])
    for i in range(len(sets)):
        for input in [data.copy(), text.copy()]:
            var chunks = List[Tuple[Int, List[UInt8]]]()
            chunks.append((ResolvedBrotli.PROCESS, input.copy()))
            var stream = ResolvedBrotli.stream(chunks, sets[i])
            _check(lib, input, stream, String(t"parameter set {i}"))


def test_brotli_decodes_flushes_and_metadata() raises:
    """Flushing pads with an empty metadata block or ends a meta-block where
    it stands, and metadata blocks are skipped."""
    var lib = ResolvedBrotli()
    var text = words(100_000)
    for q in [0, 1, 4, 11]:
        var chunks = List[Tuple[Int, List[UInt8]]]()
        var want = List[UInt8]()
        var at = 0
        var step = 1
        var rng = Rng(9)
        while at < len(text):
            var n = min(len(text) - at, step)
            var part = _slice(text, at, at + n)
            want.extend(Span(part))
            chunks.append((ResolvedBrotli.PROCESS, part^))
            chunks.append((ResolvedBrotli.FLUSH, List[UInt8]()))
            if rng.below(3) == 0:
                chunks.append(
                    (ResolvedBrotli.EMIT_METADATA, random_bytes(rng.below(300)))
                )
            at += n
            step = 2 * step + 1
        var stream = ResolvedBrotli.stream(
            chunks, [(ResolvedBrotli.QUALITY, q)]
        )
        _check(lib, want, stream, String(t"flushed at q {q}"))


def test_brotli_decodes_large_streams() raises:
    """More than a meta-block holds: several meta-blocks, and distances
    across them."""
    var lib = ResolvedBrotli()
    var data = words(17 << 20, seed=11)
    for q in [0, 1, 2]:
        var stream = lib.compress(Span(data), q, 24)
        _check(lib, data, stream, String(t"17 MiB at q {q}"))


def test_brotli_decodes_edge_sizes() raises:
    var lib = ResolvedBrotli()
    var sizes: List[Int] = [
        0,
        1,
        2,
        3,
        4,
        15,
        16,
        17,
        63,
        64,
        65,
        1007,
        1008,
        1009,
        1024,
    ]
    for lgwin in [10, 16]:
        var window = 1 << lgwin
        for d in [-17, -16, -15, -1, 0, 1, 16]:
            sizes.append(window + d)
            sizes.append(2 * window + d)
    for n in sizes:
        var data = _repeat("abcabdabcabeXYZ0123456789", n)
        var noise = random_bytes(n, seed=UInt64(n))
        for i in range(n // 3):
            data[3 * i] = noise[i]
        for q in [0, 1, 5, 11]:
            for lgwin in [10, 16, 22]:
                var stream = lib.compress(Span(data), q, lgwin)
                _check(
                    lib,
                    data,
                    stream,
                    String(t"{n} bytes, q {q}, lgwin {lgwin}"),
                )


def test_brotli_reference_vectors() raises:
    """Hand-made streams from libbrotli's test data: every way to write an
    empty stream, metadata, stored and compressed meta-blocks, and long
    copies at the window's edge."""
    var lib = ResolvedBrotli()
    var empty: List[StaticString] = [
        "06",
        "8101",
        "a101",
        "b101",
        "c101",
        "d101",
        "e101",
        "f101",
        "33",
        "35",
        "37",
        "39",
        "3b",
        "3d",
        "3f",
        "1a",
        "81160058",
    ]
    for s in empty:
        var stream = _hex(s)
        assert_true(lib.accepts(Span(stream), 0), String(s))
        assert_equal(len(_decode(stream, 0)), 0, String(s))
    var x: List[StaticString] = [
        "0b00805803",
        "0000105803",
        "2c00580000085803",
        "000010580d",
        "a1000000008115080400",
    ]
    for s in x:
        _check(lib, _repeat("X", 1), _hex(s), String(s))
    _check(
        lib,
        _repeat("XXXXXXXXXXYYYYYYYYYY", 20),
        _hex("1b130000a4b0b2ea8147028a"),
        "10x10y",
    )
    _check(lib, _repeat("X", 64), _hex("1b3f000024b0e2998012"), "64x")
    _check(lib, _repeat("Xyzzy", 5), _hex("0b028058797a7a7903"), "xyzzy")
    comptime fox = "The quick brown fox jumps over the lazy dog"
    _check(
        lib,
        _repeat(fox, 43),
        _hex(
            "0b158054686520717569636b2062726f776e20666f78206a756d7073206f7665"
            "7220746865206c617a7920646f6703"
        ),
        "quickfox",
    )
    _check(
        lib,
        _repeat(fox, 176128),
        _hex(
            "5bffaf02c022795cfb5a8c423bf42555195a9299b135c8199e9e0a7b4b90b93c"
            "98c80940f3e6d94de46d651b2787135fa6e930967b3c15d8531c"
        ),
        "quickfox_repeated",
    )
    _check(
        lib,
        List[UInt8](length=262144, fill=0),
        _hex("5bffff036002201e0b28f77e00"),
        "zeros",
    )
    var stream = _hex("5bff0001400a00ab167bac00484e73ed019203")
    var want = List[UInt8](length=65792, fill=0)
    lib.decompress(Span(stream), Span(want))
    assert_bytes(_decode(stream, 65792), want, "backward65536")


# ---------------------------------------------------------------------------
# corrupt streams
# ---------------------------------------------------------------------------


def _agrees(
    mut lib: ResolvedBrotli, stream: List[UInt8], n: Int, what: String
) raises:
    """The native decoder accepts `stream` as an `n`-byte output exactly when
    libbrotli does, to the same bytes, and writes nothing past `n`
    either way."""
    var out = canary(n)
    var ok = True
    try:
        Brotli.decompress(Span(stream), Span(out)[:n])
    except e:
        ok = False
        assert_true(String(e).startswith("CorruptError"), String(e))
    assert_canary(out, n, what)
    var accepted = lib.accepts(Span(stream), n)
    assert_equal(ok, accepted, what + ": native and libbrotli disagree")
    if ok:
        var want = List[UInt8](length=n, fill=0)
        lib.decompress(Span(stream), Span(want))
        out.shrink(n)
        assert_bytes(out, want, what)


def test_brotli_refuses_truncated_streams() raises:
    var lib = ResolvedBrotli()
    var data = words(5000)
    for q in [0, 1, 5, 11]:
        var stream = lib.compress(Span(data), q)
        for cut in range(len(stream)):
            var part = _slice(stream, 0, cut)
            _agrees(lib, part, len(data), String(t"q {q} cut at {cut}"))


def test_brotli_bit_flips() raises:
    """Every bit of small streams flipped, and random flips of larger ones:
    the native decoder agrees with libbrotli on every one."""
    var lib = ResolvedBrotli()
    var small = words(700)
    for q in [0, 1, 5, 11]:
        var stream = lib.compress(Span(small), q)
        for bit in range(8 * len(stream)):
            var flipped = stream.copy()
            flipped[bit // 8] ^= UInt8(1 << (bit % 8))
            _agrees(lib, flipped, len(small), String(t"q {q} bit {bit}"))
    var rng = Rng(7)
    var inputs = [words(50_000), small_ints(50_000), corpus("strings", 50_000)]
    for i in range(len(inputs)):
        for q in [1, 6, 11]:
            var stream = lib.compress(Span(inputs[i]), q)
            for _ in range(300):
                var flipped = stream.copy()
                var bit = rng.below(8 * len(stream))
                flipped[bit // 8] ^= UInt8(1 << (bit % 8))
                _agrees(
                    lib,
                    flipped,
                    len(inputs[i]),
                    String(t"input {i} q {q} bit {bit}"),
                )


def test_brotli_refuses_garbage() raises:
    var lib = ResolvedBrotli()
    var rng = Rng(3)
    for k in range(2000):
        var stream = random_bytes(1 + rng.below(64), seed=UInt64(k))
        _agrees(lib, stream, rng.below(200), String(t"garbage {k}"))


def test_brotli_refuses_wrong_sizes() raises:
    var lib = ResolvedBrotli()
    var data = words(10_000)
    var stream = lib.compress(Span(data), 5)
    for n in [0, 1, len(data) - 1, len(data) + 1, 2 * len(data)]:
        _agrees(lib, stream, n, String(t"size {n}"))
    var empty = List[UInt8]()
    _agrees(lib, empty, 0, "no stream")


# ---------------------------------------------------------------------------
# compressing
# ---------------------------------------------------------------------------


def _compress(data: List[UInt8]) raises -> List[UInt8]:
    var stream = List[UInt8]()
    Brotli.compress(Span(data), stream)
    return stream^


def _assert_identical(
    mut lib: ResolvedBrotli, data: List[UInt8], what: String
) raises:
    """`Brotli.compress` writes libbrotli's quality-1 stream, which decodes
    back to `data` both ways."""
    var want = lib.compress(Span(data), 1, 22)
    var got = _compress(data)
    assert_bytes(got, want, what + ": against libbrotli")
    _check(lib, data, got, what)


def test_brotli_compress_matches_libbrotli() raises:
    var lib = ResolvedBrotli()
    var inputs = shapes()
    for kind in ["strings", "ints", "floats"]:
        for n in [1 << 16, 1 << 20, 3 << 20]:
            inputs.append(corpus(kind, n))
    inputs.append(words(300_000, seed=12))
    inputs.append(random_bytes(1 << 17))
    for i in range(len(inputs)):
        _assert_identical(lib, inputs[i], String(t"input {i}"))


def test_brotli_compress_edge_sizes() raises:
    """Sizes around the 128 KiB block, the hash table's growth and the
    16-byte margin, and a run of one byte."""
    var lib = ResolvedBrotli()
    var sizes: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 21, 22, 31]
    for base in [256, 1 << 15, 1 << 16, 1 << 17, 2 << 17]:
        for d in [-17, -16, -1, 0, 1, 16]:
            sizes.append(base + d)
    for n in sizes:
        var text = words(n, seed=UInt64(n))
        _assert_identical(lib, text, String(t"{n} bytes of words"))
        var noise = random_bytes(n, seed=UInt64(n))
        _assert_identical(lib, noise, String(t"{n} random bytes"))
        var run = List[UInt8](length=n, fill=7)
        _assert_identical(lib, run, String(t"{n} equal bytes"))


def test_brotli_compress_large() raises:
    """More than a 4 MiB fragment: the hash table restarts, and so do the
    matches."""
    var lib = ResolvedBrotli()
    _assert_identical(lib, words(9 << 20, seed=13), "9 MiB of words")
    _assert_identical(lib, small_ints(17 << 20, seed=14), "17 MiB of ints")
