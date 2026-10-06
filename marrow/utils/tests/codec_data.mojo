# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Inputs and checks the codec tests and benchmarks share: deterministic byte
generators, the bench corpora, the canary that proves a decoder wrote nothing
past its destination, and the oracles: libsnappy, liblz4 and libzstd,
pyarrow, and the codecs' command-line tools."""

from std.python import Python, PythonObject
from std.testing import assert_equal

from ..compression import Codecs
from ..lz4 import Lz4
from ..testing import Rng
from ...codecs.byteorder import LittleEndian


# ---------------------------------------------------------------------------
# test inputs
# ---------------------------------------------------------------------------


def random_bytes(n: Int, seed: UInt64 = 1) -> List[UInt8]:
    var rng = Rng(seed)
    var out = List[UInt8](capacity=n)
    for _ in range(n):
        out.append(UInt8(rng.next() >> 56))
    return out^


def words(n: Int, seed: UInt64 = 2) -> List[UInt8]:
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


def small_ints(n: Int, seed: UInt64 = 3) -> List[UInt8]:
    """PLAIN int64 with a small range: offset-8 patterns and zero runs."""
    var rng = Rng(seed)
    var out = List[UInt8](capacity=n + 8)
    while len(out) < n:
        LittleEndian.put_le(out, UInt64(rng.below(1000)), 8)
    out.shrink(n)
    return out^


def shapes() -> List[List[UInt8]]:
    """Empty, one byte, a zero run, random bytes, text and ints, and texts
    straddling the 64 KiB fragment size."""
    var out = List[List[UInt8]]()
    out.append(List[UInt8]())
    out.append([UInt8(7)])
    out.append(List[UInt8](length=100_000, fill=0))
    out.append(random_bytes(200_000))
    out.append(words(300_000))
    out.append(small_ints(1 << 20))
    out.append(words(65_535, seed=5))
    out.append(words(65_536, seed=6))
    out.append(words(65_537, seed=7))
    out.append(small_ints(200_003, seed=8))
    return out^


# ---------------------------------------------------------------------------
# bench corpora
# ---------------------------------------------------------------------------


def _urls(n: Int) -> List[UInt8]:
    var hosts: List[String] = [
        "www.example.com",
        "shop.example.org",
        "news.site.net",
        "cdn.host.io",
    ]
    var parts: List[String] = [
        "item",
        "product",
        "category",
        "search",
        "article",
        "user",
        "page",
        "view",
        "list",
        "detail",
        "cart",
        "checkout",
        "2026",
        "en",
        "static",
    ]
    var rng = Rng(1)
    var out = List[UInt8](capacity=n + 256)
    while len(out) < n:
        var url = String("https://") + hosts[rng.below(len(hosts))]
        for _ in range(1 + rng.below(4)):
            url += "/" + parts[rng.below(len(parts))]
        url += "?id=" + String(rng.below(10_000_000))
        LittleEndian.put_le(out, UInt64(url.byte_length()), 4)
        for b in url.as_bytes():
            out.append(b)
    out.shrink(n)
    return out^


def _timestamps(n: Int) -> List[UInt8]:
    var rng = Rng(2)
    var out = List[UInt8](capacity=n + 8)
    var t = UInt64(1_790_000_000_000_000)
    while len(out) < n:
        t += UInt64(rng.below(5_000_000))
        LittleEndian.put_le(out, t, 8)
    out.shrink(n)
    return out^


def _floats(n: Int) -> List[UInt8]:
    var rng = Rng(3)
    var out = List[UInt8](capacity=n + 8)
    while len(out) < n:
        var f = Float64(rng.next() >> 11) / Float64(1 << 53)
        LittleEndian.put_le(out, UInt64(f.to_bits[DType.uint64]()), 8)
    out.shrink(n)
    return out^


def corpus(kind: String, n: Int) -> List[UInt8]:
    """`n` bytes of a bench corpus, the PLAIN encoding of a column that stays
    PLAIN in a real file: `strings` (BYTE_ARRAY URLs), `ints` (int64
    microsecond timestamps with random gaps) or `floats` (float64 in [0, 1),
    near-incompressible)."""
    if kind == "strings":
        return _urls(n)
    elif kind == "ints":
        return _timestamps(n)
    else:
        return _floats(n)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def assert_bytes(got: List[UInt8], want: List[UInt8], what: String) raises:
    assert_equal(len(got), len(want), what + ": length")
    for i in range(len(want)):
        if got[i] != want[i]:
            assert_equal(Int(got[i]), Int(want[i]), String(t"{what}: byte {i}"))


def canary(n: Int) -> List[UInt8]:
    """An `n`-byte destination followed by 256 canary bytes."""
    return List[UInt8](length=n + 256, fill=0xA5)


def assert_canary(buf: List[UInt8], n: Int, what: String) raises:
    """Nothing was written past `n`."""
    for i in range(n, len(buf)):
        if buf[i] != 0xA5:
            assert_equal(Int(buf[i]), 0xA5, String(t"{what}: canary byte {i}"))


# ---------------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------------


comptime _PYTHON_ORACLES = """
import shutil, subprocess
import pyarrow as pa

def pa_compress(src, dst, codec, level):
    data = open(src, "rb").read()
    kw = {} if level < 0 else {"compression_level": level}
    open(dst, "wb").write(pa.Codec(codec, **kw).compress(data, asbytes=True))

def pa_decompress(src, dst, codec, n):
    data = open(src, "rb").read()
    out = pa.decompress(data, decompressed_size=n, codec=codec, asbytes=True)
    open(dst, "wb").write(out)

def cli(tool, args, src, dst):
    with open(dst, "wb") as out:
        subprocess.run([shutil.which(tool), *args, "-c", src], stdout=out, check=True)
"""


def python_oracles() raises -> PythonObject:
    """`pa_compress`, `pa_decompress` and `cli` -- pyarrow's codecs and a
    codec's command-line tool -- exchanging data through files."""
    var g = Python.dict()
    _ = Python.import_module("builtins").exec(_PYTHON_ORACLES, g)
    return g


def write_file(path: String, data: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def py_list(items: List[String]) raises -> PythonObject:
    var out = Python.list()
    for item in items:
        out.append(item)
    return out^


comptime _SnappyFn = def(
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
    Pointer[UInt8, MutUntrackedOrigin],
    Pointer[UInt, MutUntrackedOrigin],
) thin abi("C") -> Int32
"""`snappy_compress` and `snappy_uncompress` share one C signature."""


struct LibSnappy(Movable):
    """The two libsnappy entry points, resolved once. `CompressionLibs` looks a
    symbol up on every call -- deliberately, see CLAUDE.md -- and at 64 KiB a
    `dlsym` is a measurable share of a decode that is not the codec's."""

    var _compress: _SnappyFn
    var _uncompress: _SnappyFn
    var _size: List[UInt]
    """The `size_t*` in-out argument, on the heap: a local reached only
    through an untracked pointer need not be in memory at the call."""

    def __init__(out self) raises:
        var h = Codecs.handle["snappy"]()
        self._compress = h.get_function[_SnappyFn]("snappy_compress")
        self._uncompress = h.get_function[_SnappyFn]("snappy_uncompress")
        self._size = [UInt(0)]

    def compress(
        mut self, src: Span[UInt8, _], mut dst: List[UInt8]
    ) raises -> Int:
        """Compress into `dst`, sized to `max_compressed_length`; return the
        compressed length."""
        self._size[0] = UInt(len(dst))
        var status = self._compress(
            _c(src.unsafe_ptr()),
            len(src),
            _c(dst.unsafe_ptr()),
            _c(self._size.unsafe_ptr()),
        )
        if status != 0:
            raise Error("snappy_compress failed")
        return Int(self._size[0])

    def decompress[
        o: MutOrigin
    ](mut self, src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """Decompress into exactly `dst`."""
        self._size[0] = UInt(len(dst))
        var status = self._uncompress(
            _c(src.unsafe_ptr()),
            len(src),
            _c(dst.unsafe_ptr()),
            _c(self._size.unsafe_ptr()),
        )
        if status != 0:
            raise Error("snappy_uncompress failed")


comptime _Lz4Fn = def(
    Pointer[UInt8, MutUntrackedOrigin],
    Pointer[UInt8, MutUntrackedOrigin],
    Int32,
    Int32,
) thin abi("C") -> Int32
"""`LZ4_compress_default` and `LZ4_decompress_safe` share one C signature."""


@always_inline
def _c[T: AnyType](p: Pointer[T, _]) -> Pointer[T, MutUntrackedOrigin]:
    """`p` for the C ABI, which tracks no origins."""
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(p))


struct LibLz4(Movable):
    """The liblz4 block calls, resolved once, and its frame writer: the
    oracle the LZ4 tests check against and the benchmarks race."""

    var _compress: _Lz4Fn
    var _decompress: _Lz4Fn

    def __init__(out self) raises:
        var h = Codecs.handle["lz4"]()
        self._compress = h.get_function[_Lz4Fn]("LZ4_compress_default")
        self._decompress = h.get_function[_Lz4Fn]("LZ4_decompress_safe")

    def compress_into(
        self, src: Span[UInt8, _], mut dst: List[UInt8]
    ) raises -> Int:
        """Append `LZ4_compress_default`'s block for `src` to `dst`; return
        its length. A benchmark reserves `dst` first, so the timed call does
        not allocate."""
        var at = len(dst)
        var bound = Lz4.max_block_length(len(src))
        dst.resize(unsafe_uninit_length=at + bound)
        var n = self._compress(
            _c(src.unsafe_ptr()),
            _c(dst.unsafe_ptr().unsafe_offset(at)),
            Int32(len(src)),
            Int32(bound),
        )
        if n <= 0:
            raise Error("LZ4_compress_default failed")
        dst.shrink(at + Int(n))
        return Int(n)

    def compress(self, src: Span[UInt8, _]) raises -> List[UInt8]:
        """`LZ4_compress_default`'s block for `src`."""
        var dst = List[UInt8]()
        _ = self.compress_into(src, dst)
        return dst^

    def compress_frame(self, src: Span[UInt8, _]) raises -> List[UInt8]:
        """`LZ4F_compressFrame`'s frame for `src` with default preferences,
        as Arrow C++ calls it."""
        var h = Codecs.handle["lz4"]()
        var bound = h.call["LZ4F_compressFrameBound", Int](len(src), 0)
        var dst = List[UInt8](length=bound, fill=0)
        var n = h.call["LZ4F_compressFrame", Int](
            dst.unsafe_ptr(), bound, src.unsafe_ptr(), len(src), 0
        )
        if h.call["LZ4F_isError", UInt32](n) != 0:
            raise Error("LZ4F_compressFrame failed")
        dst.shrink(n)
        return dst^

    def version(self) raises -> Int:
        """`LZ4_versionNumber`: 11000 for 1.10.0."""
        return Int(Codecs.handle["lz4"]().call["LZ4_versionNumber", Int32]())

    def decompress[
        o: MutOrigin
    ](self, src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """Decompress into exactly `dst`."""
        var n = self._decompress(
            _c(src.unsafe_ptr()),
            _c(dst.unsafe_ptr()),
            Int32(len(src)),
            Int32(len(dst)),
        )
        if Int(n) != len(dst):
            raise Error("LZ4_decompress_safe failed")


comptime _ZstdCompressFn = def(
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
    Int32,
) thin abi("C") -> Int
comptime _ZstdDecompressFn = def(
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
) thin abi("C") -> Int
comptime _ZstdBoundFn = def(Int) thin abi("C") -> Int


struct LibZstd(Movable):
    """The one-shot libzstd entry points, resolved once: what the ZSTD
    benchmarks race."""

    var _compress: _ZstdCompressFn
    var _decompress: _ZstdDecompressFn
    var _bound: _ZstdBoundFn

    def __init__(out self) raises:
        var h = Codecs.handle["zstd"]()
        self._compress = h.get_function[_ZstdCompressFn]("ZSTD_compress")
        self._decompress = h.get_function[_ZstdDecompressFn]("ZSTD_decompress")
        self._bound = h.get_function[_ZstdBoundFn]("ZSTD_compressBound")

    def version(self) raises -> Int:
        """`ZSTD_versionNumber`: 10507 for 1.5.7."""
        return Int(Codecs.handle["zstd"]().call["ZSTD_versionNumber", Int32]())

    def compress_into(
        self, src: Span[UInt8, _], mut dst: List[UInt8], level: Int
    ) raises -> Int:
        """Append `ZSTD_compress`'s frame for `src` at `level` to `dst`;
        return its length. A benchmark reserves `dst` first, so the timed
        call does not allocate."""
        var at = len(dst)
        var bound = self._bound(len(src))
        dst.resize(unsafe_uninit_length=at + bound)
        var n = self._compress(
            _c(dst.unsafe_ptr().unsafe_offset(at)),
            bound,
            _c(src.unsafe_ptr()),
            len(src),
            Int32(level),
        )
        if n <= 0 or n > bound:
            raise Error("ZSTD_compress failed")
        dst.shrink(at + n)
        return n

    def compress(self, src: Span[UInt8, _], level: Int) raises -> List[UInt8]:
        """`ZSTD_compress`'s frame for `src` at `level`."""
        var dst = List[UInt8]()
        _ = self.compress_into(src, dst, level)
        return dst^

    def decompress[
        o: MutOrigin
    ](self, src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """Decompress into exactly `dst`."""
        var n = self._decompress(
            _c(dst.unsafe_ptr()), len(dst), _c(src.unsafe_ptr()), len(src)
        )
        if n != len(dst):
            raise Error("ZSTD_decompress failed")
