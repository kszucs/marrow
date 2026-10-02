# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Snappy` against libsnappy, on the page bodies Parquet actually feeds it.

Three corpora, each the PLAIN encoding of a column that stays PLAIN in a real
file -- high cardinality, so no dictionary:

- `strings`: BYTE_ARRAY URLs, a `u32` length then the bytes -- short literals
  and short copies, the tag loop at its busiest;
- `ints`: int64 microsecond timestamps with random gaps -- the high bytes
  repeat every 8, so a 2-3 byte literal and an offset-8 copy per value;
- `floats`: float64 in [0, 1) -- near-incompressible, the compressor's skip
  heuristic and the decoder's long literals.

Each at 64 KiB (one fragment, a dictionary or small page) and 1 MiB (the
default data page of pyarrow, arrow-rs and marrow). Throughput counts one
element per uncompressed byte, both directions -- the competition report
reads `n` off an elements throughput only -- so GElems/s reads as GB/s.

Both sides get the same contract, so the comparison is of the codec alone:
libsnappy's entry points are resolved once (`LibSnappy`) and called into a
buffer allocated once, and `Snappy` compresses into a reused `List` and
decompresses into an exact-size span (`decompress_into`). `decompress2` is two
pages of a column chunk: libsnappy decodes them one call after the other,
`Snappy.decompress_pair_into` in one interleaved loop.

Run with:
    pixi run -e dev pytest marrow/utils/tests/bench_snappy.mojo --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ..byteorder import LittleEndian
from ..compression import Codecs, CompressionLibs
from ..snappy import Snappy
from ..testing import Benchmark, Rng


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------


def _strings(n: Int) -> List[UInt8]:
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


def _ints(n: Int) -> List[UInt8]:
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
    """`n` bytes of the `strings`, `ints` or `floats` corpus."""
    if kind == "strings":
        return _strings(n)
    elif kind == "ints":
        return _ints(n)
    else:
        return _floats(n)


# ---------------------------------------------------------------------------
# libsnappy, resolved once
# ---------------------------------------------------------------------------


comptime _SnappyFn = def(
    Pointer[UInt8, MutUntrackedOrigin],
    Int,
    Pointer[UInt8, MutUntrackedOrigin],
    Pointer[UInt, MutUntrackedOrigin],
) thin abi("C") -> Int32
"""`snappy_compress` and `snappy_uncompress` share one C signature."""


@always_inline
def _c[T: AnyType](p: Pointer[T, _]) -> Pointer[T, MutUntrackedOrigin]:
    """`p` for the C ABI, which tracks no origins."""
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(p))


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


# ---------------------------------------------------------------------------
# benchmark bodies
# ---------------------------------------------------------------------------


def _bench_compress(mut b: Benchmark, lib: String, kind: String, n: Int) raises:
    var data = corpus(kind, n)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    if lib == "libsnappy":
        var snappy = LibSnappy()
        var dst = List[UInt8](length=Snappy.max_compressed_length(n), fill=0)

        @always_inline
        def call_lib() raises {mut snappy, mut dst, imm}:
            keep(snappy.compress(Span(data), dst))

        b.iter(call_lib)
        keep(snappy)
        keep(dst)
    else:
        var dst = List[UInt8](capacity=Snappy.max_compressed_length(n))

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            dst.clear()
            Snappy.compress(Span(data), dst)
            keep(len(dst))

        b.iter(call_mojo)
        keep(dst)
    keep(data)


def _bench_decompress(
    mut b: Benchmark, lib: String, kind: String, n: Int
) raises:
    var data = corpus(kind, n)
    var libs = CompressionLibs()
    var comp = libs.snappy_compress(Span(data))
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](length=n, fill=0)
    if lib == "libsnappy":
        var snappy = LibSnappy()

        @always_inline
        def call_lib() raises {mut snappy, mut dst, imm}:
            snappy.decompress(Span(comp), Span(dst))
            keep(dst[n - 1])

        b.iter(call_lib)
        keep(snappy)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            Snappy.decompress_into(Span(comp), Span(dst))
            keep(dst[n - 1])

        b.iter(call_mojo)
    if dst != data:
        raise Error(t"{lib} decoded {kind} wrong")
    keep(comp)
    keep(data)
    keep(dst)


def _bench_decompress2(
    mut b: Benchmark, lib: String, kind: String, n: Int
) raises:
    """Two `n`-byte pages, consecutive in the corpus, so they differ."""
    var data = corpus(kind, 2 * n)
    var page_a = List[UInt8](Span(data)[:n])
    var page_b = List[UInt8](Span(data)[n:])
    var libs = CompressionLibs()
    var comp_a = libs.snappy_compress(Span(page_a))
    var comp_b = libs.snappy_compress(Span(page_b))
    b.throughput(BenchMetric.elements, 2 * n)
    b.extra_info("lib", lib)
    var dst_a = List[UInt8](length=n, fill=0)
    var dst_b = List[UInt8](length=n, fill=0)
    if lib == "libsnappy":
        var snappy = LibSnappy()

        @always_inline
        def call_lib() raises {mut snappy, mut dst_a, mut dst_b, imm}:
            snappy.decompress(Span(comp_a), Span(dst_a))
            snappy.decompress(Span(comp_b), Span(dst_b))
            keep(dst_a[n - 1])
            keep(dst_b[n - 1])

        b.iter(call_lib)
        keep(snappy)
    else:

        @always_inline
        def call_mojo() raises {mut dst_a, mut dst_b, imm}:
            Snappy.decompress_pair_into(
                Span(comp_a), Span(dst_a), Span(comp_b), Span(dst_b)
            )
            keep(dst_a[n - 1])
            keep(dst_b[n - 1])

        b.iter(call_mojo)
    if dst_a != page_a or dst_b != page_b:
        raise Error(t"{lib} decoded {kind} pair wrong")
    keep(comp_a)
    keep(comp_b)
    keep(dst_a)
    keep(dst_b)


comptime _64K = 1 << 16
comptime _1M = 1 << 20


def bench_libsnappy_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "strings", _64K)


def bench_mojo_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", _64K)


def bench_libsnappy_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "strings", _1M)


def bench_mojo_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", _1M)


def bench_libsnappy_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "ints", _64K)


def bench_mojo_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", _64K)


def bench_libsnappy_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "ints", _1M)


def bench_mojo_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", _1M)


def bench_libsnappy_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "floats", _64K)


def bench_mojo_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", _64K)


def bench_libsnappy_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libsnappy", "floats", _1M)


def bench_mojo_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", _1M)


def bench_libsnappy_decompress_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "strings", _64K)


def bench_mojo_decompress_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", _64K)


def bench_libsnappy_decompress_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "strings", _1M)


def bench_mojo_decompress_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", _1M)


def bench_libsnappy_decompress_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "ints", _64K)


def bench_mojo_decompress_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", _64K)


def bench_libsnappy_decompress_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "ints", _1M)


def bench_mojo_decompress_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", _1M)


def bench_libsnappy_decompress_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "floats", _64K)


def bench_mojo_decompress_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", _64K)


def bench_libsnappy_decompress_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libsnappy", "floats", _1M)


def bench_mojo_decompress_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", _1M)


def bench_libsnappy_decompress2_strings_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "strings", _64K)


def bench_mojo_decompress2_strings_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "strings", _64K)


def bench_libsnappy_decompress2_strings_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "strings", _1M)


def bench_mojo_decompress2_strings_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "strings", _1M)


def bench_libsnappy_decompress2_ints_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "ints", _64K)


def bench_mojo_decompress2_ints_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "ints", _64K)


def bench_libsnappy_decompress2_ints_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "ints", _1M)


def bench_mojo_decompress2_ints_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "ints", _1M)


def bench_libsnappy_decompress2_floats_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "floats", _64K)


def bench_mojo_decompress2_floats_64k(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "floats", _64K)


def bench_libsnappy_decompress2_floats_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "libsnappy", "floats", _1M)


def bench_mojo_decompress2_floats_1m(mut b: Benchmark) raises:
    _bench_decompress2(b, "mojo", "floats", _1M)
