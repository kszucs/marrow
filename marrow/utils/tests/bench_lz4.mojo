# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Lz4` against liblz4, on the Parquet-shaped corpora `bench_snappy.mojo`
uses (`codec_data.corpus`), at 64 KiB and 1 MiB.

Both sides get the same contract: liblz4's entry points are resolved once
(`LibLz4`) and called into a buffer allocated once; `Lz4` compresses into a
reused `List` and decompresses into an exact-size span. Both decode the same
liblz4-compressed block. Throughput counts one element per uncompressed byte.

Run with:
    pixi run -e dev pytest marrow/utils/tests/bench_lz4.mojo --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ..lz4 import Lz4
from ..testing import Benchmark
from .codec_data import LibLz4, corpus


def _bench_compress(mut b: Benchmark, lib: String, kind: String, n: Int) raises:
    var data = corpus(kind, n)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    if lib == "liblz4":
        var lz4 = LibLz4()
        var dst = List[UInt8](capacity=Lz4.max_block_length(n))

        @always_inline
        def call_lib() raises {mut dst, imm}:
            dst.clear()
            keep(lz4.compress_into(Span(data), dst))

        b.iter(call_lib)
        keep(lz4)
        keep(dst)
    else:
        var dst = List[UInt8](capacity=Lz4.max_block_length(n))

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            dst.clear()
            Lz4.compress_block(Span(data), dst)
            keep(len(dst))

        b.iter(call_mojo)
        keep(dst)
    keep(data)


def _bench_decompress(
    mut b: Benchmark, lib: String, kind: String, n: Int
) raises:
    var data = corpus(kind, n)
    var comp = LibLz4().compress(Span(data))
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](length=n, fill=0)
    if lib == "liblz4":
        var lz4 = LibLz4()

        @always_inline
        def call_lib() raises {mut dst, imm}:
            lz4.decompress(Span(comp), Span(dst))
            keep(dst[n - 1])

        b.iter(call_lib)
        keep(lz4)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            Lz4.decompress_block_into(Span(comp), Span(dst))
            keep(dst[n - 1])

        b.iter(call_mojo)
    if dst != data:
        raise Error(t"{lib} decoded {kind} wrong")
    keep(comp)
    keep(data)
    keep(dst)


# --- compress ---------------------------------------------------------------


def bench_liblz4_lz4_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "strings", 1 << 16)


def bench_mojo_lz4_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 16)


def bench_liblz4_lz4_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "strings", 1 << 20)


def bench_mojo_lz4_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 20)


def bench_liblz4_lz4_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "ints", 1 << 16)


def bench_mojo_lz4_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 16)


def bench_liblz4_lz4_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "ints", 1 << 20)


def bench_mojo_lz4_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 20)


def bench_liblz4_lz4_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "floats", 1 << 16)


def bench_mojo_lz4_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 16)


def bench_liblz4_lz4_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "liblz4", "floats", 1 << 20)


def bench_mojo_lz4_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 20)


# --- decompress -------------------------------------------------------------


def bench_liblz4_lz4_decompress_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "strings", 1 << 16)


def bench_mojo_lz4_decompress_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16)


def bench_liblz4_lz4_decompress_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "strings", 1 << 20)


def bench_mojo_lz4_decompress_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20)


def bench_liblz4_lz4_decompress_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "ints", 1 << 16)


def bench_mojo_lz4_decompress_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16)


def bench_liblz4_lz4_decompress_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "ints", 1 << 20)


def bench_mojo_lz4_decompress_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20)


def bench_liblz4_lz4_decompress_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "floats", 1 << 16)


def bench_mojo_lz4_decompress_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16)


def bench_liblz4_lz4_decompress_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "liblz4", "floats", 1 << 20)


def bench_mojo_lz4_decompress_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20)
