# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Brotli` against libbrotli, on the Parquet-shaped corpora the other codec
benchmarks use, at 64 KiB and 1 MiB: compression at quality 1, and
decompression of streams libbrotli wrote at qualities 1, 5 and 11 -- 11 being
what the Parquet writer uses.

Both sides decode the same stream into an exact-size span, and compress into
a buffer reused across calls; libbrotli's entry points are resolved once
(`ResolvedBrotli`). Throughput counts one element per uncompressed byte.

Run with:
    pixi run -e dev pytest marrow/compress/tests/bench_brotli.mojo --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ...utils.testing import Benchmark
from ..brotli import Brotli
from .codec_data import ResolvedBrotli, corpus


def _bench_compress(mut b: Benchmark, lib: String, kind: String, n: Int) raises:
    var data = corpus(kind, n)
    var brotli = ResolvedBrotli()
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](capacity=2 * n + 1024)
    if lib == "libbrotli":

        @always_inline
        def call_lib() raises {mut dst, mut brotli, imm}:
            dst.clear()
            keep(brotli.compress_into(Span(data), dst, 1))

        b.iter(call_lib)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            dst.clear()
            Brotli.compress(Span(data), dst)
            keep(len(dst))

        b.iter(call_mojo)
    keep(brotli)
    keep(data)
    keep(dst)


def _bench_decompress(
    mut b: Benchmark, lib: String, kind: String, n: Int, quality: Int
) raises:
    var data = corpus(kind, n)
    var brotli = ResolvedBrotli()
    var stream = brotli.compress(data, quality)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](length=n, fill=0)
    if lib == "libbrotli":

        @always_inline
        def call_lib() raises {mut dst, mut brotli, imm}:
            brotli.decompress(Span(stream), Span(dst))
            keep(dst[n - 1])

        b.iter(call_lib)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            Brotli.decompress(Span(stream), Span(dst))
            keep(dst[n - 1])

        b.iter(call_mojo)
    if dst != data:
        raise Error(t"{lib} decoded {kind} wrong")
    keep(brotli)
    keep(stream)
    keep(data)
    keep(dst)


# --- compress, quality 1 -----------------------------------------------------


def bench_libbrotli_brotli_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "strings", 1 << 16)


def bench_mojo_brotli_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 16)


def bench_libbrotli_brotli_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "strings", 1 << 20)


def bench_mojo_brotli_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 20)


def bench_libbrotli_brotli_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "ints", 1 << 16)


def bench_mojo_brotli_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 16)


def bench_libbrotli_brotli_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "ints", 1 << 20)


def bench_mojo_brotli_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 20)


def bench_libbrotli_brotli_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "floats", 1 << 16)


def bench_mojo_brotli_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 16)


def bench_libbrotli_brotli_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libbrotli", "floats", 1 << 20)


def bench_mojo_brotli_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 20)


# --- decompress, quality 1 -------------------------------------------------


def bench_libbrotli_brotli_decompress_q1_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 16, 1)


def bench_mojo_brotli_decompress_q1_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16, 1)


def bench_libbrotli_brotli_decompress_q1_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 20, 1)


def bench_mojo_brotli_decompress_q1_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20, 1)


def bench_libbrotli_brotli_decompress_q1_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 16, 1)


def bench_mojo_brotli_decompress_q1_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16, 1)


def bench_libbrotli_brotli_decompress_q1_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 20, 1)


def bench_mojo_brotli_decompress_q1_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20, 1)


def bench_libbrotli_brotli_decompress_q1_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 16, 1)


def bench_mojo_brotli_decompress_q1_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16, 1)


def bench_libbrotli_brotli_decompress_q1_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 20, 1)


def bench_mojo_brotli_decompress_q1_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20, 1)


# --- decompress, quality 5 -------------------------------------------------


def bench_libbrotli_brotli_decompress_q5_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 16, 5)


def bench_mojo_brotli_decompress_q5_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16, 5)


def bench_libbrotli_brotli_decompress_q5_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 20, 5)


def bench_mojo_brotli_decompress_q5_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20, 5)


def bench_libbrotli_brotli_decompress_q5_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 16, 5)


def bench_mojo_brotli_decompress_q5_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16, 5)


def bench_libbrotli_brotli_decompress_q5_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 20, 5)


def bench_mojo_brotli_decompress_q5_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20, 5)


def bench_libbrotli_brotli_decompress_q5_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 16, 5)


def bench_mojo_brotli_decompress_q5_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16, 5)


def bench_libbrotli_brotli_decompress_q5_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 20, 5)


def bench_mojo_brotli_decompress_q5_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20, 5)


# --- decompress, quality 11 ------------------------------------------------


def bench_libbrotli_brotli_decompress_q11_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 16, 11)


def bench_mojo_brotli_decompress_q11_strings_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16, 11)


def bench_libbrotli_brotli_decompress_q11_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "strings", 1 << 20, 11)


def bench_mojo_brotli_decompress_q11_strings_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20, 11)


def bench_libbrotli_brotli_decompress_q11_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 16, 11)


def bench_mojo_brotli_decompress_q11_ints_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16, 11)


def bench_libbrotli_brotli_decompress_q11_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "ints", 1 << 20, 11)


def bench_mojo_brotli_decompress_q11_ints_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20, 11)


def bench_libbrotli_brotli_decompress_q11_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 16, 11)


def bench_mojo_brotli_decompress_q11_floats_64k(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16, 11)


def bench_libbrotli_brotli_decompress_q11_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "libbrotli", "floats", 1 << 20, 11)


def bench_mojo_brotli_decompress_q11_floats_1m(
    mut b: Benchmark,
) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20, 11)
