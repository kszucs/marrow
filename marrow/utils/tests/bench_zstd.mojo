# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Zstd` against libzstd, on the Parquet-shaped corpora the Snappy and LZ4
benchmarks use, at 64 KiB and 1 MiB: compression against level 1, Arrow's
default, and decompression of frames libzstd wrote at levels 1 and 9.

Both sides decode the same frame into an exact-size span, and compress into
a buffer reused across calls; libzstd's entry points are resolved once
(`LibZstd`). Throughput counts one element per uncompressed byte.

Run with:
    pixi run -e dev pytest marrow/utils/tests/bench_zstd.mojo --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ..testing import Benchmark
from ..zstd import Zstd
from .codec_data import LibZstd, corpus


def _bench_compress(mut b: Benchmark, lib: String, kind: String, n: Int) raises:
    var data = corpus(kind, n)
    var zstd = LibZstd()
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](capacity=Zstd.max_compressed_length(n))
    if lib == "libzstd":

        @always_inline
        def call_lib() raises {mut dst, imm}:
            dst.clear()
            keep(zstd.compress_into(Span(data), dst, 1))

        b.iter(call_lib)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            dst.clear()
            Zstd.compress(Span(data), dst)
            keep(len(dst))

        b.iter(call_mojo)
    keep(zstd)
    keep(data)
    keep(dst)


def _bench_decompress(
    mut b: Benchmark, lib: String, kind: String, n: Int, level: Int
) raises:
    var data = corpus(kind, n)
    var zstd = LibZstd()
    var frame = zstd.compress(data, level)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", lib)
    var dst = List[UInt8](length=n, fill=0)
    if lib == "libzstd":

        @always_inline
        def call_lib() raises {mut dst, imm}:
            zstd.decompress(Span(frame), Span(dst))
            keep(dst[n - 1])

        b.iter(call_lib)
    else:

        @always_inline
        def call_mojo() raises {mut dst, imm}:
            Zstd.decompress_into(Span(frame), Span(dst))
            keep(dst[n - 1])

        b.iter(call_mojo)
    if dst != data:
        raise Error(t"{lib} decoded {kind} wrong")
    keep(zstd)
    keep(frame)
    keep(data)
    keep(dst)


# --- compress, level 1 -------------------------------------------------------


def bench_libzstd_zstd_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "strings", 1 << 16)


def bench_mojo_zstd_compress_strings_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 16)


def bench_libzstd_zstd_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "strings", 1 << 20)


def bench_mojo_zstd_compress_strings_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "strings", 1 << 20)


def bench_libzstd_zstd_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "ints", 1 << 16)


def bench_mojo_zstd_compress_ints_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 16)


def bench_libzstd_zstd_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "ints", 1 << 20)


def bench_mojo_zstd_compress_ints_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "ints", 1 << 20)


def bench_libzstd_zstd_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "floats", 1 << 16)


def bench_mojo_zstd_compress_floats_64k(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 16)


def bench_libzstd_zstd_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "libzstd", "floats", 1 << 20)


def bench_mojo_zstd_compress_floats_1m(mut b: Benchmark) raises:
    _bench_compress(b, "mojo", "floats", 1 << 20)


# --- decompress, level 1 ----------------------------------------------------


def bench_libzstd_zstd_decompress_l1_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "strings", 1 << 16, 1)


def bench_mojo_zstd_decompress_l1_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16, 1)


def bench_libzstd_zstd_decompress_l1_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "strings", 1 << 20, 1)


def bench_mojo_zstd_decompress_l1_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20, 1)


def bench_libzstd_zstd_decompress_l1_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "ints", 1 << 16, 1)


def bench_mojo_zstd_decompress_l1_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16, 1)


def bench_libzstd_zstd_decompress_l1_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "ints", 1 << 20, 1)


def bench_mojo_zstd_decompress_l1_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20, 1)


def bench_libzstd_zstd_decompress_l1_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "floats", 1 << 16, 1)


def bench_mojo_zstd_decompress_l1_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16, 1)


def bench_libzstd_zstd_decompress_l1_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "floats", 1 << 20, 1)


def bench_mojo_zstd_decompress_l1_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20, 1)


# --- decompress, level 9 ----------------------------------------------------


def bench_libzstd_zstd_decompress_l9_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "strings", 1 << 16, 9)


def bench_mojo_zstd_decompress_l9_strings_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 16, 9)


def bench_libzstd_zstd_decompress_l9_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "strings", 1 << 20, 9)


def bench_mojo_zstd_decompress_l9_strings_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "strings", 1 << 20, 9)


def bench_libzstd_zstd_decompress_l9_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "ints", 1 << 16, 9)


def bench_mojo_zstd_decompress_l9_ints_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 16, 9)


def bench_libzstd_zstd_decompress_l9_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "ints", 1 << 20, 9)


def bench_mojo_zstd_decompress_l9_ints_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "ints", 1 << 20, 9)


def bench_libzstd_zstd_decompress_l9_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "floats", 1 << 16, 9)


def bench_mojo_zstd_decompress_l9_floats_64k(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 16, 9)


def bench_libzstd_zstd_decompress_l9_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "libzstd", "floats", 1 << 20, 9)


def bench_mojo_zstd_decompress_l9_floats_1m(mut b: Benchmark) raises:
    _bench_decompress(b, "mojo", "floats", 1 << 20, 9)
