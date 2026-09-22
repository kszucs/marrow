"""Benchmarks for packing bools into a Bitmap.

Exercises `BitmapView.store` at widths 8, 32 and 64, over 1k–100M bits, with
throughput reported in elements/second.

One file per unit, and the boundaries here are measured rather than tidy: see
`bench_bitmap.mojo` for why the bitmap benchmarks are spread out at all, and
`bench_bitmap_filter.mojo` for why the two filters are not allowed to share a
unit with each other.

Run with: pixi run pytest marrow/tests/bench_bitmap_pack.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..buffers import Bitmap
from ..utils.testing import Benchmark


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=8
# ---------------------------------------------------------------------------


def _bench_pack_bools[W: Int](mut b: Benchmark, size: Int) raises:
    var bm = Bitmap.alloc_zeroed(size)
    var bv = bm.view()
    # Alternating True/False, built rather than spelled out: the width-8, -32
    # and -64 bodies were identical apart from the literal's length. Built once
    # here, exactly like the literals it replaces — outside `b.iter`, so nothing
    # about the measurement changes.
    var pattern = SIMD[DType.bool, W](fill=False)
    for i in range(0, W, 2):
        pattern[i] = True
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {imm}:
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load_bytes[DType.uint8](0))

    b.iter(call)


def bench_pack_bools_w8_1k(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 1_000)


def bench_pack_bools_w8_10k(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 10_000)


def bench_pack_bools_w8_100k(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 100_000)


def bench_pack_bools_w8_1m(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 1_000_000)


def bench_pack_bools_w8_10m(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 10_000_000)


def bench_pack_bools_w8_100m(mut b: Benchmark) raises:
    _bench_pack_bools[8](b, 100_000_000)


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=32
# ---------------------------------------------------------------------------


def bench_pack_bools_w32_1k(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 1_000)


def bench_pack_bools_w32_10k(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 10_000)


def bench_pack_bools_w32_100k(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 100_000)


def bench_pack_bools_w32_1m(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 1_000_000)


def bench_pack_bools_w32_10m(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 10_000_000)


def bench_pack_bools_w32_100m(mut b: Benchmark) raises:
    _bench_pack_bools[32](b, 100_000_000)


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=64
# ---------------------------------------------------------------------------


def bench_pack_bools_w64_1k(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 1_000)


def bench_pack_bools_w64_10k(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 10_000)


def bench_pack_bools_w64_100k(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 100_000)


def bench_pack_bools_w64_1m(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 1_000_000)


def bench_pack_bools_w64_10m(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 10_000_000)


def bench_pack_bools_w64_100m(mut b: Benchmark) raises:
    _bench_pack_bools[64](b, 100_000_000)
