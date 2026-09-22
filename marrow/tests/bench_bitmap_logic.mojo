"""Benchmarks for the Bitmap SIMD boolean operators, on the aligned path.

Exercises:
  - bitmap_and    — SIMD & loop
  - bitmap_or     — SIMD | loop
  - bitmap_invert — SIMD ~ loop

Sizes: 1k–100M bits.  Throughput reported in bits/second.  What the same
operators cost at an unaligned offset is `bench_bitmap_offsets.mojo`; see
`bench_bitmap.mojo` for why these are four files.

Run with: pixi run pytest marrow/tests/bench_bitmap_logic.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..utils.testing import Benchmark
from .bitmap_fixtures import make_alternating, make_half_set


# ---------------------------------------------------------------------------
# bitmap_and
# ---------------------------------------------------------------------------


def _bench_and(mut b: Benchmark, size: Int) raises:
    var lhs = make_half_set(size)
    var rhs = make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(lhs_view & rhs_view))

    b.iter(call)
    keep(len(lhs))
    keep(len(rhs))
    keep(len(lhs_view))
    keep(len(rhs_view))


def bench_and_1k(mut b: Benchmark) raises:
    _bench_and(b, 1_000)


def bench_and_10k(mut b: Benchmark) raises:
    _bench_and(b, 10_000)


def bench_and_100k(mut b: Benchmark) raises:
    _bench_and(b, 100_000)


def bench_and_1m(mut b: Benchmark) raises:
    _bench_and(b, 1_000_000)


def bench_and_10m(mut b: Benchmark) raises:
    _bench_and(b, 10_000_000)


def bench_and_100m(mut b: Benchmark) raises:
    _bench_and(b, 100_000_000)


# ---------------------------------------------------------------------------
# bitmap_or
# ---------------------------------------------------------------------------


def _bench_or(mut b: Benchmark, size: Int) raises:
    var lhs = make_half_set(size)
    var rhs = make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(lhs_view | rhs_view))

    b.iter(call)
    keep(len(lhs))
    keep(len(rhs))
    keep(len(lhs_view))
    keep(len(rhs_view))


def bench_or_1k(mut b: Benchmark) raises:
    _bench_or(b, 1_000)


def bench_or_10k(mut b: Benchmark) raises:
    _bench_or(b, 10_000)


def bench_or_100k(mut b: Benchmark) raises:
    _bench_or(b, 100_000)


def bench_or_1m(mut b: Benchmark) raises:
    _bench_or(b, 1_000_000)


def bench_or_10m(mut b: Benchmark) raises:
    _bench_or(b, 10_000_000)


def bench_or_100m(mut b: Benchmark) raises:
    _bench_or(b, 100_000_000)


# ---------------------------------------------------------------------------
# bitmap_invert
# ---------------------------------------------------------------------------


def _bench_invert(mut b: Benchmark, size: Int) raises:
    var bitmap = make_alternating(size)
    var bitmap_view = bitmap.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(~bitmap_view))

    b.iter(call)
    keep(len(bitmap))
    keep(len(bitmap_view))


def bench_invert_1k(mut b: Benchmark) raises:
    _bench_invert(b, 1_000)


def bench_invert_10k(mut b: Benchmark) raises:
    _bench_invert(b, 10_000)


def bench_invert_100k(mut b: Benchmark) raises:
    _bench_invert(b, 100_000)


def bench_invert_1m(mut b: Benchmark) raises:
    _bench_invert(b, 1_000_000)


def bench_invert_10m(mut b: Benchmark) raises:
    _bench_invert(b, 10_000_000)


def bench_invert_100m(mut b: Benchmark) raises:
    _bench_invert(b, 100_000_000)
