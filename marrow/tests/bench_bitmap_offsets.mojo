"""What alignment and offset cost the Bitmap SIMD operators.

The same `invert` and `and` loops as `bench_bitmap_logic.mojo`, measured where
the operands do not start on a 64-byte boundary:

  - a cache-line-aligned offset (lead_bytes=0) against a non-aligned one
  - two operands at the same sub-byte offset (pure SIMD, no shift)
  - two operands at different sub-byte offsets (one-sided shift-combine)

Sizes: 1k–100M bits.  Throughput reported in bits/second.  See
`bench_bitmap.mojo` for why these are four files.

Run with: pixi run pytest marrow/tests/bench_bitmap_offsets.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..utils.testing import Benchmark
from .bitmap_fixtures import make_alternating, make_half_set


# ---------------------------------------------------------------------------
# Cache-alignment: invert with 64-byte-aligned offset (lead_bytes=0)
# ---------------------------------------------------------------------------


def _bench_invert_cache_aligned(mut b: Benchmark, size: Int) raises:
    var bitmap = make_alternating(size + 2048).slice(128 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(~bitmap))

    b.iter(call)
    keep(len(bitmap))


def bench_invert_cache_aligned_1k(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 1_000)


def bench_invert_cache_aligned_10k(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 10_000)


def bench_invert_cache_aligned_100k(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 100_000)


def bench_invert_cache_aligned_1m(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 1_000_000)


def bench_invert_cache_aligned_10m(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 10_000_000)


def bench_invert_cache_aligned_100m(mut b: Benchmark) raises:
    _bench_invert_cache_aligned(b, 100_000_000)


# ---------------------------------------------------------------------------
# Cache-alignment: invert with non-aligned offset (lead_bytes=32)
# ---------------------------------------------------------------------------


def _bench_invert_cache_unaligned(mut b: Benchmark, size: Int) raises:
    var bitmap = make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(~bitmap))

    b.iter(call)
    keep(len(bitmap))


def bench_invert_cache_unaligned_1k(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 1_000)


def bench_invert_cache_unaligned_10k(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 10_000)


def bench_invert_cache_unaligned_100k(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 100_000)


def bench_invert_cache_unaligned_1m(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 1_000_000)


def bench_invert_cache_unaligned_10m(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 10_000_000)


def bench_invert_cache_unaligned_100m(mut b: Benchmark) raises:
    _bench_invert_cache_unaligned(b, 100_000_000)


# ---------------------------------------------------------------------------
# Cache-alignment: AND of two bitmaps both at non-aligned offset (lead_bytes=32)
# ---------------------------------------------------------------------------


def _bench_and_cache_unaligned(mut b: Benchmark, size: Int) raises:
    var lhs = make_half_set(size + 2048).slice(96 << 3, size)
    var rhs = make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(lhs & rhs))

    b.iter(call)
    keep(len(lhs))
    keep(len(rhs))


def bench_and_cache_unaligned_1k(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 1_000)


def bench_and_cache_unaligned_10k(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 10_000)


def bench_and_cache_unaligned_100k(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 100_000)


def bench_and_cache_unaligned_1m(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 1_000_000)


def bench_and_cache_unaligned_10m(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 10_000_000)


def bench_and_cache_unaligned_100m(mut b: Benchmark) raises:
    _bench_and_cache_unaligned(b, 100_000_000)


# ---------------------------------------------------------------------------
# Sub-byte alignment: same offset (pure SIMD, no shift)
# ---------------------------------------------------------------------------


def _bench_and_same_offset(mut b: Benchmark, size: Int) raises:
    var lhs = make_half_set(size).slice(3, size - 8)
    var rhs = make_alternating(size).slice(3, size - 8)
    b.throughput(BenchMetric.elements, size - 8)

    @always_inline
    def call() raises {imm}:
        keep(len(lhs & rhs))

    b.iter(call)
    keep(len(lhs))
    keep(len(rhs))


def bench_and_same_offset_1k(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 1_000)


def bench_and_same_offset_10k(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 10_000)


def bench_and_same_offset_100k(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 100_000)


def bench_and_same_offset_1m(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 1_000_000)


def bench_and_same_offset_10m(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 10_000_000)


def bench_and_same_offset_100m(mut b: Benchmark) raises:
    _bench_and_same_offset(b, 100_000_000)


# ---------------------------------------------------------------------------
# Sub-byte alignment: different offsets (one-sided shift-combine)
# ---------------------------------------------------------------------------


def _bench_and_diff_offset(mut b: Benchmark, size: Int) raises:
    var lhs = make_half_set(size).slice(3, size - 8)
    var rhs = make_alternating(size).slice(5, size - 8)
    b.throughput(BenchMetric.elements, size - 8)

    @always_inline
    def call() raises {imm}:
        keep(len(lhs & rhs))

    b.iter(call)
    keep(len(lhs))
    keep(len(rhs))


def bench_and_diff_offset_1k(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 1_000)


def bench_and_diff_offset_10k(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 10_000)


def bench_and_diff_offset_100k(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 100_000)


def bench_and_diff_offset_1m(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 1_000_000)


def bench_and_diff_offset_10m(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 10_000_000)


def bench_and_diff_offset_100m(mut b: Benchmark) raises:
    _bench_and_diff_offset(b, 100_000_000)
