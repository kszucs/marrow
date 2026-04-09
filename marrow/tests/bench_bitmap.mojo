"""Benchmarks for Bitmap SIMD operations.

Exercises the hot paths in Bitmap:
  - count_set_bits   — SIMD popcount loop
  - bitmap_and       — SIMD & loop (aligned path)
  - bitmap_or        — SIMD | loop (aligned path)
  - bitmap_invert    — SIMD ~ loop (aligned path)
  - set_range(True)  — bulk-set via memset (BitmapBuilder)

Sizes: 1k–100M bits.  Throughput reported in bits/second.

Run with: pixi run pytest marrow/tests/bench_bitmap.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.buffers import Bitmap
from marrow.testing import BenchSuite, Benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_alternating(size: Int) -> Bitmap[mut=False]:
    """Bitmap with alternating 0/1 bits (worst-case for popcount branching)."""
    var b = Bitmap.alloc_zeroed(size)
    var i = 0
    while i < size:
        b.set(i)
        i += 2
    return b.to_immutable()


def _make_half_set(size: Int) -> Bitmap[mut=False]:
    """Bitmap with the first half of bits set."""
    var b = Bitmap.alloc_zeroed(size)
    b.set_range(0, size // 2, True)
    return b.to_immutable()


# ---------------------------------------------------------------------------
# count_set_bits
# ---------------------------------------------------------------------------


def _bench_count_set_bits(mut b: Benchmark, size: Int) raises:
    var bm = _make_alternating(size)
    var bm_view = bm.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        keep(bm_view.count_set_bits())

    b.iter[call]()


def bench_count_set_bits_1k(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 1_000)


def bench_count_set_bits_10k(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 10_000)


def bench_count_set_bits_100k(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 100_000)


def bench_count_set_bits_1m(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 1_000_000)


def bench_count_set_bits_10m(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 10_000_000)


def bench_count_set_bits_100m(mut b: Benchmark) raises:
    _bench_count_set_bits(b, 100_000_000)


# ---------------------------------------------------------------------------
# count_set_bits — cache-line-aligned offset (byte_offset=128, lead_bytes=0)
# ---------------------------------------------------------------------------


def _bench_count_set_bits_aligned(mut b: Benchmark, size: Int) raises:
    var bm = _make_alternating(size + 2048).slice(128 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        keep(bm.count_set_bits())

    b.iter[call]()


def bench_count_set_bits_aligned_1k(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 1_000)


def bench_count_set_bits_aligned_10k(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 10_000)


def bench_count_set_bits_aligned_100k(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 100_000)


def bench_count_set_bits_aligned_1m(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 1_000_000)


def bench_count_set_bits_aligned_10m(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 10_000_000)


def bench_count_set_bits_aligned_100m(mut b: Benchmark) raises:
    _bench_count_set_bits_aligned(b, 100_000_000)


# ---------------------------------------------------------------------------
# count_set_bits — non-aligned offset (byte_offset=96, lead_bytes=32)
# ---------------------------------------------------------------------------


def _bench_count_set_bits_unaligned(mut b: Benchmark, size: Int) raises:
    var bm = _make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        keep(bm.count_set_bits())

    b.iter[call]()


def bench_count_set_bits_unaligned_1k(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 1_000)


def bench_count_set_bits_unaligned_10k(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 10_000)


def bench_count_set_bits_unaligned_100k(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 100_000)


def bench_count_set_bits_unaligned_1m(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 1_000_000)


def bench_count_set_bits_unaligned_10m(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 10_000_000)


def bench_count_set_bits_unaligned_100m(mut b: Benchmark) raises:
    _bench_count_set_bits_unaligned(b, 100_000_000)


# ---------------------------------------------------------------------------
# bitmap_and
# ---------------------------------------------------------------------------


def _bench_and(mut b: Benchmark, size: Int) raises:
    var lhs = _make_half_set(size)
    var rhs = _make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(lhs_view & rhs_view))

    b.iter[call]()
    keep(len(lhs))
    keep(len(rhs))


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
    var lhs = _make_half_set(size)
    var rhs = _make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(lhs_view | rhs_view))

    b.iter[call]()
    keep(len(lhs))
    keep(len(rhs))


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
    var bitmap = _make_alternating(size)
    var bitmap_view = bitmap.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(~bitmap_view))

    b.iter[call]()
    keep(len(bitmap))


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


# ---------------------------------------------------------------------------
# set_range
# ---------------------------------------------------------------------------


def _bench_set_range(mut b: Benchmark, size: Int) raises:
    var builder = Bitmap.alloc_zeroed(size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        builder.set_range(0, size, True)
        keep(builder.view().load[DType.uint8](0))

    b.iter[call]()


def bench_set_range_1k(mut b: Benchmark) raises:
    _bench_set_range(b, 1_000)


def bench_set_range_10k(mut b: Benchmark) raises:
    _bench_set_range(b, 10_000)


def bench_set_range_100k(mut b: Benchmark) raises:
    _bench_set_range(b, 100_000)


def bench_set_range_1m(mut b: Benchmark) raises:
    _bench_set_range(b, 1_000_000)


def bench_set_range_10m(mut b: Benchmark) raises:
    _bench_set_range(b, 10_000_000)


def bench_set_range_100m(mut b: Benchmark) raises:
    _bench_set_range(b, 100_000_000)


# ---------------------------------------------------------------------------
# Cache-alignment: invert with 64-byte-aligned offset (lead_bytes=0)
# ---------------------------------------------------------------------------


def _bench_invert_cache_aligned(mut b: Benchmark, size: Int) raises:
    var bitmap = _make_alternating(size + 2048).slice(128 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(~bitmap))

    b.iter[call]()
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
    var bitmap = _make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(~bitmap))

    b.iter[call]()
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
    var lhs = _make_half_set(size + 2048).slice(96 << 3, size)
    var rhs = _make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call() raises:
        keep(len(lhs & rhs))

    b.iter[call]()
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
    var lhs = _make_half_set(size).slice(3, size - 8)
    var rhs = _make_alternating(size).slice(3, size - 8)
    b.throughput(BenchMetric.elements, size - 8)

    @always_inline
    @parameter
    def call() raises:
        keep(len(lhs & rhs))

    b.iter[call]()
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
    var lhs = _make_half_set(size).slice(3, size - 8)
    var rhs = _make_alternating(size).slice(5, size - 8)
    b.throughput(BenchMetric.elements, size - 8)

    @always_inline
    @parameter
    def call() raises:
        keep(len(lhs & rhs))

    b.iter[call]()
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


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=8
# ---------------------------------------------------------------------------


def _bench_pack_bools_w8(mut b: Benchmark, size: Int) raises:
    alias W = 8
    var bm = Bitmap.alloc_zeroed(size)
    var bv = bm.view()
    var pattern = SIMD[DType.bool, W](
        True, False, True, False, True, False, True, False
    )
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call]()


def bench_pack_bools_w8_1k(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 1_000)


def bench_pack_bools_w8_10k(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 10_000)


def bench_pack_bools_w8_100k(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 100_000)


def bench_pack_bools_w8_1m(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 1_000_000)


def bench_pack_bools_w8_10m(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 10_000_000)


def bench_pack_bools_w8_100m(mut b: Benchmark) raises:
    _bench_pack_bools_w8(b, 100_000_000)


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=32
# ---------------------------------------------------------------------------


def _bench_pack_bools_w32(mut b: Benchmark, size: Int) raises:
    alias W = 32
    var bm = Bitmap.alloc_zeroed(size)
    var bv = bm.view()
    var pattern = SIMD[DType.bool, W](
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    )
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call]()


def bench_pack_bools_w32_1k(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 1_000)


def bench_pack_bools_w32_10k(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 10_000)


def bench_pack_bools_w32_100k(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 100_000)


def bench_pack_bools_w32_1m(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 1_000_000)


def bench_pack_bools_w32_10m(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 10_000_000)


def bench_pack_bools_w32_100m(mut b: Benchmark) raises:
    _bench_pack_bools_w32(b, 100_000_000)


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store width=64
# ---------------------------------------------------------------------------


def _bench_pack_bools_w64(mut b: Benchmark, size: Int) raises:
    alias W = 64
    var bm = Bitmap.alloc_zeroed(size)
    var bv = bm.view()
    var pattern = SIMD[DType.bool, W](
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    )
    b.throughput(BenchMetric.elements, size)

    @always_inline
    @parameter
    def call():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call]()


def bench_pack_bools_w64_1k(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 1_000)


def bench_pack_bools_w64_10k(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 10_000)


def bench_pack_bools_w64_100k(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 100_000)


def bench_pack_bools_w64_1m(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 1_000_000)


def bench_pack_bools_w64_10m(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 10_000_000)


def bench_pack_bools_w64_100m(mut b: Benchmark) raises:
    _bench_pack_bools_w64(b, 100_000_000)


def main() raises:
    BenchSuite.run[__functions_in_module()]()
