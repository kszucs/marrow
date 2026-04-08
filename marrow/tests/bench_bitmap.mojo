"""Benchmarks for Bitmap SIMD operations.

Exercises the hot paths in Bitmap:
  - count_set_bits   — SIMD popcount loop
  - bitmap_and       — SIMD & loop (aligned path)
  - bitmap_or        — SIMD | loop (aligned path)
  - bitmap_invert    — SIMD ~ loop (aligned path)
  - set_range(True)  — bulk-set via memset (BitmapBuilder)

Sizes: 1k–10M bits.  Throughput reported in bits/second.

Run with: pixi run bench_bitmap

NOTE: bench_with_input is used instead of bench_function to avoid a Mojo
codegen bug (~25.7) where registering multiple size-parameterized
instantiations of the same function crashes at runtime.
bench_with_input[Int, bench_fn] passes size as a runtime Int so only ONE
template instantiation of bench_fn is created; the same function pointer is
called multiple times with different runtime inputs.
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)

from marrow.buffers import Bitmap


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
# Benchmark functions
# ---------------------------------------------------------------------------


@parameter
def bench_count_set_bits(mut b: Bencher, size: Int) raises:
    var bm = _make_alternating(size)
    var bm_view = bm.view()

    @always_inline
    @parameter
    def call_fn():
        var n = bm_view.count_set_bits()
        keep(n)

    b.iter[call_fn]()


@parameter
def bench_count_set_bits_aligned(mut b: Bencher, size: Int) raises:
    """Count_set_bits with byte_offset=128 (64-byte aligned, lead_bytes=0)."""
    var bm = _make_alternating(size + 2048).slice(128 << 3, size)

    @always_inline
    @parameter
    def call_fn():
        var n = bm.count_set_bits()
        keep(n)

    b.iter[call_fn]()


@parameter
def bench_count_set_bits_unaligned(mut b: Bencher, size: Int) raises:
    """Count_set_bits with byte_offset=96 (NOT 64-byte aligned, lead_bytes=32).
    """
    var bm = _make_alternating(size + 2048).slice(96 << 3, size)

    @always_inline
    @parameter
    def call_fn():
        var n = bm.count_set_bits()
        keep(n)

    b.iter[call_fn]()


@parameter
def bench_and(mut b: Bencher, size: Int) raises:
    var lhs = _make_half_set(size)
    var rhs = _make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()

    @always_inline
    @parameter
    def call_fn() raises:
        var r = lhs_view & rhs_view
        keep(len(r))

    b.iter[call_fn]()
    keep(len(lhs))
    keep(len(rhs))


@parameter
def bench_or(mut b: Bencher, size: Int) raises:
    var lhs = _make_half_set(size)
    var rhs = _make_alternating(size)
    var lhs_view = lhs.view()
    var rhs_view = rhs.view()

    @always_inline
    @parameter
    def call_fn() raises:
        var r = lhs_view | rhs_view
        keep(len(r))

    b.iter[call_fn]()
    keep(len(lhs))
    keep(len(rhs))


@parameter
def bench_invert(mut b: Bencher, size: Int) raises:
    var bitmap = _make_alternating(size)
    var bitmap_view = bitmap.view()

    @always_inline
    @parameter
    def call_fn() raises:
        var r = ~bitmap_view
        keep(len(r))

    b.iter[call_fn]()
    keep(len(bitmap))


@parameter
def bench_set_range(mut b: Bencher, size: Int) raises:
    var builder = Bitmap.alloc_zeroed(size)

    @always_inline
    @parameter
    def call_fn():
        builder.set_range(0, size, True)
        keep(builder.view().load[DType.uint8](0))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Cache-alignment benchmarks: large byte offsets
# ---------------------------------------------------------------------------
#
# These benchmarks exercise the cache-alignment optimization in __invert__ and
# _binop when the source bitmap has a large byte offset into its buffer.
#
# aligned_cache: byte_offset = 128 (multiple of 64) → lead_bytes = 0, no change
# unaligned_cache: byte_offset = 96  (96 & 63 = 32)  → lead_bytes = 32
#
# Before the optimization both paths run the same code.  After, the unaligned
# case backs src up 32 bytes to a cache-line boundary; the aligned case is
# unchanged.  Compare the two to quantify the alignment benefit.


@parameter
def bench_invert_cache_aligned(mut b: Bencher, size: Int) raises:
    """Invert with a byte_offset that is already 64-byte aligned (lead_bytes=0).
    """
    # 128-byte = 1024-bit offset → byte_offset=128, 128 & 63 == 0, no lead bytes.
    var bitmap = _make_alternating(size + 2048).slice(128 << 3, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var r = ~bitmap
        keep(len(r))

    b.iter[call_fn]()
    keep(len(bitmap))


@parameter
def bench_invert_cache_unaligned(mut b: Bencher, size: Int) raises:
    """Invert with a byte_offset that is NOT 64-byte aligned (lead_bytes=32)."""
    # 96-byte = 768-bit offset → byte_offset=96, 96 & 63 == 32, lead_bytes=32.
    var bitmap = _make_alternating(size + 2048).slice(96 << 3, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var r = ~bitmap
        keep(len(r))

    b.iter[call_fn]()
    keep(len(bitmap))


@parameter
def bench_and_cache_unaligned(mut b: Bencher, size: Int) raises:
    """AND of two bitmaps both at byte_offset=96 (lead_bytes=32, same shift)."""
    var lhs = _make_half_set(size + 2048).slice(96 << 3, size)
    var rhs = _make_alternating(size + 2048).slice(96 << 3, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var r = lhs & rhs
        keep(len(r))

    b.iter[call_fn]()
    keep(len(lhs))
    keep(len(rhs))


# ---------------------------------------------------------------------------
# Non-aligned benchmarks: same-shift vs different-shift code paths
# ---------------------------------------------------------------------------
#
# Both bitmaps are sliced to force non-zero sub-byte offsets.
# same_offset: both at offset=3 → pure SIMD, no shift (~360 GElems/s at 10M)
# diff_offset: offsets 3 vs 5   → one-sided shift-combine  (~305 GElems/s at 10M)


@parameter
def bench_and_same_offset(mut b: Bencher, size: Int) raises:
    """AND of two bitmaps sharing the same non-zero sub-byte offset (pure SIMD, no shift).
    """
    var lhs = _make_half_set(size).slice(3, size - 8)
    var rhs = _make_alternating(size).slice(3, size - 8)

    @always_inline
    @parameter
    def call_fn() raises:
        var r = lhs & rhs
        keep(len(r))

    b.iter[call_fn]()
    keep(len(lhs))
    keep(len(rhs))


@parameter
def bench_and_diff_offset(mut b: Bencher, size: Int) raises:
    """AND of two bitmaps with different sub-byte offsets (one-sided shift-combine).
    """
    var lhs = _make_half_set(size).slice(3, size - 8)
    var rhs = _make_alternating(size).slice(5, size - 8)

    @always_inline
    @parameter
    def call_fn() raises:
        var r = lhs & rhs
        keep(len(r))

    b.iter[call_fn]()
    keep(len(lhs))
    keep(len(rhs))


# ---------------------------------------------------------------------------
# pack_bools — BitmapView.store (exercises _pack_bools)
# ---------------------------------------------------------------------------
#
# Directly measures the throughput of packing SIMD[bool, 8] into a bitmap
# via BitmapView.store.  This is the hot path in compare, filter-mask, and
# any kernel that produces boolean output.


@parameter
def bench_pack_bools_w8(mut b: Bencher, size: Int) raises:
    """Pack bools via BitmapView.store[8] (one byte per call)."""
    alias W = 8
    var bm = Bitmap.alloc_zeroed(size)
    var bv = bm.view()
    var pattern = SIMD[DType.bool, W](
        True, False, True, False, True, False, True, False
    )

    @always_inline
    @parameter
    def call_fn():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call_fn]()


@parameter
def bench_pack_bools_w32(mut b: Bencher, size: Int) raises:
    """Pack bools via BitmapView.store[32] (four bytes per call)."""
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

    @always_inline
    @parameter
    def call_fn():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call_fn]()


@parameter
def bench_pack_bools_w64(mut b: Bencher, size: Int) raises:
    """Pack bools via BitmapView.store[64] (eight bytes per call)."""
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

    @always_inline
    @parameter
    def call_fn():
        for i in range(0, size - W + 1, W):
            bv.store[W](i, pattern)
        keep(bv.load[DType.uint8](0))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    comptime sizes = (
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    )
    comptime labels = ("1k", "10k", "100k", "1M", "10M", "100M")

    comptime for si in range(6):
        m.bench_with_input[Int, bench_count_set_bits](
            BenchId("count_set_bits", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_count_set_bits_aligned](
            BenchId("count_set_bits_aligned", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_count_set_bits_unaligned](
            BenchId("count_set_bits_unaligned", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_and](
            BenchId("bitmap_and", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_or](
            BenchId("bitmap_or", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_invert](
            BenchId("bitmap_invert", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_set_range](
            BenchId("set_range", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_and_same_offset](
            BenchId("and_same_offset", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_and_diff_offset](
            BenchId("and_diff_offset", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_invert_cache_aligned](
            BenchId("invert_cache_aligned", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_invert_cache_unaligned](
            BenchId("invert_cache_unaligned", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_and_cache_unaligned](
            BenchId("and_cache_unaligned", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_pack_bools_w8](
            BenchId("pack_bools_w8", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_pack_bools_w32](
            BenchId("pack_bools_w32", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(6):
        m.bench_with_input[Int, bench_pack_bools_w64](
            BenchId("pack_bools_w64", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()
