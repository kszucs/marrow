"""Benchmarks for Bitmap SIMD reads and bulk writes.

Exercises the hot paths in Bitmap:
  - load[W]          — the bit-addressed reader the masked `apply` lane runs
  - count_set_bits   — SIMD popcount loop, at three offsets
  - set_range(True)  — bulk-set via memset (BitmapBuilder)

Sizes: 1k–100M bits.  Throughput reported in bits/second.

Spread over four files -- `bench_bitmap_logic`, `bench_bitmap_offsets` and
`bench_bitmap_pack` are the others -- because a file is one `-O3` compilation
unit and the benchmark workflow gives each unit a single deadline.  All 104
cases as one unit ran in 102 s on one CI runner and then blew past 1800 s on
the next, failing the whole job; four units of thirty-odd cases each put this
file in line with the rest of the suite.  No case was renamed, so the recorded
benchmark history still lines up.

Run with: pixi run pytest marrow/tests/bench_bitmap.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..buffers import Bitmap
from ..utils.testing import Benchmark
from .bitmap_fixtures import make_alternating

# ---------------------------------------------------------------------------
# load[W] — the bit-addressed reader the masked `apply` lane runs per chunk
# ---------------------------------------------------------------------------


def _bench_load(mut b: Benchmark, size: Int) raises:
    """Sweep `load[8]` across a bitmap, one call per 8 bits.

    This is the primitive every masked kernel reads validity through, so it is
    the one place a change to it can be measured without a surrounding kernel's
    noise.
    """
    var bm = make_alternating(size)
    var bm_view = bm.view()
    var chunks = size // 8
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {imm}:
        var acc = SIMD[DType.bool, 8](fill=False)
        for i in range(chunks):
            acc |= bm_view.load[8](i * 8)
        keep(acc)

    b.iter(call)
    keep(bm)


def bench_load_10k(mut b: Benchmark) raises:
    _bench_load(b, 10_000)


def bench_load_1m(mut b: Benchmark) raises:
    _bench_load(b, 1_000_000)


def bench_load_10m(mut b: Benchmark) raises:
    _bench_load(b, 10_000_000)


# ---------------------------------------------------------------------------
# count_set_bits
# ---------------------------------------------------------------------------


def _bench_count_set_bits(mut b: Benchmark, size: Int) raises:
    var bm = make_alternating(size)
    var bm_view = bm.view()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {imm}:
        keep(bm_view.count_set_bits())

    b.iter(call)


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
    var bm = make_alternating(size + 2048).slice(128 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {imm}:
        keep(bm.count_set_bits())

    b.iter(call)


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
    var bm = make_alternating(size + 2048).slice(96 << 3, size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {imm}:
        keep(bm.count_set_bits())

    b.iter(call)


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
# set_range
# ---------------------------------------------------------------------------


def _bench_set_range(mut b: Benchmark, size: Int) raises:
    var builder = Bitmap.alloc_zeroed(size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() {mut builder, imm}:
        builder.set_range(0, size, True)
        keep(builder.view().load_bytes[DType.uint8](0))

    b.iter(call)


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
