"""Benchmarks for BufferView.filter — compacting int64 values by a selection.

Sizes 1k–10M elements, throughput in elements/second.

Alone in its unit; `bench_bitmap_filter.mojo` explains why the two filters
cannot share one.

Run with: pixi run pytest marrow/tests/bench_buffer_filter.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..buffers import Buffer
from ..utils.testing import Benchmark
from .bitmap_fixtures import make_alternating


# ---------------------------------------------------------------------------
# BufferView.filter — compact fixed-width (int64) values by a selection.
# Alternating selection => the compress-store mixed path (not the memcpy
# run-merge); this is the primitive `filter` hot loop.
# ---------------------------------------------------------------------------


def _bench_filter_values(mut b: Benchmark, size: Int) raises:
    var buf = Buffer.alloc_uninit[DType.int64](size)
    var sel = make_alternating(size)
    var src_view = buf.view[DType.int64](0, size)
    var sel_view = sel.view()
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(src_view.filter(sel_view, sel_start, sel_end, out_len))

    b.iter(call)
    keep(len(buf))
    keep(len(sel))
    keep(len(src_view))
    keep(out_len)
    keep(sel_start)
    keep(sel_end)


def bench_filter_values_1k(mut b: Benchmark) raises:
    _bench_filter_values(b, 1_000)


def bench_filter_values_10k(mut b: Benchmark) raises:
    _bench_filter_values(b, 10_000)


def bench_filter_values_100k(mut b: Benchmark) raises:
    _bench_filter_values(b, 100_000)


def bench_filter_values_1m(mut b: Benchmark) raises:
    _bench_filter_values(b, 1_000_000)


def bench_filter_values_10m(mut b: Benchmark) raises:
    _bench_filter_values(b, 10_000_000)
