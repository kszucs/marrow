"""Benchmarks for BitmapView.filter — compacting a bitmap by a selection.

Sizes 1k–100M bits, throughput in elements/second.

Alone in its unit, and `BufferView.filter` is alone in
`bench_buffer_filter.mojo`, because the two together do not compile in any
reasonable time. Measured on the CI runner: six of these cases build and run
in 13 s and the five `BufferView` ones in 9 s, while the eleven as a single
`-O3` unit were still going at 300 s, burning a full core, with the whole
29-case file having already run past 1800 s. Both are masked-compress
lowerings, which `filter`/`take` are known to be delicate about; combining
them is what blows up, not the number of cases -- `bench_bitmap_offsets.mojo`
holds thirty and takes 39 s.

Run with: pixi run pytest marrow/tests/bench_bitmap_filter.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..utils.testing import Benchmark
from .bitmap_fixtures import make_alternating


# ---------------------------------------------------------------------------
# BitmapView.filter — compact a bitmap by a selection (validity / bool filter)
#
# Alternating selection => every 64-bit word is "mixed", exercising the
# pext + compressed_store path (the interesting case; all-ones / all-zeros
# words hit the cheaper run-merge branches).
# ---------------------------------------------------------------------------


def _bench_filter_bits(mut b: Benchmark, size: Int) raises:
    var src = make_alternating(size)
    var sel = make_alternating(size)
    var src_view = src.view()
    var sel_view = sel.view()
    var out_len, sel_start, sel_end = sel_view.count_set_bits_with_range()
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        var res = src_view.filter(sel_view, sel_start, sel_end, out_len)
        keep(res[0])
        keep(res[1])

    b.iter(call)
    keep(len(src))
    keep(len(sel))
    keep(len(src_view))
    keep(out_len)
    keep(sel_start)
    keep(sel_end)


def bench_filter_bits_1k(mut b: Benchmark) raises:
    _bench_filter_bits(b, 1_000)


def bench_filter_bits_10k(mut b: Benchmark) raises:
    _bench_filter_bits(b, 10_000)


def bench_filter_bits_100k(mut b: Benchmark) raises:
    _bench_filter_bits(b, 100_000)


def bench_filter_bits_1m(mut b: Benchmark) raises:
    _bench_filter_bits(b, 1_000_000)


def bench_filter_bits_10m(mut b: Benchmark) raises:
    _bench_filter_bits(b, 10_000_000)


def bench_filter_bits_100m(mut b: Benchmark) raises:
    _bench_filter_bits(b, 100_000_000)
