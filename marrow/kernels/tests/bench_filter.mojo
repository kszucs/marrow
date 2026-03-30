"""Benchmarks for filter kernel.

End-to-end benchmarks across sizes 10k–1M, plus _filter_block micro-benchmarks
with different selectivities, distributions, and compile-time parameters.

Uses manual perf_counter_ns timing rather than the Bench framework, because
the Bench framework's tight loop rapidly allocates and frees the filter
result buffer each iteration, crashing libKGENCompilerRTShared.

Run with: pixi run bench_mojo -k bench_filter
"""

from std.benchmark import keep
from std.bit import pop_count
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray
from marrow.buffers import Buffer, Bitmap
from marrow.builders import arange, PrimitiveBuilder
from marrow.views import BitmapView
from marrow.dtypes import int64, bool_
from marrow.kernels.filter import filter_, _filter_block


# ---------------------------------------------------------------------------
# Helpers — mask generation
# ---------------------------------------------------------------------------


def _make_mask(size: Int, selectivity_pct: Int) raises -> PrimitiveArray[bool_]:
    var b = PrimitiveBuilder[bool_](size)
    for i in range(size):
        b.append(Bool((i * 100) // size < selectivity_pct))
    return b.finish()


def _make_array_with_nulls(size: Int) raises -> PrimitiveArray[int64]:
    var b = PrimitiveBuilder[int64](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[int64.native](i))
    return b.finish()


def _make_sel_word_uniform(pct: Int) -> UInt64:
    """Evenly spaced bits: predictable for branch predictor."""
    if pct == 0:
        return UInt64(0)
    if pct >= 100:
        return ~UInt64(0)
    var w = UInt64(0)
    var step = 100 // pct
    for i in range(0, 64, step):
        w |= UInt64(1) << UInt64(i)
    return w


def _make_sel_word_random(pct: Int) -> UInt64:
    """Pseudo-random bits: unpredictable for branch predictor."""
    var w = UInt64(0)
    var rng = UInt64(0xDEADBEEF)
    for i in range(64):
        rng = rng * 6364136223846793005 + 1442695040888963407
        if Int((rng >> 33) % 100) < pct:
            w |= UInt64(1) << UInt64(i)
    return w


def _make_sel_word_clustered(pct: Int) -> UInt64:
    """Single contiguous run of ~pct% bits: best case for memcpy."""
    var n_bits = 64 * pct // 100
    if n_bits == 0:
        return UInt64(0)
    if n_bits >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(n_bits)) - 1


# ---------------------------------------------------------------------------
# End-to-end filter benchmarks
# ---------------------------------------------------------------------------


def _bench_filter(
    size: Int, selectivity_pct: Int, with_nulls: Bool, iters: Int
) raises -> Float64:
    var arr: PrimitiveArray[int64]
    if with_nulls:
        arr = _make_array_with_nulls(size)
    else:
        arr = arange[int64](0, size)
    var mask = _make_mask(size, selectivity_pct)

    for _ in range(3):
        var r = filter_[int64](arr, mask)
        keep(len(r))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = filter_[int64](arr, mask)
        keep(len(r))
    return Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0


# ---------------------------------------------------------------------------
# _filter_block micro-benchmarks — processes N_BLOCKS in a loop to simulate
# realistic throughput (not single-block latency).
# ---------------------------------------------------------------------------


def _bench_block(sel_word: UInt64, n_blocks: Int, iters: Int) raises -> Float64:
    """Benchmark _filter_block by processing n_blocks consecutive blocks.

    Allocates n_blocks * 64 source elements and filters them in a loop,
    measuring the average ns per block.
    """
    comptime native = int64.native
    comptime BLOCK = 64
    var total_elems = n_blocks * BLOCK
    var arr = arange[int64](0, total_elems)
    var src = arr.buffer.view[native](0)
    var out_buf = Buffer.alloc_zeroed[native](total_elems)
    var dst = out_buf.view[native]()

    for _ in range(3):
        var out_pos = 0
        for blk in range(n_blocks):
            out_pos += _filter_block[native](
                dst, out_pos, src, blk * BLOCK, sel_word
            )
        keep(out_pos)

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var out_pos = 0
        for blk in range(n_blocks):
            out_pos += _filter_block[native](
                dst, out_pos, src, blk * BLOCK, sel_word
            )
        keep(out_pos)
    var total = perf_counter_ns() - t0
    return Float64(total) / Float64(iters) / Float64(n_blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    # ── End-to-end ───────────────────────────────────────────────────────
    comptime sizes = (10_000, 100_000, 1_000_000)
    comptime labels = ("10k", "100k", "1M")
    comptime iters_ = (500, 100, 20)

    print("=== End-to-end filter[int64] ===")
    print("bench_id                              us/call")
    print("--------                              -------")

    comptime for si in range(3):
        var us = _bench_filter(sizes[si], 50, False, iters_[si])
        print(t"filter_50pct/{labels[si]}                   {us} us")
    comptime for si in range(3):
        var us = _bench_filter(sizes[si], 10, False, iters_[si])
        print(t"filter_10pct/{labels[si]}                   {us} us")
    comptime for si in range(3):
        var us = _bench_filter(sizes[si], 90, False, iters_[si])
        print(t"filter_90pct/{labels[si]}                   {us} us")
    comptime for si in range(3):
        var us = _bench_filter(sizes[si], 50, True, iters_[si])
        print(t"filter_50pct_nulls/{labels[si]}             {us} us")

    # ── _filter_block: default params, varying density ───────────────────
    comptime n_blocks = 1024  # 1024 blocks × 64 = 64k elements per iteration
    comptime blk_iters = 500

    print()
    print("=== _filter_block[int64] — sparse/dense adaptive ===")
    print("pattern                    ns/block  popcount")
    print("-------                    --------  --------")

    comptime pcts = (0, 1, 10, 25, 50, 75, 90, 100)
    comptime for pi in range(8):
        var w = _make_sel_word_uniform(pcts[pi])
        var ns = _bench_block(w, n_blocks, blk_iters)
        var pc = Int(pop_count(w))
        print(t"  uniform_{pcts[pi]}pct              {ns} ns   {pc}")

    comptime rpcts = (10, 50, 90)
    comptime for pi in range(3):
        var w = _make_sel_word_random(rpcts[pi])
        var ns = _bench_block(w, n_blocks, blk_iters)
        var pc = Int(pop_count(w))
        print(t"  random_{rpcts[pi]}pct               {ns} ns   {pc}")

    comptime for pi in range(3):
        var w = _make_sel_word_clustered(rpcts[pi])
        var ns = _bench_block(w, n_blocks, blk_iters)
        var pc = Int(pop_count(w))
        print(t"  clustered_{rpcts[pi]}pct            {ns} ns   {pc}")
