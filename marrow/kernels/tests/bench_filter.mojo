"""Benchmarks for filter kernel.

Run with:
    pixi run bench_mojo -k bench_filter
    pixi run pytest marrow/kernels/tests/bench_filter.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import BoolArray, PrimitiveArray, Int64Array, Int32Array
from ...builders import (
    arange,
    BoolBuilder,
    PrimitiveBuilder,
    Int64Builder,
    Int32Builder,
)
from ...dtypes import int64, Int64Type
from ...kernels.filter import Filter, Take
from ...execution import ExecContext
from ...testing import Benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask(size: Int, selectivity_pct: Int) raises -> BoolArray:
    var b = BoolBuilder(size)
    for i in range(size):
        b.append(Bool((i * 100) // size < selectivity_pct))
    return b.finish()


def _make_array_with_nulls(size: Int) raises -> Int64Array:
    var b = Int64Builder(size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[int64.native](i))
    return b.finish()


# ---------------------------------------------------------------------------
# End-to-end filter benchmarks — 50% selectivity
# ---------------------------------------------------------------------------


def bench_filter50pct_10k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 10_000)
    var mask = _make_mask(10_000, 50)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


def bench_filter50pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 50)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


def bench_filter50pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 50)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


# ---------------------------------------------------------------------------
# 10% selectivity
# ---------------------------------------------------------------------------


def bench_filter10pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 10)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


def bench_filter10pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 10)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


# ---------------------------------------------------------------------------
# 90% selectivity
# ---------------------------------------------------------------------------


def bench_filter90pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 90)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


def bench_filter90pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 90)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


# ---------------------------------------------------------------------------
# 50% selectivity with nulls
# ---------------------------------------------------------------------------


def bench_filter50pct_nulls_100k(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls(100_000)
    var mask = _make_mask(100_000, 50)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


def bench_filter50pct_nulls_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls(1_000_000)
    var mask = _make_mask(1_000_000, 50)

    @always_inline
    def call() raises {imm}:
        keep(len(Filter.apply(arr, mask.values())))

    b.iter(call)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Take (gather) benchmarks
#
# `Take.apply` is the hot gather behind joins, group-by and sort, and it owns
# one of the hand-rolled `sync_parallelize` stripe loops (Q2.4). It had no
# benchmark at all, so any change to that loop was unmeasurable — these exist to
# make it measurable. Both the serial default and a forced-parallel context are
# covered, since the stripe loop only runs on the latter.
# ---------------------------------------------------------------------------


def _shuffled_indices(size: Int) raises -> Int32Array:
    """A permutation-ish index array with no nulls — the SIMD-gather fast path.

    Strided by a co-prime of `size` so the gather is scattered rather than
    sequential, which is what a join or sort permutation actually looks like.
    """
    var b = Int32Builder(size)
    for i in range(size):
        b.append(Int32((i * 7919) % size))
    return b.finish()


def _indices_with_nulls(size: Int) raises -> Int32Array:
    """Index array with nulls — the serial slow path (outer-join shape)."""
    var b = Int32Builder(size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Int32((i * 7919) % size))
    return b.finish()


def _bench_take(mut b: Benchmark, size: Int, ctx: ExecContext) raises:
    var arr = arange[Int64Type](0, size)
    var idx = _shuffled_indices(size)
    b.throughput(BenchMetric.elements, size)

    @always_inline
    def call() raises {imm}:
        keep(len(Take.apply(arr, idx, ctx)))

    b.iter(call)
    keep(arr)
    keep(idx)


def bench_take_100k(mut b: Benchmark) raises:
    _bench_take(b, 100_000, ExecContext.serial())


def bench_take_1m(mut b: Benchmark) raises:
    _bench_take(b, 1_000_000, ExecContext.serial())


def bench_take_parallel_1m(mut b: Benchmark) raises:
    """Forces the striped path — `ExecContext.serial()` never reaches it."""
    _bench_take(b, 1_000_000, ExecContext(num_threads=0))


def bench_take_nulls_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var idx = _indices_with_nulls(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(Take.apply(arr, idx, ExecContext.serial())))

    b.iter(call)
    keep(arr)
    keep(idx)
