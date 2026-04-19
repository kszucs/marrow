"""Benchmarks for filter kernel.

Run with:
    pixi run bench_mojo -k bench_filter
    pixi run pytest marrow/kernels/tests/bench_filter.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import BoolArray, PrimitiveArray, Int64Array
from marrow.builders import arange, BoolBuilder, PrimitiveBuilder, Int64Builder
from marrow.dtypes import int64, Int64Type
from marrow.kernels.filter import filter_
from marrow.testing import BenchSuite, Benchmark


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


def bench_filter_50pct_10k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 10_000)
    var mask = _make_mask(10_000, 50)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


def bench_filter_50pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 50)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


def bench_filter_50pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 50)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# 10% selectivity
# ---------------------------------------------------------------------------


def bench_filter_10pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 10)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


def bench_filter_10pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 10)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# 90% selectivity
# ---------------------------------------------------------------------------


def bench_filter_90pct_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    var mask = _make_mask(100_000, 90)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


def bench_filter_90pct_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    var mask = _make_mask(1_000_000, 90)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# 50% selectivity with nulls
# ---------------------------------------------------------------------------


def bench_filter_50pct_nulls_100k(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls(100_000)
    var mask = _make_mask(100_000, 50)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


def bench_filter_50pct_nulls_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls(1_000_000)
    var mask = _make_mask(1_000_000, 50)

    @always_inline
    @parameter
    def call() raises:
        keep(len(filter_[Int64Type](arr, mask)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
