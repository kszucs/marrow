"""Benchmarks for the groupby kernel.

Run with:
    pixi run pytest marrow/kernels/tests/bench_groupby.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import AnyArray
from marrow.builders import PrimitiveBuilder, Int32Builder, Float64Builder
from marrow.dtypes import int32, float64, Int32Type, Float64Type
from marrow.kernels.groupby import group_by
from marrow.kernels.aggregate import (
    AggKernel,
    SumKernel,
    MinKernel,
    MaxKernel,
    MeanKernel,
)
from marrow.testing import BenchSuite, Benchmark


def _make_keys(n: Int, num_groups: Int) raises -> AnyArray:
    var b = Int32Builder(n)
    for i in range(n):
        b.append(Scalar[int32.native](i % num_groups))
    return b.finish()


def _make_vals(n: Int) raises -> AnyArray:
    var b = Float64Builder(n)
    for i in range(n):
        b.append(Scalar[float64.native](Float64(i)))
    return b.finish()


def _bench_group_by[K: AggKernel](mut b: Benchmark, n: Int) raises:
    var keys = _make_keys(n, 10)
    var vals = _make_vals(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(group_by[K](keys, vals))

    b.iter[call]()
    keep(keys)
    keep(vals)


# ---------------------------------------------------------------------------
# group_by sum — 10K / 100K / 1M rows
# ---------------------------------------------------------------------------


def bench_groupby_sum_10k(mut b: Benchmark) raises:
    _bench_group_by[SumKernel](b, 10_000)


def bench_groupby_sum_100k(mut b: Benchmark) raises:
    _bench_group_by[SumKernel](b, 100_000)


def bench_groupby_sum_1m(mut b: Benchmark) raises:
    _bench_group_by[SumKernel](b, 1_000_000)


# ---------------------------------------------------------------------------
# group_by min / max / mean — 100K rows
# ---------------------------------------------------------------------------


def bench_groupby_min_100k(mut b: Benchmark) raises:
    _bench_group_by[MinKernel](b, 100_000)


def bench_groupby_max_100k(mut b: Benchmark) raises:
    _bench_group_by[MaxKernel](b, 100_000)


def bench_groupby_mean_100k(mut b: Benchmark) raises:
    _bench_group_by[MeanKernel](b, 100_000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
