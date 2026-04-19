"""Benchmarks for the groupby kernel.

Run with:
    pixi run pytest marrow/kernels/tests/bench_groupby.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import AnyArray
from marrow.builders import PrimitiveBuilder, Int32Builder, Float64Builder
from marrow.dtypes import int32, float64, Int32Type, Float64Type
from marrow.kernels.groupby import groupby
from marrow.testing import BenchSuite, Benchmark


def _make_keys(n: Int, num_groups: Int) raises -> AnyArray:
    var b = Int32Builder(n)
    for i in range(n):
        b.append(Scalar[int32.native](i % num_groups))
    return AnyArray(b.finish())


def _make_vals(n: Int) raises -> List[AnyArray]:
    var b = Float64Builder(n)
    for i in range(n):
        b.append(Scalar[float64.native](Float64(i)))
    var vals = List[AnyArray]()
    vals.append(AnyArray(b.finish()))
    return vals^


def _aggs(s: String) -> List[String]:
    var a = List[String]()
    a.append(s)
    return a^


# ---------------------------------------------------------------------------
# groupby sum
# ---------------------------------------------------------------------------


def _bench_groupby_sum(mut b: Benchmark, n: Int) raises:
    var keys = _make_keys(n, 10)
    var vals = _make_vals(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(groupby(keys, vals, _aggs("sum")))

    b.iter[call]()
    keep(keys)
    keep(vals)


def bench_groupby_sum_10k(mut b: Benchmark) raises:
    _bench_groupby_sum(b, 10_000)


def bench_groupby_sum_100k(mut b: Benchmark) raises:
    _bench_groupby_sum(b, 100_000)


def bench_groupby_sum_1m(mut b: Benchmark) raises:
    _bench_groupby_sum(b, 1_000_000)


# ---------------------------------------------------------------------------
# groupby min / max / mean — 100K rows
# ---------------------------------------------------------------------------


def _bench_groupby_agg(mut b: Benchmark, agg: String, n: Int) raises:
    var keys = _make_keys(n, 10)
    var vals = _make_vals(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(groupby(keys, vals, _aggs(agg)))

    b.iter[call]()
    keep(keys)
    keep(vals)


def bench_groupby_min_100k(mut b: Benchmark) raises:
    _bench_groupby_agg(b, "min", 100_000)


def bench_groupby_max_100k(mut b: Benchmark) raises:
    _bench_groupby_agg(b, "max", 100_000)


def bench_groupby_mean_100k(mut b: Benchmark) raises:
    _bench_groupby_agg(b, "mean", 100_000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
