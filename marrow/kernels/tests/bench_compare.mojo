"""Benchmarks for comparison kernels.

Run with: pixi run pytest marrow/kernels/tests/bench_compare.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.builders import arange
from marrow.dtypes import Int32Type, Int64Type
from marrow.kernels.compare import equal
from marrow.testing import BenchSuite, Benchmark


# ---------------------------------------------------------------------------
# equal — int32
# ---------------------------------------------------------------------------


def bench_equal_int32_10k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 10_000)
    var rhs = arange[Int32Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int32Type](lhs, rhs)))

    b.iter[call]()


def bench_equal_int32_100k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 100_000)
    var rhs = arange[Int32Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int32Type](lhs, rhs)))

    b.iter[call]()


def bench_equal_int32_1m(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 1_000_000)
    var rhs = arange[Int32Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int32Type](lhs, rhs)))

    b.iter[call]()


# ---------------------------------------------------------------------------
# equal — int64
# ---------------------------------------------------------------------------


def bench_equal_int64_10k(mut b: Benchmark) raises:
    var lhs = arange[Int64Type](0, 10_000)
    var rhs = arange[Int64Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int64Type](lhs, rhs)))

    b.iter[call]()


def bench_equal_int64_100k(mut b: Benchmark) raises:
    var lhs = arange[Int64Type](0, 100_000)
    var rhs = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int64Type](lhs, rhs)))

    b.iter[call]()


def bench_equal_int64_1m(mut b: Benchmark) raises:
    var lhs = arange[Int64Type](0, 1_000_000)
    var rhs = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(equal[Int64Type](lhs, rhs)))

    b.iter[call]()


def main() raises:
    BenchSuite.run[__functions_in_module()]()
