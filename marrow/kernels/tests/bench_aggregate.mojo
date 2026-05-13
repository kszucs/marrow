"""Benchmarks for aggregate kernels (sum, product, min, max).

Run with:
    pixi run bench_mojo -k bench_aggregate
    pixi run pytest marrow/kernels/tests/bench_aggregate.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray
from marrow.builders import arange, PrimitiveBuilder
from marrow.dtypes import (
    int64,
    float64,
    Int64Type,
    Float64Type,
    PrimitiveType,
    NumericType,
)
from marrow.kernels.aggregate import sum, product, min, max
from marrow.testing import BenchSuite, Benchmark


def _make_array_with_nulls[
    T: NumericType
](size: Int) raises -> PrimitiveArray[T]:
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[T.native](i))
    return b.finish()


# ---------------------------------------------------------------------------
# sum — int64
# ---------------------------------------------------------------------------


def bench_sumint64_1k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Int64Type](arr))

    b.iter[call]()


def bench_sumint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Int64Type](arr))

    b.iter[call]()


def bench_sumint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# sum — float64
# ---------------------------------------------------------------------------


def bench_sumfloat64_1k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Float64Type](arr))

    b.iter[call]()


def bench_sumfloat64_100k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Float64Type](arr))

    b.iter[call]()


def bench_sumfloat64_1m(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Float64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# sum — with nulls
# ---------------------------------------------------------------------------


def bench_sumnulls_int64_100k(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Int64Type](arr))

    b.iter[call]()


def bench_sumnulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def bench_product_int64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](1, 100_001)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(product[Int64Type](arr))

    b.iter[call]()


def bench_product_int64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](1, 1_000_001)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(product[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------


def bench_minint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min[Int64Type](arr))

    b.iter[call]()


def bench_minint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min[Int64Type](arr))

    b.iter[call]()


def bench_maxint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(max[Int64Type](arr))

    b.iter[call]()


def bench_maxint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(max[Int64Type](arr))

    b.iter[call]()


def bench_minnulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
