"""Benchmarks for aggregate kernels (sum, product, min, max).

Run with:
    pixi run bench_mojo -k bench_aggregate
    pixi run pytest marrow/kernels/tests/bench_aggregate.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray
from marrow.builders import arange, PrimitiveBuilder
from marrow.dtypes import int64, float64, Int64Type, Float64Type, PrimitiveType
from marrow.kernels.aggregate import sum_, product, min_, max_
from marrow.testing import BenchSuite, Benchmark


def _make_array_with_nulls[
    T: PrimitiveType
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


def bench_sum_int64_1k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Int64Type](arr))

    b.iter[call]()


def bench_sum_int64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Int64Type](arr))

    b.iter[call]()


def bench_sum_int64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# sum — float64
# ---------------------------------------------------------------------------


def bench_sum_float64_1k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Float64Type](arr))

    b.iter[call]()


def bench_sum_float64_100k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Float64Type](arr))

    b.iter[call]()


def bench_sum_float64_1m(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Float64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# sum — with nulls
# ---------------------------------------------------------------------------


def bench_sum_nulls_int64_100k(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Int64Type](arr))

    b.iter[call]()


def bench_sum_nulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(sum_[Int64Type](arr))

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


def bench_min_int64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min_[Int64Type](arr))

    b.iter[call]()


def bench_min_int64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min_[Int64Type](arr))

    b.iter[call]()


def bench_max_int64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(max_[Int64Type](arr))

    b.iter[call]()


def bench_max_int64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(max_[Int64Type](arr))

    b.iter[call]()


def bench_min_nulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(min_[Int64Type](arr))

    b.iter[call]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
