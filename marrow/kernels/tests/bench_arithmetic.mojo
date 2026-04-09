"""Benchmarks for arithmetic kernel variants.

CPU: add() with no nulls and with 10% nulls, across sizes 1k–1M for int32
and float64.

Run with: pixi run pytest marrow/kernels/tests/bench_arithmetic.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray
from marrow.builders import arange, PrimitiveBuilder
from marrow.dtypes import Int32Type, Float64Type, PrimitiveType
from marrow.kernels.arithmetic import add
from marrow.testing import BenchSuite, Benchmark


def _make_array_with_nulls[T: PrimitiveType](size: Int) raises -> PrimitiveArray[T]:
    """Build an array with 10% nulls (every 10th element is null)."""
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[T.native](i))
    return b.finish()


# ---------------------------------------------------------------------------
# add — int32
# ---------------------------------------------------------------------------


def bench_add_int32_1k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 1_000)
    var rhs = arange[Int32Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_int32_10k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 10_000)
    var rhs = arange[Int32Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_int32_100k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 100_000)
    var rhs = arange[Int32Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_int32_1m(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 1_000_000)
    var rhs = arange[Int32Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


# ---------------------------------------------------------------------------
# add with nulls — int32
# ---------------------------------------------------------------------------


def bench_add_nulls_int32_1k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](1_000)
    var rhs = _make_array_with_nulls[Int32Type](1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_int32_10k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](10_000)
    var rhs = _make_array_with_nulls[Int32Type](10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_int32_100k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](100_000)
    var rhs = _make_array_with_nulls[Int32Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_int32_1m(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](1_000_000)
    var rhs = _make_array_with_nulls[Int32Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


# ---------------------------------------------------------------------------
# add — float64
# ---------------------------------------------------------------------------


def bench_add_float64_1k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 1_000)
    var rhs = arange[Float64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_float64_10k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 10_000)
    var rhs = arange[Float64Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_float64_100k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 100_000)
    var rhs = arange[Float64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


def bench_add_float64_1m(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 1_000_000)
    var rhs = arange[Float64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter[call]()


# ---------------------------------------------------------------------------
# add with nulls — float64
# ---------------------------------------------------------------------------


def bench_add_nulls_float64_1k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](1_000)
    var rhs = _make_array_with_nulls[Float64Type](1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_float64_10k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](10_000)
    var rhs = _make_array_with_nulls[Float64Type](10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_float64_100k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](100_000)
    var rhs = _make_array_with_nulls[Float64Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def bench_add_nulls_float64_1m(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](1_000_000)
    var rhs = _make_array_with_nulls[Float64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    @parameter
    def call() raises:
        keep(add[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter[call]()


def main() raises:
    BenchSuite.run[__functions_in_module()]()
