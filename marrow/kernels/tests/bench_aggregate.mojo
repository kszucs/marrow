"""Benchmarks for aggregate kernels (sum, product, min, max).

Run with:
    pixi run bench_mojo -k bench_aggregate
    pixi run pytest marrow/kernels/tests/bench_aggregate.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import PrimitiveArray
from ...builders import arange, PrimitiveBuilder
from ...dtypes import (
    int64,
    float64,
    Int64Type,
    Float64Type,
    PrimitiveType,
    NumericType,
)
from ...kernels.aggregate import (
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
)
from ...testing import Benchmark


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
    def call() raises {imm}:
        keep(SumKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_sumint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_sumint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Int64Type](arr))

    b.iter(call)


# ---------------------------------------------------------------------------
# sum — float64
# ---------------------------------------------------------------------------


def bench_sumfloat64_1k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Float64Type](arr))

    b.iter(call)


def bench_sumfloat64_100k(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Float64Type](arr))

    b.iter(call)


def bench_sumfloat64_1m(mut b: Benchmark) raises:
    var arr = arange[Float64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Float64Type](arr))

    b.iter(call)


# ---------------------------------------------------------------------------
# sum — with nulls
# ---------------------------------------------------------------------------


def bench_sumnulls_int64_100k(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_sumnulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(SumKernel.reduce[Int64Type](arr))

    b.iter(call)


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def bench_product_int64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](1, 100_001)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(ProductKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_product_int64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](1, 1_000_001)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(ProductKernel.reduce[Int64Type](arr))

    b.iter(call)


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------


def bench_minint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(MinKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_minint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(MinKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_maxint64_100k(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(MaxKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_maxint64_1m(mut b: Benchmark) raises:
    var arr = arange[Int64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(MaxKernel.reduce[Int64Type](arr))

    b.iter(call)


def bench_minnulls_int64_1m(mut b: Benchmark) raises:
    var arr = _make_array_with_nulls[Int64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(MinKernel.reduce[Int64Type](arr))

    b.iter(call)
