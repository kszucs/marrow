"""Benchmarks for aggregate kernels (sum, product, min, max).

Covers no-null and 10%-null arrays across sizes 1k–1M, dtypes int64 and float64.

Run with: pixi run bench_mojo
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)

from marrow.arrays import PrimitiveArray
from marrow.builders import arange, PrimitiveBuilder
from marrow.dtypes import int64, float64, PrimitiveType
from marrow.kernels.aggregate import sum_, product, min_, max_


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
# sum
# ---------------------------------------------------------------------------


@parameter
def bench_sum[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = sum_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_sum_nulls[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = _make_array_with_nulls[T](size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = sum_[T](arr)
        keep(result)

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


@parameter
def bench_product[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](1, size + 1)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = product[T](arr)
        keep(result)

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------


@parameter
def bench_min[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = min_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_max[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = max_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_min_nulls[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var arr = _make_array_with_nulls[T](size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = min_[T](arr)
        keep(result)

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    comptime sizes = (1_000, 10_000, 100_000, 1_000_000)
    comptime size_labels = ("1k", "10k", "100k", "1M")

    # --- sum ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_sum[Int64Type]](
            BenchId("sum[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_sum[Float64Type]](
            BenchId("sum[Float64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_sum_nulls[Int64Type]](
            BenchId("sum_nulls[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    # --- product ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_product[Int64Type]](
            BenchId("product[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    # --- min / max ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_min[Int64Type]](
            BenchId("min[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_max[Int64Type]](
            BenchId("max[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_min_nulls[Int64Type]](
            BenchId("min_nulls[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()
