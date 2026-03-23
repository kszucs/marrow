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
from marrow.dtypes import int64, float64, DataType
from marrow.kernels.aggregate import sum_, product, min_, max_


def _make_array_with_nulls[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[T.native](i))
    return b.finish_typed()


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


@parameter
def bench_sum[T: DataType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = sum_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_sum_nulls[T: DataType](mut b: Bencher, size: Int) raises:
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
def bench_product[T: DataType](mut b: Bencher, size: Int) raises:
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
def bench_min[T: DataType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = min_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_max[T: DataType](mut b: Bencher, size: Int) raises:
    var arr = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = max_[T](arr)
        keep(result)

    b.iter[call_fn]()


@parameter
def bench_min_nulls[T: DataType](mut b: Bencher, size: Int) raises:
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
        m.bench_with_input[Int, bench_sum[int64]](
            BenchId("sum[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_sum[float64]](
            BenchId("sum[float64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_sum_nulls[int64]](
            BenchId("sum_nulls[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    # --- product ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_product[int64]](
            BenchId("product[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    # --- min / max ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_min[int64]](
            BenchId("min[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_max[int64]](
            BenchId("max[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_min_nulls[int64]](
            BenchId("min_nulls[int64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()
