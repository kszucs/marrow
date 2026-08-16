"""Benchmarks for arithmetic kernel variants.

CPU: AddKernel with no nulls and with 10% nulls, across sizes 1k–1M for int32
and float64.

Run with: pixi run pytest marrow/kernels/tests/bench_arithmetic.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import PrimitiveArray
from ...builders import arange, PrimitiveBuilder
from ...dtypes import Int32Type, Float64Type, NumericType
from ...kernels.numeric import AddKernel
from ...execution import ExecContext
from ...utils.testing import Benchmark


def _make_array_with_nulls[
    T: NumericType
](size: Int) raises -> PrimitiveArray[T]:
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
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_int32_10k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 10_000)
    var rhs = arange[Int32Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_int32_100k(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 100_000)
    var rhs = arange[Int32Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_int32_1m(mut b: Benchmark) raises:
    var lhs = arange[Int32Type](0, 1_000_000)
    var rhs = arange[Int32Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


# ---------------------------------------------------------------------------
# add with nulls — int32
# ---------------------------------------------------------------------------


def bench_add_nulls_int32_1k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](1_000)
    var rhs = _make_array_with_nulls[Int32Type](1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_int32_10k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](10_000)
    var rhs = _make_array_with_nulls[Int32Type](10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_int32_100k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](100_000)
    var rhs = _make_array_with_nulls[Int32Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_int32_1m(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Int32Type](1_000_000)
    var rhs = _make_array_with_nulls[Int32Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


# ---------------------------------------------------------------------------
# add — float64
# ---------------------------------------------------------------------------


def bench_add_float64_1k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 1_000)
    var rhs = arange[Float64Type](0, 1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_float64_10k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 10_000)
    var rhs = arange[Float64Type](0, 10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_float64_100k(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 100_000)
    var rhs = arange[Float64Type](0, 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


def bench_add_float64_1m(mut b: Benchmark) raises:
    var lhs = arange[Float64Type](0, 1_000_000)
    var rhs = arange[Float64Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(0))

    b.iter(call)


# ---------------------------------------------------------------------------
# add with nulls — float64
# ---------------------------------------------------------------------------


def bench_add_nulls_float64_1k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](1_000)
    var rhs = _make_array_with_nulls[Float64Type](1_000)
    b.throughput(BenchMetric.elements, 1_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_float64_10k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](10_000)
    var rhs = _make_array_with_nulls[Float64Type](10_000)
    b.throughput(BenchMetric.elements, 10_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_float64_100k(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](100_000)
    var rhs = _make_array_with_nulls[Float64Type](100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


def bench_add_nulls_float64_1m(mut b: Benchmark) raises:
    var lhs = _make_array_with_nulls[Float64Type](1_000_000)
    var rhs = _make_array_with_nulls[Float64Type](1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Float64Type](lhs, rhs).unsafe_get(1))

    b.iter(call)


# ---------------------------------------------------------------------------
# add — parallel scaling on a 1M int32 input
#
# All benchmarks run on the same array (size 1M). They differ only in the
# ExecContext passed: serial vs forced parallel(2/4/8) vs auto. Use
# --competition to compare side-by-side and watch the scaling.
# ---------------------------------------------------------------------------


def _bench_add_1m_ctx(mut b: Benchmark, ctx: ExecContext) raises:
    var lhs = arange[Int32Type](0, 1_000_000)
    var rhs = arange[Int32Type](0, 1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(AddKernel.apply[Int32Type](lhs, rhs, ctx).unsafe_get(0))

    b.iter(call)
    keep(lhs)
    keep(rhs)


def bench_add_int32_1m_serial(mut b: Benchmark) raises:
    _bench_add_1m_ctx(b, ExecContext.serial())


def bench_add_int32_1m_parallel_2(mut b: Benchmark) raises:
    _bench_add_1m_ctx(b, ExecContext.parallel(2))


def bench_add_int32_1m_parallel_4(mut b: Benchmark) raises:
    _bench_add_1m_ctx(b, ExecContext.parallel(4))


def bench_add_int32_1m_parallel_8(mut b: Benchmark) raises:
    _bench_add_1m_ctx(b, ExecContext.parallel(8))


def bench_add_int32_1m_auto(mut b: Benchmark) raises:
    _bench_add_1m_ctx(b, ExecContext.auto())
