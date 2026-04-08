"""Benchmarks for comparison kernels.

Run with: pixi run bench_mojo -k bench_compare
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
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray
from marrow.builders import arange
from marrow.dtypes import (
    int32,
    float32,
    int64,
    float64,
    PrimitiveType,
    Int32Type,
    Int64Type,
    Float32Type,
)
from marrow.kernels.compare import equal, less


@parameter
def bench_equal[T: PrimitiveType](mut b: Bencher, size: Int) raises:
    var lhs = arange[T](0, size)
    var rhs = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = equal[T](lhs, rhs)
        keep(len(result))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# GPU helper — manual timing (cannot use Bench framework: result buffer
# allocation/free in tight loop crashes libKGENCompilerRTShared)
# ---------------------------------------------------------------------------


def _bench_gpu_equal[
    T: PrimitiveType
](size: Int, iters: Int, ctx: DeviceContext) raises -> Float64:
    """Returns mean microseconds per kernel dispatch with pre-loaded data."""
    var lhs = arange[T](0, size).to_device(ctx)
    var rhs = arange[T](0, size).to_device(ctx)
    ctx.synchronize()

    # Warmup
    for _ in range(3):
        var r = equal[T](lhs, rhs, ctx)
        keep(len(r))
    ctx.synchronize()

    # Timed runs
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = equal[T](lhs, rhs, ctx)
        keep(len(r))
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    comptime sizes = (10_000, 100_000, 1_000_000)
    comptime size_labels = ("10k", "100k", "1M")

    comptime for si in range(3):
        m.bench_with_input[Int, bench_equal[Int32Type]](
            BenchId("equal[Int32Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    comptime for si in range(3):
        m.bench_with_input[Int, bench_equal[Int64Type]](
            BenchId("equal[Int64Type]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()

    if has_accelerator():
        var ctx = DeviceContext()
        comptime gpu_sizes = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
        comptime gpu_iters = (500, 100, 20, 5, 2)

        print("\n=== GPU equal benchmarks (data pre-loaded on device) ===")
        print("bench_id                        us/call")
        print("--------                        -------")

        comptime for si in range(5):
            var us = _bench_gpu_equal[Int32Type](
                gpu_sizes[si], gpu_iters[si], ctx
            )
            print(t"gpu_equal[Int32Type]/{gpu_sizes[si]}      {us} us")

        comptime for si in range(5):
            var us = _bench_gpu_equal[Float32Type](
                gpu_sizes[si], gpu_iters[si], ctx
            )
            print(t"gpu_equal[Float32Type]/{gpu_sizes[si]}    {us} us")
