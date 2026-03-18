"""Benchmarks for arithmetic kernel variants.

CPU section (Bench framework):
  - add:   add() with no nulls (CPU SIMD path)
  - nulls: add() where inputs have 10% nulls
  Across sizes 1k–1M, dtypes int32 and float64.

GPU section (manual timing, skipped when no GPU present):
  - gpu: GPU add with data pre-loaded on device (mean us/call)
  Across sizes 1k–10M, dtypes int32 and float32.
  Uses manual perf_counter_ns timing (like bench_similarity) rather than the
  Bench framework, because the Bench framework's tight loop rapidly allocates
  and frees the GPU result buffer each iteration, crashing libKGENCompilerRTShared.

Run with: pixi run bench

NOTE: bench_with_input is used instead of bench_function to avoid a Mojo
codegen bug (~25.7) where registering multiple size-parameterized instantiations
of the same function crashes at runtime inside bitmap_and():

    # This crashes:
    m.bench_function[bench_add[1_000]](...)
    m.bench_function[bench_add[10_000]](...)  # panic in bitmap_and

bench_with_input[Int, bench_add[T]] passes size as a runtime Int, so only
ONE template instantiation of bench_add[T] is created per dtype. The same
function pointer is called multiple times with different runtime inputs,
which does not trigger the crash.
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

from marrow.arrays import PrimitiveArray, arange
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, float32, float64, DataType
from marrow.kernels.arithmetic import add


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_array_with_nulls[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    """Build an array with 10% nulls (every 10th element is null)."""
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[T.native](i))
    return b.finish_typed()


# ---------------------------------------------------------------------------
# Benchmark functions — parameterized by dtype, size passed at runtime
# ---------------------------------------------------------------------------


@parameter
def bench_add[T: DataType](mut b: Bencher, size: Int) raises:
    var lhs = arange[T](0, size)
    var rhs = arange[T](0, size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = add[T](lhs, rhs)
        keep(result.unsafe_get(0))

    b.iter[call_fn]()


@parameter
def bench_add_nulls[T: DataType](mut b: Bencher, size: Int) raises:
    """Add() with 10% nulls in both inputs."""
    var lhs = _make_array_with_nulls[T](size)
    var rhs = _make_array_with_nulls[T](size)

    @always_inline
    @parameter
    def call_fn() raises:
        var result = add[T](lhs, rhs)
        keep(result.unsafe_get(1))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# GPU helper — manual timing (cannot use Bench framework: result buffer
# allocation/free in tight loop crashes libKGENCompilerRTShared)
# ---------------------------------------------------------------------------


def _bench_gpu_add[
    T: DataType
](size: Int, iters: Int, ctx: DeviceContext,) raises -> Float64:
    """Returns mean microseconds per kernel dispatch with pre-loaded data."""
    var lhs = arange[T](0, size).to_device(ctx)
    var rhs = arange[T](0, size).to_device(ctx)
    ctx.synchronize()

    # Warmup
    for _ in range(3):
        var r = add[T](lhs, rhs, ctx)
        keep(len(r))
    ctx.synchronize()

    # Timed runs
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = add[T](lhs, rhs, ctx)
        keep(len(r))
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    comptime sizes = (1_000, 10_000, 100_000, 1_000_000)
    comptime size_labels = ("1k", "10k", "100k", "1M")

    # --- int32 ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_add[int32]](
            BenchId("add[int32]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_add_nulls[int32]](
            BenchId("nulls[int32]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    # --- float64 ---
    comptime for si in range(4):
        m.bench_with_input[Int, bench_add[float64]](
            BenchId("add[float64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )
    comptime for si in range(4):
        m.bench_with_input[Int, bench_add_nulls[float64]](
            BenchId("nulls[float64]", size_labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()

    if has_accelerator():
        var ctx = DeviceContext()
        comptime gpu_sizes = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
        comptime gpu_iters = (500, 100, 20, 5, 2)

        print("\n=== GPU add benchmarks (data pre-loaded on device) ===")
        print("bench_id               us/call")
        print("--------               -------")

        comptime for si in range(5):
            var us = _bench_gpu_add[int32](gpu_sizes[si], gpu_iters[si], ctx)
            print(t"gpu[int32]/{gpu_sizes[si]}    {us} us")

        comptime for si in range(5):
            var us = _bench_gpu_add[float32](gpu_sizes[si], gpu_iters[si], ctx)
            print(t"gpu[float32]/{gpu_sizes[si]}    {us} us")
