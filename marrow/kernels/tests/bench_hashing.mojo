"""Benchmarks: rapidhash CPU SIMD vs GPU.

Run with: pixi run bench_mojo -k bench_hashing
"""

from std.benchmark import keep
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, BoolArray
from marrow.builders import PrimitiveBuilder, BoolBuilder
from marrow.dtypes import DataType, bool_, int32, int64, uint64
from marrow.kernels.hashing import rapidhash


def _make_int64(n: Int) raises -> PrimitiveArray[int64]:
    var b = PrimitiveBuilder[int64](capacity=n)
    for i in range(n):
        b.append(Scalar[int64.native](i))
    return b.finish()


def _make_int32(n: Int) raises -> PrimitiveArray[int32]:
    var b = PrimitiveBuilder[int32](capacity=n)
    for i in range(n):
        b.append(Scalar[int32.native](i))
    return b.finish()


def _make_bool(n: Int) raises -> BoolArray:
    var b = BoolBuilder(capacity=n)
    for i in range(n):
        b.append(Bool(i % 2 == 0))
    return b.finish()


def _fmt(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


def _bench_cpu[T: DataType](n: Int, warmup: Int, iters: Int) raises -> UInt:
    """CPU rapidhash benchmark. Returns avg ns."""
    var arr = PrimitiveBuilder[T](capacity=n)
    for i in range(n):
        arr.append(Scalar[T.native](i))
    var keys = arr.finish()

    for _ in range(warmup):
        var h = rapidhash[T](keys)
        keep(len(h))
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var h = rapidhash[T](keys)
        keep(len(h))
    return (perf_counter_ns() - t0) // UInt(iters)


def _bench_cpu_bool(n: Int, warmup: Int, iters: Int) raises -> UInt:
    """CPU bool rapidhash benchmark. Returns avg ns."""
    var keys = _make_bool(n)

    for _ in range(warmup):
        var h = rapidhash(keys)
        keep(len(h))
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var h = rapidhash(keys)
        keep(len(h))
    return (perf_counter_ns() - t0) // UInt(iters)


def _bench_gpu[
    T: DataType
](n: Int, warmup: Int, iters: Int, ctx: DeviceContext) raises -> UInt:
    """GPU rapidhash benchmark with preloaded data. Returns avg ns."""
    var arr = PrimitiveBuilder[T](capacity=n)
    for i in range(n):
        arr.append(Scalar[T.native](i))
    var keys = arr.finish().to_device(ctx)
    ctx.synchronize()

    for _ in range(warmup):
        var h = rapidhash[T](keys, ctx)
        keep(len(h))
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var h = rapidhash[T](keys, ctx)
        keep(len(h))
    ctx.synchronize()
    return (perf_counter_ns() - t0) // UInt(iters)


def _bench_gpu_bool(
    n: Int, warmup: Int, iters: Int, ctx: DeviceContext
) raises -> UInt:
    """GPU bool rapidhash benchmark with preloaded data. Returns avg ns."""
    var keys = _make_bool(n).to_device(ctx)
    ctx.synchronize()

    for _ in range(warmup):
        var h = rapidhash(keys, ctx)
        keep(len(h))
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var h = rapidhash(keys, ctx)
        keep(len(h))
    ctx.synchronize()
    return (perf_counter_ns() - t0) // UInt(iters)


def main() raises:
    print("Hashing benchmark: rapidhash CPU vs GPU")
    print("========================================")

    print("\n=== int64, 10k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int64](10_000, 5, 50)))
    print("\n=== int64, 100k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int64](100_000, 3, 20)))
    print("\n=== int64, 1M elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int64](1_000_000, 2, 10)))

    print("\n=== int32, 10k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int32](10_000, 5, 50)))
    print("\n=== int32, 100k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int32](100_000, 3, 20)))
    print("\n=== int32, 1M elements ===")
    print("  cpu:       ", _fmt(_bench_cpu[int32](1_000_000, 2, 10)))

    print("\n=== bool, 10k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu_bool(10_000, 5, 50)))
    print("\n=== bool, 100k elements ===")
    print("  cpu:       ", _fmt(_bench_cpu_bool(100_000, 3, 20)))
    print("\n=== bool, 1M elements ===")
    print("  cpu:       ", _fmt(_bench_cpu_bool(1_000_000, 2, 10)))

    if has_accelerator():
        print("\n\nGPU benchmark: rapidhash (preloaded)")
        print("====================================")
        var ctx = DeviceContext()

        print("\n=== GPU int64, 10k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int64](10_000, 5, 50, ctx)))
        print("\n=== GPU int64, 100k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int64](100_000, 3, 20, ctx)))
        print("\n=== GPU int64, 1M elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int64](1_000_000, 2, 10, ctx)))

        print("\n=== GPU int32, 10k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int32](10_000, 5, 50, ctx)))
        print("\n=== GPU int32, 100k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int32](100_000, 3, 20, ctx)))
        print("\n=== GPU int32, 1M elements ===")
        print("  gpu:       ", _fmt(_bench_gpu[int32](1_000_000, 2, 10, ctx)))

        print("\n=== GPU bool, 10k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu_bool(10_000, 5, 50, ctx)))
        print("\n=== GPU bool, 100k elements ===")
        print("  gpu:       ", _fmt(_bench_gpu_bool(100_000, 3, 20, ctx)))
        print("\n=== GPU bool, 1M elements ===")
        print("  gpu:       ", _fmt(_bench_gpu_bool(1_000_000, 2, 10, ctx)))
