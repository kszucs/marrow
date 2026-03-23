"""Benchmarks: rapidhash (vectorized) vs ahash (scalar).

Run with: pixi run bench_mojo -k bench_hashing
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64, uint64
from marrow.kernels.hashing import rapidhash, ahash


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


def _fmt(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


def bench_int64(n: Int, warmup: Int, iters: Int) raises:
    var arr = _make_int64(n)

    # rapidhash (vectorized)
    for _ in range(warmup):
        var h = rapidhash[int64](arr)
        keep(len(h))
    var t_rapid = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var h = rapidhash[int64](arr)
        t_rapid += perf_counter_ns() - t0
        keep(len(h))

    # ahash (scalar)
    for _ in range(warmup):
        var h = ahash[int64](arr)
        keep(len(h))
    var t_ahash = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var h = ahash[int64](arr)
        t_ahash += perf_counter_ns() - t0
        keep(len(h))

    var avg_rapid = t_rapid // UInt(iters)
    var avg_ahash = t_ahash // UInt(iters)
    print("  rapidhash: ", _fmt(avg_rapid))
    print("  ahash:     ", _fmt(avg_ahash))
    if avg_rapid > 0:
        print("  speedup:   ", String(Int(avg_ahash // avg_rapid)), "x")


def bench_int32(n: Int, warmup: Int, iters: Int) raises:
    var arr = _make_int32(n)

    for _ in range(warmup):
        var h = rapidhash[int32](arr)
        keep(len(h))
    var t_rapid = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var h = rapidhash[int32](arr)
        t_rapid += perf_counter_ns() - t0
        keep(len(h))

    for _ in range(warmup):
        var h = ahash[int32](arr)
        keep(len(h))
    var t_ahash = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var h = ahash[int32](arr)
        t_ahash += perf_counter_ns() - t0
        keep(len(h))

    var avg_rapid = t_rapid // UInt(iters)
    var avg_ahash = t_ahash // UInt(iters)
    print("  rapidhash: ", _fmt(avg_rapid))
    print("  ahash:     ", _fmt(avg_ahash))
    if avg_rapid > 0:
        print("  speedup:   ", String(Int(avg_ahash // avg_rapid)), "x")


def main() raises:
    print("Hashing benchmark: rapidhash (SIMD) vs ahash (scalar)")
    print("======================================================")

    print("\n=== int64, 10k elements ===")
    bench_int64(10_000, warmup=5, iters=50)

    print("\n=== int64, 100k elements ===")
    bench_int64(100_000, warmup=3, iters=20)

    print("\n=== int64, 1M elements ===")
    bench_int64(1_000_000, warmup=2, iters=10)

    print("\n=== int32, 10k elements ===")
    bench_int32(10_000, warmup=5, iters=50)

    print("\n=== int32, 100k elements ===")
    bench_int32(100_000, warmup=3, iters=20)

    print("\n=== int32, 1M elements ===")
    bench_int32(1_000_000, warmup=2, iters=10)
