"""Benchmarks for SwissHashTable: build, find, find_or_insert.

Run with: pixi run bench_mojo -k bench_hash_table
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import uint64
from marrow.kernels.hash_table import SwissHashTable
from marrow.kernels.hashing import rapidhash


def _make_hashes(n: Int) raises -> PrimitiveArray[uint64]:
    """Generate n distinct uint64 hashes (sequential values through rapidhash)."""
    var b = PrimitiveBuilder[uint64](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    return b.finish()


def _fmt(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


def bench_build(n: Int, warmup: Int, iters: Int) raises:
    var hashes = _make_hashes(n)

    for _ in range(warmup):
        var t = SwissHashTable[rapidhash]()
        t.build(hashes)
        keep(t.num_buckets())

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var t = SwissHashTable[rapidhash]()
        t.build(hashes)
        total += perf_counter_ns() - t0
        keep(t.num_buckets())

    print("  build:          ", _fmt(total // UInt(iters)))


def bench_find(n: Int, warmup: Int, iters: Int) raises:
    var hashes = _make_hashes(n)
    var table = SwissHashTable[rapidhash]()
    table.build(hashes)

    for _ in range(warmup):
        var s = 0
        for i in range(n):
            s += table.find(UInt64(hashes.unsafe_get(i)))
        keep(s)

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var s = 0
        for i in range(n):
            s += table.find(UInt64(hashes.unsafe_get(i)))
        total += perf_counter_ns() - t0
        keep(s)

    print("  find (hit):     ", _fmt(total // UInt(iters)))


def bench_find_miss(n: Int, warmup: Int, iters: Int) raises:
    var hashes = _make_hashes(n)
    var table = SwissHashTable[rapidhash]()
    table.build(hashes)

    # Generate hashes that are NOT in the table.
    var miss_hashes = _make_hashes(n)
    var b = PrimitiveBuilder[uint64](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](UInt64(miss_hashes.unsafe_get(i)) + UInt64(n * 2)))
    var misses = b.finish()

    for _ in range(warmup):
        var s = 0
        for i in range(n):
            s += table.find(UInt64(misses.unsafe_get(i)))
        keep(s)

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var s = 0
        for i in range(n):
            s += table.find(UInt64(misses.unsafe_get(i)))
        total += perf_counter_ns() - t0
        keep(s)

    print("  find (miss):    ", _fmt(total // UInt(iters)))


def bench_find_or_insert(n: Int, warmup: Int, iters: Int) raises:
    var hashes = _make_hashes(n)

    for _ in range(warmup):
        var t = SwissHashTable[rapidhash]()
        t.reserve(n)
        for i in range(n):
            _ = t.find_or_insert(UInt64(hashes.unsafe_get(i)))
        keep(t.num_buckets())

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var t = SwissHashTable[rapidhash]()
        t.reserve(n)
        for i in range(n):
            _ = t.find_or_insert(UInt64(hashes.unsafe_get(i)))
        total += perf_counter_ns() - t0
        keep(t.num_buckets())

    print("  find_or_insert: ", _fmt(total // UInt(iters)))


def run_size(n: Int, warmup: Int, iters: Int) raises:
    print("\n=== SwissHashTable", n, "entries ===")
    bench_build(n, warmup, iters)
    bench_find(n, warmup, iters)
    bench_find_miss(n, warmup, iters)
    bench_find_or_insert(n, warmup, iters)

    # Throughput summary.
    var hashes = _make_hashes(n)
    var t0 = perf_counter_ns()
    var t = SwissHashTable[rapidhash]()
    t.build(hashes)
    var build_ns = perf_counter_ns() - t0
    t0 = perf_counter_ns()
    for i in range(n):
        _ = t.find(UInt64(hashes.unsafe_get(i)))
    var find_ns = perf_counter_ns() - t0
    print(
        "  throughput:      build",
        String(Int(UInt(n) * 1_000 // (build_ns + 1))),
        "Mops/s, find",
        String(Int(UInt(n) * 1_000 // (find_ns + 1))),
        "Mops/s",
    )


def main() raises:
    print("SwissHashTable benchmark")
    print("========================")

    run_size(10_000, warmup=10, iters=100)
    run_size(100_000, warmup=5, iters=50)
    run_size(1_000_000, warmup=3, iters=20)
    run_size(10_000_000, warmup=1, iters=5)
