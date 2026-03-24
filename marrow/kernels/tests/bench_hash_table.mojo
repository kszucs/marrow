"""Benchmarks for SwissHashTable: build, insert, probe.

Run with: pixi run bench_mojo -k bench_hash_table
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.bitmap import Bitmap
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import uint64, struct_, Field
from marrow.kernels.hashtable import SwissHashTable
from marrow.kernels.hashing import rapidhash


def _make_keys(n: Int) raises -> StructArray:
    """Generate a single-column StructArray with n distinct uint64 keys."""
    var b = PrimitiveBuilder[uint64](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    var children = List[AnyArray]()
    children.append(b.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", uint64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=Optional[Bitmap](None),
        children=children^,
    )


def _fmt(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


def bench_build(n: Int, warmup: Int, iters: Int) raises:
    var keys = _make_keys(n)

    for _ in range(warmup):
        var t = SwissHashTable[rapidhash]()
        t.build(keys)
        keep(t.num_keys())

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var t = SwissHashTable[rapidhash]()
        t.build(keys)
        total += perf_counter_ns() - t0
        keep(t.num_keys())

    print("  build:                ", _fmt(total // UInt(iters)))


def bench_insert(n: Int, warmup: Int, iters: Int) raises:
    var keys = _make_keys(n)

    for _ in range(warmup):
        var t = SwissHashTable[rapidhash]()
        var bids = t.insert(keys)
        keep(t.num_keys())

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var t = SwissHashTable[rapidhash]()
        var bids = t.insert(keys)
        total += perf_counter_ns() - t0
        keep(t.num_keys())

    print("  insert: ", _fmt(total // UInt(iters)))


def bench_probe(n: Int, warmup: Int, iters: Int) raises:
    var keys = _make_keys(n)
    var table = SwissHashTable[rapidhash]()
    table.build(keys)

    for _ in range(warmup):
        var pairs = table.probe(keys, keys, n)
        keep(len(pairs[0]))

    var total = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var pairs = table.probe(keys, keys, n)
        total += perf_counter_ns() - t0
        keep(len(pairs[0]))

    print("  probe:          ", _fmt(total // UInt(iters)))


def run_size(n: Int, warmup: Int, iters: Int) raises:
    print("\n=== SwissHashTable", n, "entries ===")
    bench_insert(n, warmup, iters)
    bench_build(n, warmup, iters)
    bench_probe(n, warmup, iters)

    # Throughput summary.
    var keys = _make_keys(n)
    var t0 = perf_counter_ns()
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var build_ns = perf_counter_ns() - t0
    t0 = perf_counter_ns()
    var pairs = t.probe(keys, keys, n)
    var probe_ns = perf_counter_ns() - t0
    print(
        "  throughput:           build",
        String(Int(UInt(n) * 1_000 // (build_ns + 1))),
        "Mops/s, probe",
        String(Int(UInt(n) * 1_000 // (probe_ns + 1))),
        "Mops/s",
    )


def main() raises:
    print("SwissHashTable benchmark")
    print("========================")

    run_size(10_000, warmup=10, iters=100)
    run_size(100_000, warmup=5, iters=50)
    run_size(1_000_000, warmup=3, iters=20)
    run_size(10_000_000, warmup=1, iters=5)
