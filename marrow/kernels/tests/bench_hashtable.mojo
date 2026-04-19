"""Benchmarks for SwissHashTable: build, insert, probe.

Run with:
    pixi run bench_mojo -k bench_hash_table
    pixi run pytest marrow/kernels/tests/bench_hash_table.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.buffers import Bitmap
from marrow.builders import PrimitiveBuilder, UInt64Builder
from marrow.dtypes import uint64, UInt64Type, struct_, Field
from marrow.kernels.hashtable import SwissHashTable
from marrow.kernels.hashing import rapidhash
from marrow.testing import BenchSuite, Benchmark


def _make_keys(n: Int) raises -> StructArray:
    """Generate a single-column StructArray with n distinct uint64 keys."""
    var b = UInt64Builder(capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    var children = List[AnyArray]()
    children.append(b.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", uint64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=Optional[Bitmap[]](None),
        children=children^,
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def bench_hash_table_build_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)

    @always_inline
    @parameter
    def call() raises:
        var t = SwissHashTable[rapidhash]()
        t.build(keys)
        keep(t.num_keys())

    b.iter[call]()
    keep(keys)


def bench_hash_table_build_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)

    @always_inline
    @parameter
    def call() raises:
        var t = SwissHashTable[rapidhash]()
        t.build(keys)
        keep(t.num_keys())

    b.iter[call]()
    keep(keys)


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def bench_hash_table_insert_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)

    @always_inline
    @parameter
    def call() raises:
        var t = SwissHashTable[rapidhash]()
        var bids = t.insert(keys)
        keep(t.num_keys())

    b.iter[call]()
    keep(keys)


def bench_hash_table_insert_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)

    @always_inline
    @parameter
    def call() raises:
        var t = SwissHashTable[rapidhash]()
        var bids = t.insert(keys)
        keep(t.num_keys())

    b.iter[call]()
    keep(keys)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def bench_hash_table_probe_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)
    var table = SwissHashTable[rapidhash]()
    table.build(keys)

    @always_inline
    @parameter
    def call() raises:
        var pairs = table.probe(keys, keys, 100_000)
        keep(len(pairs[0]))

    b.iter[call]()
    keep(keys)
    keep(table)


def bench_hash_table_probe_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)
    var table = SwissHashTable[rapidhash]()
    table.build(keys)

    @always_inline
    @parameter
    def call() raises:
        var pairs = table.probe(keys, keys, 1_000_000)
        keep(len(pairs[0]))

    b.iter[call]()
    keep(keys)
    keep(table)


# ---------------------------------------------------------------------------
# probe — semi-join (single_match=True)
# ---------------------------------------------------------------------------


def bench_hash_table_probe_semi_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)
    var table = SwissHashTable[rapidhash]()
    table.build(keys)

    @always_inline
    @parameter
    def call() raises:
        var pairs = table.probe(keys, keys, 100_000, single_match=True)
        keep(len(pairs[0]))

    b.iter[call]()
    keep(keys)
    keep(table)


def bench_hash_table_probe_semi_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)
    var table = SwissHashTable[rapidhash]()
    table.build(keys)

    @always_inline
    @parameter
    def call() raises:
        var pairs = table.probe(keys, keys, 1_000_000, single_match=True)
        keep(len(pairs[0]))

    b.iter[call]()
    keep(keys)
    keep(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    BenchSuite.run[__functions_in_module()]()
