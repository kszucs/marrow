"""Benchmarks for the sort kernel.

Run with:
    pixi run -e bench pytest marrow/kernels/tests/bench_sort.mojo --benchmark
    pixi run -e bench pytest python/marrow/tests/bench_sort.py --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ...testing import Benchmark
from ...arrays import DynArray, StructArray
from ...builders import (
    Int32Builder,
    Int64Builder,
    Float64Builder,
)
from ...dtypes import int32, int64, float64
from ...kernels.sort import sort_indices, SortIndices
from ...tabular import record_batch


# ---------------------------------------------------------------------------
# Data generators (xorshift64 pseudo-random, reproducible)
# ---------------------------------------------------------------------------


def _random_int32(n: Int) raises -> DynArray:
    var b = Int32Builder(capacity=n)
    var s: UInt64 = 0x123456789ABCDEF0
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        b.append(s.cast[int32.native]())
    return b.finish().to_dyn()


def _random_int64(n: Int) raises -> DynArray:
    var b = Int64Builder(capacity=n)
    var s: UInt64 = 0xFEDCBA9876543210
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        b.append(s.cast[int64.native]())
    return b.finish().to_dyn()


def _random_float64(n: Int) raises -> DynArray:
    var b = Float64Builder(capacity=n)
    var s: UInt64 = 0xABCDEF0123456789
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        # Map to [0, 1) by using the mantissa bits.
        var f = (s >> 11).cast[float64.native]() * (1.0 / Float64(1 << 53))
        b.append(f)
    return b.finish().to_dyn()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_sort_int32(mut b: Benchmark, n: Int) raises:
    var data = _random_int32(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(sort_indices(data.copy()))

    b.iter(call)
    keep(data)


def _bench_sort_int64(mut b: Benchmark, n: Int) raises:
    var data = _random_int64(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(sort_indices(data.copy()))

    b.iter(call)
    keep(data)


def _bench_sort_float64(mut b: Benchmark, n: Int) raises:
    var data = _random_float64(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(sort_indices(data.copy()))

    b.iter(call)
    keep(data)


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------


def bench_sort_int32_10k(mut b: Benchmark) raises:
    _bench_sort_int32(b, 10_000)


def bench_sort_int32_100k(mut b: Benchmark) raises:
    _bench_sort_int32(b, 100_000)


def bench_sort_int32_1m(mut b: Benchmark) raises:
    _bench_sort_int32(b, 1_000_000)


def bench_sort_int64_100k(mut b: Benchmark) raises:
    _bench_sort_int64(b, 100_000)


def bench_sort_int64_1m(mut b: Benchmark) raises:
    _bench_sort_int64(b, 1_000_000)


def bench_sort_float64_100k(mut b: Benchmark) raises:
    _bench_sort_float64(b, 100_000)


def bench_sort_float64_1m(mut b: Benchmark) raises:
    _bench_sort_float64(b, 1_000_000)


# ---------------------------------------------------------------------------
# Multi-key sort — column-wise LSD, one stable pass per key.
#
# These span `_RADIX_THRESHOLD` (32,768) deliberately. Below it each pass takes
# the comparison path, where `stable` makes the comparator a total order
# (ties break on the original row index); at or above it each pass takes LSD
# radix, which is stable by construction and ignores the flag. So the two sizes
# measure the two regimes, and only the small one can carry any cost from
# honouring `stable`.
# ---------------------------------------------------------------------------


def _multi_key_batch(n: Int, keys: Int) raises -> StructArray:
    # low-cardinality leading key => large tie groups => the stable comparator's
    # tie-break branch is taken often, which is the worst case for it.
    var cols = List[DynArray]()
    var names = List[String]()
    var s: UInt64 = 0x9E3779B97F4A7C15
    for k in range(keys):
        var col = Int32Builder(capacity=n)
        for _ in range(n):
            s ^= s << 13
            s ^= s >> 7
            s ^= s << 17
            # leading key has 8 distinct values, later keys are dense
            var v = (s % 8) if k == 0 else s
            col.append(v.cast[int32.native]())
        cols.append(col.finish().to_dyn())
        names.append(String("k", k))
    return record_batch(cols^, names=names^).to_struct_array()


def _bench_sort_multi(mut b: Benchmark, n: Int, keys: Int) raises:
    var data = _multi_key_batch(n, keys)
    var key_indices = List[Int]()
    var ascending = List[Bool]()
    for k in range(keys):
        key_indices.append(k)
        ascending.append(True)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(SortIndices.multi(data, key_indices, ascending))

    b.iter(call)
    keep(data)
    keep(key_indices)
    keep(ascending)


def bench_sort_multi_2col_10k(mut b: Benchmark) raises:
    _bench_sort_multi(b, 10_000, 2)


def bench_sort_multi_2col_1m(mut b: Benchmark) raises:
    _bench_sort_multi(b, 1_000_000, 2)


def bench_sort_multi_3col_1m(mut b: Benchmark) raises:
    _bench_sort_multi(b, 1_000_000, 3)
