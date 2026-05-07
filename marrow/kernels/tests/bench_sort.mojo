"""Benchmarks for the sort kernel.

Run with:
    pixi run -e bench pytest marrow/kernels/tests/bench_sort.mojo --benchmark
    pixi run -e bench pytest python/tests/bench_sort.py --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from marrow.testing import BenchSuite, Benchmark
from marrow.arrays import AnyArray
from marrow.builders import (
    Int32Builder,
    Int64Builder,
    Float64Builder,
)
from marrow.dtypes import int32, int64, float64
from marrow.kernels.sort import argsort


# ---------------------------------------------------------------------------
# Data generators (xorshift64 pseudo-random, reproducible)
# ---------------------------------------------------------------------------


def _random_int32(n: Int) raises -> AnyArray:
    var b = Int32Builder(capacity=n)
    var s: UInt64 = 0x123456789ABCDEF0
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        b.append(s.cast[int32.native]())
    return b.finish().to_any()


def _random_int64(n: Int) raises -> AnyArray:
    var b = Int64Builder(capacity=n)
    var s: UInt64 = 0xFEDCBA9876543210
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        b.append(s.cast[int64.native]())
    return b.finish().to_any()


def _random_float64(n: Int) raises -> AnyArray:
    var b = Float64Builder(capacity=n)
    var s: UInt64 = 0xABCDEF0123456789
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        # Map to [0, 1) by using the mantissa bits.
        var f = (s >> 11).cast[float64.native]() * (1.0 / Float64(1 << 53))
        b.append(f)
    return b.finish().to_any()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_sort_int32(mut b: Benchmark, n: Int) raises:
    var data = _random_int32(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(argsort(data.copy()))

    b.iter[call]()
    keep(data)


def _bench_sort_int64(mut b: Benchmark, n: Int) raises:
    var data = _random_int64(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(argsort(data.copy()))

    b.iter[call]()
    keep(data)


def _bench_sort_float64(mut b: Benchmark, n: Int) raises:
    var data = _random_float64(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(argsort(data.copy()))

    b.iter[call]()
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


def main() raises:
    BenchSuite.run[__functions_in_module()]()
