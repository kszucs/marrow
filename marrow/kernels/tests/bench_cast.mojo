"""Benchmarks for cast kernels.

Run with: pixi run -e dev pytest marrow/kernels/tests/bench_cast.mojo --benchmark

Covers the numeric SIMD path (widen / narrow, safe vs unsafe) and the
temporal reinterpret path. String parse/format is builder-based and benchmarked
against PyArrow/Polars in ``python/marrow/tests/bench_cast.py``.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray
from ...builders import arange
from ...dtypes import (
    Int32Type,
    Int64Type,
    Float64Type,
    int8,
    int64,
    float64,
    timestamp,
    second,
    millisecond,
)
from ...kernels.cast import cast, NumericCast
from ...testing import Benchmark


# ---------------------------------------------------------------------------
# Typed numeric cast — int32 → float64 (widen, lossless, safe)
# ---------------------------------------------------------------------------


def _bench_int32_to_float64(mut b: Benchmark, n: Int) raises:
    var src = arange[Int32Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(NumericCast.apply[Int32Type, Float64Type, safe=True](src)))

    b.iter[call]()
    keep(src)


def bench_int32_to_float64_10k(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 10_000)


def bench_int32_to_float64_100k(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 100_000)


def bench_int32_to_float64_1m(mut b: Benchmark) raises:
    _bench_int32_to_float64(b, 1_000_000)


# ---------------------------------------------------------------------------
# Typed numeric cast — int64 → int32 (narrow, unsafe/truncating)
# ---------------------------------------------------------------------------


def _bench_int64_to_int32_unsafe(mut b: Benchmark, n: Int) raises:
    var src = arange[Int64Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(NumericCast.apply[Int64Type, Int32Type, safe=False](src)))

    b.iter[call]()
    keep(src)


def bench_int64_to_int32_unsafe_10k(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 10_000)


def bench_int64_to_int32_unsafe_100k(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 100_000)


def bench_int64_to_int32_unsafe_1m(mut b: Benchmark) raises:
    _bench_int64_to_int32_unsafe(b, 1_000_000)


# ---------------------------------------------------------------------------
# Runtime dispatch — int64 → float64 via the type-erased router
# ---------------------------------------------------------------------------


def _bench_dispatch_int64_to_float64(mut b: Benchmark, n: Int) raises:
    var src: DynArray = arange[Int64Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(cast(src, float64, safe=True)))

    b.iter[call]()
    keep(src)


def bench_dispatch_int64_to_float64_100k(mut b: Benchmark) raises:
    _bench_dispatch_int64_to_float64(b, 100_000)


def bench_dispatch_int64_to_float64_1m(mut b: Benchmark) raises:
    _bench_dispatch_int64_to_float64(b, 1_000_000)


# ---------------------------------------------------------------------------
# Temporal — timestamp[s] → timestamp[ms] (unit upscale)
# ---------------------------------------------------------------------------


def _bench_timestamp_upscale(mut b: Benchmark, n: Int) raises:
    var src = cast(DynArray(arange[Int64Type](0, n)), timestamp(second))
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(cast(src, timestamp(millisecond))))

    b.iter[call]()
    keep(src)


def bench_timestamp_upscale_100k(mut b: Benchmark) raises:
    _bench_timestamp_upscale(b, 100_000)


def bench_timestamp_upscale_1m(mut b: Benchmark) raises:
    _bench_timestamp_upscale(b, 1_000_000)
