"""Benchmarks for the string compute kernels.

Run with: pixi run pytest marrow/kernels/tests/bench_string.mojo --benchmark

The LIKE benchmarks contrast the two shapes of the same kernel: the array x
array overload, which recompiles the pattern once per row, against the
array x scalar-pattern overload, which compiles it once per call.
"""

from std.benchmark import BenchMetric, keep

from marrow.arrays import StringArray
from marrow.builders import StringBuilder
from marrow.kernels.string import LikeKernel, ILikeKernel
from marrow.testing import BenchSuite, Benchmark


def _urls(n: Int) raises -> StringArray:
    """ClickBench-ish URLs, roughly a quarter of which contain 'google'."""
    var b = StringBuilder(capacity=n)
    for i in range(n):
        var r = i % 4
        if r == 0:
            b.append("http://www.google.com/search?q=" + String(i))
        elif r == 1:
            b.append("http://example.org/page/" + String(i))
        elif r == 2:
            b.append("https://news.site.ru/article/" + String(i))
        else:
            b.append("http://shop.example.com/item/" + String(i))
    return b.finish()


def _broadcast(pattern: String, n: Int) raises -> StringArray:
    var b = StringBuilder(capacity=n)
    for _ in range(n):
        b.append(pattern)
    return b.finish()


# ---------------------------------------------------------------------------
# LIKE '%google%' — scalar pattern (compiled once)
# ---------------------------------------------------------------------------


def _bench_like_scalar(mut b: Benchmark, n: Int) raises:
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(LikeKernel.apply(data, "%google%")))

    b.iter[call]()
    keep(data)


def bench_like_scalar_10k(mut b: Benchmark) raises:
    _bench_like_scalar(b, 10_000)


def bench_like_scalar_100k(mut b: Benchmark) raises:
    _bench_like_scalar(b, 100_000)


def bench_like_scalar_1m(mut b: Benchmark) raises:
    _bench_like_scalar(b, 1_000_000)


# ---------------------------------------------------------------------------
# LIKE '%google%' — array x array (pattern recompiled per row)
# ---------------------------------------------------------------------------


def _bench_like_array(mut b: Benchmark, n: Int) raises:
    var data = _urls(n)
    var pattern = _broadcast("%google%", n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(LikeKernel.apply(data, pattern)))

    b.iter[call]()
    keep(data)
    keep(pattern)


def bench_like_array_10k(mut b: Benchmark) raises:
    _bench_like_array(b, 10_000)


def bench_like_array_100k(mut b: Benchmark) raises:
    _bench_like_array(b, 100_000)


def bench_like_array_1m(mut b: Benchmark) raises:
    _bench_like_array(b, 1_000_000)


# ---------------------------------------------------------------------------
# LIKE with a wildcard in the middle — the general backtracking matcher
# ---------------------------------------------------------------------------


def _bench_like_general(mut b: Benchmark, n: Int) raises:
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(LikeKernel.apply(data, "http%google%search%")))

    b.iter[call]()
    keep(data)


def bench_like_general_100k(mut b: Benchmark) raises:
    _bench_like_general(b, 100_000)


# ---------------------------------------------------------------------------
# ILIKE '%GOOGLE%' — scalar pattern, case-folded per row
# ---------------------------------------------------------------------------


def _bench_ilike_scalar(mut b: Benchmark, n: Int) raises:
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(len(ILikeKernel.apply(data, "%GOOGLE%")))

    b.iter[call]()
    keep(data)


def bench_ilike_scalar_100k(mut b: Benchmark) raises:
    _bench_ilike_scalar(b, 100_000)


def bench_ilike_array_100k(mut b: Benchmark) raises:
    var data = _urls(100_000)
    var pattern = _broadcast("%GOOGLE%", 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    @parameter
    def call() raises:
        keep(len(ILikeKernel.apply(data, pattern)))

    b.iter[call]()
    keep(data)
    keep(pattern)


def main() raises:
    BenchSuite.run[__functions_in_module()]()
