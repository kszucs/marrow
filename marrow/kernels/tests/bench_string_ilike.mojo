"""Benchmarks for ILIKE, and for the general backtracking LIKE matcher.

ILIKE case-folds per row; the general matcher is what a wildcard in the middle
of the pattern falls back to, rather than the prefix/suffix fast paths.

See `bench_string.mojo` for why the string benchmarks are spread over four
compilation units.

Run with: pixi run pytest marrow/kernels/tests/bench_string_ilike.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...kernels.string import ILikeKernel, LikeKernel
from ...utils.testing import Benchmark
from .string_fixtures import broadcast, urls


# ---------------------------------------------------------------------------
# LIKE with a wildcard in the middle — the general backtracking matcher
# ---------------------------------------------------------------------------


def _bench_like_general(mut b: Benchmark, n: Int) raises:
    var data = urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, "http%google%search%")))

    b.iter(call)
    keep(data)


def bench_like_general_100k(mut b: Benchmark) raises:
    _bench_like_general(b, 100_000)


# ---------------------------------------------------------------------------
# ILIKE '%GOOGLE%' — scalar pattern, case-folded per row
# ---------------------------------------------------------------------------


def _bench_ilike_scalar(mut b: Benchmark, n: Int) raises:
    var data = urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(ILikeKernel.apply(data, "%GOOGLE%")))

    b.iter(call)
    keep(data)


def bench_ilike_scalar_100k(mut b: Benchmark) raises:
    _bench_ilike_scalar(b, 100_000)


def bench_ilike_array_100k(mut b: Benchmark) raises:
    var data = urls(100_000)
    var pattern = broadcast("%GOOGLE%", 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(ILikeKernel.apply(data, pattern)))

    b.iter(call)
    keep(data)
    keep(pattern)
