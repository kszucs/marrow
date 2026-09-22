"""Benchmarks for LIKE with a per-row pattern — the array x array overload.

The pattern is recompiled once per row here, against once per call in
`bench_string.mojo`; contrasting the two is what these measure. Covers
10k-1M rows plus the ClickBench q21 shape as it actually executes, dense and
sparse.

See `bench_string.mojo` for why the string benchmarks are spread over four
compilation units.

Run with: pixi run pytest marrow/kernels/tests/bench_string_array.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...kernels.string import LikeKernel
from ...utils.testing import Benchmark
from .string_fixtures import broadcast, urls


# ---------------------------------------------------------------------------
# LIKE '%google%' — array x array (pattern recompiled per row)
# ---------------------------------------------------------------------------


def _bench_like_array(mut b: Benchmark, n: Int) raises:
    var data = urls(n)
    var pattern = broadcast("%google%", n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, pattern)))

    b.iter(call)
    keep(data)
    keep(pattern)


def bench_like_array_10k(mut b: Benchmark) raises:
    _bench_like_array(b, 10_000)


def bench_like_array_100k(mut b: Benchmark) raises:
    _bench_like_array(b, 100_000)


def bench_like_array_1m(mut b: Benchmark) raises:
    _bench_like_array(b, 1_000_000)


# ---------------------------------------------------------------------------
# The shape ClickBench q21 actually executes.
#
# The runtime expression lane (`marrow/expr/runtime/values.mojo`) evaluates a literal by
# `DynScalar.to_array(num_rows)`, so `URL LIKE '%google%'` reaches the kernel as
# array x array with n identical right-hand rows -- the `_bench_like_array`
# shape, not the `_bench_like_scalar` one.  Dense and sparse variants of it
# pin down whether the cost tracks the number of matches (compare-bound) or
# the number of rows (per-row set-up).
# ---------------------------------------------------------------------------


def _bench_like_array_pattern(mut b: Benchmark, pattern: String, n: Int) raises:
    var data = urls(n)
    var pat = broadcast(pattern, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, pat)))

    b.iter(call)
    keep(data)
    keep(pat)


def bench_like_array_dense_1m(mut b: Benchmark) raises:
    _bench_like_array_pattern(b, "%http%", 1_000_000)


def bench_like_array_sparse_1m(mut b: Benchmark) raises:
    _bench_like_array_pattern(b, "%zqxjv%", 1_000_000)
