"""Benchmarks for the string compute kernels.

Run with: pixi run pytest marrow/kernels/tests/bench_string.mojo --benchmark

The LIKE benchmarks contrast the two shapes of the same kernel: the array x
array overload, which recompiles the pattern once per row, against the
array x scalar-pattern overload, which compiles it once per call.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import StringArray
from ...builders import StringBuilder
from ...kernels.string import (
    ContainsKernel,
    ILikeKernel,
    LengthKernel,
    LikeKernel,
    UpperKernel,
)
from ...utils.testing import Benchmark


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
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, "%google%")))

    b.iter(call)
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
# LIKE with a wildcard in the middle — the general backtracking matcher
# ---------------------------------------------------------------------------


def _bench_like_general(mut b: Benchmark, n: Int) raises:
    var data = _urls(n)
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
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(ILikeKernel.apply(data, "%GOOGLE%")))

    b.iter(call)
    keep(data)


def bench_ilike_scalar_100k(mut b: Benchmark) raises:
    _bench_ilike_scalar(b, 100_000)


def bench_ilike_array_100k(mut b: Benchmark) raises:
    var data = _urls(100_000)
    var pattern = _broadcast("%GOOGLE%", 100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(ILikeKernel.apply(data, pattern)))

    b.iter(call)
    keep(data)
    keep(pattern)


# ---------------------------------------------------------------------------
# ClickBench q21 shape at scale: `URL LIKE '%google%'` over 1M rows, in a
# matching-dense and a matching-sparse variant.
#
# The pair separates a compare-bound implementation from an allocation-bound
# one: both scan every row, but the dense case makes ~every row a hit and the
# sparse case ~none, so any per-hit or per-output allocation shows up as a gap
# between the two.  `_urls` is ~25% hits, sitting between them.
# ---------------------------------------------------------------------------


def _bench_like_dense(mut b: Benchmark, n: Int) raises:
    """`%http%` — every row matches (the substring is the scheme prefix)."""
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, "%http%")))

    b.iter(call)
    keep(data)


def bench_like_dense_1m(mut b: Benchmark) raises:
    _bench_like_dense(b, 1_000_000)


def _bench_like_sparse(mut b: Benchmark, n: Int) raises:
    """`%zqxjv%` — no row matches, so every row is scanned in full."""
    var data = _urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, "%zqxjv%")))

    b.iter(call)
    keep(data)


def bench_like_sparse_1m(mut b: Benchmark) raises:
    _bench_like_sparse(b, 1_000_000)


# ---------------------------------------------------------------------------
# Neighbouring kernels over the same data.
#
# `contains` is the same scan under a different entry point (it should track
# `like_scalar`); `length` is offset arithmetic only and touches no character
# data at all, so it is the drift control -- nothing done to the matching path
# can move it.
# ---------------------------------------------------------------------------


def bench_contains_1m(mut b: Benchmark) raises:
    var data = _urls(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(ContainsKernel.apply_scalar(data, "google")))

    b.iter(call)
    keep(data)


def bench_length_1m(mut b: Benchmark) raises:
    var data = _urls(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(LengthKernel.apply(data)))

    b.iter(call)
    keep(data)


def bench_upper_100k(mut b: Benchmark) raises:
    """A string -> string map: builds a whole new `StringArray`, so it is the
    allocation-heavy control next to the predicates' single bitmap."""
    var data = _urls(100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(UpperKernel.apply(data)))

    b.iter(call)
    keep(data)


# ---------------------------------------------------------------------------
# The shape ClickBench q21 actually executes.
#
# The runtime expression lane (`marrow.exprold.dynamic`) evaluates a literal by
# `DynScalar.repeat(num_rows)`, so `URL LIKE '%google%'` reaches the kernel as
# array x array with n identical right-hand rows -- the `_bench_like_array`
# shape, not the `_bench_like_scalar` one.  Dense and sparse variants of it
# pin down whether the cost tracks the number of matches (compare-bound) or
# the number of rows (per-row set-up).
# ---------------------------------------------------------------------------


def _bench_like_array_pattern(mut b: Benchmark, pattern: String, n: Int) raises:
    var data = _urls(n)
    var pat = _broadcast(pattern, n)
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
