"""Benchmarks for LIKE with a scalar pattern — the pattern compiled once.

Covers `URL LIKE '%google%'` over 10k-1M rows, and the ClickBench q21 shape at
1M rows in a matching-dense and a matching-sparse variant.

One file is one `-O3` compilation unit, and these benchmarks are spread over
four of them -- `bench_string_array`, `bench_string_ilike` and
`bench_string_kernels` are the others -- because the sixteen together do not
compile in any reasonable time. Measured on a CI runner: each group builds and
runs in 15-81 s and all six together in 245 s, while the single unit of
sixteen was still going at 300 s burning a full core, and had already passed
1800 s on an earlier run of the same source. The cost is superlinear in what a
unit holds, so the boundaries below are the ones that were measured rather
than the ones that read best. No case was renamed, so the recorded benchmark
history still lines up.

The array x array shape, which recompiles the pattern once per row, is the
contrast this one is written against; it lives in `bench_string_array.mojo`.

Run with: pixi run pytest marrow/kernels/tests/bench_string.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...kernels.string import LikeKernel
from ...utils.testing import Benchmark
from .string_fixtures import urls


# ---------------------------------------------------------------------------
# LIKE '%google%' — scalar pattern (compiled once)
# ---------------------------------------------------------------------------


def _bench_like_scalar(mut b: Benchmark, n: Int) raises:
    var data = urls(n)
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
# ClickBench q21 shape at scale: `URL LIKE '%google%'` over 1M rows, in a
# matching-dense and a matching-sparse variant.
#
# The pair separates a compare-bound implementation from an allocation-bound
# one: both scan every row, but the dense case makes ~every row a hit and the
# sparse case ~none, so any per-hit or per-output allocation shows up as a gap
# between the two.  `urls` is ~25% hits, sitting between them.
# ---------------------------------------------------------------------------


def _bench_like_dense(mut b: Benchmark, n: Int) raises:
    """`%http%` — every row matches (the substring is the scheme prefix)."""
    var data = urls(n)
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
    var data = urls(n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(LikeKernel.apply(data, "%zqxjv%")))

    b.iter(call)
    keep(data)


def bench_like_sparse_1m(mut b: Benchmark) raises:
    _bench_like_sparse(b, 1_000_000)
