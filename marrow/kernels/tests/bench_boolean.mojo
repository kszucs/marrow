"""Benchmarks for the boolean kernels.

Run with: pixi run pytest marrow/kernels/tests/bench_boolean.mojo --benchmark

`and_` / `or_` are three-valued, so they have two distinct paths and both are
covered here: the nullable path computes Kleene validity, and the non-nullable
path returns early with no validity buffer at all. That early return is the
common case for real predicates, and a change to the validity algebra must not
cost it anything — which is only visible if both are measured.

`not_` is the **control**: it shares the module and the array construction but
not the Kleene path, so its median is what a whole-run drift is measured
against.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import BoolArray
from ...builders import BoolBuilder
from ...kernels.boolean import AndKernel, NotKernel, OrKernel
from ...utils.testing import Benchmark


def _bools(n: Int, *, nullable: Bool, phase: Int) raises -> BoolArray:
    """`n` bools alternating on `phase`, every 7th null when `nullable`."""
    var b = BoolBuilder(capacity=n)
    for i in range(n):
        if nullable and i % 7 == 0:
            b.append_null()
        else:
            b.append((i // phase) % 2 == 0)
    return b.finish()


# ---------------------------------------------------------------------------
# and_ / or_ — nullable operands, the Kleene validity path
# ---------------------------------------------------------------------------


def _bench_and_nullable(mut b: Benchmark, n: Int) raises:
    var lhs = _bools(n, nullable=True, phase=1)
    var rhs = _bools(n, nullable=True, phase=3)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(AndKernel.apply(lhs, rhs)))

    b.iter(call)
    keep(lhs)
    keep(rhs)


def bench_and_nullable_100k(mut b: Benchmark) raises:
    _bench_and_nullable(b, 100_000)


def bench_and_nullable_1m(mut b: Benchmark) raises:
    _bench_and_nullable(b, 1_000_000)


def bench_or_nullable_1m(mut b: Benchmark) raises:
    var lhs = _bools(1_000_000, nullable=True, phase=1)
    var rhs = _bools(1_000_000, nullable=True, phase=3)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(OrKernel.apply(lhs, rhs)))

    b.iter(call)
    keep(lhs)
    keep(rhs)


# ---------------------------------------------------------------------------
# and_ — non-nullable operands, the early return that allocates nothing
# ---------------------------------------------------------------------------


def _bench_and_nonnull(mut b: Benchmark, n: Int) raises:
    var lhs = _bools(n, nullable=False, phase=1)
    var rhs = _bools(n, nullable=False, phase=3)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(len(AndKernel.apply(lhs, rhs)))

    b.iter(call)
    keep(lhs)
    keep(rhs)


def bench_and_nonnull_100k(mut b: Benchmark) raises:
    _bench_and_nonnull(b, 100_000)


def bench_and_nonnull_1m(mut b: Benchmark) raises:
    _bench_and_nonnull(b, 1_000_000)


# ---------------------------------------------------------------------------
# not_ — the control: same module, no Kleene path
# ---------------------------------------------------------------------------


def bench_not_control_1m(mut b: Benchmark) raises:
    var arr = _bools(1_000_000, nullable=True, phase=1)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(NotKernel.apply(arr)))

    b.iter(call)
    keep(arr)
