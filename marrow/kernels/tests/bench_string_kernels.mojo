"""Benchmarks for the string kernels neighbouring LIKE, over the same data.

`contains`, `length` and `upper` at 100k-1M rows.

See `bench_string.mojo` for why the string benchmarks are spread over four
compilation units.

Run with: pixi run pytest marrow/kernels/tests/bench_string_kernels.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import StringArray
from ...kernels.string import ContainsKernel, LengthKernel, UpperKernel
from ...utils.testing import Benchmark
from .string_fixtures import urls


# ---------------------------------------------------------------------------
# Neighbouring kernels over the same data.
#
# `contains` is the same scan under a different entry point (it should track
# `like_scalar`); `length` is offset arithmetic only and touches no character
# data at all, so it is the drift control -- nothing done to the matching path
# can move it.
# ---------------------------------------------------------------------------


def bench_contains_1m(mut b: Benchmark) raises:
    var data = urls(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(ContainsKernel.apply_scalar(data, "google")))

    b.iter(call)
    keep(data)


def bench_length_1m(mut b: Benchmark) raises:
    var data = urls(1_000_000)
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() raises {imm}:
        keep(len(LengthKernel.apply(data)))

    b.iter(call)
    keep(data)


def bench_upper_100k(mut b: Benchmark) raises:
    """A string -> string map: builds a whole new `StringArray`, so it is the
    allocation-heavy control next to the predicates' single bitmap."""
    var data = urls(100_000)
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def call() raises {imm}:
        keep(len(UpperKernel.apply(data)))

    b.iter(call)
    keep(data)
