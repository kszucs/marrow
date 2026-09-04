"""Benchmarks for `BufferView`\'s value-level scans.

`__contains__` is the one every caller shares — `RuntimeValue._null_zeros` and
`DivisionBinary` both ask "is there a zero in this column" before deciding
whether a division needs a validity bitmap at all, and a plan that divides
asks it once per batch. It was a scalar loop until 2026-09-04; the scalar form
cost +45% on `bench_floordiv_int32_*`, which is what these exist to keep from
coming back.

Two shapes, because the loop has an early exit and they answer differently:
a **miss** scans the whole view and is what a column with no zero costs, which
is the common case; a **hit at the front** returns from the first chunk and is
what the loop saves when there is one.

Run with: pixi run -e dev pytest marrow/tests/bench_views.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..buffers import Buffer
from ..utils.testing import Benchmark


def _ones(n: Int) raises -> Buffer[mut=True]:
    var buf = Buffer.alloc_zeroed[DType.int32](n)
    for i in range(n):
        buf.unsafe_set[DType.int32](i, Int32(1))
    return buf^


def _bench_miss(mut b: Benchmark, n: Int) raises:
    """No match: the whole view is scanned."""
    var buf = _ones(n)
    var view = buf.view[DType.int32]()
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() {imm}:
        keep(0 in view)

    b.iter(call)
    keep(buf)


def bench_contains_miss_10k(mut b: Benchmark) raises:
    _bench_miss(b, 10_000)


def bench_contains_miss_100k(mut b: Benchmark) raises:
    _bench_miss(b, 100_000)


def bench_contains_miss_1m(mut b: Benchmark) raises:
    _bench_miss(b, 1_000_000)


def bench_contains_hit_first_chunk_1m(mut b: Benchmark) raises:
    """A match in element 0 — what the early exit is worth against the miss
    above at the same size."""
    var buf = _ones(1_000_000)
    buf.unsafe_set[DType.int32](0, Int32(0))
    var view = buf.view[DType.int32]()
    b.throughput(BenchMetric.elements, 1_000_000)

    @always_inline
    def call() {imm}:
        keep(0 in view)

    b.iter(call)
    keep(buf)
