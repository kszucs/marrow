# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for the order kernels in `kernels/scan.mojo`.

- `bench_sum_scan_int64_1m`: `SumScan().run` over a column of ones into a
  second buffer, compared against `bench_views.bench_scan_int64_1m`, the
  in-place running sum it replaced.
- `bench_cumulative_sum_int64_*`: `cumulative_sum` over a whole column, new.

Run with: pixi run -e dev pytest marrow/kernels/tests/bench_scan.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...buffers import Buffer
from ...builders import arange
from ...dtypes import Int64Type
from ...kernels.numeric import AddKernel
from ...kernels.scan import CumulativeKernel, SumScan
from ...kernels.segments import BySize
from ...utils.testing import Benchmark


def _ones_int64(n: Int) raises -> Buffer[mut=True]:
    var buf = Buffer.alloc_zeroed[DType.int64](n)
    for i in range(n):
        buf.unsafe_set[DType.int64](i, 1)
    return buf^


def bench_sum_scan_int64_1m(mut b: Benchmark) raises:
    comptime n = 1_000_000
    var src = _ones_int64(n)
    var dst = _ones_int64(n)
    b.throughput(BenchMetric.elements, n)

    # `run` reads one view and writes another: the origin checker refuses
    # one buffer as both.
    @always_inline
    def call() {imm src, mut dst}:
        var out = dst.view[DType.int64](0, n)
        SumScan().run(src.view[DType.int64](0, n), out, BySize.whole(n))
        keep(out[n - 1])

    b.iter(call)
    keep(src)
    keep(dst)


def _bench_cumulative(mut b: Benchmark, n: Int) raises:
    var values = arange[Int64Type](0, n)
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(
            CumulativeKernel[AddKernel]
            .apply(values, False, BySize.whole(n))
            .unsafe_get(n - 1)
        )

    b.iter(call)
    keep(values)


def bench_cumulative_sum_int64_1k(mut b: Benchmark) raises:
    _bench_cumulative(b, 1_000)


def bench_cumulative_sum_int64_64k(mut b: Benchmark) raises:
    _bench_cumulative(b, 64_000)


def bench_cumulative_sum_int64_1m(mut b: Benchmark) raises:
    _bench_cumulative(b, 1_000_000)
