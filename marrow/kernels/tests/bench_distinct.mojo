# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for `count_distinct` over a whole column: a million values at
low and high cardinality, fixed- and variable-width, on one thread and on
eight."""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray
from ...builders import Int64Builder, StringBuilder
from ...execution import ExecContext
from ...kernels.distinct import count_distinct
from ...utils.testing import Benchmark

comptime _N = 1_000_000


def _ints(card: Int) raises -> DynArray:
    var b = Int64Builder(capacity=_N)
    for i in range(_N):
        b.append(Int64((i * 7919) % card))
    return b.finish()


def _strings(card: Int) raises -> DynArray:
    var b = StringBuilder(_N)
    for i in range(_N):
        b.append(String("key-") + String((i * 7919) % card))
    return b.finish()


def _bench_count_distinct(
    mut b: Benchmark, var values: DynArray, threads: Int
) raises:
    b.throughput(BenchMetric.elements, len(values))
    var ctx = ExecContext.serial() if threads <= 1 else ExecContext.parallel(
        threads
    )

    @always_inline
    def call() raises {imm}:
        keep(count_distinct(values, ctx.copy()).value())

    b.iter(call)
    keep(values)
    keep(ctx)


def bench_count_distinct_int64_1m_card1k_t1(mut b: Benchmark) raises:
    _bench_count_distinct(b, _ints(1_000), 1)


def bench_count_distinct_int64_1m_card500k_t1(mut b: Benchmark) raises:
    _bench_count_distinct(b, _ints(500_000), 1)


def bench_count_distinct_int64_1m_card500k_t8(mut b: Benchmark) raises:
    _bench_count_distinct(b, _ints(500_000), 8)


def bench_count_distinct_string_1m_card10k_t1(mut b: Benchmark) raises:
    _bench_count_distinct(b, _strings(10_000), 1)
