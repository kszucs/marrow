# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for `is_in`: a million values probed against sets of different
sizes, for a fixed-width and a variable-width type."""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray
from ...builders import Int64Builder, StringBuilder
from ...execution import ExecContext
from ...kernels.membership import is_in
from ...utils.testing import Benchmark

comptime _N = 1_000_000


def _ints(n: Int, card: Int) raises -> DynArray:
    var b = Int64Builder(capacity=n)
    for i in range(n):
        b.append(Int64((i * 7919) % card))
    return b.finish()


def _strings(n: Int, card: Int) raises -> DynArray:
    var b = StringBuilder(n)
    for i in range(n):
        b.append(String("key-") + String((i * 7919) % card))
    return b.finish()


def _bench_is_in(
    mut b: Benchmark, var values: DynArray, var value_set: DynArray
) raises:
    b.throughput(BenchMetric.elements, len(values))

    @always_inline
    def call() raises {imm}:
        keep(len(is_in(values, value_set, ExecContext.serial())))

    b.iter(call)
    keep(values)
    keep(value_set)


def bench_is_in_int64_1m_set1k(mut b: Benchmark) raises:
    # Half the probed values are in the set.
    _bench_is_in(b, _ints(_N, 2_000), _ints(1_000, 1_000))


def bench_is_in_int64_1m_set100k(mut b: Benchmark) raises:
    _bench_is_in(b, _ints(_N, 200_000), _ints(100_000, 100_000))


def bench_is_in_string_1m_set1k(mut b: Benchmark) raises:
    _bench_is_in(b, _strings(_N, 2_000), _strings(1_000, 1_000))


def bench_is_in_string_1m_set100k(mut b: Benchmark) raises:
    _bench_is_in(b, _strings(_N, 200_000), _strings(100_000, 100_000))
