# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for the hash tables: a join build and probe (`JoinHashTable`),
and a batch insert (`SwissHashTable`). Each includes hashing the keys.

Run with:
    pixi run bench-mojo -k bench_hash_table
    pixi run pytest marrow/kernels/tests/bench_hash_table.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray
from ...builders import UInt64Builder
from ...dtypes import uint64
from ...kernels.hashtable import SwissHashTable
from ...kernels.join import JoinHashTable
from ...kernels.hashing import RapidHashKernel
from ...utils import RapidHash64
from ...utils.testing import Benchmark


def _make_keys(n: Int) raises -> List[DynArray]:
    """One key column of n distinct uint64 keys."""
    var b = UInt64Builder(capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    var keys = List[DynArray]()
    keys.append(b.finish().to_dyn())
    return keys^


def _join_table(keys: List[DynArray]) raises -> JoinHashTable[]:
    return JoinHashTable(keys)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def bench_hash_table_build_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)

    @always_inline
    def call() raises {imm}:
        var t = _join_table(keys)
        keep(t)

    b.iter(call)
    keep(keys)


def bench_hash_table_build_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)

    @always_inline
    def call() raises {imm}:
        var t = _join_table(keys)
        keep(t)

    b.iter(call)
    keep(keys)


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def bench_hash_table_insert_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)

    @always_inline
    def call() raises {imm}:
        var t = SwissHashTable()
        t.reserve(len(keys[0]))
        var placed = t.insert_hashes(RapidHashKernel.dispatch(keys[0]))
        keep(len(placed.ids))
        keep(len(t))

    b.iter(call)
    keep(keys)


def bench_hash_table_insert_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)

    @always_inline
    def call() raises {imm}:
        var t = SwissHashTable()
        t.reserve(len(keys[0]))
        var placed = t.insert_hashes(RapidHashKernel.dispatch(keys[0]))
        keep(len(placed.ids))
        keep(len(t))

    b.iter(call)
    keep(keys)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def bench_hash_table_probe_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)
    var table = _join_table(keys)

    @always_inline
    def call() raises {imm}:
        var pairs = table.candidates(keys)
        keep(len(pairs))

    b.iter(call)
    keep(keys)
    keep(table)


def bench_hash_table_probe_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)
    var table = _join_table(keys)

    @always_inline
    def call() raises {imm}:
        var pairs = table.candidates(keys)
        keep(len(pairs))

    b.iter(call)
    keep(keys)
    keep(table)


# ---------------------------------------------------------------------------
# probe — semi-join (single_match=True)
# ---------------------------------------------------------------------------


def bench_hash_table_probe_semi_100k(mut b: Benchmark) raises:
    var keys = _make_keys(100_000)
    var table = _join_table(keys)

    @always_inline
    def call() raises {imm}:
        var pairs = table.candidates(keys, single_match=True)
        keep(len(pairs))

    b.iter(call)
    keep(keys)
    keep(table)


def bench_hash_table_probe_semi_1m(mut b: Benchmark) raises:
    var keys = _make_keys(1_000_000)
    var table = _join_table(keys)

    @always_inline
    def call() raises {imm}:
        var pairs = table.candidates(keys, single_match=True)
        keep(len(pairs))

    b.iter(call)
    keep(keys)
    keep(table)
