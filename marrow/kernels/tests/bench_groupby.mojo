"""Benchmarks for `HashGrouping` — serial versus radix-partitioned placement.

Run with:
    pixi run -e dev pytest marrow/kernels/tests/bench_groupby.mojo --benchmark

Every row groups the same 1M-row column; only the `ExecContext` differs, so a
`serial` row and a `par8` row of the same cardinality are directly comparable
and the speedup is the ratio between them. `worth_parallel` reads a forced
thread count as a budget, so `parallel(N)` really does run N workers here.

`bench_groupby_anchor_*` touches no group-by code at all — it is a raw
`SwissHashTable` insert. It is here to be *ignored*, which is the point: this
machine drifts up to ~8% per case, so a batch that reads as a uniform
regression should move the anchor too. Normalise against it before attributing
any delta to placement.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray, UInt64Array
from ...builders import Int32Builder, UInt64Builder, StringBuilder
from ...dtypes import uint64
from ...execution import ExecContext
from ...kernels.groupby import HashGrouping
from ...kernels.hashtable import SwissHashTable
from ...utils import RapidHash64
from ...utils.testing import Benchmark


comptime _N: Int = 1_000_000


def _int_keys(n: Int, card: Int) raises -> List[DynArray]:
    """`n` int32 keys over `card` distinct values, interleaved rather than
    blocked so the table is probed in a realistic order."""
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32((i * 7919) % card))
    var cols = List[DynArray]()
    cols.append(b.finish())
    return cols^


def _string_keys(n: Int, card: Int) raises -> List[DynArray]:
    var b = StringBuilder(n)
    for i in range(n):
        b.append(String("key-") + String((i * 7919) % card))
    var cols = List[DynArray]()
    cols.append(b.finish())
    return cols^


def _bench_group(
    mut b: Benchmark, var cols: List[DynArray], n: Int, var ctx: ExecContext
) raises:
    """One grouping per iteration — a fresh grouper, since `assign` accumulates.

    That includes constructing the 64 per-partition tables on the radix path,
    which is real per-query cost and should not be hidden from the measurement.
    """
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        var g = HashGrouping(ctx.copy())
        var groups = g.assign(cols.copy(), n)
        keep(groups.num_groups)

    b.iter(call)
    keep(cols)
    keep(ctx)


# ---------------------------------------------------------------------------
# Low cardinality — 1,000 groups. Almost every probe hits an existing key, so
# this is the case radix has the least to win and the most overhead to lose.
# ---------------------------------------------------------------------------


def bench_groupby_serial_1m_card1k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 1_000), _N, ExecContext.serial())


def bench_groupby_par2_1m_card1k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 1_000), _N, ExecContext.parallel(2))


def bench_groupby_par4_1m_card1k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 1_000), _N, ExecContext.parallel(4))


def bench_groupby_par8_1m_card1k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 1_000), _N, ExecContext.parallel(8))


# ---------------------------------------------------------------------------
# High cardinality — 500,000 groups over 1M rows. Insert-heavy: the table
# resizes repeatedly and every miss walks a ctrl group. This is what the
# partitioning is for.
# ---------------------------------------------------------------------------


def bench_groupby_serial_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.serial())


def bench_groupby_par2_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.parallel(2))


def bench_groupby_par4_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.parallel(4))


def bench_groupby_par8_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.parallel(8))


# ---------------------------------------------------------------------------
# String keys — hashing is a much larger share of the work here, and it used to
# run on the calling thread whatever context the caller passed.
# ---------------------------------------------------------------------------


def bench_groupby_string_serial_1m_card10k(mut b: Benchmark) raises:
    _bench_group(b, _string_keys(_N, 10_000), _N, ExecContext.serial())


def bench_groupby_string_par4_1m_card10k(mut b: Benchmark) raises:
    _bench_group(b, _string_keys(_N, 10_000), _N, ExecContext.parallel(4))


def bench_groupby_string_par8_1m_card10k(mut b: Benchmark) raises:
    _bench_group(b, _string_keys(_N, 10_000), _N, ExecContext.parallel(8))


# ---------------------------------------------------------------------------
# Drift anchor — no group-by code on this path. If this row moves, the batch
# moved; subtract it before reading anything above.
# ---------------------------------------------------------------------------


def bench_groupby_anchor_swiss_insert_1m(mut b: Benchmark) raises:
    var hb = UInt64Builder(capacity=_N)
    for i in range(_N):
        hb.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    var hashes = hb.finish()
    b.throughput(BenchMetric.elements, _N)

    @always_inline
    def call() raises {imm}:
        var t = SwissHashTable[RapidHash64]()
        var bids = t.insert_hashes(hashes, grow_adaptively=True)
        keep(t.num_keys())

    b.iter(call)
    keep(hashes)
