"""Benchmarks for `HashGrouping` — serial versus radix-partitioned placement.

Run with:
    pixi run -e dev pytest marrow/kernels/tests/bench_groupby.mojo --benchmark

Every row groups the same column; only the `ExecContext` differs, so a `serial`
row and a `par8` row of the same cardinality are directly comparable and the
speedup is the ratio between them.

**`parallel(N)` does not bound the radix path to N workers, so read the `parN`
rows as one "parallel" data point and not as a scaling curve.** `ctx` governs
only the *striped* phases — key hashing, the radix histogram and scatter, and
the closing `take` — because `ExecContext.stripe` sets concurrency by choosing
`resolved_num_threads()` work items. The dominant phase does not go through
`stripe`: the 64 per-partition `SwissHashTable` inserts are dispatched as
`sync_parallelize(worker, 64)` in `RadixPartitioner.map_partitions`, and the id
write-back as `sync_parallelize(finish_partition, 64)` below it. Both hand 64
work items to the global runtime pool, which sizes itself and never sees `ctx`.

**That split is intended, not a defect to be fixed here.** A partitioned phase
is sized by the partition count, and the runtime owns how many threads drain a
64-item queue; threading `ctx` into `map_partitions` would put a second thread
budget next to the one the pool already keeps, for the join as well as for the
group-by. The consequence to know about is that a thread-limited host is
oversubscribed by this path regardless of what `ctx` says.

The measurement says so on its own: `par2_10m_card5m` runs at **2.82x** serial,
and a genuine two-worker budget cannot exceed 2.00x. That is also why the sweep
is flat from `par4` on — every `parN` row already inserts at full machine width,
and only the striped remainder responds to N.

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

comptime _N10: Int = 10_000_000
"""The scaling tier. 1M rows leaves too little work per worker for the thread
count to matter — see the `_10m_` rows below."""


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


def bench_groupby_par12_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.parallel(12))


def bench_groupby_par16_1m_card500k(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N, 500_000), _N, ExecContext.parallel(16))


# ---------------------------------------------------------------------------
# Scaling tier — 10M rows, 5M groups. Same shape as the 1M/500k rows above at
# ten times the size, which is what tells a *fixed* per-grouping cost (the 64
# tables, the thread pool, the O(groups) key materialisation at the end) apart
# from one that scales with rows.
#
# This machine is an M4 Max: 12 performance cores and 4 efficiency ones, so
# `par16` is not 16 equal workers and a drop from `par12` to `par16` is the
# topology, not the algorithm.
# ---------------------------------------------------------------------------


def bench_groupby_serial_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.serial())


def bench_groupby_par2_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.parallel(2))


def bench_groupby_par4_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.parallel(4))


def bench_groupby_par8_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.parallel(8))


def bench_groupby_par12_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.parallel(12))


def bench_groupby_par16_10m_card5m(mut b: Benchmark) raises:
    _bench_group(b, _int_keys(_N10, 5_000_000), _N10, ExecContext.parallel(16))


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
        _ = t.insert_hashes(hashes, grow_adaptively=True)
        keep(t.num_keys())

    b.iter(call)
    keep(hashes)
