# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""How the engine scales: each query run by `Driver.dynamic()` on 1, 2, 4, 8
and 12 threads, as contenders `w<threads>` over the same in-memory batch of
`ROWS` rows, read in morsels.

Run with `pixi run -e dev bench-scaling`, which records the comparison in
`docs/data/scaling.json` for `docs/reference/engine.qmd`. Each 12-thread
contender also carries the query's `analyze` report as JSON in its
`extra_info`, so a regression in a snapshot can be traced to a phase and a
stage.
"""

from std.benchmark import BenchMetric, keep

from ...builders import array
from ...dtypes import Float64Type, Int64Type, float64, int64
from ...engine import Driver
from ...execution import ExecContext
from ...tabular import RecordBatch, record_batch
from ...utils.testing import Benchmark
from ..builders import col, count_star, lit, table
from ..logical import DynRelation
from ..optimizer import AllRules


comptime ROWS = 8_000_000


def _batch() raises -> RecordBatch:
    """A key `g` of 100,000 groups, a key `g1m` of 1,000,000, a key `g16` of
    16, an integer `a` and a float `x`; co-prime moduli, so a filter keeps a
    scattered subset."""
    var g = List[Int64](capacity=ROWS)
    var g1m = List[Int64](capacity=ROWS)
    var g16 = List[Int64](capacity=ROWS)
    var a = List[Int64](capacity=ROWS)
    var x = List[Float64](capacity=ROWS)
    for i in range(ROWS):
        g.append(Int64((i * 7_919) % 100_000))
        g1m.append(Int64((i * 7_919) % 1_000_000))
        g16.append(Int64(i % 16))
        a.append(Int64(i % 1_009))
        x.append(Float64(i % 101) * 0.5)
    return record_batch(
        [
            array[Int64Type](g, int64).to_dyn(),
            array[Int64Type](g1m, int64).to_dyn(),
            array[Int64Type](g16, int64).to_dyn(),
            array[Int64Type](a, int64).to_dyn(),
            array[Float64Type](x, float64).to_dyn(),
        ],
        names=["g", "g1m", "g16", "a", "x"],
    )


def _bench(mut b: Benchmark, threads: Int, plan: DynRelation) raises:
    """Time lowering and draining `plan` on `threads`."""
    var ctx = ExecContext.parallel(threads)
    b.throughput(BenchMetric.elements, ROWS)
    b.extra_info("lib", String("w", threads))
    if threads == 12:
        b.extra_info("report", plan.profile(ctx).to_json())

    @always_inline
    def call() raises {imm}:
        var driver = Driver.dynamic(ctx)
        keep(len(driver.collect(plan.lower(driver), plan.schema())))

    b.iter(call)
    keep(plan)


def _filter(batch: RecordBatch) raises -> DynRelation:
    return (
        table(batch.copy())
        .filter(col("a", int64) > lit(500, int64))
        .select("g", "x")
    )


def _sum16(batch: RecordBatch) raises -> DynRelation:
    return table(batch.copy()).aggregate(
        [col("x", float64).sum().alias("s"), count_star().alias("n")],
        [col("g16", int64)],
    )


def _sum100k(batch: RecordBatch) raises -> DynRelation:
    return table(batch.copy()).aggregate(
        [col("x", float64).sum().alias("s"), count_star().alias("n")],
        [col("g", int64)],
    )


def _sum1m(batch: RecordBatch) raises -> DynRelation:
    return table(batch.copy()).aggregate(
        [col("x", float64).sum().alias("s"), count_star().alias("n")],
        [col("g1m", int64)],
    )


def _join(batch: RecordBatch) raises -> DynRelation:
    var dim = table(batch.copy()).aggregate(
        [count_star().alias("n")], [col("g", int64)]
    )
    return table(batch.copy()).select("g", "a").join(dim^, [0], [0])


def _sort(batch: RecordBatch) raises -> DynRelation:
    return (
        table(batch.copy())
        .select("x", "a")
        .sort_by([col("x", float64)], [False])
    )


def _sort2(batch: RecordBatch) raises -> DynRelation:
    return (
        table(batch.copy())
        .select("g16", "x")
        .sort_by([col("g16", int64), col("x", float64)], [True, False])
    )


def _topn(batch: RecordBatch) raises -> DynRelation:
    return (
        table(batch.copy())
        .sort_by([col("x", float64)], [False])
        .limit(10)
        .optimize[AllRules]()
    )


def bench_w1_filter(mut b: Benchmark) raises:
    _bench(b, 1, _filter(_batch()))


def bench_w2_filter(mut b: Benchmark) raises:
    _bench(b, 2, _filter(_batch()))


def bench_w4_filter(mut b: Benchmark) raises:
    _bench(b, 4, _filter(_batch()))


def bench_w8_filter(mut b: Benchmark) raises:
    _bench(b, 8, _filter(_batch()))


def bench_w12_filter(mut b: Benchmark) raises:
    _bench(b, 12, _filter(_batch()))


def bench_w1_sum16(mut b: Benchmark) raises:
    _bench(b, 1, _sum16(_batch()))


def bench_w2_sum16(mut b: Benchmark) raises:
    _bench(b, 2, _sum16(_batch()))


def bench_w4_sum16(mut b: Benchmark) raises:
    _bench(b, 4, _sum16(_batch()))


def bench_w8_sum16(mut b: Benchmark) raises:
    _bench(b, 8, _sum16(_batch()))


def bench_w12_sum16(mut b: Benchmark) raises:
    _bench(b, 12, _sum16(_batch()))


def bench_w1_sum100k(mut b: Benchmark) raises:
    _bench(b, 1, _sum100k(_batch()))


def bench_w2_sum100k(mut b: Benchmark) raises:
    _bench(b, 2, _sum100k(_batch()))


def bench_w4_sum100k(mut b: Benchmark) raises:
    _bench(b, 4, _sum100k(_batch()))


def bench_w8_sum100k(mut b: Benchmark) raises:
    _bench(b, 8, _sum100k(_batch()))


def bench_w12_sum100k(mut b: Benchmark) raises:
    _bench(b, 12, _sum100k(_batch()))


def bench_w1_join(mut b: Benchmark) raises:
    _bench(b, 1, _join(_batch()))


def bench_w2_join(mut b: Benchmark) raises:
    _bench(b, 2, _join(_batch()))


def bench_w4_join(mut b: Benchmark) raises:
    _bench(b, 4, _join(_batch()))


def bench_w8_join(mut b: Benchmark) raises:
    _bench(b, 8, _join(_batch()))


def bench_w12_join(mut b: Benchmark) raises:
    _bench(b, 12, _join(_batch()))


def bench_w1_sort(mut b: Benchmark) raises:
    _bench(b, 1, _sort(_batch()))


def bench_w2_sort(mut b: Benchmark) raises:
    _bench(b, 2, _sort(_batch()))


def bench_w4_sort(mut b: Benchmark) raises:
    _bench(b, 4, _sort(_batch()))


def bench_w8_sort(mut b: Benchmark) raises:
    _bench(b, 8, _sort(_batch()))


def bench_w12_sort(mut b: Benchmark) raises:
    _bench(b, 12, _sort(_batch()))


def bench_w1_topn(mut b: Benchmark) raises:
    _bench(b, 1, _topn(_batch()))


def bench_w2_topn(mut b: Benchmark) raises:
    _bench(b, 2, _topn(_batch()))


def bench_w4_topn(mut b: Benchmark) raises:
    _bench(b, 4, _topn(_batch()))


def bench_w8_topn(mut b: Benchmark) raises:
    _bench(b, 8, _topn(_batch()))


def bench_w12_topn(mut b: Benchmark) raises:
    _bench(b, 12, _topn(_batch()))


def bench_w1_sum1m(mut b: Benchmark) raises:
    _bench(b, 1, _sum1m(_batch()))


def bench_w2_sum1m(mut b: Benchmark) raises:
    _bench(b, 2, _sum1m(_batch()))


def bench_w4_sum1m(mut b: Benchmark) raises:
    _bench(b, 4, _sum1m(_batch()))


def bench_w8_sum1m(mut b: Benchmark) raises:
    _bench(b, 8, _sum1m(_batch()))


def bench_w12_sum1m(mut b: Benchmark) raises:
    _bench(b, 12, _sum1m(_batch()))


def bench_w1_sort2(mut b: Benchmark) raises:
    _bench(b, 1, _sort2(_batch()))


def bench_w2_sort2(mut b: Benchmark) raises:
    _bench(b, 2, _sort2(_batch()))


def bench_w4_sort2(mut b: Benchmark) raises:
    _bench(b, 4, _sort2(_batch()))


def bench_w8_sort2(mut b: Benchmark) raises:
    _bench(b, 8, _sort2(_batch()))


def bench_w12_sort2(mut b: Benchmark) raises:
    _bench(b, 12, _sort2(_batch()))
