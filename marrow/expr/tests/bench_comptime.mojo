# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The comptime lane against the runtime lane: each query as two contenders,
`bench_comptime_<op>` and `bench_runtime_<op>`, over the same batch.

Run with `pixi run -e dev bench-comptime`; `docs/reference/comptime.qmd` says
what is compared and why.
"""

from std.benchmark import BenchMetric, keep

from ...builders import array
from ...dtypes import Float64Type, Int64Type, float64, int64
from ...scalars import Float64Scalar, Int64Scalar
from ...tabular import RecordBatch, record_batch
from ...utils.testing import Benchmark
from ..builders import col, lit, table
from ..logical import DynRelation


comptime ROWS = 1_000_000


def _batch() raises -> RecordBatch:
    """Group keys `g1`, `g4`, `g100` (1, 4 and 100 groups), integer columns
    `a`, `b`, `c` and float columns `x`, `y`.

    Co-prime moduli, so no two columns move together and a filter keeps a
    scattered subset rather than a prefix.
    """
    var g1 = List[Int64](capacity=ROWS)
    var g4 = List[Int64](capacity=ROWS)
    var g100 = List[Int64](capacity=ROWS)
    var a = List[Int64](capacity=ROWS)
    var b = List[Int64](capacity=ROWS)
    var c = List[Int64](capacity=ROWS)
    var x = List[Float64](capacity=ROWS)
    var y = List[Float64](capacity=ROWS)
    for i in range(ROWS):
        g1.append(0)
        g4.append(Int64(i % 4))
        g100.append(Int64(i % 100))
        a.append(Int64(i % 1_009))
        b.append(Int64(100 + i % 23))
        c.append(Int64(i % 17))
        x.append(Float64(i % 101) * 0.5)
        y.append(Float64(i % 37) * 0.25)
    return record_batch(
        [
            array[Int64Type](g1^, int64).to_dyn(),
            array[Int64Type](g4^, int64).to_dyn(),
            array[Int64Type](g100^, int64).to_dyn(),
            array[Int64Type](a^, int64).to_dyn(),
            array[Int64Type](b^, int64).to_dyn(),
            array[Int64Type](c^, int64).to_dyn(),
            array[Float64Type](x^, float64).to_dyn(),
            array[Float64Type](y^, float64).to_dyn(),
        ],
        names=["g1", "g4", "g100", "a", "b", "c", "x", "y"],
    )


def _bench[
    P: def(RecordBatch) raises -> DynRelation
](mut b: Benchmark, lib: String, plan: P) raises:
    """Time building, lowering and draining `plan` over the batch, as `lib`."""
    var batch = _batch()
    b.throughput(BenchMetric.elements, ROWS)
    b.extra_info("lib", lib)

    @always_inline
    def call() raises {imm}:
        keep(plan(batch).execute().num_rows())

    b.iter(call)
    keep(batch)


# ---------------------------------------------------------------------------
# project — an expression per row, no rows dropped
# ---------------------------------------------------------------------------
def bench_comptime_project_int_arithmetic(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).project(
            ["r"], [col("a", int64) * col("b", int64) + col("c", int64)]
        )

    _bench(b, "comptime", plan)


def bench_runtime_project_int_arithmetic(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).project(
            ["r"], [col("a") * col("b") + col("c")]
        )

    _bench(b, "runtime", plan)


def bench_comptime_project_float_arithmetic(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var x = col("x", float64)
        var y = col("y", float64)
        return table(batch.copy()).project(
            ["r"], [x * x + y * y - lit(1.5, float64)]
        )

    _bench(b, "comptime", plan)


def bench_runtime_project_float_arithmetic(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var x = col("x")
        var y = col("y")
        return table(batch.copy()).project(
            ["r"], [x * x + y * y - lit(Float64Scalar(1.5))]
        )

    _bench(b, "runtime", plan)


# ---------------------------------------------------------------------------
# filter — a predicate per row, then a gather of the survivors
# ---------------------------------------------------------------------------
def bench_comptime_filter_range(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var a = col("a", int64)
        return table(batch.copy()).filter(
            (a > lit(200, int64)) & (a < lit(800, int64))
        )

    _bench(b, "comptime", plan)


def bench_runtime_filter_range(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var a = col("a")
        return table(batch.copy()).filter(
            (a > lit(Int64Scalar(200))) & (a < lit(Int64Scalar(800)))
        )

    _bench(b, "runtime", plan)


def bench_comptime_filter_computed(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).filter(
            (col("a", int64) * col("b", int64) > lit(50_000, int64))
            | (col("c", int64) < lit(3, int64))
        )

    _bench(b, "comptime", plan)


def bench_runtime_filter_computed(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).filter(
            (col("a") * col("b") > lit(Int64Scalar(50_000)))
            | (col("c") < lit(Int64Scalar(3)))
        )

    _bench(b, "runtime", plan)


# ---------------------------------------------------------------------------
# aggregate — whole-table and grouped
#
# Each comptime plan asserts `fuses`, so this file does not compile if the
# aggregate it times would run through the buffered (materialising) operator
# rather than `RegisterAggregateOperator` (whole-table) or
# `ScatteredAggregateOperator` (grouped). The runtime lane never fuses.
# ---------------------------------------------------------------------------
def bench_comptime_aggregate_sum(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var agg = (col("a", int64) * col("b", int64)).sum().alias("r")
        comptime assert type_of(agg).fuses, "sum(a * b) must fuse"
        return table(batch.copy()).aggregate(aggs=[agg^])

    _bench(b, "comptime", plan)


def bench_runtime_aggregate_sum(mut b: Benchmark) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).aggregate(
            aggs=[(col("a") * col("b")).sum().alias("r")]
        )

    _bench(b, "runtime", plan)


def _sum_grouped_comptime(mut b: Benchmark, key: String) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        var agg = (col("a", int64) * col("b", int64)).sum().alias("r")
        comptime assert type_of(agg).fuses, "sum(a * b) must fuse"
        return table(batch.copy()).aggregate([agg^], [col(key.copy(), int64)])

    _bench(b, "comptime", plan)


def _sum_grouped_runtime(mut b: Benchmark, key: String) raises:
    def plan(batch: RecordBatch) raises {imm} -> DynRelation:
        return table(batch.copy()).aggregate(
            [(col("a") * col("b")).sum().alias("r")], [col(key.copy())]
        )

    _bench(b, "runtime", plan)


def bench_comptime_aggregate_sum_grouped_1(mut b: Benchmark) raises:
    _sum_grouped_comptime(b, "g1")


def bench_runtime_aggregate_sum_grouped_1(mut b: Benchmark) raises:
    _sum_grouped_runtime(b, "g1")


def bench_comptime_aggregate_sum_grouped_4(mut b: Benchmark) raises:
    _sum_grouped_comptime(b, "g4")


def bench_runtime_aggregate_sum_grouped_4(mut b: Benchmark) raises:
    _sum_grouped_runtime(b, "g4")


def bench_comptime_aggregate_sum_grouped_100(mut b: Benchmark) raises:
    _sum_grouped_comptime(b, "g100")


def bench_runtime_aggregate_sum_grouped_100(mut b: Benchmark) raises:
    _sum_grouped_runtime(b, "g100")
