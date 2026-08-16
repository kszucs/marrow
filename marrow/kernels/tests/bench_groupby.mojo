"""Benchmarks for the groupby kernel.

Run with:
    pixi run pytest marrow/kernels/tests/bench_groupby.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray
from ...builders import PrimitiveBuilder, Int32Builder, Float64Builder
from ...dtypes import int32, float64, Int32Type, Float64Type
from ...kernels.groupby import GroupBy
from ...kernels.aggregate import (
    Aggregation,
    SumKernel,
    MinKernel,
    MaxKernel,
    MeanKernel,
    NumericAgg,
    CountKernel,
    CountAgg,
)


from ...testing import Benchmark


def _make_keys(n: Int, num_groups: Int) raises -> DynArray:
    var b = Int32Builder(n)
    for i in range(n):
        b.append(Scalar[int32.native](i % num_groups))
    return b.finish()


def _make_vals(n: Int) raises -> DynArray:
    var b = Float64Builder(n)
    for i in range(n):
        b.append(Scalar[float64.native](Float64(i)))
    return b.finish()


def _bench_group_by[
    A: Aggregation
](mut b: Benchmark, n: Int, num_groups: Int = 10) raises:
    var keys = _make_keys(n, num_groups)
    var vals = A.from_any(_make_vals(n))
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(GroupBy(keys).aggregate[A](vals))

    b.iter(call)
    keep(keys)
    keep(vals)


# ---------------------------------------------------------------------------
# group_by sum — 10K / 100K / 1M rows
# ---------------------------------------------------------------------------


def bench_groupby_sum_10k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[SumKernel, Float64Type]](b, 10_000)


def bench_groupby_sum_100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[SumKernel, Float64Type]](b, 100_000)


def bench_groupby_sum_1m(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[SumKernel, Float64Type]](b, 1_000_000)


# ---------------------------------------------------------------------------
# group_by sum — 1M rows by cardinality.
#
# Cardinality, not row count, is what picks the execution strategy: g10 and g1k
# fold thread-local partials, g100k partitions by key hash. Without the g100k
# case the radix path is only reachable through the Python competition harness,
# where Python overhead and machine noise hide exactly the regressions this is
# meant to catch.
# ---------------------------------------------------------------------------


def bench_groupby_sum_1m_g1k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[SumKernel, Float64Type]](b, 1_000_000, 1_000)


def bench_groupby_sum_1m_g100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[SumKernel, Float64Type]](b, 1_000_000, 100_000)


def bench_groupby_mean_1m_g100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[MeanKernel, Float64Type]](b, 1_000_000, 100_000)


# ---------------------------------------------------------------------------
# group_by min / max / mean — 100K rows
# ---------------------------------------------------------------------------


def bench_groupby_min_100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[MinKernel, Float64Type]](b, 100_000)


def bench_groupby_max_100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[MaxKernel, Float64Type]](b, 100_000)


def bench_groupby_mean_100k(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[MeanKernel, Float64Type]](b, 100_000)


def _make_vals_nulls(n: Int) raises -> DynArray:
    """Values with every third row null — enough nulls that the validity check
    cannot be branch-predicted away."""
    var b = Float64Builder(n)
    for i in range(n):
        if i % 3 == 0:
            b.append_null()
        else:
            b.append(Scalar[float64.native](Float64(i)))
    return b.finish()


def _bench_group_by_nulls[
    A: Aggregation
](mut b: Benchmark, n: Int, num_groups: Int = 10) raises:
    var keys = _make_keys(n, num_groups)
    var vals = A.from_any(_make_vals_nulls(n))
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() raises {imm}:
        keep(GroupBy(keys).aggregate[A](vals))

    b.iter(call)
    keep(keys)
    keep(vals)


# ---------------------------------------------------------------------------
# grouped count — the A/B for Q7.3.
#
# Two implementations exist and the two expression lanes disagree about which
# to use: `CountValid.resolve` picks `NumericAgg[CountKernel, V]` for numeric
# columns, while the AOT lane uses `CountKernel.Grouped`, which is `CountAgg`.
#
# They are not obviously ordered. `CountAgg` takes a `DynArray` and calls
# `values.is_valid(i)` per row — erased dispatch — but skips it entirely when
# the column is null-free, in which case it never loads a value at all.
# `AggState` pays a typed validity check and does load the value. So null-free
# should favour `CountAgg` and nullable may favour `AggState`.
#
# Both live in this one binary so the harness interleaves them: measuring one
# variant, rebuilding, then measuring the other invents regressions that are not
# there.
#
# g100k, never g10 — cardinality picks the execution strategy.
# ---------------------------------------------------------------------------


def bench_groupby_count_1m_g100k_aggstate(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[CountKernel, Float64Type]](b, 1_000_000, 100_000)


def bench_groupby_count_1m_g100k_countagg(mut b: Benchmark) raises:
    _bench_group_by[CountAgg](b, 1_000_000, 100_000)


def bench_groupby_count_nulls_1m_g100k_aggstate(mut b: Benchmark) raises:
    _bench_group_by_nulls[NumericAgg[CountKernel, Float64Type]](
        b, 1_000_000, 100_000
    )


def bench_groupby_count_nulls_1m_g100k_countagg(mut b: Benchmark) raises:
    _bench_group_by_nulls[CountAgg](b, 1_000_000, 100_000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
