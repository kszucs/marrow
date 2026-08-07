"""AOT-resolved aggregates against runtime-named ones, in one binary.

This is the measurement Q6.1 called missing, and it is the one that matters
most: marrow's claim is that resolving an aggregate at compile time is cheaper
than naming it at run time, and until now that claim was only ever checked by
*binary size* (`benchmarks/binary_size/query_streaming_agg{,_fused}.mojo`). Size
says the interpreted machinery is absent from the fused binary. It says nothing
about whether the fused one is faster, which is the part a user cares about.

The two paths differ in exactly one thing, which is what makes the comparison
honest:

    AggFunc.of[NumericAgg[SumKernel, Int64Type]](int64)   # kernel + dtype comptime
    AggFunc("sum", int64)                                 # both resolved at run time

Everything else — the plan, the grouper, the input batch, the number of
aggregates — is identical, and both live in this one file so the harness builds
them into a single binary and interleaves them. Measuring one variant, rebuilding
and measuring the other is the trap `docs/backlog.md` §0 names: it invents
regressions that are not there. That mistake was made on the sort work in this
repo and produced two phantom regressions.

**Read the group count, not just the row count.** Cardinality picks the
execution strategy, so `g100k` exercises the radix path and `g10` the
thread-local fold. §0 says to validate aggregate work at `g100k`, never `g10` —
at ten groups the per-group work is a rounding error and both paths look the
same.
"""

from std.benchmark import BenchMetric, keep

from ...testing import Benchmark
from ...arrays import DynArray
from ...builders import array, Int64Builder, StringBuilder
from ...dtypes import DynType, Int64Type, int64, string, field
from ...schema import schema
from ...tabular import record_batch, RecordBatch
from ...kernels.aggregate import NumericAgg, SumKernel, MinKernel
from ...expr.aggregates import AggFunc
from ...expr.values import col
from ...expr.relations import (
    BoxedValue,
    InMemoryTable,
    Aggregate,
    DynRelation,
)


def _batch(n: Int, num_groups: Int) raises -> RecordBatch:
    """`n` rows over `num_groups` distinct string keys."""
    var a = Int64Builder(n)
    var b = Int64Builder(n)
    var nm = StringBuilder(n)
    for i in range(n):
        a.append(Scalar[int64.native](i))
        b.append(Scalar[int64.native](n - i))
        nm.append(String("k", i % num_groups))
    return record_batch(
        [a.finish().to_dyn(), b.finish().to_dyn(), nm.finish().to_dyn()],
        names=["a", "b", "name"],
    )


def _plan(
    var batch: RecordBatch, var funcs: List[AggFunc]
) raises -> DynRelation:
    """`SELECT name, sum(a), min(b) FROM t GROUP BY name` over `funcs`."""
    var keys = List[BoxedValue]()
    keys.append(BoxedValue(col("name", string)))

    var inputs = List[BoxedValue]()
    inputs.append(BoxedValue(col("a", int64)))
    inputs.append(BoxedValue(col("b", int64)))

    return DynRelation(
        Aggregate(
            input=DynRelation(InMemoryTable(batch=batch^)),
            keys=keys^,
            inputs=inputs^,
            aggs=funcs^,
            schema=schema(
                [field("name", string), field("a", int64), field("b", int64)]
            ),
        )
    )


def _fused_funcs() raises -> List[AggFunc]:
    var funcs = List[AggFunc]()
    funcs.append(AggFunc.of[NumericAgg[SumKernel, Int64Type]](DynType(int64)))
    funcs.append(AggFunc.of[NumericAgg[MinKernel, Int64Type]](DynType(int64)))
    return funcs^


def _named_funcs() raises -> List[AggFunc]:
    var funcs = List[AggFunc]()
    funcs.append(AggFunc("sum", DynType(int64)))
    funcs.append(AggFunc("min", DynType(int64)))
    return funcs^


def _bench_fused(mut bm: Benchmark, n: Int, num_groups: Int) raises:
    var batch = _batch(n, num_groups)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(_plan(batch.copy(), _fused_funcs()).execute())

    bm.iter[call]()
    keep(batch)


def _bench_named(mut bm: Benchmark, n: Int, num_groups: Int) raises:
    var batch = _batch(n, num_groups)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(_plan(batch.copy(), _named_funcs()).execute())

    bm.iter[call]()
    keep(batch)


# ---------------------------------------------------------------------------
# g100k — the radix path, and the case §0 says to validate on.
# ---------------------------------------------------------------------------


def bench_agg_aot_1m_g100k_fused(mut b: Benchmark) raises:
    _bench_fused(b, 1_000_000, 100_000)


def bench_agg_aot_1m_g100k_named(mut b: Benchmark) raises:
    _bench_named(b, 1_000_000, 100_000)


# ---------------------------------------------------------------------------
# g1k — the thread-local fold, where per-group work is amortised over more rows.
# Kept because the two strategies can order differently, which is exactly the
# kind of thing a single data point hides.
# ---------------------------------------------------------------------------


def bench_agg_aot_1m_g1k_fused(mut b: Benchmark) raises:
    _bench_fused(b, 1_000_000, 1_000)


def bench_agg_aot_1m_g1k_named(mut b: Benchmark) raises:
    _bench_named(b, 1_000_000, 1_000)
