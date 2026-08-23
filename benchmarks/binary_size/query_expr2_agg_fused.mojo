"""Binary-size gate for a fully **fused** `expr2` aggregation.

`SELECT g, sum(a), min(b) FROM orders GROUP BY g` — the `expr2` counterpart of
`query_streaming_agg_fused.mojo`. Keys and aggregate inputs are comptime
`Column[Int64Type]` nodes and the aggregates are `Sum` / `Min`, so kernel *and*
input dtype are known at compile time: the plan holds a direct
`AggState[SumKernel, Int64Type]` / `AggState[MinKernel, Int64Type]` and nothing
interprets an aggregate at run time.

**Why this file exists.** Before it, `benchmarks/binary_size/` gated five
programs and **not one of them built anything from `expr2`**, so
`pixi run binary_size` reported ~0.00% no matter what the expression rewrite
did. Every size claim about that work was unfalsifiable. This is the gate that
makes the aggregation path measurable; `query_expr2_streaming.mojo` covers
filter and projection.

It differs from its `expr/` twin in one respect worth stating rather than
hiding: the group key here is `int64`, not `string`. `expr2`'s comptime
`Column[T]` is bound on `NumericType`, so a fused string key cannot be spelled
yet. The two numbers are therefore **not** directly comparable across packages;
this gate's job is to catch `expr2` regressing against *itself*.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr2.builders import col
from marrow.expr2.`comptime`.aggregates import Min, Sum
from marrow.expr2.logical import DynValue
from marrow.expr2.logical import Aggregate, DynRelation, InMemoryTable
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var g = array([1, 2, 1, 2, 1], int64)
    var batch = record_batch(
        [a.copy(), b.copy(), g.copy()], names=["a", "b", "g"]
    )

    var keys = List[DynValue]()
    keys.append(DynValue(col("g", int64)))

    var aggs = List[DynValue]()
    aggs.append(DynValue(Sum(col("a", int64), "total")))
    aggs.append(DynValue(Min(col("b", int64), "smallest")))

    var agg = Aggregate(DynRelation(InMemoryTable(batch^)), keys^, aggs^)
    print(DynRelation(agg^).execute())
