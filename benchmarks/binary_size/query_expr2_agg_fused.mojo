"""Binary-size gate: a fully **fused** aggregation over an `int64` key.

`SELECT g, sum(a), min(b) FROM orders GROUP BY g`. Keys and aggregate inputs
are comptime `Column[Int64Type]` nodes and the aggregates are `Sum` / `Min`, so
kernel *and* input dtype are known at compile time: the plan holds a direct
`AggState[SumFold, Int64Type]` / `AggState[MinFold, Int64Type]` and nothing
interprets an aggregate at run time.

**It is `query_streaming_agg_fused.mojo` with a numeric group key**, and that
is now its whole reason to exist. It was written when this directory measured
`marrow/expr2/` — the package that has since replaced the old one and taken
the name `marrow/expr/` — as the gate that first made that lane's
aggregation path measurable at all. `query_streaming_agg_fused.mojo` was
ported onto the same package on 2026-08-29 and groups by `col("name", string)`,
so the key's dtype is the only difference left between the two.

**Keep the key numeric.** This docstring used to say a fused string key
"cannot be spelled" because the comptime `Column[T]` is bound on `NumericType`.
That is false — `StringColumn[T]` is a separate leaf and `col(name, string)`
returns one. What is true is that this gate's recorded baseline was measured
against the `int64` key, so changing it would silently invalidate the only
number CI checks for it. The pair earns its keep as a pair: the delta against
`query_streaming_agg_fused` is what grouping by a string costs over grouping by
an `int64`, with everything else equal.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr import col
from marrow.expr import Min, Sum
from marrow.expr import DynValue
from marrow.expr import Aggregate, DynRelation, InMemoryTable
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
    # The fluent spelling CLAUDE.md mandates, and the only one that compiles:
    # `Sum[A]` puts `A` under a projection (`Fold[SumFold, A.Type]`), so `A`
    # is not inferrable from a constructor argument. `col(...).sum()` names it.
    aggs.append(DynValue(col("a", int64).sum().alias("total")))
    aggs.append(DynValue(col("b", int64).min().alias("smallest")))

    var agg = Aggregate(DynRelation(InMemoryTable(batch^)), keys^, aggs^)
    print(DynRelation(agg^).execute())
