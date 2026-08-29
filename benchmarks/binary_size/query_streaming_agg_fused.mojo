"""Binary-size gate for a fully **fused** aggregation.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name` — the same query as
`query_streaming_agg.mojo`, but every value is comptime: the key is a
`StringColumn[StringType]`, and `col("a", int64).sum()` resolves to
`Aggregate[Fold[SumFold, Int64Type], Column[Int64Type]]`. Kernel *and* input
dtype are known at compile time, so the plan holds a direct
`AggState[SumFold, Int64Type]` / `AggState[MinFold, Int64Type]`.

Nothing interprets an aggregate here — no function-name switch, no per-dtype
`resolve_aggregate` ladder over ten kernels, and no erased operand. Everything the
runtime-named variant must keep alive is dead code in this binary, and the
`__text` delta between the two is the cost of resolving an aggregate at run
time.

    pixi run binary_size

**Ported from the old expression package on 2026-08-29; the recorded baseline
predates the port and is stale.** The aggregates are now spelled fluently
(`col("a", int64).sum().alias(...)`) rather than by naming a kernel through
`AggFunc.of[Fold[SumFold, Int64Type]]()` — which is what CLAUDE.md mandates
and, more to the point, the only spelling that compiles: `Sum[A]` puts `A`
under a projection, so it is not inferrable from a constructor argument. The
resolved node is the same one. `Aggregate` also derives its output schema
rather than taking one.

Its near-twin `query_expr2_agg_fused.mojo` groups by an `int64` key because it
was written before `col(name, string)` existed; that is the only difference,
and it is kept so that gate's recorded number stays comparable.
"""

from marrow.builders import array
from marrow.dtypes import int64, string
from marrow.expr import col, table
from marrow.expr import DynValue
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "p", "q", "p"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var keys: List[DynValue] = [col("name", string)]
    var aggs: List[DynValue] = [
        col("a", int64).sum().alias("a"),
        col("b", int64).min().alias("b"),
    ]
    print(table(batch^).aggregate(aggs^, keys^).execute())
