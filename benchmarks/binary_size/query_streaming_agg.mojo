"""Binary-size gate for aggregation with a **runtime-named** aggregate.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name` — a fused comptime key
(`col("name", string)`), but the aggregates resolved from a function *name*
through `RuntimeAggregate` (`marrow/expr/runtime/aggregates.mojo`), as the
Python / ibis frontend does. Resolution goes through `dispatch_agg`, which
switches on the name and then on the operand's runtime dtype, so every kernel
in the catalog and every dtype arm each one accepts stays reachable.

Why this file exists: `query_streaming.mojo` is filter+project only, so the AOT
size gate was blind to the aggregate path. Fusion monomorphises per
aggregate-set — exactly the change that can blow code size up — and without an
aggregate query in the gate that regression would go unnoticed.

Its pair is `query_streaming_agg_fused.mojo`, which expresses the same query
with comptime aggregates.

    pixi run binary_size

**Ported from the old expression package on 2026-08-29, and the pair now
measures more than it did.** The old package could hand a *fused* operand to a
runtime-named aggregate (`AggFunc("sum")` over a boxed `col("a", int64)`), so
the delta against the fused gate isolated the aggregate *identity* alone.
`RuntimeAggregate` stores its operand as a `RuntimeValue` and cannot take a
fused one — the node's own docstring says a caller who names its aggregate with
a string built its operand at run time too — so `col("a").sum()` here is
runtime in both respects. The delta against the fused gate therefore now
conflates the runtime aggregate identity with the runtime *operand*, and is
correspondingly larger — measured 2026-08-29 at 10,711,044 bytes of `__text`
against the fused gate's 1,466,328, with `marrow::expr::runtime` at 864 symbols
here and 0 there. The group key is still fused, so that half is unchanged. The
recorded baseline predates the port and is stale.
"""

from marrow.builders import array
from marrow.dtypes import int64, string
from marrow.expr.builders import col, table
from marrow.expr.logical import DynValue
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
        col("a").sum().alias("a"),
        col("b").min().alias("b"),
    ]
    print(table(batch^).aggregate(aggs^, keys^).execute())
