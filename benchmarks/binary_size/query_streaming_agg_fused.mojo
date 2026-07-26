"""Binary-size gate for a fully **fused** aggregation.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name` — the same query as
`query_streaming_agg.mojo`, but the aggregates are `AggFunc.of[NumericAgg[K, V]]()`:
kernel *and* input dtype are comptime, so the plan holds a direct pointer to
`AggState[SumKernel, Int64Type]` / `AggState[MinKernel, Int64Type]`.

Nothing interprets an aggregate here — no function-name switch, no per-dtype
`dispatch_numeric` ladder over six kernels. Everything the runtime-named variant
must keep alive is dead code in this binary, and the delta between the two
stripped sizes is precisely the cost of a runtime aggregate identity.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import AnyDataType, Int64Type, int64, string, field
from marrow.kernels.aggregate import NumericAgg, SumKernel, MinKernel
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.aggregates import Aggregates
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import InMemoryTable, Aggregate, AnyRelation, execute


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "p", "q", "p"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var keys = List[AnyValue]()
    keys.append(AnyValue(col("name", string)))

    var aggs = List[AnyValue]()
    aggs.append(AnyValue(col("a", int64)))
    aggs.append(AnyValue(col("b", int64)))

    var funcs = Aggregates()
    funcs.append[NumericAgg[SumKernel, Int64Type]](AnyDataType(int64))
    funcs.append[NumericAgg[MinKernel, Int64Type]](AnyDataType(int64))

    var agg = Aggregate(
        input=AnyRelation(InMemoryTable(batch=batch)),
        keys=keys^,
        inputs=aggs^,
        aggs=funcs^,
        schema=schema(
            [field("name", string), field("a", int64), field("b", int64)]
        ),
    )
    print(execute(AnyRelation(agg^)))
