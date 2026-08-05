"""Binary-size gate for *aggregation* with a **runtime-named** aggregate.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name` — fused comptime values
(`col`) for the key and the aggregate inputs, but the aggregate *identity*
resolved from a function name (`AggFunc("sum")`), as the Python / ibis frontend
does.

Why this file exists: `query_streaming.mojo` is filter+project only, so the AOT
size gate was blind to the aggregate path. Fusion monomorphises per
aggregate-set — exactly the change that can blow code size up — and without an
aggregate query in the gate that regression would go unnoticed.

Its pair, `query_streaming_agg_fused.mojo`, expresses the **same** query with
comptime aggregations (`AggFunc.of[NumericAgg[SumKernel, Int64Type]]()`). The delta between
the two is the measurement: it is exactly the cost of the aggregate identity
(and the input dtype) being runtime rather than comptime.

    pixi run binary_size
"""

from marrow.expr.relations import BoxedValue
from marrow.builders import array
from marrow.dtypes import DynType, int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.aggregates import AggFunc
from marrow.expr.values import col, DynValue
from marrow.expr.relations import InMemoryTable, Aggregate, DynRelation


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "p", "q", "p"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var keys = List[BoxedValue]()
    keys.append(BoxedValue(col("name", string)))

    var aggs = List[BoxedValue]()
    aggs.append(BoxedValue(col("a", int64)))
    aggs.append(BoxedValue(col("b", int64)))

    var funcs = List[AggFunc]()
    funcs.append(AggFunc("sum", DynType(int64)))
    funcs.append(AggFunc("min", DynType(int64)))

    var agg = Aggregate(
        input=DynRelation(InMemoryTable(batch=batch)),
        keys=keys^,
        inputs=aggs^,
        aggs=funcs^,
        schema=schema(
            [field("name", string), field("a", int64), field("b", int64)]
        ),
    )
    print(DynRelation(agg^).execute())
