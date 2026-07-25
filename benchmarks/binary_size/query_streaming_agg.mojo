"""Binary-size gate for *aggregation* through the fused relational layer.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name`, built from fused
comptime values (`col`) over the self-executing fat nodes.

Why this file exists: `query_streaming.mojo` is filter+project only, so the
AOT size gate was blind to the aggregate path. Fusion monomorphises per
aggregate-set — exactly the change that can blow code size up — and without an
aggregate query in the gate that regression would go unnoticed.

Note the node is constructed directly rather than via `AnyRelation.aggregate()`:
that builder takes `List[DynValue]`, so it forces the runtime interpreter and
cannot express a fused aggregation. `Aggregate` itself holds `List[AnyValue]`,
so the node *can* carry fused values — only the builder cannot produce them.
Closing that F1/F2 gap is part of the aggregate work; until then, this file
documents it by construction.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
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

    var funcs = List[String]()
    funcs.append("sum")
    funcs.append("min")

    var agg = Aggregate(
        input=AnyRelation(InMemoryTable(batch=batch)),
        keys=keys^,
        aggs=aggs^,
        funcs=funcs^,
        schema=schema(
            [field("name", string), field("a", int64), field("b", int64)]
        ),
    )
    print(execute(AnyRelation(agg^)))
