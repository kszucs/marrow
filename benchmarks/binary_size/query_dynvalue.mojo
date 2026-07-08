"""Binary-size demo: interpreter values through the fat-node relational layer.

Same query as the other variants (`SELECT a, name WHERE a > b`), same
self-executing `marrow.expr.relations` fat nodes as `query_streaming` — but the
values are `DynValue` interpreter nodes (built the Python way via `col()` +
operators), boxed into `AnyValue`. Constructing a `DynValue` links its
interpreter (and the per-dtype kernel fanout); `query_streaming`, which only
boxes fused nodes, stays tiny. That delta is the unification's DCE proof.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.runtime import col
from marrow.expr.values import AnyValue
from marrow.expr.relations import InMemoryTable, Project, AnyRelation, execute


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var filtered = AnyRelation(InMemoryTable(batch=batch)).filter(
        AnyValue(col("a") > col("b"))
    )
    var values = List[AnyValue]()
    values.append(AnyValue(col("a")))
    values.append(AnyValue(col("name")))
    var proj = Project(
        input=filtered,
        names=["a", "name"],
        values=values^,
        schema=schema([field("a", int64), field("name", string)]),
    )
    print(execute(AnyRelation(proj^)))
