"""Binary-size demo: fused values through the fat-node relational layer.

Same query as the other variants (`SELECT a, name FROM orders WHERE a > b`),
built with `marrow.expr.relations` — the self-executing fat nodes (`InMemoryTable`
/`Filter`/`Project`, `pull()`-based, no `Planner`) over fused `AnyValue` values.
`collect()` uses the closed flat concat, so the fused path never links the open
`AnyBuilder`. This should land near `query_erased_aot`, not the runtime path.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.values import AnyValue
from marrow.expr.relations import InMemoryTable, Project, AnyRelation, execute


struct Orders:
    var a: Int64Type
    var b: Int64Type
    var name: StringType


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var t = Table[Orders]()
    var filtered = AnyRelation(InMemoryTable(batch=batch)).filter(
        AnyValue(Gt(t.a, t.b))
    )
    var values = List[AnyValue]()
    values.append(AnyValue(t.a))
    values.append(AnyValue(t.name))
    var proj = Project(
        input=filtered,
        names=["a", "name"],
        exprs_=values^,
        schema_=schema([field("a", int64), field("name", string)]),
    )
    print(execute(AnyRelation(proj^)))
