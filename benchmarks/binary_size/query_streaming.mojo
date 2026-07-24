"""Binary-size demo: fused values through the fat-node relational layer.

Same query as the other variants (`SELECT a, name FROM orders WHERE a > b`),
built with `marrow.expr.relations` — the self-executing fat nodes (`InMemoryTable`
/`Filter`/`Project`, `pull()`-based, no `Planner`) over fused `AnyValue` values.
Only fused comptime nodes (`col`/`>`) are boxed, so the `DynValue` interpreter and
its per-dtype kernel fanout are dead-code-eliminated — this should land near the
fused path, far below the runtime path. That delta is the unification's DCE proof.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import InMemoryTable, Project, AnyRelation, execute


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var filtered = AnyRelation(InMemoryTable(batch=batch)).filter(
        AnyValue(col("a", int64) > col("b", int64))
    )
    var values = List[AnyValue]()
    values.append(AnyValue(col("a", int64)))
    values.append(AnyValue(col("name", string)))
    var proj = Project(
        input=filtered,
        names=["a", "name"],
        values=values^,
        schema=schema([field("a", int64), field("name", string)]),
    )
    print(execute(AnyRelation(proj^)))
