"""Binary-size demo: fused values through the fat-node relational layer.

Same query as the other variants (`SELECT a, name FROM orders WHERE a > b`),
built with `marrow.expr.relations` — the self-executing fat nodes (`InMemoryTable`
/`Filter`/`Project`, `pull()`-based, no `Planner`) over fused `AnyValue` values.
Only fused comptime nodes (`col`/`>`) are boxed, so the `DynValue` interpreter and
its per-dtype kernel fanout are dead-code-eliminated — this should land near the
fused path, far below the runtime path. That delta is the unification's DCE proof.

    pixi run binary_size

**These four gate programs are the one place that builds plan nodes directly,
and that is deliberate — do not "fix" them to use the plan-building API.**
Everywhere else (tests included) should use `in_memory_table(...).filter(...)
.project(...)`, because those verbs *derive* the output schema instead of taking
one. Deriving it means probing each expression against a 0-row batch, and the
probe is real code: converting these four measured **+16,528 bytes on this gate
alone** (1,307,624 -> 1,324,152, +1.26%), with `marrow::tabular` 8 -> 9 symbols,
`marrow::schema` 2 -> 3 and `marrow::expr::relations` 20 -> 24.

A size-critical AOT program legitimately skips that probe: it knows its own
output schema at compile time. So the gate keeps measuring the floor, and the
+16,528 is the standing measurement of what schema derivation costs. It is also
avoidable — a *fused* value's `OutType` is statically known, so probing it by
execution is unnecessary in principle (see Q4.x in
`docs/code-quality-tasks.md`); if that lands, these can be converted.
"""

from marrow.builders import array
from marrow.dtypes import int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import InMemoryTable, Project, DynRelation


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var filtered = DynRelation(InMemoryTable(batch=batch)).filter(
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
    print(DynRelation(proj^).execute())
