"""Binary-size demo: the type-erased runtime relational layer, end to end.

    SELECT a, name FROM orders WHERE a > b

built with the `DynRelation`/`DynValue` layer
(`marrow.expr.relations`, `marrow.expr.dynamic`, `marrow.expr.execution`)
-- `in_memory_table(batch).filter(...).select(...)` then `plan.execute()`,
which builds each node's own processor via `Relation.to_operator()` into a
pull-based pipeline. Filter comes before select in the chain because
`DynRelation.filter()` resolves `col()` names against its *input*'s schema --
`b` must still be present when the predicate is resolved, so it has to run
before the projection drops it.

Build + strip + compare against the other gates:

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.tabular import record_batch
from marrow.expr import col, in_memory_table


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var plan = (
        in_memory_table(batch).filter(col("a") > col("b")).select("a", "name")
    )
    var result = plan.execute()
    print(result)
