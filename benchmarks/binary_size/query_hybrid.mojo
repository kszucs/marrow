"""Binary-size demo: hybrid -- runtime relational plan, AOT-fused predicate.

Same query as `query_comptime.mojo` / `query_runtime.mojo`:

    SELECT a, name FROM orders WHERE a > b

The relational *structure* (`Scan`/`Filter`/`Project` via `AnyRelation`,
`Planner.build()`, the pull-based `RelationProcessor` pipeline) stays fully
runtime/type-erased -- exactly like `query_runtime.mojo`, and just as capable
of being built dynamically (parsed SQL, a Python-driven plan, ...). The
*predicate* itself, though, is a comptime-typed `Gt(NumericColumn, NumericColumn)` node
(`marrow.expr.values`) boxed into a runtime `DynValue` via the `FUSED` tag --
`DynValue(gt_node)` -- so evaluating it is a direct call into the fused
vectorize loop, not a walk through `DynValue.eval()`'s full tag-dispatch
interpreter.

This sits in between the other two files: the executor/`AnyRelation`
machinery this query touches (`Filter`, `Project`, `Scan`, `Planner`) is
still linked in wholesale, same as `query_runtime.mojo` -- but the predicate
itself doesn't pull in `DynValue.eval()`'s other op branches (`ADD`, `SUB`,
`AND`, `IF_ELSE`, ...), since it's boxed rather than built from `col(...) >
col(...)`.

Build + strip + compare against the other two:

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import record_batch
from marrow.expr import DynValue, in_memory_table, execute
from marrow.expr.values import NumericColumn, Gt


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var predicate = DynValue(Gt(NumericColumn[Int64Type](0), NumericColumn[Int64Type](1)))
    var plan = in_memory_table(batch).filter(predicate).select("a", "name")
    var result = execute(plan)
    print(result)
