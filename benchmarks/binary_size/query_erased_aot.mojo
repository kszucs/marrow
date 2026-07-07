"""Binary-size demo: erased-relation, fused-value AOT layer ("option 1").

Same query as `query_comptime.mojo` / `query_runtime.mojo`:

    SELECT a, name FROM orders WHERE a > b

built with `marrow.expr.erased` — a **runtime**, walkable plan tree
(`Project`/`Filter` as plain structs over `List[AnyValue]`, not a `*Es` type
pack) whose values are **fused-only** boxes (`AnyValue`, trampoline into each
node's own `execute()`, no `eval()` tag interpreter). The operators execute
themselves single-shot — no `Planner`/`RelationProcessor` open dispatch.

The question this file answers: does the fused-only value box + self-executing
runtime tree keep the binary near `query_comptime` (~250 KB, fully typed) even
though the plan is now a rewritable runtime object — or does erasing the
relational layer to runtime structs cost the ~30x like `query_runtime`?

Build + strip + compare against the other three:

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import record_batch
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.erased import AnyValue, Project


struct Orders:
    var a: Int64Type
    var b: Int64Type
    var name: StringType


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var t = Table[Orders]()
    var exprs = List[AnyValue]()
    exprs.append(AnyValue(t.a))
    exprs.append(AnyValue(t.name))
    var plan = Project(exprs^).filter(AnyValue(Gt(t.a, t.b)))
    var result = plan.execute(batch)
    print(result)
