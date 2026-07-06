"""Binary-size demo: the fully-monomorphized (AOT) comptime relational layer.

Same query as `query_runtime.mojo`:

    SELECT a, name FROM orders WHERE a > b

built with `marrow.aot.relations` (`Project`, `Filter`, `Column`,
`StringColumn`) instead of the type-erased `marrow.dyn`
`AnyRelation`/`Expr` layer. No `Expr.eval()` tag interpreter, no
`AnyRelation` vtable/trampolines, no `executor.mojo`
`Planner`/`RelationProcessor` pull pipeline get linked in at all — the whole
plan is one nested generic type, and `.execute(batch)` compiles straight to
column loads, a SIMD comparison, and a filter call.

Build + strip + compare against `query_runtime.mojo`:

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.aot.relations import Column, StringColumn, Table, Project
from marrow.aot.values import Gt


struct Orders(Table):
    var a: Column[Orders, "a", Int64Type]
    var b: Column[Orders, "b", Int64Type]
    var name: StringColumn[Orders, "name"]

    def __init__(out self):
        self.a = {}
        self.b = {}
        self.name = {}


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var t = Orders()
    var plan = Project(Tuple(t.a, t.name)).filter(Gt(t.a, t.b))
    var result = plan.execute(batch)
    print(result)
