"""Binary-size demo: the fully-monomorphized (AOT) comptime relational layer.

Same query as `query_runtime.mojo`:

    SELECT a, name FROM orders WHERE a > b

built with `marrow.expr.relations` (`Table`, `Project`, `Filter`,
`NumericColumn`, `StringColumn`) instead of the type-erased `marrow.expr`
`AnyRelation`/`DynValue` layer. No `DynValue.eval()` tag interpreter, no
`AnyRelation` vtable/trampolines, no `executor.mojo`
`Planner`/`RelationProcessor` pull pipeline get linked in at all — the whole
plan is one nested generic type, and `.execute(batch)` compiles straight to
column loads, a SIMD comparison, and a filter call.

Build + strip + compare against `query_runtime.mojo`:

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.relations import Table, Project
from marrow.expr.values import Gt


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
    var plan = Project(Tuple(t.a, t.name)).filter(Gt(t.a, t.b))
    var result = plan.execute(batch)
    print(result)
