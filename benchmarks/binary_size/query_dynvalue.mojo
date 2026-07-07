"""Binary-size demo: interpreter values through the *same* erased relations.

Same query as the other variants (`SELECT a, name WHERE a > b`) and the same
runtime, self-executing `marrow.expr.erased` `Project`/`Filter` as
`query_erased_aot.mojo` — but the values are boxed `DynValue` **interpreter**
nodes (tag-dispatched `to_array()`) built the Python way (`col(...)` +
operators), instead of fused nodes.

This is the other half of the unification's DCE proof: the relational ops are
identical to `query_erased_aot`, so any size difference is purely the value
representation. Constructing a `DynValue` links its interpreter (and the
per-dtype kernel fanout); `query_erased_aot`, which never constructs one, stays
tiny. If both hold, one `AnyValue` box serving fused and interpreted values is
confirmed size-safe.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.tabular import record_batch
from marrow.expr.runtime import col
from marrow.expr.erased import AnyValue, Project


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var exprs = List[AnyValue]()
    exprs.append(AnyValue(col("a")))
    exprs.append(AnyValue(col("name")))
    var predicate = AnyValue(col("a") > col("b"))
    var plan = Project(exprs^).filter(predicate^)
    var result = plan.execute(batch)
    print(result)
