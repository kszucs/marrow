"""Binary-size demo: interpreter values through the *same* erased relations.

Same query as the other variants (`SELECT a, name WHERE a > b`) and the same
runtime, self-executing `marrow.expr.erased` `Project`/`Filter` as
`query_erased_aot.mojo` — but the values are boxed `DynValue` **interpreter**
nodes (tag-dispatched `to_array()`) instead of fused nodes.

This is the other half of the unification's DCE proof: the relational ops are
identical to `query_erased_aot`, so any size difference is purely the value
representation. Constructing a `DynValue` links its interpreter (and the
`AnyArray`-dispatching `greater` kernel's per-dtype fanout); `query_erased_aot`,
which never constructs one, must stay tiny. If both hold, one `AnyValue` box
serving fused and interpreted values is confirmed size-safe.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.tabular import record_batch
from marrow.expr.erased import AnyValue, DynValue, Project


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var exprs = List[AnyValue]()
    exprs.append(AnyValue(DynValue.load(0, "a")))
    exprs.append(AnyValue(DynValue.load(2, "name")))
    var predicate = AnyValue(
        DynValue.gt(AnyValue(DynValue.load(0, "a")), AnyValue(DynValue.load(1, "b")))
    )
    var plan = Project(exprs^).filter(predicate^)
    var result = plan.execute(batch)
    print(result)
