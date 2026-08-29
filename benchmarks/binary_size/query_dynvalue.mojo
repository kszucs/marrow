"""Binary-size gate: runtime values through an explicitly built plan.

Same query as the other variants (`SELECT a, name FROM orders WHERE a > b`) and
the same logical nodes as `query_streaming.mojo` — but the values are erased
(`RuntimeValue`, built by `col(name)` with no dtype and compared with `gt`)
rather than fused. Constructing one links the interpreter and, behind it, the
per-dtype kernel fanout; `query_streaming`, which boxes only comptime nodes,
stays small. That delta is the erasure boundary's DCE proof.

**This is the gate that watches `marrow.kernels.cast`.** The four gates that
predate it link *zero* symbols from it, so the entire cast family was ungated
and a change that added +435,072 bytes here measured 0.00% on everything CI
checked. The interpreter reaches `cast` from `RuntimeValue`'s binary arms,
which widen the narrower operand before applying the kernel.

Its near-twin is `query_runtime.mojo`, which builds the *same* erased values
through the fluent verbs and `select`. The two are deliberately close: what
separates them is how the projection is spelled, not what it computes.

    pixi run binary_size

**Ported from the old expression package on 2026-08-29.** Two spelling
changes, neither of which alters what is measured: the runtime lane has no
operator sugar (`gt(l, r)`, not `l > r`), and `Project` derives its output
schema rather than taking one — see `query_streaming.mojo`. The recorded
baseline predates the port and is stale.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.logical import DynRelation, DynValue, InMemoryTable, Project
from marrow.expr.runtime.values import gt
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var filtered = DynRelation(InMemoryTable(batch^)).filter(
        gt(col("a"), col("b"))
    )
    var values: List[DynValue] = [col("a"), col("name")]
    var proj = Project(filtered^, ["a", "name"], values^)
    print(DynRelation(proj^).execute())
