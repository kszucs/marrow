"""Binary-size gate: the type-erased runtime lane, end to end.

    SELECT a, name FROM orders WHERE a > b

built entirely from `marrow.expr`'s runtime lane (`marrow/expr/runtime/`):
`table(batch).filter(gt(col("a"), col("b"))).select(["a", "name"])`, then
`plan.execute()`. Nothing here names a dtype, so nothing fuses — `col("a")` is
a `RuntimeValue`, the comparison is a tag the interpreter switches on, and
`select` builds runtime column reads of its own. That pulls in the per-dtype
kernel fanout the fused gates dead-code-eliminate, including the whole
`marrow.kernels.cast` ladder the interpreter uses to align operand widths.

The delta against `query_streaming.mojo` — the same query, fused — is the cost
of not knowing the dtypes at compile time.

    pixi run binary_size

**Ported from the old expression package on 2026-08-29.** One spelling
change: the runtime lane has no operator sugar, so the predicate is
`gt(col("a"), col("b"))` where the old package wrote `col("a") > col("b")`.
It is the same node.
`select` takes a list rather than varargs. The recorded baseline predates the
port and is stale.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr.builders import col, table
from marrow.expr.runtime.values import gt
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    # Filter before select: the predicate resolves `col()` names against its
    # *input*'s schema, so `b` must still be present when it runs.
    var plan = (
        table(batch^).filter(gt(col("a"), col("b"))).select(["a", "name"])
    )
    print(plan.execute())
