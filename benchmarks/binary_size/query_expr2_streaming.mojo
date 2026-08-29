"""Binary-size gate: filter + projection over two `int64` columns.

`SELECT a, b FROM orders WHERE a > b`, built through `marrow.expr`'s
plan-building verbs with fused comptime values. Only comptime nodes (`col`,
`Gt`) are boxed into `DynValue`, so the runtime lane and its per-dtype kernel
fanout are dead-code-eliminated.

**It is `query_streaming.mojo` with both projected columns numeric**, and that
is now its whole reason to exist. It was written when this directory measured
`marrow/expr2/` — the package that has since replaced the old one and taken
the name `marrow/expr/` — as the first gate that built anything from that
lane at all, back when `pixi run binary_size` reported ~0.00% no matter what
the expression rewrite did. `query_streaming.mojo` was ported onto the same
package on 2026-08-29, so the two now differ only in that one projects
`col("name", string)` where this projects `col("b", int64)`.

**Keep it numeric.** The docstring here used to claim a fused string column
"cannot be spelled" because the comptime `Column[T]` is bound on `NumericType`.
That is false: `StringColumn[T]` is a separate leaf and `col(name, string)`
returns one. What is true is that this gate's recorded baseline was measured
against the all-numeric program, so changing it to a string column would
silently invalidate the only number CI checks for it. The pair is worth
keeping either way — the delta against `query_streaming` is what a fused
*string* column costs over a fused numeric one, with everything else equal.

**`project`, not `select`.** `select` takes names only and builds *runtime*
column reads, which would link the very lane this gate exists to prove absent.
`project` takes fused values, so the projection stays in the comptime lane.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.`comptime`.numeric import Gt
from marrow.expr.logical import DynRelation, DynValue, InMemoryTable
from marrow.tabular import record_batch


def main() raises:
    var batch = record_batch(
        [array([1, 5, 3, 8, 2], int64), array([4, 4, 4, 4, 4], int64)],
        names=["a", "b"],
    )
    var orders: DynRelation = InMemoryTable(batch^)
    var values: List[DynValue] = [col("a", int64), col("b", int64)]
    print(
        orders.filter(Gt(col("a", int64), col("b", int64)))
        .project(["a", "b"], values^)
        .execute()
    )
