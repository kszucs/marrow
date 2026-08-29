"""Binary-size floor: fused values through the plan/operator layers.

`SELECT a, name FROM orders WHERE a > b`, built from `marrow.expr`'s logical
nodes (`InMemoryTable`/`Filter`/`Project`, `marrow/expr/logical.mojo`) and run
through the push operators (`marrow/expr/physical.mojo`) over fused comptime
values. Only comptime nodes (`col`, `Gt`) are boxed into `DynValue`, so the
runtime lane (`marrow/expr/runtime/`) and its per-dtype kernel fanout are
dead-code-eliminated — this should land far below `query_runtime` and
`query_dynvalue`, and that delta is the erasure boundary's DCE proof.

    pixi run binary_size

**Ported from the old expression package on 2026-08-29; the recorded
baseline predates the port and is stale.** Two things changed with the
package, both of which move the number:

- **`project` derives its output schema; it can no longer be handed one.**
  The old `Project` took a `schema=` argument, and these gate programs
  passed one deliberately: deriving it there meant probing every expression
  against a 0-row batch, which measured **+16,528 bytes on this gate alone**.
  `marrow.expr` derives it from `DynValue.dtype(schema)` — a type query, not an
  execution — so the probe, and the reason to bypass it, are both gone. The
  gate now measures the same floor the verbs give an ordinary caller.
- **The projected `name` column is fused.** `col("name", string)` is a
  `StringColumn[StringType]`, so both projected columns stay in the comptime
  lane. `query_expr2_streaming.mojo` — written before `StringColumn` had a
  `col` overload — projects two `int64` columns instead, which is the only
  difference between the two programs.
"""

from marrow.builders import array
from marrow.dtypes import int64, string
from marrow.expr.builders import col, table
from marrow.expr.`comptime`.numeric import Gt
from marrow.expr.logical import DynValue
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), nm.copy()], names=["a", "b", "name"]
    )

    var values: List[DynValue] = [col("a", int64), col("name", string)]
    print(
        table(batch^)
        .filter(Gt(col("a", int64), col("b", int64)))
        .project(["a", "name"], values^)
        .execute()
    )
