"""Binary-size gate for `expr2` filter + projection.

`SELECT a, b FROM orders WHERE a > b`, built through `expr2`'s plan-building
verbs with fused comptime values. Only comptime nodes (`col`, `Gt`) are boxed
into `DynValue`, so the runtime lane and its per-dtype kernel fanout are
dead-code-eliminated — this gate is the DCE proof for `expr2`, the same role
`query_streaming.mojo` plays for `expr/`.

Paired with `query_expr2_agg_fused.mojo`, which covers the aggregation path.
Together they are the first gates in this directory that build anything from
`expr2` at all.

**`project`, not `select`.** `select` takes names only and builds *runtime*
column reads, which would link the very lane this gate exists to prove absent.
`project` takes fused values, so the projection stays in the comptime lane.

Unlike its `expr/` twin this projects two `int64` columns rather than an
`int64` and a `string`: `expr2`'s comptime `Column[T]` is bound on
`NumericType`, so a fused string column cannot be spelled yet.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr2.builders import col
from marrow.expr2.`comptime`.numeric import Gt
from marrow.expr2.logical import DynRelation, DynValue, InMemoryTable
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
