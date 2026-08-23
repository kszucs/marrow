"""Binary-size gate for `expr2` filter + projection.

`SELECT a, b FROM orders WHERE a > b`, built from `marrow.expr2.logical` with
fused comptime values. Only comptime nodes (`col`, `Gt`) are boxed into
`DynValue`, so the runtime lane and its per-dtype kernel fanout are
dead-code-eliminated — this gate is the DCE proof for `expr2`, the same role
`query_streaming.mojo` plays for `expr/`.

Paired with `query_expr2_agg_fused.mojo`, which covers the aggregation path.
Together they are the first gates in this directory that build anything from
`expr2` at all.

Like its `expr/` twin this builds plan nodes directly, and that is deliberate —
a size-critical AOT program knows its own output schema at compile time. Unlike
that twin it projects two `int64` columns rather than an `int64` and a
`string`: `expr2`'s comptime `Column[T]` is bound on `NumericType`, so a fused
string column cannot be spelled yet.

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr2.builders import col
from marrow.expr2.`comptime`.numeric import Gt
from marrow.expr2.logical import DynValue
from marrow.expr2.logical import DynRelation, Filter, InMemoryTable, Project
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var filtered = DynRelation(
        Filter(
            DynRelation(InMemoryTable(batch^)),
            DynValue(Gt(col("a", int64), col("b", int64))),
        )
    )
    var values = List[DynValue]()
    values.append(DynValue(col("a", int64)))
    values.append(DynValue(col("b", int64)))
    var proj = Project(filtered^, ["a", "b"], values^)
    print(DynRelation(proj^).execute())
