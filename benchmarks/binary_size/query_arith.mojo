"""Binary-size gate: fused **arithmetic**, which nothing else measures.

`SELECT (a + b) * a - b FROM t WHERE a > b` — the same relational shape as
`query_streaming.mojo`, so the delta between the two is exactly what the
arithmetic nodes (`Add`/`Sub`/`Mul` over `NumericBinary`) cost on top of a
column reference and a comparison.

**This gate exists because that cost was invisible.** Q0.4 rewrote all twelve of
`DynValue.eval`'s binary arms and the fused gates came back byte-identical —
true, and worthless: neither of them contains a single arithmetic expression.
A change to the fused numeric algebra could regress arbitrarily without any gate
noticing.

    pixi run binary_size query_arith
"""

from marrow.builders import array
from marrow.dtypes import int64, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import InMemoryTable, Project, AnyRelation


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var filtered = AnyRelation(InMemoryTable(batch=batch)).filter(
        AnyValue(col("a", int64) > col("b", int64))
    )
    var values = List[AnyValue]()
    values.append(
        AnyValue(
            (col("a", int64) + col("b", int64)) * col("a", int64)
            - col("b", int64)
        )
    )
    var proj = Project(
        input=filtered,
        names=["z"],
        values=values^,
        schema=schema([field("z", int64)]),
    )
    print(AnyRelation(proj^).execute())
