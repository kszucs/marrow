"""Binary-size gate: **Sort + Limit** (top-K), which nothing else measures.

`SELECT a, name FROM t ORDER BY a LIMIT 3` over the same batch as
`query_streaming.mojo`, so the delta is what the sorting operators cost: the
`SortIndices` kernel's comparison and radix paths, its key encoding, and the
`take` that applies the permutation.

`Sort` carries the `limit` itself (the plan builder folds a following `Limit`
into it), so this links the top-K path rather than a full sort plus a slice.

    pixi run binary_size query_sort
"""

from marrow.builders import array
from marrow.dtypes import int64, string, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import InMemoryTable, Sort, AnyRelation


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch([a.copy(), nm.copy()], names=["a", "name"])
    var sch = schema([field("a", int64), field("name", string)])

    var keys = List[AnyValue]()
    keys.append(AnyValue(col("a", int64)))

    var sorted = Sort(
        input=AnyRelation(InMemoryTable(batch=batch)),
        keys=keys^,
        ascending=[True],
        nulls_first=True,
        stable=False,
        limit=Optional(3),
        schema=sch,
    )
    print(AnyRelation(sorted^).execute())
