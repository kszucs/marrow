"""Binary-size gate: **Sort + Limit**, which nothing else measures.

`SELECT a, name FROM t ORDER BY a LIMIT 3` over the same batch as
`query_streaming.mojo`, so the delta is what the sorting operators cost: the
`SortIndices` kernel's comparison and radix paths, its key encoding, and the
`take` that applies the permutation.

    pixi run binary_size query_sort

**Ported from the old expression package on 2026-08-29, and it lost the top-K
path.** The old `Sort` node carried the `limit` itself — the plan builder
folded a following `Limit` into it — so this gate linked a bounded top-K sort.
`marrow.expr`'s `Sort` (`marrow/expr/logical.mojo`) has no `limit` field and
`SortOperator` (`marrow/expr/physical.mojo`) buffers, sorts in full at
`finish`, and hands the result to a separate `LimitOperator` that slices it.
So this gate now measures **a full sort plus a zero-copy slice**, not top-K.
Nothing in `marrow.expr` can express top-K today; when that returns, restore
the fold and re-record. The recorded baseline predates the port and is stale.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr import col, table
from marrow.expr import DynValue
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var nm = array(["p", "q", "r", "s", "t"])
    var batch = record_batch([a.copy(), nm.copy()], names=["a", "name"])

    var keys: List[DynValue] = [col("a", int64)]
    print(
        table(batch^)
        .sort_by(keys^, [True], nulls_first=True)
        .limit(3)
        .execute()
    )
