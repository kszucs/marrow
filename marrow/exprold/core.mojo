"""The vocabulary both expression lanes share, and nothing else.

`Datum` is the strategy-agnostic wire format between stages — Arrow's Datum /
DataFusion's ColumnarValue — and `into_array` is the one place a scalar stops
being lazy and becomes a column. Both are spoken by the AOT lane
(`marrow.exprold.values`) and the runtime lane (`marrow.exprold.dynamic`), so neither
belongs to either; putting them here is what lets `dynamic` stop importing
`values` for them.

This module is a **leaf**: it imports the array and scalar containers and
nothing else under `marrow.expr`.
"""

from std.utils import Variant

from ..arrays import DynArray
from ..scalars import DynScalar


# ---------------------------------------------------------------------------
# Datum — Scalar | Array, the uniform `execute` result.
# ---------------------------------------------------------------------------
comptime Datum = Variant[DynScalar, DynArray]


def into_array(d: Datum, n: Int) raises -> DynArray:
    """Force `d` to an array of length `n` — broadcasting a scalar (lazy until here).
    """
    if d.isa[DynScalar]():
        return d[DynScalar].repeat(n)
    return d[DynArray].copy()


# ---------------------------------------------------------------------------
# Plan analysis — order-preserving dedup union of column-name lists, so a
# composite node can fold its children's `referenced_columns()` without repeats.
# ---------------------------------------------------------------------------
def _union_columns(var acc: List[String], names: List[String]) -> List[String]:
    for i in range(len(names)):
        var seen = False
        for j in range(len(acc)):
            if acc[j] == names[i]:
                seen = True
                break
        if not seen:
            acc.append(names[i].copy())
    return acc^
