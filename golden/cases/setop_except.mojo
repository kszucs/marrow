from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales EXCEPT SELECT region FROM regions ORDER BY region NULLS FIRST

    Set difference, also deduplicating and also with NULL equal to itself: the
    NULL region survives because `regions` has none, and `north` disappears
    although `sales` has two of it.

    There is no set-operation node in `marrow/expr/logical.mojo`.

    -- skip mojo

    -- expected
    region:string
    NULL
    'east'
    """
    var t = table("sales").select(["region"])
    var d = t.except_(table("regions").select(["region"]))
    return d.sort_by([col("region", string)], [True])
