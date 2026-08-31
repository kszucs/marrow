from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales INTERSECT SELECT region FROM regions ORDER BY region NULLS FIRST

    Set intersection. `west` is only on the right and `east` and NULL only on
    the left, so the answer is the two names both sides carry — once each,
    however many times `sales` repeats them.

    There is no set-operation node in `marrow/expr/logical.mojo`.

    -- skip mojo
    -- skip python

    -- expected
    region:string
    'north'
    'south'
    """
    var t = table("sales").select(["region"])
    var i = t.intersect(table("regions").select(["region"]))
    return i.sort_by([col("region", string)], [True])
