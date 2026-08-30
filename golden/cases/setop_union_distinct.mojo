from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales UNION SELECT region FROM regions ORDER BY region NULLS FIRST

    Bare `UNION` deduplicates, and it does so with NULL treated as **equal to
    itself** — the opposite of `=` and of a join key. `sales` holds one NULL
    region and the result holds one, not zero.

    There is no set-operation node in `marrow/expr/logical.mojo`.

    -- skip mojo

    -- expected
    region:string
    NULL
    'east'
    'north'
    'south'
    'west'
    """
    var t = table("sales").select(["region"])
    var u = t.union(table("regions").select(["region"]))
    return u.sort_by([col("region", string)], [True])
