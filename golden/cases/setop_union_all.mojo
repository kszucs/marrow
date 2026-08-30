from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales UNION ALL SELECT region FROM regions ORDER BY region NULLS FIRST

    `UNION ALL` keeps multiplicity: `north` appears twice from `sales` and once
    from `regions`, so the answer has three of it. That is what separates it
    from `UNION`, and it is the cheaper of the two because no dedup is needed.

    There is no set-operation node in `marrow/expr/logical.mojo`.

    -- skip mojo

    -- expected
    region:string
    NULL
    'east'
    'north'
    'north'
    'north'
    'south'
    'south'
    'south'
    'west'
    """
    var t = table("sales").select(["region"])
    var u = t.union_all(table("regions").select(["region"]))
    return u.sort_by([col("region", string)], [True])
