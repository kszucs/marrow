from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT price, region FROM sales ORDER BY price NULLS FIRST

    `select` projects in the order given, not the schema's order.

    -- expected
    price:double	region:string
    NULL	NULL
    -1.25	'south'
    0.5	'north'
    1.5	'north'
    2.25	'south'
    4.0	'east'
    """
    var t = table("sales")
    var picked = t.select("price", "region")
    var q = picked.sort([col("price", float64)], [True])
    return q
