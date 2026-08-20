from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region AS area, qty AS units, price FROM sales ORDER BY units NULLS FIRST

    `rename` is a schema-level rewrite: order, dtypes and the untouched
    columns are all preserved.

    -- expected
    area:string	units:int32	price:double
    'north'	NULL	0.5
    'south'	5	-1.25
    'north'	10	1.5
    'south'	20	2.25
    NULL	40	NULL
    'east'	50	4.0
    """
    var t = table("sales")
    var picked = t.select("region", "qty", "price")
    var renamed = picked.rename(["region", "qty"], ["area", "units"])
    var q = renamed.sort([col("units", int32)], [True])
    return q
