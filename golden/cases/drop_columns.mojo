from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, price FROM sales ORDER BY price NULLS FIRST

    `drop` says what goes rather than what stays, and the survivors keep
    their original order.

    -- expected
    region:string	price:double
    NULL	NULL
    'south'	-1.25
    'north'	0.5
    'north'	1.5
    'south'	2.25
    'east'	4.0
    """
    var t = table("sales")
    var dropped = t.drop(["qty", "active", "ref"])
    var q = dropped.sort([col("price", float64)], [True])
    return q
