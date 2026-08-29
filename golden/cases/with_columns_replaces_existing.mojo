from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, qty * 2 AS qty, price, active, ref FROM sales

    `with_columns` on a name that already exists **replaces it in place**
    rather than appending a second column of the same name — Polars' rule, and
    the only one that keeps the output schema free of duplicates. Position is
    part of the contract: `qty` stays second, between `region` and `price`.

    `with_columns_appends` covers the other half, where the name is new.

    -- expected
    region:string	qty:int32	price:double	active:bool	ref:int64
    'north'	20	1.5	True	1
    'south'	40	2.25	False	2
    'north'	NULL	0.5	True	2
    NULL	80	NULL	NULL	3
    'east'	100	4.0	True	NULL
    'south'	10	-1.25	False	99
    """
    var t = table("sales")
    return t.with_columns(["qty"], [col("qty", int32) * lit(2, int32)])
