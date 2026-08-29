from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, qty, price FROM sales WHERE region = 'south' AND price < 0.0 ORDER BY qty NULLS FIRST

    A string comparison and a float comparison combined under AND.

    -- expected
    region:string	qty:int32	price:double
    'south'	5	-1.25
    """
    var t = table("sales")
    var picked = t.select(["region", "qty", "price"])
    var filtered = picked.filter(
        (col("region", string) == lit("south", string))
        & (col("price", float64) < lit(0.0, float64))
    )
    return filtered.sort_by([col("qty", int32)], [True])
