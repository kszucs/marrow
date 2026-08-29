from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT sum(price) AS total FROM sales

    Every price is an exact binary fraction, so the sum is exact in any
    order — this compares two implementations, not two roundings.

    -- expected
    total:double
    7.0
    """
    var t = table("sales")
    return t.aggregate(aggs=[col("price", float64).sum().alias("total")])
