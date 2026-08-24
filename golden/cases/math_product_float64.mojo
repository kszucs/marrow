from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT product(y) AS p FROM floats

    Every value in `y` is a power of two, so the running product is exact at
    every step and this compares two implementations rather than two roundings.
    The null is skipped, as it is for `sum`.

    -- expected
    p:double
    256.0
    """
    var t = table("floats")
    var q = t.aggregate(aggs=[col("y", float64).product().alias("p")])
    return q
