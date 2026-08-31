from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT mode(k) AS m FROM basic

    The most frequent value. `basic.k` has a unique winner (`a`, three times),
    so the answer does not depend on the tie-breaking rule — which is
    unspecified and would make the case non-deterministic if it did.

    `marrow/kernels/aggregate.mojo` has no mode kernel.

    -- skip mojo
    -- skip python

    -- expected
    m:string
    'a'
    """
    var t = table("basic")
    return t.aggregate(aggs=[col("k", string).mode().alias("m")])
