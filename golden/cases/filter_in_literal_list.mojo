from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE v IN (1, 3, 99)

    `IN` against a constant set. `marrow/kernels/membership.mojo` **has**
    `is_in`, so this is a missing expression node: a value whose payload is a
    set rather than a scalar, which no existing leaf shape covers.

    Spelling it as `v = 1 OR v = 3 OR v = 99` gives the same answer here and is
    what a plan must do today — at O(n) comparisons instead of one hash probe.

    -- skip mojo

    -- expected
    k:string	v:int64	w:int64
    'a'	1	10
    'a'	3	30
    """
    var t = table("basic")
    return t.filter(col("v", int64).is_in([1, 3, 99]))
