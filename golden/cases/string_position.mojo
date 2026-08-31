from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(position(',' IN t) AS BIGINT) AS p FROM text

    1-based, and **0** — not null — when the needle is absent. That zero is
    what makes the result an index into a 1-based world rather than an offset,
    and it is the convention a 0-based engine cannot reuse.

    `marrow/kernels/string.mojo` has no such kernel. `ContainsKernel` answers
    the boolean question; this one needs the offset.

    -- skip mojo
    -- skip python

    -- expected
    p:int64
    2
    0
    0
    NULL
    0
    0
    """
    var t = table("text")
    return t.project(["p"], [col("t", string).position(",")])
