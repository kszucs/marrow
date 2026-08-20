from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT x, y, (x > 0) OR (y > 0) AS r FROM kleene

    -- expected
    x:int64	y:int64	r:bool
    1	1	True
    1	-1	True
    1	NULL	True
    -1	1	True
    -1	-1	False
    -1	NULL	NULL
    NULL	1	True
    NULL	-1	NULL
    NULL	NULL	NULL
    """
    var t = table("kleene")
    var q = t.project(
        ["x", "y", "r"],
        [
            col("x", int64),
            col("y", int64),
            (col("x", int64) > lit(0, int64))
            | (col("y", int64) > lit(0, int64)),
        ],
    )
    return q
