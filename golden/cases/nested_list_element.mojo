from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT l[1] AS first, l[3] AS third FROM lists

    Subscripting a list, 1-based, with an out-of-bounds index answering
    **NULL** rather than raising — the two conventions this family is built on.
    Reading both index 1 and index 3 puts every row of the fixture on one of
    the two sides: `[7]` is in bounds for the first and out for the second, and
    `[NULL, 5]` returns a genuine null element from index 1.

    `ListValue` exposes only `array_length`; there is no element-access node,
    and an element is not a value a fused lane can hold.

    -- skip mojo
    -- skip python

    -- expected
    first:int64	third:int64
    1	3
    NULL	NULL
    NULL	NULL
    NULL	NULL
    7	NULL
    """
    var t = table("lists")
    return t.project(
        ["first", "third"],
        [
            col("l", list_(int64)).element(1),
            col("l", list_(int64)).element(3),
        ],
    )
