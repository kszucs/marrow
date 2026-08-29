from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY v + w NULLS FIRST, v NULLS FIRST

    A sort key that is computed rather than read. Every other `ORDER BY` in the
    corpus names a column, so this is where a `Sort` that assumed its keys were
    columns — and looked them up in the input schema — would fail.

    `v + w` is null wherever either operand is, which puts two rows in the
    leading null block; `v` breaks that tie, itself with nulls first.

    -- expected
    k:string	v:int64	w:int64
    'b'	NULL	50
    'b'	2	NULL
    'a'	1	10
    'a'	3	30
    'c'	4	40
    'a'	6	60
    NULL	7	70
    """
    var t = table("basic")
    return t.sort_by(
        [col("v", int64) + col("w", int64), col("v", int64)], [True, True]
    )
