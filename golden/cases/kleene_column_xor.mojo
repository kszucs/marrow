from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT p, q, (p != q) AS r FROM flags

    Exclusive or over bool columns, the one connective `kleene_column_and` and
    `kleene_column_or` leave out. SQL has no `XOR` keyword, and `p != q` is the
    same function: unlike `AND` and `OR` it has no absorbing value, so every
    row with a null operand is null and the 3x3 table has six of them.

    -- expected
    p:bool	q:bool	r:bool
    True	True	False
    True	False	True
    True	NULL	NULL
    False	True	True
    False	False	False
    False	NULL	NULL
    NULL	True	NULL
    NULL	False	NULL
    NULL	NULL	NULL
    """
    var t = table("flags")
    return t.project(
        ["p", "q", "r"],
        [col("p", bool_), col("q", bool_), col("p", bool_) ^ col("q", bool_)],
    )
