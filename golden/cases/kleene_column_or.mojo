def test_golden_kleene_column_or() raises:
    """
    SELECT p, q, p OR q AS r FROM flags

    -- expected
    p:bool	q:bool	r:bool
    True	True	True
    True	False	True
    True	NULL	True
    False	True	True
    False	False	False
    False	NULL	NULL
    NULL	True	True
    NULL	False	NULL
    NULL	NULL	NULL
    """
    var t = table("flags")
    var q = t.project(
        ["p", "q", "r"],
        [col("p", bool_), col("q", bool_), col("p", bool_) | col("q", bool_)],
    )
    check(q)
