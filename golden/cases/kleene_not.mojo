def test_golden_kleene_not() raises:
    """
    SELECT x, NOT (x > 0) AS r FROM kleene

    -- expected
    x:int64	r:bool
    1	False
    1	False
    1	False
    -1	True
    -1	True
    -1	True
    NULL	NULL
    NULL	NULL
    NULL	NULL
    """
    var t = table("kleene")
    var q = t.project(
        ["x", "r"], [col("x", int64), ~(col("x", int64) > lit(0, int64))]
    )
    check(q)
