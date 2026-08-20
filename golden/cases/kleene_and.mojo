def test_golden_kleene_and() raises:
    """
    SELECT x, y, (x > 0) AND (y > 0) AS r FROM kleene

    -- expected
    x:int64	y:int64	r:bool
    1	1	True
    1	-1	False
    1	NULL	NULL
    -1	1	False
    -1	-1	False
    -1	NULL	False
    NULL	1	NULL
    NULL	-1	False
    NULL	NULL	NULL
    """
    var t = table("kleene")
    var q = t.project(
        ["x", "y", "r"],
        [
            col("x", int64),
            col("y", int64),
            (col("x", int64) > lit(0, int64))
            & (col("y", int64) > lit(0, int64)),
        ],
    )
    check(q)
