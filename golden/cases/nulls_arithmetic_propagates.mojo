def test_golden_nulls_arithmetic_propagates() raises:
    """
    SELECT a + b AS s FROM nulls

    -- expected
    s:int64
    NULL
    NULL
    NULL
    NULL
    """
    var t = table("nulls")
    var q = t.project(["s"], [col("a", int64) + col("b", int64)])
    check(q)
