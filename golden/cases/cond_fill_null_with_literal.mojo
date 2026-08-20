def test_golden_cond_fill_null_with_literal() raises:
    """
    SELECT coalesce(v, 0) AS c FROM basic

    -- expected
    c:int64
    1
    2
    3
    4
    0
    6
    7
    """
    var t = table("basic")
    var q = t.project(["c"], [FillNull(col("v", int64), lit(0, int64))])
    check(q)
