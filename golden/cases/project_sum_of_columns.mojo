def test_golden_project_sum_of_columns() raises:
    """
    SELECT v + w AS s FROM basic

    -- expected
    s:int64
    11
    NULL
    33
    44
    NULL
    66
    77
    """
    var t = table("basic")
    var q = t.project(["s"], [col("v", int64) + col("w", int64)])
    check(q)
