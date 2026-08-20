def test_golden_project_predicate_is_boolean() raises:
    """
    SELECT v > 3 AS gt FROM basic

    -- expected
    gt:bool
    False
    False
    False
    True
    NULL
    True
    True
    """
    var t = table("basic")
    var q = t.project(["gt"], [col("v", int64) > lit(3, int64)])
    check(q)
