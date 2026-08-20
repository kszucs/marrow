def test_golden_cast_int_to_bool() raises:
    """
    SELECT CAST(i AS BOOLEAN) AS c FROM nums

    -- expected
    c:bool
    True
    True
    True
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumToBool(col("i", int64))])
    check(q)
