def test_golden_cast_bool_to_int() raises:
    """
    SELECT CAST(b AS BIGINT) AS c FROM nums

    -- expected
    c:int64
    1
    0
    1
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [BoolToNum[Int64Type](col("b", bool_))])
    check(q)
