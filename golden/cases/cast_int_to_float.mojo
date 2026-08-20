def test_golden_cast_int_to_float() raises:
    """
    SELECT CAST(i AS DOUBLE) AS c FROM nums

    -- expected
    c:double
    1.0
    -2.0
    300.0
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumericCast[Float64Type](col("i", int64))])
    check(q)
