def test_golden_cast_int_to_string() raises:
    """
    SELECT CAST(i AS VARCHAR) AS c FROM nums

    -- expected
    c:string
    '1'
    '-2'
    '300'
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumToString[StringType](col("i", int64))])
    check(q)
