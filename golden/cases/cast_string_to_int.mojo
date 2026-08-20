def test_golden_cast_string_to_int() raises:
    """
    SELECT TRY_CAST(s AS BIGINT) AS c FROM nums

    `TRY_CAST`, because 'abc' does not parse: DuckDB's plain CAST raises and
    marrow nulls the value, so the twin has to ask DuckDB the same question
    marrow answers.

    -- expected
    c:int64
    1
    -2
    NULL
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [StringToNum[Int64Type](col("s", string))])
    check(q)
