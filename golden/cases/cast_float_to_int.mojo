def test_golden_cast_float_to_int() raises:
    """
    SELECT CAST(TRUNC(f) AS BIGINT) AS c FROM nums

    A recorded divergence. DuckDB's `CAST(1.7 AS BIGINT)` **rounds** to 2 and
    `0.5` to 0 (half-to-even); Arrow **truncates** toward zero, giving 1 and
    0. PyArrow confirms marrow's answer — `pc.cast(..., safe=False)` returns
    `[1, -2, 0, None]` and `safe=True` raises "Float value 1.700000 was
    truncated converting to int64".

    So the twin says `TRUNC` to ask DuckDB the question marrow answers.
    Writing the twin as a bare CAST would assert DuckDB's rounding rule and
    report Arrow-correct behaviour as a defect.

    -- expected
    c:int64
    1
    -2
    0
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumericCast[Int64Type](col("f", float64))])
    check(q)
