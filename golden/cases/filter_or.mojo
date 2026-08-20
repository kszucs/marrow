def test_golden_filter_or() raises:
    """
    SELECT k, v, w FROM basic WHERE v < 2 OR w > 60

    -- expected
    k:string	v:int64	w:int64
    'a'	1	10
    NULL	7	70
    """
    var t = table("basic")
    var q = t.filter(
        (col("v", int64) < lit(2, int64)) | (col("w", int64) > lit(60, int64))
    )
    check(q)
