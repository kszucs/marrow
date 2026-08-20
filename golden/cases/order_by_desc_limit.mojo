def test_golden_order_by_desc_limit() raises:
    """
    SELECT k, v, w FROM basic ORDER BY v DESC NULLS FIRST LIMIT 3

    -- expected
    k:string	v:int64	w:int64
    'b'	NULL	50
    NULL	7	70
    'a'	6	60
    """
    var t = table("basic")
    var q = t.sort([col("v", int64)], [False]).limit(3)
    check(q)
