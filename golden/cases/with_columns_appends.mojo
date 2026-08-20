def test_golden_with_columns_appends() raises:
    """
    SELECT k, v, w, v + w AS s FROM basic

    -- expected
    k:string	v:int64	w:int64	s:int64
    'a'	1	10	11
    'b'	2	NULL	NULL
    'a'	3	30	33
    'c'	4	40	44
    'b'	NULL	50	NULL
    'a'	6	60	66
    NULL	7	70	77
    """
    var t = table("basic")
    var q = t.with_columns(["s"], [col("v", int64) + col("w", int64)])
    check(q)
