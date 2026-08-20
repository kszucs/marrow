def test_golden_sort_top_k() raises:
    """
    SELECT k, v, w FROM basic ORDER BY v DESC NULLS LAST LIMIT 2

    -- expected
    k:string	v:int64	w:int64
    NULL	7	70
    'a'	6	60
    """
    var t = table("basic")
    var q = t.sort([col("v", int64)], [False], nulls_first=False).limit(2)
    check(q)
