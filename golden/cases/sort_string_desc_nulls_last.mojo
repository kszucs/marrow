def test_golden_sort_string_desc_nulls_last() raises:
    """
    SELECT k, v, w FROM basic ORDER BY k DESC NULLS LAST

    -- expected
    k:string	v:int64	w:int64
    'c'	4	40
    'b'	2	NULL
    'b'	NULL	50
    'a'	1	10
    'a'	3	30
    'a'	6	60
    NULL	7	70
    """
    var t = table("basic")
    var q = t.sort([col("k", string)], [False], nulls_first=False)
    check(q)
