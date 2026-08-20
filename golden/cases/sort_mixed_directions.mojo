def test_golden_sort_mixed_directions() raises:
    """
    SELECT k, v, w FROM basic ORDER BY k ASC NULLS FIRST, v DESC NULLS FIRST

    -- expected
    k:string	v:int64	w:int64
    NULL	7	70
    'a'	6	60
    'a'	3	30
    'a'	1	10
    'b'	NULL	50
    'b'	2	NULL
    'c'	4	40
    """
    var t = table("basic")
    var q = t.sort([col("k", string), col("v", int64)], [True, False])
    check(q)
