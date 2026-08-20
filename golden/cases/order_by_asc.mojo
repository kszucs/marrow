def test_golden_order_by_asc() raises:
    """
    SELECT k, v, w FROM basic ORDER BY v NULLS FIRST

    -- expected
    k:string	v:int64	w:int64
    'b'	NULL	50
    'a'	1	10
    'b'	2	NULL
    'a'	3	30
    'c'	4	40
    'a'	6	60
    NULL	7	70
    """
    var t = table("basic")
    var q = t.sort([col("v", int64)], [True])
    check(q)
