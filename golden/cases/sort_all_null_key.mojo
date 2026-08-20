def test_golden_sort_all_null_key() raises:
    """
    SELECT a, b, g FROM nulls ORDER BY a NULLS FIRST, b NULLS FIRST

    -- expected
    a:int64	b:int64	g:string
    NULL	2	'x'
    NULL	4	NULL
    NULL	6	'x'
    NULL	8	'y'
    """
    var t = table("nulls")
    var q = t.sort([col("a", int64), col("b", int64)], [True, True])
    check(q)
