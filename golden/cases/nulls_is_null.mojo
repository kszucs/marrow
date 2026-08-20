def test_golden_nulls_is_null() raises:
    """
    SELECT a, b, g FROM nulls WHERE a IS NULL

    -- expected
    a:int64	b:int64	g:string
    NULL	2	'x'
    NULL	4	NULL
    NULL	6	'x'
    NULL	8	'y'
    """
    var t = table("nulls")
    var q = t.filter(IsNull(col("a", int64)))
    check(q)
