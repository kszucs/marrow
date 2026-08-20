def test_golden_nulls_is_not_null() raises:
    """
    SELECT a, b, g FROM nulls WHERE b IS NOT NULL

    -- expected
    a:int64	b:int64	g:string
    NULL	2	'x'
    NULL	4	NULL
    NULL	6	'x'
    NULL	8	'y'
    """
    var t = table("nulls")
    var q = t.filter(NotNull(col("b", int64)))
    check(q)
