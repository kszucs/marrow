def test_golden_nulls_predicate_excludes_null() raises:
    """
    SELECT a, b, g FROM nulls WHERE a > 0

    -- expected
    a:int64	b:int64	g:string
    """
    var t = table("nulls")
    var q = t.filter(col("a", int64) > lit(0, int64))
    check(q)
