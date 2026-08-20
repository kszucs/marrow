def test_golden_kleene_column_filter() raises:
    """
    SELECT p, q FROM flags WHERE p

    A null predicate does not select, so only the three TRUE rows survive.

    -- expected
    p:bool	q:bool
    True	True
    True	False
    True	NULL
    """
    var t = table("flags")
    var q = t.filter(col("p", bool_))
    check(q)
