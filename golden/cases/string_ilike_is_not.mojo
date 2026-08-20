def test_golden_string_ilike_is_not() raises:
    """
    SELECT s ILIKE 'h%' AS b FROM words

    -- expected
    b:bool
    True
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    var q = t.project(["b"], [ILike(col("s", string), lit("h%"))])
    check(q)
