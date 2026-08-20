def test_golden_string_like_is_case_sensitive() raises:
    """
    SELECT s LIKE 'h%' AS b FROM words

    -- expected
    b:bool
    False
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    var q = t.project(["b"], [Like(col("s", string), lit("h%"))])
    check(q)
