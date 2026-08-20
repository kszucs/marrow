def test_golden_string_filter_on_like() raises:
    """
    SELECT s FROM words WHERE s LIKE '%o%'

    -- expected
    s:string
    'Hello'
    'héllo'
    """
    var t = table("words")
    var q = t.filter(Like(col("s", string), lit("%o%")))
    check(q)
