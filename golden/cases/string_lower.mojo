def test_golden_string_lower() raises:
    """
    SELECT lower(s) AS l FROM words

    -- expected
    l:string
    'hello'
    'world'
    '  pad  '
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["l"], [Lower(col("s", string))])
    check(q)
