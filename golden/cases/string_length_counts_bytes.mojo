def test_golden_string_length_counts_bytes() raises:
    """
    SELECT CAST(strlen(s) AS INTEGER) AS n FROM words

    Bytes, not codepoints: `héllo` is 6. The twin says `octet_length`
    deliberately — DuckDB's `length` counts characters, which is the other
    answer and would make this case assert the wrong thing.

    -- expected
    n:int32
    5
    5
    7
    0
    6
    NULL
    """
    var t = table("words")
    var q = t.project(["n"], [StringLength(col("s", string))])
    check(q)
