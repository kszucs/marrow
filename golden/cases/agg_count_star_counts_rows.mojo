def test_golden_agg_count_star_counts_rows() raises:
    """
    SELECT CAST(count(*) AS BIGINT) AS n FROM basic

    -- expected
    n:int64
    7
    """
    var t = table("basic")
    var q = t.aggregate(keys=List[BoxedValue](), aggs=[count_star().alias("n")])
    check(q)
