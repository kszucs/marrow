def test_golden_agg_count_column_skips_nulls() raises:
    """
    SELECT CAST(count(v) AS BIGINT) AS n FROM basic

    The contrast with `count(*)`: `v` has one null, so this is 6 where the
    row count is 7.

    -- expected
    n:int64
    6
    """
    var t = table("basic")
    var q = t.aggregate(
        keys=List[BoxedValue](),
        aggs=[
            AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                col("v", int64)
            ).alias("n")
        ],
    )
    check(q)
