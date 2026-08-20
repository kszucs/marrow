def test_golden_nulls_count_of_all_null_is_zero() raises:
    """
    SELECT CAST(count(a) AS BIGINT) AS n FROM nulls

    -- expected
    n:int64
    0
    """
    var t = table("nulls")
    var q = t.aggregate(
        keys=List[BoxedValue](),
        aggs=[
            AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                col("a", int64)
            ).alias("n")
        ],
    )
    check(q)
