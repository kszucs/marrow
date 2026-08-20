def test_golden_nulls_mean_ignores_nulls() raises:
    """
    SELECT avg(b) AS m FROM nulls

    -- expected
    m:double
    5.0
    """
    var t = table("nulls")
    var q = t.aggregate(
        keys=List[BoxedValue](),
        aggs=[
            AggExpr.of[NumericAgg[MeanKernel, Int64Type]](
                col("b", int64)
            ).alias("m")
        ],
    )
    check(q)
