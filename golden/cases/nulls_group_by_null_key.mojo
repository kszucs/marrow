def test_golden_nulls_group_by_null_key() raises:
    """
    SELECT g, CAST(sum(b) AS BIGINT) AS total FROM nulls GROUP BY g ORDER BY g NULLS FIRST

    -- expected
    g:string	total:int64
    NULL	4
    'x'	8
    'y'	8
    """
    var t = table("nulls")
    var agg = t.aggregate(
        keys=[col("g", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("b", int64)).alias(
                "total"
            )
        ],
    )
    var q = agg.sort([col("g", string)], [True])
    check(q)
