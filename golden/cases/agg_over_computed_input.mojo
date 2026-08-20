def test_golden_agg_over_computed_input() raises:
    """
    SELECT k, CAST(sum(v * 2) AS BIGINT) AS d FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	d:int64
    NULL	14
    'a'	20
    'b'	4
    'c'	8
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](
                col("v", int64) * lit(2, int64)
            ).alias("d")
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    check(q)
