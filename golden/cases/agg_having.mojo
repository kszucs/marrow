def test_golden_agg_having() raises:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic GROUP BY k HAVING sum(v) > 5 ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64
    NULL	7
    'a'	10
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            )
        ],
    )
    var q = agg.filter(col("total", int64) > lit(5, int64)).sort(
        [col("k", string)], [True]
    )
    check(q)
