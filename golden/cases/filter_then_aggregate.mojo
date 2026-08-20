def test_golden_filter_then_aggregate() raises:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic WHERE w > 20 GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64
    NULL	7
    'a'	9
    'b'	NULL
    'c'	4
    """
    var t = table("basic")
    var agg = t.filter(col("w", int64) > lit(20, int64)).aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            )
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    check(q)
