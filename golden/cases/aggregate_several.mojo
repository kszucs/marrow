def test_golden_aggregate_several() raises:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total, max(v) AS biggest, CAST(count(w) AS BIGINT) AS n FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64	biggest:int64	n:int64
    NULL	7	7	1
    'a'	10	6	3
    'b'	2	2	1
    'c'	4	4	1
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            ),
            AggExpr.of[NumericAgg[MaxKernel, Int64Type]](col("v", int64)).alias(
                "biggest"
            ),
            AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                col("w", int64)
            ).alias("n"),
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    check(q)
