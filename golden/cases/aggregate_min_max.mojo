def test_golden_aggregate_min_max() raises:
    """
    SELECT k, min(v) AS lo, max(w) AS hi FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	lo:int64	hi:int64
    NULL	7	70
    'a'	1	60
    'b'	2	50
    'c'	4	40
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[MinKernel, Int64Type]](col("v", int64)).alias(
                "lo"
            ),
            AggExpr.of[NumericAgg[MaxKernel, Int64Type]](col("w", int64)).alias(
                "hi"
            ),
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    check(q)
