def test_golden_aggregate_sum_by_key() raises:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64
    NULL	7
    'a'	10
    'b'	2
    'c'	4
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
    var q = agg.sort([col("k", string)], [True])
    check(q)
