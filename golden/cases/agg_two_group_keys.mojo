def test_golden_agg_two_group_keys() raises:
    """
    SELECT k, v, CAST(sum(w) AS BIGINT) AS s FROM basic GROUP BY k, v ORDER BY k NULLS FIRST, v NULLS FIRST

    -- expected
    k:string	v:int64	s:int64
    NULL	7	70
    'a'	1	10
    'a'	3	30
    'a'	6	60
    'b'	NULL	50
    'b'	2	NULL
    'c'	4	40
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string), col("v", int64)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("w", int64)).alias(
                "s"
            )
        ],
    )
    var q = agg.sort([col("k", string), col("v", int64)], [True, True])
    check(q)
