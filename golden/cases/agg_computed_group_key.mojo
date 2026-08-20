def test_golden_agg_computed_group_key() raises:
    """
    SELECT (v > 3) AS key0, CAST(sum(w) AS BIGINT) AS s FROM basic GROUP BY (v > 3) ORDER BY key0 NULLS FIRST

    A computed key has no source column to name it after, so the plan calls
    it `key0` (`relations.mojo`); the twin matches that rather than the other
    way round.

    -- expected
    key0:bool	s:int64
    NULL	50
    False	40
    True	170
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("v", int64) > lit(3, int64)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("w", int64)).alias(
                "s"
            )
        ],
    )
    var q = agg.sort([col("key0", int64)], [True])
    check(q)
