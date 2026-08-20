def test_golden_aggregate_no_keys() raises:
    """
    SELECT CAST(sum(v) AS BIGINT) AS total FROM basic

    -- expected
    total:int64
    23
    """
    var t = table("basic")
    var q = t.aggregate(
        keys=List[BoxedValue](),
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            )
        ],
    )
    check(q)
