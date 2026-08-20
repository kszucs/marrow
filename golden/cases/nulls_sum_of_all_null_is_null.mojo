def test_golden_nulls_sum_of_all_null_is_null() raises:
    """
    SELECT CAST(sum(a) AS BIGINT) AS total FROM nulls

    -- expected
    total:int64
    NULL
    """
    var t = table("nulls")
    var q = t.aggregate(
        keys=List[BoxedValue](),
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("a", int64)).alias(
                "total"
            )
        ],
    )
    check(q)
