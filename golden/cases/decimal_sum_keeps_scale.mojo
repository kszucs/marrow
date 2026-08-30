from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(CAST(price AS DECIMAL(10, 2))) AS VARCHAR) AS total FROM sales

    Summing a decimal is exact and keeps the scale — `1.50 + 2.25 + ...` is not
    a float sum that happens to look right. The twin renders the result as text
    so that the *scale* is asserted and not just the value: `7.00` and `7.0`
    are the same number and different answers.

    marrow has `Decimal128Array` and the decimal dtypes, but `Column[T]` binds
    `T: NumericType`, which decimal is not, so no decimal column can enter an
    expression.

    -- skip mojo

    -- expected
    total:string
    '7.00'
    """
    var t = table("sales")
    var d = t.project(
        ["d"], [NumericCast[Decimal128Type](col("price", float64))]
    )
    return d.aggregate(
        aggs=[
            NumToString[StringType](col("d", decimal128(10, 2)).sum()).alias(
                "total"
            )
        ]
    )
