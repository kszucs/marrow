from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(CAST(price AS DECIMAL(10, 2)) / 3 AS VARCHAR) AS d FROM sales

    Decimal division cannot be exact, so the engine has to choose a result
    scale and round to it — the one place in decimal arithmetic where
    information is lost, and where two engines that agree on `+`, `-` and `*`
    can still disagree.

    -- skip mojo

    -- expected
    d:string
    '0.5'
    '0.75'
    '0.16666666666666666'
    NULL
    '1.3333333333333333'
    '-0.4166666666666667'
    """
    var t = table("sales")
    var d = t.project(
        ["a"], [NumericCast[Decimal128Type](col("price", float64))]
    )
    return d.project(
        ["d"],
        [
            NumToString[StringType](
                col("a", decimal128(10, 2)) / lit(3, decimal128(10, 2))
            )
        ],
    )
