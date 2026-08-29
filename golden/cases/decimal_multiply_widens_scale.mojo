from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(CAST(price AS DECIMAL(10, 2)) * CAST(qty AS DECIMAL(10, 2)) AS VARCHAR) AS p FROM sales

    Multiplying two decimals **adds** their scales, so a `(10, 2)` times a
    `(10, 2)` is a `(20, 4)` and the text shows four decimal places. That rule
    is the whole reason decimal arithmetic is not float arithmetic on a scaled
    integer, and no other case in the corpus asks about a result type computed
    from the operands' *parameters* rather than their families.

    -- skip mojo

    -- expected
    p:string
    '15.0000'
    '45.0000'
    NULL
    NULL
    '200.0000'
    '-6.2500'
    """
    var t = table("sales")
    var d = t.project(
        ["a", "b"],
        [
            NumericCast[Decimal128Type](col("price", float64)),
            NumericCast[Decimal128Type](col("qty", int32)),
        ],
    )
    return d.project(
        ["p"],
        [
            NumToString[StringType](
                col("a", decimal128(10, 2)) * col("b", decimal128(10, 2))
            )
        ],
    )
