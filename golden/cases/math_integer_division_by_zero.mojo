from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT 10 // n AS q, 10 % n AS r FROM floats WHERE n >= 0

    Division and modulo by zero. A SIMD lane can neither raise nor produce a
    null, so `FloordivKernel` and `ModKernel` still compute against a
    substituted divisor of 1 — and the row is masked out afterwards, by
    `BinaryKernel.domain` in the erased path and by `DivisionBinary`'s `Bound` in
    the fused one. Both answer NULL, as SQL does.

    The predicate keeps only the non-negative divisors so that this case asks
    *only* about zero: with negatives in the column the rounding rule
    (`math_floordiv_truncates_toward_zero`) would be mixed in and a failure
    would no longer identify which one is wrong.

    -- expected
    q:int64	r:int64
    2	2
    NULL	NULL
    10	0
    5	0
    3	1
    """
    var t = table("floats")
    var nonneg = t.filter(col("n", int64) >= lit(0, int64))
    return nonneg.project(
        ["q", "r"],
        [
            lit(10, int64) // col("n", int64),
            lit(10, int64) % col("n", int64),
        ],
    )
