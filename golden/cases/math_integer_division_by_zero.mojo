from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT 10 // n AS q, 10 % n AS r FROM floats WHERE n >= 0

    Division and modulo by zero. `FloordivKernel` and `ModKernel` both replace
    a zero divisor with 1 — a lane cannot raise and cannot produce a null from
    inside a SIMD loop — so marrow answers with the dividend and with zero
    where SQL answers null.

    The predicate keeps only the non-negative divisors so that this case asks
    *only* about zero: with negatives in the column the rounding divergence
    (`math_floordiv_truncates_toward_zero`) would be mixed in and the failure
    would no longer identify which rule is wrong.

    -- xfail marrow substitutes 1 for a zero divisor, so 10 // 0 is 10 and 10 % 0 is 0; SQL says NULL for both

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
