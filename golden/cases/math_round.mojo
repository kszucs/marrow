from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT round(x) AS r FROM floats WHERE NOT isnan(x) AND NOT isinf(x)

    The rows are restricted to the finite ones because the expectation block
    cannot represent a NaN or an infinity: it renders a float with `repr`, and
    `nan` / `inf` are not Python literals, so `ast.literal_eval` rejects them on
    the way back in. The predicate does double duty as `is_nan` / `is_inf`
    coverage, and it drops the null row too -- `NOT NULL` is NULL, not true.

    This case does not pin down the halfway rule: the only fractional input is
    1.5, which rounds to 2.0 under both half-away-from-zero and half-to-even. A
    case that discriminates the two needs an input like 0.5 or 2.5, which this
    fixture does not have.

    -- expected
    r:double
    2.0
    -2.0
    0.0
    -0.0
    """
    var t = table("floats")
    var finite = t.filter(
        (~col("x", float64).is_nan()) & (~col("x", float64).is_inf())
    )
    var q = finite.project(["r"], [col("x", float64).round()])
    return q
