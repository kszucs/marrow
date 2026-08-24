from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT x / y AS d FROM floats WHERE NOT isnan(x) AND NOT isinf(x)

    The rows are restricted to the finite ones because the expectation block
    cannot represent a NaN or an infinity: it renders a float with `repr`, and
    `nan` / `inf` are not Python literals, so `ast.literal_eval` rejects them on
    the way back in. The predicate does double duty as `is_nan` / `is_inf`
    coverage, and it drops the null row too -- `NOT NULL` is NULL, not true.

    Division by zero is a separate question -- `y` has no zero, so this case asks
    about arithmetic only.

    -- expected
    d:double
    0.75
    -0.5
    0.0
    -0.0
    """
    var t = table("floats")
    var finite = t.filter(
        (~col("x", float64).is_nan()) & (~col("x", float64).is_inf())
    )
    var q = finite.project(["d"], [col("x", float64) / col("y", float64)])
    return q
