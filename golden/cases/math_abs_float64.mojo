from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT abs(x) AS a FROM floats WHERE NOT isnan(x) AND NOT isinf(x)

    The rows are restricted to the finite ones because the expectation block
    cannot represent a NaN or an infinity: it renders a float with `repr`, and
    `nan` / `inf` are not Python literals, so `ast.literal_eval` rejects them on
    the way back in. The predicate does double duty as `is_nan` / `is_inf`
    coverage, and it drops the null row too -- `NOT NULL` is NULL, not true.

    -- expected
    a:double
    1.5
    2.0
    0.0
    0.0
    """
    var t = table("floats")
    var finite = t.filter(
        (~col("x", float64).is_nan()) & (~col("x", float64).is_inf())
    )
    var q = finite.project(["a"], [col("x", float64).abs()])
    return q
