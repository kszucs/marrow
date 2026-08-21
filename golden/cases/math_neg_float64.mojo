from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT -x AS m FROM floats WHERE NOT isnan(x) AND NOT isinf(x)

    The rows are restricted to the finite ones because the expectation block
    cannot represent a NaN or an infinity: it renders a float with `repr`, and
    `nan` / `inf` are not Python literals, so `ast.literal_eval` rejects them on
    the way back in. The predicate does double duty as `is_nan` / `is_inf`
    coverage, and it drops the null row too -- `NOT NULL` is NULL, not true.

    Note that the two zero rows are a weak assertion: Arrow's array comparison
    holds 0.0 and -0.0 to be equal, so a sign-bit-only difference would not be
    caught here.

    -- expected
    m:double
    -1.5
    2.0
    -0.0
    0.0
    """
    var t = table("floats")
    var finite = t.filter(
        (~col("x", float64).is_nan()) & (~col("x", float64).is_inf())
    )
    var q = finite.project(["m"], [-col("x", float64)])
    return q
