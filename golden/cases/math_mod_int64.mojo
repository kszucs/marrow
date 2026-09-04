from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ((n % 3) + 3) % 3 AS m FROM floats

    Floored modulo spelled in SQL, and a three-level nested expression whose
    intermediate value is negative — the shape a single `%` cannot ask.

    It used to be here for the opposite reason: marrow's `%` was Python's, so
    a bare `n % 3` twin would have asserted SQL's convention against marrow's
    deliberate one. That divergence is gone — `%` takes the dividend's sign
    now, and `math_modulo_sign_follows_dividend` asks the bare form directly —
    so both sides of this case spell the same expression.

    -- expected
    m:int64
    1
    0
    0
    1
    2
    0
    2
    NULL
    """
    var t = table("floats")
    var n = col("n", int64)
    return t.project(
        ["m"], [((n % lit(3, int64)) + lit(3, int64)) % lit(3, int64)]
    )
