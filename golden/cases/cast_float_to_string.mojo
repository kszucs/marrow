from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(f AS VARCHAR) AS c FROM nums

    Floating point to text, where `cast_int_to_string` covers the integer form.
    This is the cast whose answer is a *formatting* decision rather than a
    value one — how many digits, whether a trailing `.0` is written, when an
    exponent appears — so it is where two engines that agree on every number
    can still disagree on the string.

    -- expected
    c:string
    '1.7'
    '-2.7'
    '0.5'
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [NumToString[StringType](col("f", float64))])
