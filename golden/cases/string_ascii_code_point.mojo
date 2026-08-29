from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(ascii(t) AS BIGINT) AS a FROM text

    The first character's code point — the smallest function that has to decode
    UTF-8 rather than move bytes. `héllo wörld` starts with `h` (104) and the
    empty string answers 0.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    a:int64
    97
    120
    0
    NULL
    32
    104
    """
    var t = table("text")
    return t.project(["a"], [col("t", string).ascii()])
