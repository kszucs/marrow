from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ends_with(s, 'o') AS b FROM words

    The suffix counterpart of `starts_with`. The empty string ends with nothing, and the null row answers NULL.

    -- expected
    b:bool
    True
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    var q = t.project(["b"], [col("s", string).endswith(lit("o"))])
    return q
