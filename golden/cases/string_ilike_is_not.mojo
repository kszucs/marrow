from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s ILIKE 'h%' AS b FROM words

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
    return t.project(["b"], [col("s", string).ilike(lit("h%", string))])
