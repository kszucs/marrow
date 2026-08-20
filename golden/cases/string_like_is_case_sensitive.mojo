from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s LIKE 'h%' AS b FROM words

    -- expected
    b:bool
    False
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    var q = t.project(["b"], [Like(col("s", string), lit("h%"))])
    return q
