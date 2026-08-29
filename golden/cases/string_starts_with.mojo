from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT starts_with(s, 'H') AS b FROM words

    -- expected
    b:bool
    True
    False
    False
    False
    False
    NULL
    """
    var t = table("words")
    return t.project(["b"], [StartsWith(col("s", string), lit("H", string))])
