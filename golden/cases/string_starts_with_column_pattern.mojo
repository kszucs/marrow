from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT starts_with(s, lower(s)) AS b FROM words

    A prefix that is a *column* rather than a literal. Every other string
    predicate in the corpus passes a constant, which `StringPredicate` compiles
    once and reuses; a column operand forces the per-row path where both sides
    are loaded from arrays. Only the rows with no upper case can match.

    -- expected
    b:bool
    False
    False
    True
    True
    True
    NULL
    """
    var t = table("words")
    return t.project(
        ["b"], [StartsWith(col("s", string), Lower(col("s", string)))]
    )
