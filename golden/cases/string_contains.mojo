from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT contains(s, 'l') AS b FROM words

    A substring search, not a pattern match: the needle is case-sensitive, so
    `wORLD` is false on a lowercase `l` while `Hello` is true. The empty string
    contains nothing, and the null row answers NULL.

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
    var q = t.project(["b"], [col("s", string).contains(lit("l"))])
    return q
