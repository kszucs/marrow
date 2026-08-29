from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT trim(s) AS p FROM words

    -- expected
    p:string
    'Hello'
    'wORLD'
    'pad'
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    return t.project(["p"], [Strip(col("s", string))])
