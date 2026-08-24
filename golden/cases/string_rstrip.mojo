from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT rtrim(s) AS r FROM words

    Trailing whitespace only -- the mirror of `lstrip`, and `  pad  ` keeps its leading spaces.

    -- expected
    r:string
    'Hello'
    'wORLD'
    '  pad'
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["r"], [col("s", string).rstrip()])
    return q
