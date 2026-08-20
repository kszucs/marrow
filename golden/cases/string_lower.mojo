from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT lower(s) AS l FROM words

    -- expected
    l:string
    'hello'
    'world'
    '  pad  '
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["l"], [Lower(col("s", string))])
    return q
