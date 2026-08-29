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
    return t.project(["l"], [Lower(col("s", string))])
