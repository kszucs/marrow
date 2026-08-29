from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ltrim(s) AS l FROM words

    Leading whitespace only -- `  pad  ` keeps its trailing spaces, which is what separates this from `strip`.

    -- expected
    l:string
    'Hello'
    'wORLD'
    'pad  '
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    return t.project(["l"], [col("s", string).lstrip()])
