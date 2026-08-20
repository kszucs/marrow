from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT upper(s) AS u FROM words

    -- expected
    u:string
    'HELLO'
    'WORLD'
    '  PAD  '
    ''
    'HÉLLO'
    NULL
    """
    var t = table("words")
    var q = t.project(["u"], [Upper(col("s", string))])
    return q
