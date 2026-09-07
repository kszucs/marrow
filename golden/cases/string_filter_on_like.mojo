from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s FROM words WHERE s LIKE '%o%'

    -- expected
    s:string
    'Hello'
    'héllo'
    """
    var t = table("words")
    return t.filter(col("s", string).like(lit("%o%", string)))
