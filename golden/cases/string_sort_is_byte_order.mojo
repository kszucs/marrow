from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s FROM words ORDER BY s NULLS FIRST

    String ordering is bytewise, not locale-aware: `héllo` sorts between
    `Hello` and `wORLD` because its first byte is `h` (0x68), and ` pad ` sorts
    before both because a space is 0x20. A collation-aware engine would put
    `héllo` next to `Hello`. The empty string is the minimum, and the null is
    separate from it.

    -- expected
    s:string
    NULL
    ''
    '  pad  '
    'Hello'
    'héllo'
    'wORLD'
    """
    var t = table("words")
    return t.sort_by([col("s", string)], [True])
