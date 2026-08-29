from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT reverse(s) AS r FROM words

    `héllo` is the row that matters: reversing by byte would split the
    two-byte `é` and produce invalid UTF-8, so the expectation asserts a
    character-wise reversal.

    -- expected
    r:string
    'olleH'
    'DLROw'
    '  dap  '
    ''
    'olléh'
    NULL
    """
    var t = table("words")
    return t.project(["r"], [col("s", string).reverse()])
