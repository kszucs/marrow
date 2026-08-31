from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT regexp_extract(t, '([a-z]+)', 1) AS s FROM text

    Capture-group extraction. A row that does not match answers the empty
    string rather than null, which is the third "not found" convention in this
    family after `split_part`'s empty string and `position`'s zero.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    s:string
    'a'
    'xyz'
    ''
    NULL
    'b'
    'h'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).regexp_extract("([a-z]+)", 1)])
