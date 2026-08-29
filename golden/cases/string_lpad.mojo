from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT lpad(t, 8, '*') AS s FROM text

    Padding is also **truncation**: a string longer than the width is cut, not
    left alone. And the width counts characters, so `héllo wörld` (11
    characters, 13 bytes) is cut to 8 characters.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    '***a,b,c'
    '*****xyz'
    '********'
    NULL
    '**  Ab  '
    'héllo wö'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).lpad(8, "*")])
