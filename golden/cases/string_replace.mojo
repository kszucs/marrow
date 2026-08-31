from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT replace(t, ',', ';') AS s FROM text

    Literal replacement, and it replaces **every** occurrence — `a,b,c` has
    two. The rows with no match come back unchanged rather than null, and the
    null row stays null.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    s:string
    'a;b;c'
    'xyz'
    ''
    NULL
    '  Ab  '
    'héllo wörld'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).replace(",", ";")])
