from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT repeat(t, n) AS s FROM text

    A count taken from a *column*, and one that goes to zero and negative — the
    two values engines disagree about. Zero gives the empty string; a negative
    count gives the empty string in DuckDB and an error elsewhere.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip python

    -- expected
    s:string
    'a,b,c'
    'xyzxyz'
    ''
    NULL
    ''
    NULL
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).repeat(col("n", int64))])
