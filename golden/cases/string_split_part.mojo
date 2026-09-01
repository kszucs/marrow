from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT split_part(t, ',', 2) AS s FROM text

    The n-th field of a split, 1-based. The rows with no separator are the
    question: DuckDB answers the empty string for an index past the end, where
    several engines answer null.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip python

    -- expected
    s:string
    'b'
    ''
    ''
    NULL
    ''
    ''
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).split_part(",", 2)])
