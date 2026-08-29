from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sign(n) AS BIGINT) AS s FROM floats

    `sign` over an integer column. The cast is DuckDB's TINYINT result widened to the corpus type map; marrow returns the input type, int64.

    -- expected
    s:int64
    1
    -1
    0
    1
    1
    1
    -1
    NULL
    """
    var t = table("floats")
    return t.project(["s"], [col("n", int64).sign()])
