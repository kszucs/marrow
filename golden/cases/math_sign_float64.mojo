from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sign(x) AS DOUBLE) AS s FROM floats

    `sign` is the one numeric map that takes the whole `floats` column without
    a filter: every edge value collapses to a finite -1, 0 or 1, so NaN and both
    infinities are covered here rather than being filtered away.

    The cast is not cosmetic. DuckDB's `sign` returns TINYINT, which the corpus
    type map does not carry, and marrow's `sign` is a unary numeric kernel whose
    output type is its input type -- double in, double out. Casting the twin to
    DOUBLE states the type both sides must agree on.

    -- expected
    s:double
    1.0
    -1.0
    0.0
    0.0
    0.0
    1.0
    -1.0
    NULL
    """
    var t = table("floats")
    return t.project(["s"], [col("x", float64).sign()])
