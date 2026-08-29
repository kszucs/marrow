from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT exp(x) AS e FROM floats WHERE x = 0

    `exp` is **not** correctly rounded under IEEE 754, so DuckDB and marrow may
    legitimately differ in the last bit and a general input would make this case
    flake. The filter keeps only the rows where the result is exact: `exp(0)` is
    1.0 in any conforming implementation.

    Two rows survive, 0.0 and -0.0, which also asserts that the equality
    comparison treats the two zeroes as equal.

    -- expected
    e:double
    1.0
    1.0
    """
    var t = table("floats")
    var zeroes = t.filter(col("x", float64) == lit(0.0, float64))
    return zeroes.project(["e"], [col("x", float64).exp()])
