from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT isinf(10.0 / n) AND 10.0 / n > 0 AS pos, isinf(-10.0 / n) AND -10.0 / n < 0 AS neg, isnan(0.0 / n) AS undef FROM floats

    `/` by zero has a value where `//` and `%` have a null, and the value
    depends on the dividend: `+inf`, `-inf` and `nan`.
    `math_integer_division_by_zero` asks the other half on the same column.

    Asked as three booleans because the corpus cannot write `inf` or `nan` as
    a *result* -- see "Consciously omitted" in `COVERAGE.md`. The `AND` is
    what separates the signs, which `isinf` alone does not, and the divisor is
    the int64 column on purpose: `/` is `float64` in marrow as in DuckDB, so
    an integer zero divisor still reaches the float arm of `DivKernel.core`.

    -- expected
    pos:bool	neg:bool	undef:bool
    False	False	False
    False	False	False
    True	True	True
    False	False	False
    False	False	False
    False	False	False
    False	False	False
    NULL	NULL	NULL
    """
    var t = table("floats")
    var p = lit(10.0, float64) / col("n", int64)
    var m = lit(-10.0, float64) / col("n", int64)
    var pos = p.is_inf() & (p > lit(0.0, float64))
    var neg = m.is_inf() & (m < lit(0.0, float64))
    var undef = (lit(0.0, float64) / col("n", int64)).is_nan()
    return t.project(["pos", "neg", "undef"], [pos, neg, undef])
