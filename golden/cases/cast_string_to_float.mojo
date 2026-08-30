from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT TRY_CAST(s AS DOUBLE) AS c FROM nums

    String to floating point, where `cast_string_to_int` covers string to
    integer. A different parser: `'abc'` still fails to null, but the accepting
    rows have to produce exact doubles rather than integers.

    `TRY_CAST` for the same reason as the integer case — DuckDB's plain `CAST`
    raises on `'abc'` where marrow nulls it.

    -- expected
    c:double
    1.0
    -2.0
    NULL
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [StringToNum[Float64Type](col("s", string))])
