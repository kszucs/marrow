from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ((n % 3) + 3) % 3 AS m FROM floats

    A recorded divergence, and the twin is written to ask DuckDB marrow's
    question. marrow's arithmetic operators are **Python's**, coherently: `/`
    is true division returning a float, `//` floors, and `%` takes the sign of
    the divisor, so `-1 % 3` is 2 and `a == (a // b) * b + a % b` holds. SQL,
    PyArrow and arrow-rs instead truncate toward zero and answer -1, and
    PyArrow's `divide` on integers returns 0 where marrow's `/` returns
    -0.333.

    So a bare `n % 3` twin would assert SQL's convention and report marrow's
    deliberate one as a defect. `((n % 3) + 3) % 3` is floored modulo spelled
    in SQL, valid because the divisor is positive.

    -- expected
    m:int64
    1
    0
    0
    1
    2
    0
    2
    NULL
    """
    var t = table("floats")
    return t.project(["m"], [col("n", int64) % lit(3, int64)])
