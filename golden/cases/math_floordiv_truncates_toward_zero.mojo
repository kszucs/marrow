from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n // 3 AS q FROM floats

    Integer division's rounding direction, which the two languages under marrow
    disagree about. Mojo's `//` is Python's — it floors — and `FloordivKernel`
    passes it straight through. SQL's `//` truncates toward zero, so every
    negative dividend that is not an exact multiple differs by one.

    `n = -1` is the discriminating row: SQL says 0, marrow says -1. The rest of
    the column (4, -9, 0, 1, 2, 3) agrees under both rules, which is why
    `math_mod_int64` could sidestep the question with `((n % 3) + 3) % 3` and
    this one cannot.

    -- xfail marrow's // floors (Mojo/Python semantics), DuckDB's truncates toward zero: -1 // 3 is 0 in SQL and -1 here

    -- expected
    q:int64
    1
    -3
    0
    0
    0
    1
    0
    NULL
    """
    var t = table("floats")
    return t.project(["q"], [col("n", int64) // lit(3, int64)])
