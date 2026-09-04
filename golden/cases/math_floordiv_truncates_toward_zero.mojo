from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n // 3 AS q FROM floats

    Integer division's rounding direction, which the two languages under marrow
    disagree about. Mojo's `//` is Python's — it floors — and `FloordivKernel`
    corrects for it: SQL truncates toward zero, so every negative dividend that
    is not an exact multiple differs by one.

    `n = -1` is the discriminating row: both rules answer 0 only under
    truncation. The rest of the column (4, -9, 0, 1, 2, 3) agrees under either,
    which is why `math_mod_int64` could once sidestep the question with
    `((n % 3) + 3) % 3`.

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
