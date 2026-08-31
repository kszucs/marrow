from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT (n & 3) AS a, (n | 3) AS o, xor(n, 3) AS x, (abs(n) << 1) AS l FROM floats

    Bitwise and, or, exclusive-or and left shift over integers. The negative
    values are the point: these are defined on the two's-complement
    representation, so `-9 & 3` is 3 and `-1 << 1` is -2, and an implementation
    that reasons about magnitude gets them wrong.

    The shift takes `abs(n)` because DuckDB refuses to left-shift a negative
    number outright — so that one operation is a place the two engines cannot
    be compared at all, only the non-negative half of it.

    `marrow/kernels/numeric.mojo` has no such kernel.
    `marrow/kernels/boolean.mojo`'s `and`/`or`/`xor` are *logical*, over bit-
    packed booleans, which is a different operation on a different layout.

    -- skip mojo
    -- skip python

    -- expected
    a:int64	o:int64	x:int64	l:int64
    0	7	7	8
    3	-9	-12	18
    0	3	3	0
    1	3	2	2
    2	3	1	4
    3	3	0	6
    3	-1	-4	2
    NULL	NULL	NULL	NULL
    """
    var t = table("floats")
    return t.project(
        ["a", "o", "x", "l"],
        [
            col("n", int64) & lit(3, int64),
            col("n", int64) | lit(3, int64),
            col("n", int64).bit_xor(lit(3, int64)),
            col("n", int64).abs() << lit(1, int64),
        ],
    )
