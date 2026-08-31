from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(bit_and(v) AS BIGINT) AS a, CAST(bit_or(v) AS BIGINT) AS o, CAST(bit_xor(v) AS BIGINT) AS x FROM basic

    The bitwise reductions. They are ordinary folds with an ordinary identity,
    so they would fit `Fold[K, V]` unchanged — the gap is three kernels, not a
    shape. Nulls are skipped, so `bit_and` is not 0.

    -- skip mojo
    -- skip python

    -- expected
    a:int64	o:int64	x:int64
    0	7	5
    """
    var t = table("basic")
    return t.aggregate(
        aggs=[
            col("v", int64).bit_and().alias("a"),
            col("v", int64).bit_or().alias("o"),
            col("v", int64).bit_xor().alias("x"),
        ]
    )
