from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST((v + w) * 2 - v AS BIGINT) AS e FROM basic

    A three-level arithmetic tree over two columns, where the rest of the
    arithmetic family is one operator deep. The fused lane compiles the whole
    subtree into one loop, so this is where an intermediate that was
    materialised with the wrong validity — or a lane that read the wrong
    operand's bound — shows up. Null propagates through every level: a row null
    in either column is null in the result.

    -- expected
    e:int64
    21
    NULL
    63
    84
    NULL
    126
    147
    """
    var t = table("basic")
    return t.project(
        ["e"],
        [(col("v", int64) + col("w", int64)) * lit(2, int64) - col("v", int64)],
    )
