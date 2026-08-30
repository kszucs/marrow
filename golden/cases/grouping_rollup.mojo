from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total, CAST(GROUPING(k) AS BIGINT) AS g FROM basic GROUP BY ROLLUP (k) ORDER BY g, k NULLS FIRST

    `ROLLUP` adds a grand-total row whose key is NULL — which collides with the
    genuine NULL group `basic.k` already has. `GROUPING(k)` is what
    distinguishes them, and a corpus that omitted it would assert an ambiguous
    answer.

    There is no multi-grouping node: `Aggregate` carries one key list.

    -- skip mojo

    -- expected
    k:string	total:int64	g:int64
    NULL	7	0
    'a'	10	0
    'b'	2	0
    'c'	4	0
    NULL	23	1
    """
    var t = table("basic")
    var agg = t.rollup(
        keys=[col("k", string)],
        aggs=[col("v", int64).sum().alias("total")],
    )
    return agg.sort_by([col("g", int64), col("k", string)], [True, True])
