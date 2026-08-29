from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, string_agg(CAST(v AS VARCHAR), ',' ORDER BY v NULLS FIRST) AS s FROM basic GROUP BY k ORDER BY k NULLS FIRST

    Concatenating a group into one string. Nulls are skipped rather than
    stringified, so group `b` contributes one element and not two, and the
    `ORDER BY` is what makes the result assertable at all.

    This is the renderable member of the collect-into-a-container family;
    `array_agg` returns a list, which the expectation block cannot hold.

    -- skip mojo

    -- expected
    k:string	s:string
    NULL	'7'
    'a'	'1,3,6'
    'b'	'2'
    'c'	'4'
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            NumToString[StringType](col("v", int64))
            .string_agg(",", order_by=[col("v", int64)])
            .alias("s")
        ],
    )
    return agg.sort_by([col("k", string)], [True])
