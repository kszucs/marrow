from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, d.did FROM emp e JOIN dept d ON e.dept > d.did ORDER BY e.eid NULLS FIRST, d.did NULLS FIRST

    An inequality join condition. A hash join cannot answer it at all — the
    predicate is not an equality on a key — so this needs a different algorithm
    rather than a different key list, which is why it is a deeper gap than the
    cross join.

    -- skip mojo

    -- expected
    eid:int64	did:int64
    2	10
    3	10
    4	10
    4	20
    4	30
    """
    var joined = table("emp").join_on(
        table("dept"), col("dept", int64) > col("did", int64), JOIN_INNER
    )
    return joined.select(["eid", "did"]).sort_by(
        [col("eid", int64), col("did", int64)], [True, True]
    )
