from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, d.did FROM emp e CROSS JOIN dept d ORDER BY e.eid NULLS FIRST, d.did NULLS FIRST

    The Cartesian product — every left row against every right row, NULL keys
    included, because there are no keys. `DynRelation.join` requires two key
    lists, so the degenerate empty-key case has no spelling; an equi-join on a
    constant is the workaround and needs a constant column first.

    -- skip mojo
    -- skip python

    -- expected
    eid:int64	did:int64
    1	10
    1	20
    1	30
    2	10
    2	20
    2	30
    3	10
    3	20
    3	30
    4	10
    4	20
    4	30
    5	10
    5	20
    5	30
    """
    var joined = table("emp").cross_join(table("dept"))
    return joined.select(["eid", "did"]).sort_by(
        [col("eid", int64), col("did", int64)], [True, True]
    )
