from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did FROM emp e ASOF JOIN dept d ON e.dept >= d.did ORDER BY e.eid NULLS FIRST

    An as-of join takes the *closest* match rather than all of them, so `dept =
    99` pairs with `did = 30` and not with all three. It is a non-equi join
    with a tie-break, and the strategy — backward, forward or nearest — is part
    of the semantics rather than an optimisation.

    -- skip mojo
    -- skip python

    -- expected
    eid:int64	dept:int64	did:int64
    1	10	10
    2	20	20
    3	20	20
    4	99	30
    """
    var joined = table("emp").asof_join(
        table("dept"), col("dept", int64) >= col("did", int64)
    )
    return joined.select(["eid", "dept", "did"]).sort_by(
        [col("eid", int64)], [True]
    )
