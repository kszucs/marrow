def test_golden_join_right() raises:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e RIGHT JOIN dept d ON e.dept = d.did ORDER BY d.did NULLS FIRST, e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	10	'eng'
    2	20	20	'sales'
    3	20	20	'sales'
    NULL	NULL	30	'ops'
    """
    var left = table("emp")
    var right = table("dept")
    var joined = left.join(
        right,
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=JOIN_RIGHT,
        strictness=JOIN_ALL,
    )
    var q = joined.sort([col("did", int64), col("eid", int64)], [True, True])
    check(q)
