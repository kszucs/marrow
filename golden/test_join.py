"""Golden cases — joins, the runtime lane.

Every kind the engine claims to implement: inner, left, right, full, semi,
anti. `CROSS` is deliberately absent — `hash_join` rejects it and
`is_supported()` answers False, so there is nothing to compare against.

The fixtures are built so each case has something to prove: `dept` 20 matches
twice (cardinality), 99 matches nothing, and one key is NULL — which must
match nothing at all, including another NULL. On the right, `did` 30 is
unmatched, so the outer kinds widen in both directions.

Join output order is unspecified, so every case sorts before comparing.
"""


def test_golden_join_inner(golden):
    """SELECT e.eid, e.dept, d.did, d.dname FROM emp e JOIN dept d
    ON e.dept = d.did ORDER BY e.eid NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    golden.check(e.join(d, left_on="dept", right_on="did").order_by("eid"))


def test_golden_join_left(golden):
    """SELECT e.eid, e.dept, d.did, d.dname FROM emp e LEFT JOIN dept d
    ON e.dept = d.did ORDER BY e.eid NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    plan = e.join(d, left_on="dept", right_on="did", how="left")
    golden.check(plan.order_by("eid"))


def test_golden_join_right(golden):
    """SELECT e.eid, e.dept, d.did, d.dname FROM emp e RIGHT JOIN dept d
    ON e.dept = d.did ORDER BY d.did NULLS FIRST, e.eid NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    plan = e.join(d, left_on="dept", right_on="did", how="right")
    golden.check(plan.order_by("did", "eid"))


def test_golden_join_full(golden):
    """SELECT e.eid, e.dept, d.did, d.dname FROM emp e FULL OUTER JOIN dept d
    ON e.dept = d.did ORDER BY e.eid NULLS FIRST, d.did NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    plan = e.join(d, left_on="dept", right_on="did", how="full")
    golden.check(plan.order_by("eid", "did"))


def test_golden_join_semi(golden):
    """SELECT e.eid, e.dept FROM emp e WHERE EXISTS
    (SELECT 1 FROM dept d WHERE e.dept = d.did) ORDER BY e.eid NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    plan = e.join(d, left_on="dept", right_on="did", how="semi")
    golden.check(plan.order_by("eid"))


def test_golden_join_anti(golden):
    """SELECT e.eid, e.dept FROM emp e WHERE NOT EXISTS
    (SELECT 1 FROM dept d WHERE e.dept = d.did) ORDER BY e.eid NULLS FIRST"""
    e = golden.table("emp")
    d = golden.table("dept")
    plan = e.join(d, left_on="dept", right_on="did", how="anti")
    golden.check(plan.order_by("eid"))
