# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Subqueries, derived tables, CTEs and comma joins through `sql`, answered
against data.

`emp` has a NULL `dept` (eid 5) and a `dept` no department has (eid 4), the two
rows that tell `EXISTS` from `IN`, `NOT EXISTS` from `NOT IN`, and a left join's
null from `count(*)`'s 0. `part`, `partsupp`, `lineitem` and `supp` are
TPC-H's tables cut down to the columns its subquery queries read.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...dtypes import int64
from ...tabular import RecordBatch, record_batch
from ..sql import Catalog, sql


def _catalog() raises -> Catalog:
    var catalog = Catalog()
    catalog.add(
        "emp",
        record_batch(
            [
                array([1, 2, 3, 4, 5], int64).copy(),
                array([10, 20, 20, 99, None], int64).copy(),
            ],
            names=["eid", "dept"],
        ),
    )
    catalog.add(
        "dept",
        record_batch(
            [
                array([10, 20, 30], int64).copy(),
                array([100, 300, 50], int64).copy(),
            ],
            names=["did", "budget"],
        ),
    )
    catalog.add(
        "part",
        record_batch(
            [
                array([1, 2, 3], int64).copy(),
                array(["a", "a", "b"]).copy(),
            ],
            names=["p_partkey", "p_brand"],
        ),
    )
    catalog.add(
        "partsupp",
        record_batch(
            [
                array([1, 1, 2, 2, 3], int64).copy(),
                array([7, 8, 8, 9, 9], int64).copy(),
                array([5, 6, 1, 3, 100], int64).copy(),
            ],
            names=["ps_partkey", "ps_suppkey", "ps_availqty"],
        ),
    )
    catalog.add(
        "lineitem",
        record_batch(
            [
                array([1, 1, 1, 2, 2, 3], int64).copy(),
                array([7, 7, 8, 8, 9, 9], int64).copy(),
                array([1, 10, 10, 4, 4, 1], int64).copy(),
                array([100, 200, 300, 400, 500, 600], int64).copy(),
            ],
            names=["l_partkey", "l_suppkey", "l_quantity", "l_price"],
        ),
    )
    catalog.add(
        "supp",
        record_batch(
            [
                array([7, 8, 9], int64).copy(),
                array(["s7", "s8", "s9"]).copy(),
            ],
            names=["sid", "sname"],
        ),
    )
    # Order 1 has two suppliers, order 2 one, order 3 two, of which supplier
    # 9 was late.
    catalog.add(
        "lines",
        record_batch(
            [
                array([1, 1, 2, 3, 3], int64).copy(),
                array([7, 8, 7, 7, 9], int64).copy(),
                array([0, 0, 0, 0, 1], int64).copy(),
            ],
            names=["l_orderkey", "l_suppkey", "late"],
        ),
    )
    return catalog^


def _rows(query: String) raises -> String:
    """The answer to `query`, one `;`-separated row of `,`-separated values."""
    var batch = sql(query, _catalog()).execute()
    var out = String("")
    for r in range(batch.num_rows()):
        if r > 0:
            out += ";"
        for c in range(batch.num_columns()):
            if c > 0:
                out += ","
            out += String(batch.column(c)[r])
    return out^


def test_sql_in_subquery() raises:
    assert_equal(
        _rows(
            "SELECT eid FROM emp WHERE dept IN (SELECT did FROM dept) ORDER"
            " BY eid"
        ),
        "1;2;3",
    )


def test_sql_not_in_subquery_null_operand_is_unknown() raises:
    """Employee 5's dept is NULL: unknown against a non-empty subquery."""
    assert_equal(
        _rows(
            "SELECT eid FROM emp WHERE dept NOT IN (SELECT did FROM dept)"
            " ORDER BY eid"
        ),
        "4",
    )


def test_sql_not_in_subquery_holding_null_keeps_nothing() raises:
    """`x NOT IN (10, 20, 20, 99, NULL)` is never true."""
    assert_equal(
        _rows("SELECT eid FROM emp WHERE eid NOT IN (SELECT dept FROM emp)"),
        "",
    )


def test_sql_not_in_empty_subquery_keeps_everything() raises:
    assert_equal(
        _rows(
            "SELECT eid FROM emp WHERE dept NOT IN (SELECT did FROM dept"
            " WHERE did > 1000) ORDER BY eid"
        ),
        "1;2;3;4;5",
    )


def test_sql_exists_correlated() raises:
    assert_equal(
        _rows(
            "SELECT eid FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE"
            " d.did = e.dept) ORDER BY eid"
        ),
        "1;2;3",
    )


def test_sql_not_exists_keeps_null_key() raises:
    """Employee 5's NULL dept matches nothing, so `NOT EXISTS` holds."""
    assert_equal(
        _rows(
            "SELECT eid FROM emp e WHERE NOT EXISTS (SELECT 1 FROM dept d"
            " WHERE d.did = e.dept) ORDER BY eid"
        ),
        "4;5",
    )


def test_sql_unqualified_outer_column() raises:
    """`dept` is not a column of `dept`'s table, so it is the outer one."""
    assert_equal(
        _rows(
            "SELECT eid FROM emp WHERE EXISTS (SELECT * FROM dept WHERE did ="
            " dept AND budget > 200) ORDER BY eid"
        ),
        "2;3",
    )


def test_sql_uncorrelated_scalar() raises:
    assert_equal(
        _rows(
            "SELECT eid FROM emp WHERE eid > (SELECT count(*) FROM dept)"
            " ORDER BY eid"
        ),
        "4;5",
    )


def test_sql_correlated_count_is_zero_unmatched() raises:
    """`count(*)` over no rows is 0, not the null a left join leaves."""
    assert_equal(
        _rows(
            "SELECT eid, (SELECT count(*) FROM dept d WHERE d.did = e.dept)"
            " AS n FROM emp e ORDER BY eid"
        ),
        "1,1;2,1;3,1;4,0;5,0",
    )


def test_sql_correlated_aggregate_q17() raises:
    """TPC-H Q17's shape: a correlated average over the same table the outer
    query joins, read through the comma join's `part.p_partkey`."""
    assert_equal(
        _rows(
            "SELECT sum(l_price) AS total FROM lineitem, part WHERE p_partkey"
            " = l_partkey AND p_brand = 'a' AND l_quantity < (SELECT 0.5 *"
            " avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)"
        ),
        "100",
    )


def test_sql_exists_with_residual_q21() raises:
    """TPC-H Q21's shape: another line of the same order from a different
    supplier, and none of those late. `lines` is aliased in all three
    queries, so the alias alone decides which one a column belongs to."""
    assert_equal(
        _rows(
            "SELECT s.sname, l1.l_orderkey FROM supp s, lines l1 WHERE s.sid ="
            " l1.l_suppkey AND EXISTS (SELECT * FROM lines l2 WHERE"
            " l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)"
            " AND NOT EXISTS (SELECT * FROM lines l3 WHERE l3.l_orderkey ="
            " l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.late = 1)"
            " ORDER BY l1.l_orderkey, s.sname"
        ),
        "s7,1;s8,1;s9,3",
    )


def test_sql_nested_subqueries_q20() raises:
    """TPC-H Q20's shape: `IN` inside `IN`, beside a scalar correlated on two
    columns of the middle query."""
    assert_equal(
        _rows(
            "SELECT sname FROM supp WHERE sid IN (SELECT ps_suppkey FROM"
            " partsupp WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE"
            " p_brand = 'a') AND ps_availqty > (SELECT 0.5 * sum(l_quantity)"
            " FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey ="
            " ps_suppkey)) ORDER BY sname"
        ),
        "s8;s9",
    )


def test_sql_cte_read_twice_q15() raises:
    """TPC-H Q15's shape: one CTE joined and aggregated over again."""
    assert_equal(
        _rows(
            "WITH revenue AS (SELECT l_suppkey AS supplier_no, sum(l_price) AS"
            " total FROM lineitem GROUP BY l_suppkey) SELECT s.sid, s.sname,"
            " r.total FROM supp s, revenue r WHERE s.sid = r.supplier_no AND"
            " r.total = (SELECT max(total) FROM revenue) ORDER BY s.sid"
        ),
        "9,s9,1100",
    )


def test_sql_cte_reads_an_earlier_cte() raises:
    assert_equal(
        _rows(
            "WITH a AS (SELECT did FROM dept WHERE budget > 60), b AS (SELECT"
            " did FROM a WHERE did > 10) SELECT eid FROM emp WHERE dept IN"
            " (SELECT did FROM b) ORDER BY eid"
        ),
        "2;3",
    )


def test_sql_derived_table() raises:
    assert_equal(
        _rows(
            "SELECT x.dept, x.n FROM (SELECT dept, count(*) AS n FROM emp"
            " GROUP BY dept) AS x WHERE x.n > 1"
        ),
        "20,2",
    )


def test_sql_comma_join_waits_for_a_connected_item() raises:
    """`supp` shares no column with `part`; it joins once `partsupp` has."""
    assert_equal(
        _rows(
            "SELECT p.p_partkey, s.sname FROM part p, supp s, partsupp ps"
            " WHERE p.p_partkey = ps.ps_partkey AND s.sid = ps.ps_suppkey AND"
            " p.p_brand = 'b'"
        ),
        "3,s9",
    )


def test_sql_comma_join_without_an_equality_raises() raises:
    with assert_raises(contains="cross joins are not supported"):
        _ = _rows("SELECT eid FROM emp, dept WHERE eid > did")


def test_sql_correlation_two_levels_out_raises() raises:
    with assert_raises(contains="2 levels out"):
        _ = _rows(
            "SELECT eid FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE"
            " d.did = e.dept AND EXISTS (SELECT 1 FROM part p WHERE"
            " p.p_partkey = e.eid))"
        )


def test_sql_plan_has_no_subquery_left() raises:
    """`sql` answers a plan that runs: its subqueries already decorrelated."""
    var plan = sql(
        "SELECT eid FROM emp WHERE dept IN (SELECT did FROM dept)", _catalog()
    )
    assert_true("semi" in String(plan))


def test_sql_cte_read_twice_is_shared() raises:
    var plan = sql(
        (
            "WITH r AS (SELECT did FROM dept) SELECT eid FROM emp WHERE dept IN"
            " (SELECT did FROM r) AND eid IN (SELECT did FROM r)"
        ),
        _catalog(),
    )
    assert_true("Shared" in String(plan))
    var once = sql(
        "WITH r AS (SELECT did FROM dept) SELECT did FROM r", _catalog()
    )
    assert_true("Shared" not in String(once))
