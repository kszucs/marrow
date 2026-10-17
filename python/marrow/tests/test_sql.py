# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""``marrow.sql`` — the SQL front end through the Python bindings.

Two things are checked, and they are different:

- that a clause **builds the plan node it says it does**, readable off
  ``explain()`` without running anything, and
- that ``collect()`` then produces the rows a SQL engine would.

The whole golden corpus is driven through this same entry point by
``golden/test_sql.py``; what is here are the cases that pin the front end's own
decisions — how ``COUNT(*)`` differs from ``COUNT(x)``, what ``IN`` does with a
NULL, which errors are refusals — rather than the engine's behaviour.
"""

import pytest

import marrow as ma


@pytest.fixture
def basic():
    """Seven rows with nulls in both value columns, and a null key."""
    return ma.record_batch(
        {
            "k": ma.array(["a", "b", "a", "c", "b", "a", None]),
            "v": ma.array([1, 2, 3, 4, None, 6, 7], type=ma.int64()),
            "w": ma.array([10, None, 30, 40, 50, 60, 70], type=ma.int64()),
        }
    )


@pytest.fixture
def dept():
    return ma.record_batch(
        {
            "did": ma.array([1, 2, 3], type=ma.int64()),
            "dname": ma.array(["eng", "ops", "hr"]),
        }
    )


@pytest.fixture
def emp():
    return ma.record_batch(
        {
            "name": ma.array(["ann", "bob", "cid"]),
            "did": ma.array([1, 2, 1], type=ma.int64()),
        }
    )


def rows(query, **tables):
    return ma.sql(query, tables).collect().to_pylist()


# ── the API surface ────────────────────────────────────────────────────────


def test_sql_returns_a_lazy_table(basic):
    """A parsed query is an ordinary plan, so it explains without running."""
    plan = ma.sql("SELECT k FROM basic", basic=basic)
    assert isinstance(plan, ma.LazyTable)
    assert plan.column_names == ["k"]
    assert "InMemoryTable" in plan.explain()


def test_tables_may_be_a_mapping_or_keywords(basic):
    by_keyword = ma.sql("SELECT k FROM basic", basic=basic)
    by_mapping = ma.sql("SELECT k FROM basic", {"basic": basic})
    assert by_keyword.explain() == by_mapping.explain()


def test_a_parsed_plan_keeps_composing(basic):
    """The point of answering a plan rather than a table: the built verbs
    still apply to a parsed query."""
    plan = ma.sql("SELECT k, v FROM basic", basic=basic).limit(2)
    assert len(rows("SELECT k FROM basic", basic=basic)) == 7
    assert len(plan.collect().to_pylist()) == 2


def test_table_names_are_case_insensitive(basic):
    assert len(rows("SELECT k FROM BASIC", basic=basic)) == 7


def test_sql_needs_a_table():
    with pytest.raises(ValueError):
        ma.sql("SELECT 1")


# ── clauses ────────────────────────────────────────────────────────────────


def test_select_projects_in_order(basic):
    assert ma.sql("SELECT w, k FROM basic", basic=basic).column_names == ["w", "k"]


def test_star_expands_to_every_column(basic):
    assert ma.sql("SELECT * FROM basic", basic=basic).column_names == [
        "k",
        "v",
        "w",
    ]


def test_alias_names_the_output(basic):
    assert ma.sql("SELECT v AS value FROM basic", basic=basic).column_names == ["value"]


def test_where_filters(basic):
    assert [r["v"] for r in rows("SELECT v FROM basic WHERE v >= 3", basic=basic)] == [
        3,
        4,
        6,
        7,
    ]


def test_where_builds_a_filter_node(basic):
    assert "Filter" in ma.sql("SELECT v FROM basic WHERE v >= 3", basic=basic).explain()


def test_null_predicate_is_not_true(basic):
    """SQL's rule: the row with a null `v` is not returned by either
    comparison."""
    kept = rows("SELECT v FROM basic WHERE v < 4", basic=basic)
    dropped = rows("SELECT v FROM basic WHERE v >= 4", basic=basic)
    assert len(kept) + len(dropped) == 6  # not 7 — the null row is in neither


def test_order_by_and_limit(basic):
    assert [
        r["v"]
        for r in rows(
            "SELECT v FROM basic WHERE v IS NOT NULL ORDER BY v DESC LIMIT 2",
            basic=basic,
        )
    ] == [7, 6]


def test_order_by_an_ordinal(basic):
    assert [
        r["v"]
        for r in rows(
            "SELECT v FROM basic WHERE v IS NOT NULL ORDER BY 1 LIMIT 2",
            basic=basic,
        )
    ] == [1, 2]


def test_offset_without_limit(basic):
    assert len(rows("SELECT v FROM basic ORDER BY v OFFSET 5", basic=basic)) == 2


def test_distinct_groups_by_every_output_column(basic):
    out = rows("SELECT DISTINCT k FROM basic", basic=basic)
    assert sorted(r["k"] for r in out if r["k"] is not None) == ["a", "b", "c"]
    assert len(out) == 4  # three keys and the null


def test_union_all_and_union(emp, dept):
    query = "SELECT did FROM emp {} SELECT did FROM dept ORDER BY did"
    assert [r["did"] for r in rows(query.format("UNION ALL"), emp=emp, dept=dept)] == [
        1,
        1,
        1,
        2,
        2,
        3,
    ]
    assert [r["did"] for r in rows(query.format("UNION"), emp=emp, dept=dept)] == [
        1,
        2,
        3,
    ]


def test_except_and_intersect(emp, dept):
    out = rows("SELECT did FROM dept EXCEPT SELECT did FROM emp", emp=emp, dept=dept)
    assert out == [{"did": 3}]
    out = rows(
        "SELECT did FROM emp INTERSECT ALL SELECT did FROM dept ORDER BY 1",
        emp=emp,
        dept=dept,
    )
    assert [r["did"] for r in out] == [1, 2]


def test_intersect_binds_tighter_than_union(emp, dept):
    """`a UNION b INTERSECT c` is `a UNION (b INTERSECT c)`: read left to
    right it would be `{1, 2, 3} INTERSECT {3}`, just `3`."""
    out = rows(
        "SELECT did FROM emp UNION SELECT did FROM dept"
        " INTERSECT SELECT did FROM dept WHERE did = 3 ORDER BY did",
        emp=emp,
        dept=dept,
    )
    assert [r["did"] for r in out] == [1, 2, 3]


def test_a_compound_order_by_must_name_an_output_column(emp, dept):
    with pytest.raises(Exception, match="sql: ORDER BY over a set operation"):
        rows(
            "SELECT did FROM emp UNION SELECT did FROM dept ORDER BY did + 1",
            emp=emp,
            dept=dept,
        )


# ── aggregates ─────────────────────────────────────────────────────────────


def test_group_by_with_sum(basic):
    out = {
        r["k"]: r["total"]
        for r in rows("SELECT k, SUM(v) AS total FROM basic GROUP BY k", basic=basic)
    }
    assert out["a"] == 10  # 1 + 3 + 6
    assert out["b"] == 2  # 2, and the null contributes nothing


def test_count_star_counts_rows_and_count_column_skips_nulls(basic):
    """The distinction the front end has to make: `count` counts non-null
    values, so `COUNT(*)` is counted over a constant instead."""
    out = rows("SELECT COUNT(*) AS n, COUNT(v) AS nv FROM basic", basic=basic)[0]
    assert out["n"] == 7
    assert out["nv"] == 6


def test_count_distinct(basic):
    out = rows("SELECT COUNT(DISTINCT k) AS n FROM basic", basic=basic)[0]
    assert out["n"] == 3


def test_aggregate_without_group_by_is_one_row(basic):
    assert len(rows("SELECT SUM(v) AS t FROM basic", basic=basic)) == 1


def test_expression_over_two_aggregates(basic):
    """`SUM(v) / COUNT(*)` needs the aggregates hoisted and the arithmetic
    left behind — the case that made the rewrite order matter."""
    out = rows("SELECT SUM(v) / COUNT(*) AS ratio FROM basic", basic=basic)[0]
    assert out["ratio"] == pytest.approx(23 / 7)


def test_group_by_a_computed_key(basic):
    """`Aggregate` names a computed key `key0`, not "" — recording the
    expression's own (empty) name is what made this raise `column '' not
    found in schema`."""
    out = rows(
        "SELECT UPPER(k) AS u, COUNT(*) AS n FROM basic"
        " WHERE k IS NOT NULL GROUP BY UPPER(k)",
        basic=basic,
    )
    assert sorted((r["u"], r["n"]) for r in out) == [("A", 3), ("B", 2), ("C", 1)]


def test_group_by_an_ordinal(basic):
    """`GROUP BY 1` groups by the first SELECT item, as `ORDER BY 1` sorts by
    it — not by the literal 1, which would make one group of everything."""
    out = rows("SELECT k, COUNT(*) AS n FROM basic GROUP BY 1", basic=basic)
    assert len(out) == 4


def test_order_by_an_aggregate_that_is_not_selected(basic):
    """The sort key is resolved before the Aggregate node is built, and rides
    the projection as an extra column that is dropped again afterwards."""
    out = ma.sql("SELECT k FROM basic GROUP BY k ORDER BY COUNT(*) DESC", basic=basic)
    assert out.column_names == ["k"]
    assert out.collect().to_pylist()[0]["k"] == "a"


def test_order_by_mixes_directions(basic):
    """Two keys with different directions and no NULLS clause is not a
    conflict: the default placement is the same for both."""
    out = rows("SELECT k, v FROM basic ORDER BY k ASC, v DESC", basic=basic)
    assert len(out) == 7


def test_expressions_over_aggregates_of_every_shape(basic):
    """One walk translates both phases, so a `CASE`, an `IN` and a `BETWEEN`
    over an aggregate work for the same reason arithmetic over one does. The
    separate post-grouping walk this replaced handled five node kinds of
    fifteen and failed all three."""
    out = rows(
        "SELECT k, CASE WHEN SUM(v) > 3 THEN 'big' ELSE 'small' END AS size"
        " FROM basic GROUP BY k",
        basic=basic,
    )
    assert {r["k"]: r["size"] for r in out}["a"] == "big"
    assert (
        len(
            rows("SELECT k FROM basic GROUP BY k HAVING SUM(v) IN (2, 10)", basic=basic)
        )
        == 2
    )
    assert (
        len(
            rows(
                "SELECT k FROM basic GROUP BY k HAVING SUM(v) BETWEEN 1 AND 5",
                basic=basic,
            )
        )
        == 2
    )


def test_having_filters_groups(basic):
    out = rows(
        "SELECT k, COUNT(*) AS n FROM basic GROUP BY k HAVING COUNT(*) > 1",
        basic=basic,
    )
    assert sorted(r["k"] for r in out) == ["a", "b"]


def test_avg_is_mean(basic):
    out = rows("SELECT AVG(v) AS m FROM basic", basic=basic)[0]
    assert out["m"] == pytest.approx(23 / 6)


def test_where_rejects_an_aggregate(basic):
    with pytest.raises(ma.ArrowInvalid, match="WHERE cannot contain an aggregate"):
        ma.sql("SELECT k FROM basic WHERE SUM(v) > 1", basic=basic)


# ── expressions ────────────────────────────────────────────────────────────


def test_arithmetic_precedence(basic):
    out = rows("SELECT v + w * 2 AS x FROM basic WHERE v = 1", basic=basic)
    assert out[0]["x"] == 21  # 1 + 20, not (1 + 10) * 2


def test_cast(basic):
    out = ma.sql("SELECT CAST(v AS DOUBLE) AS d FROM basic", basic=basic)
    # The collected batch prints its dtypes under marrow's names, where SQL's
    # DOUBLE is `float64`; `Schema`'s own repr prints no dtype at all.
    assert "float64" in str(out.collect())


def test_case_when(basic):
    out = rows(
        "SELECT CASE WHEN v > 3 THEN 'big' ELSE 'small' END AS s FROM basic"
        " WHERE v IS NOT NULL",
        basic=basic,
    )
    assert [r["s"] for r in out] == ["small", "small", "small", "big", "big", "big"]


def test_in_list_follows_three_valued_logic(basic):
    """`IN` is an `OR` chain of equalities, so a NULL value matches nothing
    and does not become false — SQL's rule, and the reason `isin` was not
    used."""
    assert len(rows("SELECT v FROM basic WHERE v IN (1, 3)", basic=basic)) == 2
    assert len(rows("SELECT v FROM basic WHERE v NOT IN (1, 3)", basic=basic)) == 4


def test_between_is_inclusive(basic):
    assert [
        r["v"] for r in rows("SELECT v FROM basic WHERE v BETWEEN 2 AND 4", basic=basic)
    ] == [2, 3, 4]


def test_column_names_are_case_insensitive(basic):
    """SQL folds unquoted identifiers, and `Catalog.get` already folded table
    names; columns were the inconsistent half."""
    assert ma.sql("SELECT K FROM basic", basic=basic).column_names == ["K"]
    assert len(rows("SELECT k FROM basic WHERE V > 3", basic=basic)) == 3


def test_float_is_32_bit_like_duckdb(basic):
    """DuckDB's `FLOAT` is an alias for `REAL`."""
    out = ma.sql("SELECT CAST(v AS FLOAT) AS f FROM basic", basic=basic)
    assert "float32" in str(out.collect())


def test_limit_rejects_a_float(basic):
    with pytest.raises(ma.ArrowInvalid, match="whole number"):
        ma.sql("SELECT k FROM basic LIMIT 1.5", basic=basic)


def test_like(basic):
    assert len(rows("SELECT k FROM basic WHERE k LIKE 'a'", basic=basic)) == 3


def test_string_function(basic):
    out = rows("SELECT UPPER(k) AS u FROM basic WHERE k = 'a' LIMIT 1", basic=basic)
    assert out[0]["u"] == "A"


# ── joins ──────────────────────────────────────────────────────────────────


def test_inner_join(emp, dept):
    out = rows(
        "SELECT e.name, d.dname FROM emp e JOIN dept d ON e.did = d.did",
        emp=emp,
        dept=dept,
    )
    assert sorted((r["name"], r["dname"]) for r in out) == [
        ("ann", "eng"),
        ("bob", "ops"),
        ("cid", "eng"),
    ]


def test_join_disambiguates_a_shared_column_name(emp, dept):
    """Both sides have `did`; the qualified references must still resolve."""
    out = ma.sql(
        "SELECT e.did FROM emp e JOIN dept d ON e.did = d.did", emp=emp, dept=dept
    )
    assert out.column_names == ["did"]


def test_star_after_a_join_keeps_names_distinct(emp, dept):
    """Both sides carry `did`, and two output columns of one name are
    indistinguishable to every consumer — `to_pylist()` keeps one."""
    out = ma.sql("SELECT * FROM emp e JOIN dept d ON e.did = d.did", emp=emp, dept=dept)
    assert len(set(out.column_names)) == len(out.column_names)


def test_join_key_order_may_be_written_either_way(emp, dept):
    """`ON d.did = e.did` names the right table first; which side is which is
    decided by where the column actually lives."""
    swapped = rows(
        "SELECT e.name FROM emp e JOIN dept d ON d.did = e.did", emp=emp, dept=dept
    )
    assert len(swapped) == 3


# ── refusals ───────────────────────────────────────────────────────────────


def test_syntax_error_names_the_offset(basic):
    with pytest.raises(ma.ArrowInvalid, match="offset"):
        ma.sql("SELECT a # b FROM basic", basic=basic)


def test_unknown_column_is_named(basic):
    with pytest.raises(ma.ArrowKeyError, match="nope"):
        ma.sql("SELECT nope FROM basic", basic=basic)


def test_unknown_table_is_named(basic):
    with pytest.raises(ma.ArrowKeyError, match="missing"):
        ma.sql("SELECT k FROM missing", basic=basic)


def test_non_equi_join_is_refused(emp, dept):
    with pytest.raises(ma.ArrowInvalid, match="equality"):
        ma.sql(
            "SELECT e.name FROM emp e JOIN dept d ON e.did > d.did",
            emp=emp,
            dept=dept,
        )


def test_a_join_under_a_where_optimizes_to_one_join_chain(emp, dept):
    """`SELECT … FROM … JOIN … WHERE …`: the filter moves into the chain and
    the projection folds into its output, so the optimized plan is one join
    node, with the rows the written plan returns."""
    plan = ma.sql(
        "SELECT e.name, d.dname FROM emp e JOIN dept d ON e.did = d.did "
        "WHERE d.dname = 'eng'",
        emp=emp,
        dept=dept,
    )
    optimized = plan.optimize()
    text = optimized.explain()
    assert text.startswith("Join("), text
    assert sorted(r["name"] for r in optimized.collect().to_pylist()) == [
        "ann",
        "cid",
    ]
    assert sorted(r["name"] for r in plan.collect().to_pylist()) == ["ann", "cid"]


# ── subqueries, derived tables, CTEs and comma joins ───────────────────────


@pytest.fixture
def subq():
    """`staff` has a NULL `dept` (eid 5) and a `dept` no unit has (eid 4):
    the rows that tell EXISTS from IN and NOT EXISTS from NOT IN. `part`,
    `partsupp`, `lineitem` and `supp` are TPC-H's, cut down to the columns
    its subquery queries read; `lines` is three orders, one with a late
    supplier."""
    i64 = ma.int64()
    return {
        "staff": ma.record_batch(
            {
                "eid": ma.array([1, 2, 3, 4, 5], type=i64),
                "dept": ma.array([10, 20, 20, 99, None], type=i64),
            }
        ),
        "units": ma.record_batch(
            {
                "did": ma.array([10, 20, 30], type=i64),
                "budget": ma.array([100, 300, 50], type=i64),
            }
        ),
        "part": ma.record_batch(
            {
                "p_partkey": ma.array([1, 2, 3], type=i64),
                "p_brand": ma.array(["a", "a", "b"]),
            }
        ),
        "partsupp": ma.record_batch(
            {
                "ps_partkey": ma.array([1, 1, 2, 2, 3], type=i64),
                "ps_suppkey": ma.array([7, 8, 8, 9, 9], type=i64),
                "ps_availqty": ma.array([5, 6, 1, 3, 100], type=i64),
            }
        ),
        "lineitem": ma.record_batch(
            {
                "l_partkey": ma.array([1, 1, 1, 2, 2, 3], type=i64),
                "l_suppkey": ma.array([7, 7, 8, 8, 9, 9], type=i64),
                "l_quantity": ma.array([1, 10, 10, 4, 4, 1], type=i64),
                "l_price": ma.array([100, 200, 300, 400, 500, 600], type=i64),
            }
        ),
        "supp": ma.record_batch(
            {
                "sid": ma.array([7, 8, 9], type=i64),
                "sname": ma.array(["s7", "s8", "s9"]),
            }
        ),
        "lines": ma.record_batch(
            {
                "l_orderkey": ma.array([1, 1, 2, 3, 3], type=i64),
                "l_suppkey": ma.array([7, 8, 7, 7, 9], type=i64),
                "late": ma.array([0, 0, 0, 0, 1], type=i64),
            }
        ),
    }


def column(query, tables, name):
    return [r[name] for r in ma.sql(query, tables).collect().to_pylist()]


def test_in_subquery(subq):
    query = "SELECT eid FROM staff WHERE dept IN (SELECT did FROM units) ORDER BY eid"
    assert column(query, subq, "eid") == [1, 2, 3]


def test_a_planned_subquery_is_a_join(subq):
    """`sql` decorrelates, so the plan it answers runs as it stands."""
    plan = ma.sql("SELECT eid FROM staff WHERE dept IN (SELECT did FROM units)", subq)
    assert "semi" in plan.explain()


def test_not_in_subquery_is_three_valued(subq):
    """A NULL operand is unknown against a non-empty subquery, true against
    an empty one; a NULL in the subquery makes every miss unknown."""
    assert column(
        "SELECT eid FROM staff WHERE dept NOT IN (SELECT did FROM units) ORDER BY eid",
        subq,
        "eid",
    ) == [4]
    assert column(
        "SELECT eid FROM staff WHERE dept NOT IN "
        "(SELECT did FROM units WHERE did > 1000) ORDER BY eid",
        subq,
        "eid",
    ) == [1, 2, 3, 4, 5]
    assert (
        column(
            "SELECT did FROM units WHERE did NOT IN (SELECT dept FROM staff)",
            subq,
            "did",
        )
        == []
    )


def test_exists_and_not_exists(subq):
    exists = (
        "SELECT eid FROM staff s WHERE {} EXISTS "
        "(SELECT 1 FROM units u WHERE u.did = s.dept) ORDER BY eid"
    )
    assert column(exists.format(""), subq, "eid") == [1, 2, 3]
    # The NULL key matches nothing, so NOT EXISTS keeps it.
    assert column(exists.format("NOT"), subq, "eid") == [4, 5]


def test_an_unqualified_column_falls_back_to_the_enclosing_query(subq):
    query = (
        "SELECT eid FROM staff WHERE EXISTS "
        "(SELECT * FROM units WHERE did = dept AND budget > 200) ORDER BY eid"
    )
    assert column(query, subq, "eid") == [2, 3]


def test_uncorrelated_scalar_subquery(subq):
    query = (
        "SELECT eid FROM staff WHERE eid > (SELECT count(*) FROM units) ORDER BY eid"
    )
    assert column(query, subq, "eid") == [4, 5]
    query = "SELECT eid, (SELECT max(budget) FROM units) AS top FROM staff ORDER BY eid"
    assert column(query, subq, "top") == [300] * 5


def test_correlated_count_is_zero_where_nothing_matches(subq):
    query = (
        "SELECT eid, (SELECT count(*) FROM units u WHERE u.did = s.dept) AS n "
        "FROM staff s ORDER BY eid"
    )
    assert column(query, subq, "n") == [1, 1, 1, 0, 0]


def test_scalar_subquery_with_two_rows_raises(subq):
    with pytest.raises(Exception, match="at most one"):
        ma.sql(
            "SELECT eid FROM staff WHERE eid > (SELECT did FROM units)", subq
        ).collect()


def test_scalar_subquery_must_select_one_column(subq):
    with pytest.raises(ma.ArrowInvalid, match="sql: a subquery used as a value"):
        ma.sql(
            "SELECT eid FROM staff WHERE eid > (SELECT did, budget FROM units)", subq
        )


def test_q17_correlated_aggregate(subq):
    query = (
        "SELECT sum(l_price) AS total FROM lineitem, part "
        "WHERE p_partkey = l_partkey AND p_brand = 'a' AND l_quantity < "
        "(SELECT 0.5 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)"
    )
    assert column(query, subq, "total") == [100]


def test_q21_residual_correlation_by_alias(subq):
    """`lines` is read three times; only the alias says which."""
    query = (
        "SELECT s.sname, l1.l_orderkey FROM supp s, lines l1 "
        "WHERE s.sid = l1.l_suppkey "
        "AND EXISTS (SELECT * FROM lines l2 WHERE l2.l_orderkey = l1.l_orderkey "
        "AND l2.l_suppkey <> l1.l_suppkey) "
        "AND NOT EXISTS (SELECT * FROM lines l3 WHERE l3.l_orderkey = l1.l_orderkey "
        "AND l3.l_suppkey <> l1.l_suppkey AND l3.late = 1) "
        "ORDER BY l1.l_orderkey, s.sname"
    )
    assert ma.sql(query, subq).collect().to_pylist() == [
        {"sname": "s7", "l_orderkey": 1},
        {"sname": "s8", "l_orderkey": 1},
        {"sname": "s9", "l_orderkey": 3},
    ]


def test_q20_nested_subqueries(subq):
    query = (
        "SELECT sname FROM supp WHERE sid IN (SELECT ps_suppkey FROM partsupp "
        "WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_brand = 'a') "
        "AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem "
        "WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey)) ORDER BY sname"
    )
    assert column(query, subq, "sname") == ["s8", "s9"]


def test_q15_cte_read_twice(subq):
    query = (
        "WITH revenue AS (SELECT l_suppkey AS supplier_no, sum(l_price) AS total "
        "FROM lineitem GROUP BY l_suppkey) "
        "SELECT s.sid, r.total FROM supp s, revenue r "
        "WHERE s.sid = r.supplier_no AND r.total = (SELECT max(total) FROM revenue)"
    )
    assert ma.sql(query, subq).collect().to_pylist() == [{"sid": 9, "total": 1100}]


def test_a_cte_reads_an_earlier_one_and_shadows_a_table(subq):
    query = (
        "WITH units AS (SELECT did FROM units WHERE budget > 60), "
        "big AS (SELECT did FROM units WHERE did > 10) "
        "SELECT eid FROM staff WHERE dept IN (SELECT did FROM big) ORDER BY eid"
    )
    assert column(query, subq, "eid") == [2, 3]


def test_derived_table(subq):
    query = (
        "SELECT x.dept, x.n FROM (SELECT dept, count(*) AS n FROM staff "
        "GROUP BY dept) AS x WHERE x.n > 1"
    )
    assert ma.sql(query, subq).collect().to_pylist() == [{"dept": 20, "n": 2}]


def test_comma_join_picks_its_keys_out_of_where(subq):
    """`supp` shares no column with `part`, so it waits for `partsupp`; `*`
    still lists the tables as written."""
    plan = ma.sql(
        "SELECT * FROM part p, supp s, partsupp ps "
        "WHERE p.p_partkey = ps.ps_partkey AND s.sid = ps.ps_suppkey "
        "AND p.p_brand = 'b'",
        subq,
    )
    assert plan.column_names == [
        "p_partkey",
        "p_brand",
        "sid",
        "sname",
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
    ]
    assert plan.collect().to_pylist() == [
        {
            "p_partkey": 3,
            "p_brand": "b",
            "sid": 9,
            "sname": "s9",
            "ps_partkey": 3,
            "ps_suppkey": 9,
            "ps_availqty": 100,
        }
    ]


def test_comma_and_explicit_joins_mix(subq):
    query = (
        "SELECT s.sname, ps.ps_partkey FROM supp s JOIN partsupp ps "
        "ON s.sid = ps.ps_suppkey, part p "
        "WHERE p.p_partkey = ps.ps_partkey AND p.p_brand = 'b'"
    )
    assert ma.sql(query, subq).collect().to_pylist() == [
        {"sname": "s9", "ps_partkey": 3}
    ]


def test_comma_join_without_an_equality_is_refused(subq):
    with pytest.raises(NotImplementedError, match="cross joins are not supported"):
        ma.sql("SELECT eid FROM staff, units WHERE eid > did", subq)


def test_correlation_two_levels_out_is_refused(subq):
    with pytest.raises(NotImplementedError, match="2 levels out"):
        ma.sql(
            "SELECT eid FROM staff s WHERE EXISTS (SELECT 1 FROM units u "
            "WHERE u.did = s.dept AND EXISTS "
            "(SELECT 1 FROM part p WHERE p.p_partkey = s.eid))",
            subq,
        )
