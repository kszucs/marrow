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
    with pytest.raises(Exception, match="WHERE cannot contain an aggregate"):
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
    with pytest.raises(Exception, match="whole number"):
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
    with pytest.raises(Exception, match="offset"):
        ma.sql("SELECT a # b FROM basic", basic=basic)


def test_unknown_column_is_named(basic):
    with pytest.raises(Exception, match="nope"):
        ma.sql("SELECT nope FROM basic", basic=basic)


def test_unknown_table_is_named(basic):
    with pytest.raises(Exception, match="missing"):
        ma.sql("SELECT k FROM missing", basic=basic)


def test_non_equi_join_is_refused(emp, dept):
    with pytest.raises(Exception, match="equality"):
        ma.sql(
            "SELECT e.name FROM emp e JOIN dept d ON e.did > d.did",
            emp=emp,
            dept=dept,
        )
