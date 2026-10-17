# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Subqueries through the lazy frontend: ``exists``, ``isin`` over a table,
``as_scalar`` and ``outer``, planned automatically by ``collect()``.

``emp`` holds a NULL ``dept`` (eid 5) and a ``dept`` no department has (eid
4) — the rows that tell ``EXISTS`` from ``IN`` and ``NOT IN`` from
``NOT EXISTS``.
"""

import pytest

import marrow as ma
from marrow import col, count_star, exists, lit, outer


@pytest.fixture
def emp():
    return ma.memtable(
        ma.record_batch(
            {
                "eid": ma.array([1, 2, 3, 4, 5], type=ma.int64()),
                "dept": ma.array([10, 20, 20, 99, None], type=ma.int64()),
            }
        )
    )


@pytest.fixture
def dept():
    return ma.memtable(
        ma.record_batch(
            {
                "did": ma.array([10, 20, 30], type=ma.int64()),
                "budget": ma.array([100, 300, 50], type=ma.int64()),
            }
        )
    )


def eids(lazy_table):
    return sorted(r["eid"] for r in lazy_table.collect().to_pylist())


def test_exists_correlated(emp, dept):
    sub = dept.filter(col("did") == outer("dept"))
    assert eids(emp.filter(exists(sub))) == [1, 2, 3]
    assert eids(emp.filter(~exists(sub))) == [4, 5]


def test_isin_table(emp, dept):
    assert eids(emp.filter(col("dept").isin(dept.select("did")))) == [1, 2, 3]


def test_not_in_table_is_three_valued(emp, dept):
    """NULL ``dept`` is unknown against a non-empty table."""
    assert eids(emp.filter(~col("dept").isin(dept.select("did")))) == [4]


def test_as_scalar(emp, dept):
    n = dept.aggregate(n=count_star())
    assert eids(emp.filter(col("eid") > n.as_scalar())) == [4, 5]


def test_correlated_scalar_count(emp, dept):
    n = dept.filter(col("did") == outer("dept")).aggregate(n=count_star())
    got = emp.with_columns(n=n.as_scalar()).collect().to_pylist()
    assert sorted((r["eid"], r["n"]) for r in got) == [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 0),
        (5, 0),
    ]


def test_scalar_with_two_rows_raises(emp, dept):
    two = dept.filter(col("did") > lit(10)).select("did")
    with pytest.raises(Exception, match="at most one"):
        emp.filter(col("eid") > two.as_scalar()).collect()
