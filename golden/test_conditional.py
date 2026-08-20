"""Golden cases — conditional kernels, the runtime lane.

`coalesce`, filling a null with a literal, and a CASE expression. All three
exist to *consume* nulls, so they are the kernels most likely to disagree
with SQL about what a null means.
"""

import marrow


def test_golden_cond_coalesce(golden):
    """SELECT coalesce(v, w) AS c FROM basic"""
    t = golden.table("basic")
    golden.check(t.project(c=t["v"].coalesce(t["w"])))


def test_golden_cond_fill_null_with_literal(golden):
    """SELECT coalesce(v, 0) AS c FROM basic"""
    t = golden.table("basic")
    golden.check(t.project(c=t["v"].fill_null(0)))


def test_golden_cond_case_when(golden):
    """SELECT CASE WHEN v > 3 THEN v ELSE w END AS c FROM basic

    A null condition counts as false in Arrow, so the null row takes the
    ELSE branch rather than becoming null.
    """
    t = golden.table("basic")
    golden.check(t.project(c=marrow.if_else(t["v"] > 3, t["v"], t["w"])))
