"""Golden cases — sort depth, the runtime lane.

Null placement, per-key directions, the ORDER BY ... LIMIT top-K path, and
sorting a column that is entirely null.
"""


def test_golden_sort_string_desc_nulls_last(golden):
    """SELECT k, v, w FROM basic ORDER BY k DESC NULLS LAST"""
    t = golden.table("basic")
    golden.check(t.order_by(("k", "descending"), nulls_first=False))


def test_golden_sort_mixed_directions(golden):
    """SELECT k, v, w FROM basic
    ORDER BY k ASC NULLS FIRST, v DESC NULLS FIRST"""
    t = golden.table("basic")
    golden.check(t.order_by("k", ("v", "descending")))


def test_golden_sort_top_k(golden):
    """SELECT k, v, w FROM basic ORDER BY v DESC NULLS LAST LIMIT 2"""
    t = golden.table("basic")
    golden.check(t.order_by(("v", "descending"), nulls_first=False).limit(2))


def test_golden_sort_all_null_key(golden):
    """SELECT a, b, g FROM nulls ORDER BY a NULLS FIRST, b NULLS FIRST"""
    t = golden.table("nulls")
    golden.check(t.order_by("a", "b"))
