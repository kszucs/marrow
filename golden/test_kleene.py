"""Golden cases — three-valued (Kleene) logic, the runtime lane.

`NULL AND FALSE` is FALSE, but `NULL AND TRUE` is NULL; `NULL OR TRUE` is
TRUE, but `NULL OR FALSE` is NULL. Getting one of those four wrong is
invisible until a predicate combines columns that have nulls in different
places.

The operands are *derived* booleans (`x > 0`), not bool columns: the AOT lane
has no boolean column leaf, so a bool-column spelling would be runtime-only
and could not be held to the same expectation.
"""


def test_golden_kleene_and(golden):
    """SELECT x, y, (x > 0) AND (y > 0) AS r FROM kleene"""
    t = golden.table("kleene")
    golden.check(t.project(x=t["x"], y=t["y"], r=(t["x"] > 0) & (t["y"] > 0)))


def test_golden_kleene_or(golden):
    """SELECT x, y, (x > 0) OR (y > 0) AS r FROM kleene"""
    t = golden.table("kleene")
    golden.check(t.project(x=t["x"], y=t["y"], r=(t["x"] > 0) | (t["y"] > 0)))


def test_golden_kleene_not(golden):
    """SELECT x, NOT (x > 0) AS r FROM kleene"""
    t = golden.table("kleene")
    golden.check(t.project(x=t["x"], r=~(t["x"] > 0)))


def test_golden_kleene_filter_and(golden):
    """SELECT x, y FROM kleene WHERE (x > 0) AND (y > 0)"""
    t = golden.table("kleene")
    golden.check(t.filter((t["x"] > 0) & (t["y"] > 0)))
