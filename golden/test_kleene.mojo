"""Golden cases — three-valued (Kleene) logic, the AOT lane.

Operands are *derived* booleans (`x > 0`) rather than bool columns: the AOT
lane has no boolean column leaf — `col` has overloads for numeric, string,
list and temporal dtypes but none for `BoolType`, and `values.mojo` has no
`BoolColumn` node — so a bool-column spelling would be runtime-only and could
not be held to the same expectation. That gap is itself an invariant-2
violation worth recording.
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation, in_memory_table


def _kleene() raises -> DynRelation:
    return in_memory_table(_fixture("kleene"))


def test_golden_kleene_and() raises:
    _check(
        "test_golden_kleene_and",
        _kleene().project(
            ["x", "y", "r"],
            [
                col("x", int64),
                col("y", int64),
                (col("x", int64) > lit(0, int64))
                & (col("y", int64) > lit(0, int64)),
            ],
        ),
    )


def test_golden_kleene_or() raises:
    _check(
        "test_golden_kleene_or",
        _kleene().project(
            ["x", "y", "r"],
            [
                col("x", int64),
                col("y", int64),
                (col("x", int64) > lit(0, int64))
                | (col("y", int64) > lit(0, int64)),
            ],
        ),
    )


def test_golden_kleene_not() raises:
    _check(
        "test_golden_kleene_not",
        _kleene().project(
            ["x", "r"],
            [col("x", int64), ~(col("x", int64) > lit(0, int64))],
        ),
    )


def test_golden_kleene_filter_and() raises:
    _check(
        "test_golden_kleene_filter_and",
        _kleene().filter(
            (col("x", int64) > lit(0, int64))
            & (col("y", int64) > lit(0, int64))
        ),
    )
