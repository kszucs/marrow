"""Golden cases — three-valued (Kleene) logic, the AOT lane.

The first four cases use *derived* booleans (`x > 0`); the `_column_` cases
read bool columns directly. The latter were inexpressible in this lane until
`col` gained a `BoolType` overload and `values.mojo` a `BoolColumn` leaf —
the fused lane had numeric, string, list and temporal column leaves and no
boolean one, so it could not reference a `bool` column while the runtime lane
could. That was an invariant-2 violation: a feature present in only one lane.
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import bool_, int64
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


def _flags() raises -> DynRelation:
    return in_memory_table(_fixture("flags"))


def test_golden_kleene_column_and() raises:
    _check(
        "test_golden_kleene_column_and",
        _flags().project(
            ["p", "q", "r"],
            [
                col("p", bool_),
                col("q", bool_),
                col("p", bool_) & col("q", bool_),
            ],
        ),
    )


def test_golden_kleene_column_or() raises:
    _check(
        "test_golden_kleene_column_or",
        _flags().project(
            ["p", "q", "r"],
            [
                col("p", bool_),
                col("q", bool_),
                col("p", bool_) | col("q", bool_),
            ],
        ),
    )


def test_golden_kleene_column_filter() raises:
    _check(
        "test_golden_kleene_column_filter",
        _flags().filter(col("p", bool_)),
    )
