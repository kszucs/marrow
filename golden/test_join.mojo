"""Golden cases — joins, the AOT lane.

`CROSS` is absent deliberately: `hash_join` rejects it and `is_supported()`
answers False, so there is nothing to hold to an expectation.

Join output order is unspecified, so every case sorts before comparing.
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import int64, string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.kernels.join import (
    JOIN_ALL,
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
    JoinKind,
)


def _emp() raises -> DynRelation:
    return in_memory_table(_fixture("emp"))


def _dept() raises -> DynRelation:
    return in_memory_table(_fixture("dept"))


def _joined(kind: JoinKind) raises -> DynRelation:
    """`emp JOIN dept ON emp.dept = dept.did` — differently named keys, which
    is the shape that used to raise `equal: dtype mismatch`."""
    return _emp().join(
        _dept(),
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=kind,
        strictness=JOIN_ALL,
    )


def test_golden_join_inner() raises:
    _check(
        "test_golden_join_inner",
        _joined(JOIN_INNER).sort([col("eid", int64)], [True]),
    )


def test_golden_join_left() raises:
    _check(
        "test_golden_join_left",
        _joined(JOIN_LEFT).sort([col("eid", int64)], [True]),
    )


def test_golden_join_right() raises:
    _check(
        "test_golden_join_right",
        _joined(JOIN_RIGHT).sort(
            [col("did", int64), col("eid", int64)], [True, True]
        ),
    )


def test_golden_join_full() raises:
    _check(
        "test_golden_join_full",
        _joined(JOIN_FULL).sort(
            [col("eid", int64), col("did", int64)], [True, True]
        ),
    )


def test_golden_join_semi() raises:
    _check(
        "test_golden_join_semi",
        _joined(JOIN_SEMI).sort([col("eid", int64)], [True]),
    )


def test_golden_join_anti() raises:
    _check(
        "test_golden_join_anti",
        _joined(JOIN_ANTI).sort([col("eid", int64)], [True]),
    )
