"""Golden cases — conditional kernels, the AOT lane."""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import CaseWhen, Coalesce, FillNull


def _basic() raises -> DynRelation:
    return in_memory_table(_fixture("basic"))


def test_golden_cond_coalesce() raises:
    _check(
        "test_golden_cond_coalesce",
        _basic().project(["c"], [Coalesce(col("v", int64), col("w", int64))]),
    )


def test_golden_cond_fill_null_with_literal() raises:
    _check(
        "test_golden_cond_fill_null_with_literal",
        _basic().project(["c"], [FillNull(col("v", int64), lit(0, int64))]),
    )


def test_golden_cond_case_when() raises:
    _check(
        "test_golden_cond_case_when",
        _basic().project(
            ["c"],
            [
                CaseWhen(
                    col("v", int64) > lit(3, int64),
                    col("v", int64),
                    col("w", int64),
                )
            ],
        ),
    )
