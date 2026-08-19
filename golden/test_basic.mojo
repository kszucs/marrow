"""Golden cases — the AOT lane.

The same eleven queries as `golden/test_basic.py`, under the same names, built
from **typed** `col(name, dtype)` leaves so every operand's dtype is fixed at
compile time and the subtree fuses. The runtime lane resolves the same queries
from a `DynValue` tree instead.

Both lanes compare against the *same* expectation — `golden/.exp/<case>.arrow`,
materialised by `golden/conftest.py` from the committed `.exp` text, which
DuckDB produced. Neither lane is the other's reference: a bug both lanes share
still fails.

Imports are **absolute** (`from marrow.x import ...`), the opposite of the rule
for tests inside the package: `golden/` sits outside `marrow/` and is reached
through the `-I .` the harness passes, exactly as `benchmarks/profiles/` is.

Paths are relative to the repository root, which is the working directory the
harness runs the generated driver from.
"""


from marrow.dtypes import Int64Type, int64, string
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import AggExpr, BoxedValue
from golden.helpers import check as _check, fixture as _fixture
from marrow.kernels.aggregate import (
    CountKernel,
    MaxKernel,
    MinKernel,
    NumericAgg,
    SumKernel,
)


def _basic() raises -> DynRelation:
    return in_memory_table(_fixture("basic"))


def test_golden_select_two_columns() raises:
    _check("test_golden_select_two_columns", _basic().select("k", "v"))


def test_golden_filter_gt() raises:
    _check(
        "test_golden_filter_gt",
        _basic().filter(col("v", int64) > lit(3, int64)),
    )


def test_golden_filter_and() raises:
    _check(
        "test_golden_filter_and",
        _basic().filter(
            (col("v", int64) > lit(2, int64))
            & (col("w", int64) < lit(60, int64))
        ),
    )


def test_golden_project_sum_of_columns() raises:
    _check(
        "test_golden_project_sum_of_columns",
        _basic().project(["s"], [col("v", int64) + col("w", int64)]),
    )


def test_golden_with_columns_appends() raises:
    _check(
        "test_golden_with_columns_appends",
        _basic().with_columns(["s"], [col("v", int64) + col("w", int64)]),
    )


def test_golden_aggregate_sum_by_key() raises:
    var plan = _basic().aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            )
        ],
    )
    _check(
        "test_golden_aggregate_sum_by_key",
        plan.sort([col("k", string)], [True]),
    )


def test_golden_aggregate_several() raises:
    var plan = _basic().aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            ),
            AggExpr.of[NumericAgg[MaxKernel, Int64Type]](col("v", int64)).alias(
                "biggest"
            ),
            AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                col("w", int64)
            ).alias("n"),
        ],
    )
    _check(
        "test_golden_aggregate_several",
        plan.sort([col("k", string)], [True]),
    )


def test_golden_aggregate_no_keys() raises:
    _check(
        "test_golden_aggregate_no_keys",
        _basic().aggregate(
            keys=List[BoxedValue](),
            aggs=[
                AggExpr.of[NumericAgg[SumKernel, Int64Type]](
                    col("v", int64)
                ).alias("total")
            ],
        ),
    )


def test_golden_order_by_asc() raises:
    _check(
        "test_golden_order_by_asc",
        _basic().sort([col("v", int64)], [True]),
    )


def test_golden_order_by_desc_limit() raises:
    _check(
        "test_golden_order_by_desc_limit",
        _basic().sort([col("v", int64)], [False]).limit(3),
    )


def test_golden_filter_then_aggregate() raises:
    var plan = (
        _basic()
        .filter(col("w", int64) > lit(20, int64))
        .aggregate(
            keys=[col("k", string)],
            aggs=[
                AggExpr.of[NumericAgg[SumKernel, Int64Type]](
                    col("v", int64)
                ).alias("total")
            ],
        )
    )
    _check(
        "test_golden_filter_then_aggregate",
        plan.sort([col("k", string)], [True]),
    )


def test_golden_filter_lt() raises:
    _check(
        "test_golden_filter_lt",
        _basic().filter(col("v", int64) < lit(4, int64)),
    )


def test_golden_filter_or() raises:
    _check(
        "test_golden_filter_or",
        _basic().filter(
            (col("v", int64) < lit(2, int64))
            | (col("w", int64) > lit(60, int64))
        ),
    )


def test_golden_filter_not() raises:
    _check(
        "test_golden_filter_not",
        _basic().filter(~(col("v", int64) > lit(3, int64))),
    )


def test_golden_project_difference() raises:
    _check(
        "test_golden_project_difference",
        _basic().project(["d"], [col("v", int64) - col("w", int64)]),
    )


def test_golden_project_predicate_is_boolean() raises:
    _check(
        "test_golden_project_predicate_is_boolean",
        _basic().project(["gt"], [col("v", int64) > lit(3, int64)]),
    )


def test_golden_order_by_two_keys() raises:
    _check(
        "test_golden_order_by_two_keys",
        _basic().sort(
            [col("k", string), col("v", int64)],
            [True, True],
        ),
    )


def test_golden_order_by_nulls_last() raises:
    _check(
        "test_golden_order_by_nulls_last",
        _basic().sort([col("v", int64)], [True], nulls_first=False),
    )


def test_golden_limit_with_offset() raises:
    _check(
        "test_golden_limit_with_offset",
        _basic().sort([col("v", int64)], [True]).limit(3, 2),
    )


def test_golden_aggregate_min_max() raises:
    var plan = _basic().aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[MinKernel, Int64Type]](col("v", int64)).alias(
                "lo"
            ),
            AggExpr.of[NumericAgg[MaxKernel, Int64Type]](col("w", int64)).alias(
                "hi"
            ),
        ],
    )
    _check(
        "test_golden_aggregate_min_max",
        plan.sort([col("k", string)], [True]),
    )
