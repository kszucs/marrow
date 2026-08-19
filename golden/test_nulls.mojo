"""Golden cases — null semantics, the AOT lane.

The same eight queries as `golden/test_nulls.py`, under the same names, built
from typed `col(name, dtype)` leaves so the subtree fuses.

Runs against the `nulls` fixture: `a` is entirely null, `b` has none, `g` is a
string key carrying one null.
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import Float64Type, Int64Type, int64, string
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import AggExpr, BoxedValue, IsNull, NotNull
from marrow.kernels.aggregate import (
    CountKernel,
    MeanKernel,
    NumericAgg,
    SumKernel,
)


def _nulls() raises -> DynRelation:
    return in_memory_table(_fixture("nulls"))


def test_golden_nulls_is_null() raises:
    _check(
        "test_golden_nulls_is_null",
        _nulls().filter(IsNull(col("a", int64))),
    )


def test_golden_nulls_is_not_null() raises:
    _check(
        "test_golden_nulls_is_not_null",
        _nulls().filter(NotNull(col("b", int64))),
    )


def test_golden_nulls_predicate_excludes_null() raises:
    _check(
        "test_golden_nulls_predicate_excludes_null",
        _nulls().filter(col("a", int64) > lit(0, int64)),
    )


def test_golden_nulls_arithmetic_propagates() raises:
    _check(
        "test_golden_nulls_arithmetic_propagates",
        _nulls().project(["s"], [col("a", int64) + col("b", int64)]),
    )


def test_golden_nulls_sum_of_all_null_is_null() raises:
    _check(
        "test_golden_nulls_sum_of_all_null_is_null",
        _nulls().aggregate(
            keys=List[BoxedValue](),
            aggs=[
                AggExpr.of[NumericAgg[SumKernel, Int64Type]](
                    col("a", int64)
                ).alias("total")
            ],
        ),
    )


def test_golden_nulls_count_of_all_null_is_zero() raises:
    _check(
        "test_golden_nulls_count_of_all_null_is_zero",
        _nulls().aggregate(
            keys=List[BoxedValue](),
            aggs=[
                AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                    col("a", int64)
                ).alias("n")
            ],
        ),
    )


def test_golden_nulls_mean_ignores_nulls() raises:
    _check(
        "test_golden_nulls_mean_ignores_nulls",
        _nulls().aggregate(
            keys=List[BoxedValue](),
            aggs=[
                AggExpr.of[NumericAgg[MeanKernel, Int64Type]](
                    col("b", int64)
                ).alias("m")
            ],
        ),
    )


def test_golden_nulls_group_by_null_key() raises:
    var plan = _nulls().aggregate(
        keys=[col("g", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("b", int64)).alias(
                "total"
            )
        ],
    )
    _check(
        "test_golden_nulls_group_by_null_key",
        plan.sort([col("g", string)], [True]),
    )
