"""Golden cases — aggregation depth, the AOT lane."""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import Int64Type, int64, string
from marrow.expr.builders import col, count_star, lit
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import AggExpr, BoxedValue
from marrow.kernels.aggregate import CountKernel, NumericAgg, SumKernel


def _basic() raises -> DynRelation:
    return in_memory_table(_fixture("basic"))


def test_golden_agg_count_star_counts_rows() raises:
    _check(
        "test_golden_agg_count_star_counts_rows",
        _basic().aggregate(
            keys=List[BoxedValue](), aggs=[count_star().alias("n")]
        ),
    )


def test_golden_agg_count_column_skips_nulls() raises:
    _check(
        "test_golden_agg_count_column_skips_nulls",
        _basic().aggregate(
            keys=List[BoxedValue](),
            aggs=[
                AggExpr.of[NumericAgg[CountKernel, Int64Type]](
                    col("v", int64)
                ).alias("n")
            ],
        ),
    )


def test_golden_agg_over_computed_input() raises:
    var plan = _basic().aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](
                col("v", int64) * lit(2, int64)
            ).alias("d")
        ],
    )
    _check(
        "test_golden_agg_over_computed_input",
        plan.sort([col("k", string)], [True]),
    )


def test_golden_agg_two_group_keys() raises:
    var plan = _basic().aggregate(
        keys=[col("k", string), col("v", int64)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("w", int64)).alias(
                "s"
            )
        ],
    )
    _check(
        "test_golden_agg_two_group_keys",
        plan.sort([col("k", string), col("v", int64)], [True, True]),
    )


def test_golden_agg_computed_group_key() raises:
    var plan = _basic().aggregate(
        keys=[col("v", int64) > lit(3, int64)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("w", int64)).alias(
                "s"
            )
        ],
    )
    _check(
        "test_golden_agg_computed_group_key",
        plan.sort([col("key0", int64)], [True]),
    )


def test_golden_agg_having() raises:
    var agg = _basic().aggregate(
        keys=[col("k", string)],
        aggs=[
            AggExpr.of[NumericAgg[SumKernel, Int64Type]](col("v", int64)).alias(
                "total"
            )
        ],
    )
    _check(
        "test_golden_agg_having",
        agg.filter(col("total", int64) > lit(5, int64)).sort(
            [col("k", string)], [True]
        ),
    )
