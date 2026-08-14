"""Aggregation through the expression API — build a plan, execute it.

Everything here goes through the surface a query frontend uses, the way
ibis/polars do:

    var plan = in_memory_table(orders).aggregate(
        keys=[col("region")],
        aggs=[
            col("amount").sum().alias("total"),
            col("amount").aggregate("max").alias("biggest"),
        ],
    )
    var out = plan.execute()                       # region | total | biggest

Keys and inputs are arbitrary expressions (``col("a") * lit(2)``,
``col("ts").year()``), ``HAVING`` is a ``.filter(...)`` on top, and leaving
``keys`` empty is ``SELECT agg(x)`` with no GROUP BY.

An aggregate is written on the expression it aggregates, so there are no
parallel ``values``/``funcs``/``names`` lists to keep in step — a whole class of
caller mistake that used to need a test is now unrepresentable.

Deliberately nothing here touches ``Aggregates`` / ``AggFunc`` / ``Aggregation``:
the machinery underneath should stay refactorable without editing this file, so
what is pinned is what a query can observe — results, output dtypes, and which
queries are rejected.
"""

from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)


from ...builders import array, Date32Builder, PrimitiveBuilder
from ...dtypes import (
    Int64Type,
    TimestampType,
    date32,
    float64,
    int32,
    int64,
    second,
    string,
    timestamp,
)
from ...schema import schema
from ...tabular import RecordBatch, record_batch
from ...dtypes import DynType, Int32Type, StringType
from ...kernels.aggregate import (
    NumericAgg,
    StringMinMax,
    MinOp,
    SumKernel,
    MaxKernel,
)
from ...expr.aggregates import AggFunc
from ...expr.values import AggExpr
from ...expr.values import col, lit
from ...expr.relations import DynRelation, in_memory_table
from ...expr.relations import BoxedValue
from ...expr.values import col as fused_col


# ---------------------------------------------------------------------------
# A tiny orders table: region, amount, quantity, day.
# ---------------------------------------------------------------------------


def _orders() raises -> RecordBatch:
    var region = array(["east", "west", "east", "west", "east"])
    var amount = array([10, 40, 30, 20, 50], int64)
    var quantity = array([1, 2, 3, 4, 5], int32)
    var db = Date32Builder(date32(), 5)
    var days = [19000, 18500, 19200, 18800, 19100]
    for d in days:
        db.append(Int32(d))
    return record_batch(
        [region.copy(), amount.copy(), quantity.copy(), db.finish().to_dyn()],
        names=["region", "amount", "quantity", "day"],
    )


def _row_for(out_batch: RecordBatch, region: String) raises -> Int:
    """Row index of ``region`` — group order follows the key hash."""
    ref keys = out_batch.column(0).as_string()
    for i in range(out_batch.num_rows()):
        if keys[i].to_string() == region:
            return i
    raise Error("region not in result: ", region)


# ---------------------------------------------------------------------------
# GROUP BY
# ---------------------------------------------------------------------------


def test_group_by_one_aggregate() raises:
    """``SELECT region, sum(amount) FROM orders GROUP BY region``."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("amount").sum(),
        ],
    )
    var out = plan.execute()

    assert_equal(out.num_rows(), 2)
    assert_equal(out.num_columns(), 2)
    assert_true(out.schema.fields[0].name == "region")
    assert_true(out.schema.fields[1].name == "sum")
    ref total = out.column(1).as_int64()
    assert_equal(total[_row_for(out, "east")].value(), 90)
    assert_equal(total[_row_for(out, "west")].value(), 60)


def test_group_by_several_aggregates_in_one_pass() raises:
    """``sum(amount), max(amount), count(quantity), mean(amount)`` — the keys
    are grouped once and every aggregate rides along."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("amount").sum().alias("total"),
            col("amount").aggregate("max").alias("biggest"),
            col("quantity").aggregate("count").alias("n"),
            col("amount").aggregate("mean").alias("avg"),
        ],
    )
    var out = plan.execute()
    var east = _row_for(out, "east")

    assert_equal(out.num_columns(), 5)
    assert_equal(out.column(1).as_int64()[east].value(), 90)
    assert_equal(out.column(2).as_int64()[east].value(), 50)
    assert_equal(out.column(3).as_int64()[east].value(), 3)
    assert_equal(out.column(4).as_float64()[east].value(), 30.0)


def test_group_by_computed_input() raises:
    """Keys and aggregate inputs are ordinary expressions."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            (col("amount") * lit[Int64Type](2)).sum().alias("doubled"),
        ],
    )
    var out = plan.execute()
    assert_equal(out.column(1).as_int64()[_row_for(out, "east")].value(), 180)


def test_group_by_names_disambiguate_aggregates() raises:
    """Two aggregates of the same function need explicit output names."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("amount").sum().alias("amount_sum"),
            col("quantity").sum().alias("quantity_sum"),
        ],
    )
    assert_true(plan.schema().fields[1].name == "amount_sum")
    assert_true(plan.schema().fields[2].name == "quantity_sum")


def test_having_is_a_filter_on_the_aggregate() raises:
    """``HAVING`` needs no node of its own — filter the aggregate's output."""
    var plan = (
        in_memory_table(_orders())
        .aggregate(
            keys=[col("region")],
            aggs=[
                col("amount").sum().alias("total"),
            ],
        )
        .filter(col("total") > lit[Int64Type](80))
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.column(0).as_string()[0].to_string() == "east")
    assert_equal(out.column(1).as_int64()[0].value(), 90)


# ---------------------------------------------------------------------------
# no GROUP BY
# ---------------------------------------------------------------------------


def test_whole_table_aggregate_without_keys() raises:
    """``SELECT sum(amount), min(amount), count(quantity) FROM orders`` — no
    keys means one implicit group and a single output row."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=List[BoxedValue](),
        aggs=[
            col("amount").sum().alias("total"),
            col("amount").aggregate("min").alias("smallest"),
            col("quantity").aggregate("count").alias("n"),
        ],
    )
    var out = plan.execute()

    assert_equal(out.num_rows(), 1)
    assert_equal(out.num_columns(), 3)
    assert_equal(out.column(0).as_int64()[0].value(), 150)
    assert_equal(out.column(1).as_int64()[0].value(), 10)
    assert_equal(out.column(2).as_int64()[0].value(), 5)


# ---------------------------------------------------------------------------
# output dtypes — the rule each aggregate follows, seen from the plan schema
# ---------------------------------------------------------------------------


def test_output_dtypes_match_what_execution_produces() raises:
    """The plan's schema is not a guess: it is the resolved aggregate's own
    output dtype, and execution must agree with it column for column."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("quantity").sum().alias("s"),  # sum widens int32 -> int64
            col("quantity").aggregate("min").alias("mn"),  # min keeps int32
            col("amount").aggregate("mean").alias("avg"),  # mean is float64
            col("region")
            .aggregate("count")
            .alias("n"),  # count is int64, any dtype
            col("region")
            .aggregate("min")
            .alias("first_name"),  # string min keeps string
            col("day")
            .aggregate("min")
            .alias("earliest"),  # date min keeps date32
        ],
    )
    var out = plan.execute()
    for i in range(len(plan.schema().fields)):
        assert_equal(plan.schema().fields[i].dtype, out.schema.fields[i].dtype)

    assert_equal(out.schema.fields[1].dtype, int64)
    assert_equal(out.schema.fields[2].dtype, int32)
    assert_equal(out.schema.fields[3].dtype, float64)
    assert_equal(out.schema.fields[4].dtype, int64)
    assert_equal(out.schema.fields[5].dtype, string)
    assert_equal(out.schema.fields[6].dtype, date32())


def test_min_max_keep_timestamp_unit_and_timezone() raises:
    """An order-preserving aggregate carries the column's runtime parameters,
    not just its type."""
    var tb = PrimitiveBuilder[TimestampType](timestamp(second, "UTC"), 3)
    tb.append(Int64(3000))
    tb.append(Int64(1000))
    tb.append(Int64(2000))
    var batch = record_batch(
        [array(["a", "a", "a"]).to_dyn(), tb.finish().to_dyn()],
        names=["k", "ts"],
    )
    var plan = in_memory_table(batch).aggregate(
        keys=[col("k")],
        aggs=[
            col("ts").aggregate("min").alias("first_seen"),
        ],
    )
    var out = plan.execute()
    assert_equal(plan.schema().fields[1].dtype, timestamp(second, "UTC"))
    assert_equal(out.column(1).dtype(), timestamp(second, "UTC"))
    assert_equal(out.column(1).as_timestamp()[0].value(), 1000)


# ---------------------------------------------------------------------------
# aggregates over non-numeric columns
# ---------------------------------------------------------------------------


def test_min_max_over_strings_are_lexicographic() raises:
    var batch = record_batch(
        [
            array(["a", "a", "a"]).to_dyn(),
            array(["banana", "apple", "cherry"]).to_dyn(),
        ],
        names=["k", "fruit"],
    )
    var plan = in_memory_table(batch).aggregate(
        keys=[col("k")],
        aggs=[
            col("fruit").aggregate("min").alias("lo"),
            col("fruit").aggregate("max").alias("hi"),
        ],
    )
    var out = plan.execute()
    assert_true(out.column(1).as_string()[0].to_string() == "apple")
    assert_true(out.column(2).as_string()[0].to_string() == "cherry")


def test_count_distinct_over_any_column() raises:
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("region").count_distinct().alias("regions"),
            col("amount").count_distinct().alias("amounts"),
        ],
    )
    var out = plan.execute()
    var east = _row_for(out, "east")
    assert_equal(out.column(1).as_int64()[east].value(), 1)
    assert_equal(out.column(2).as_int64()[east].value(), 3)


def test_nulls_are_excluded_and_empty_groups_are_null() raises:
    """SQL semantics: nulls do not contribute; a group with nothing to fold
    yields null, while ``count`` yields 0."""
    var vb = PrimitiveBuilder[Int64Type](5)
    vb.append(Int64(10))
    vb.append_null()
    vb.append(Int64(30))
    vb.append_null()
    vb.append_null()
    var batch = record_batch(
        [array(["a", "a", "a", "b", "b"]).to_dyn(), vb.finish().to_dyn()],
        names=["k", "v"],
    )
    var plan = in_memory_table(batch).aggregate(
        keys=[col("k")],
        aggs=[
            col("v").sum().alias("total"),
            col("v").aggregate("count").alias("n"),
        ],
    )
    var out = plan.execute()

    ref keys = out.column(0).as_string()
    for i in range(out.num_rows()):
        if keys[i].to_string() == "a":
            assert_equal(out.column(1).as_int64()[i].value(), 40)
            assert_equal(out.column(2).as_int64()[i].value(), 2)
        else:
            assert_false(out.column(1).is_valid(i))  # all-null group -> null
            # ... but counting nothing is 0, and a *valid* 0, whatever the
            # column's type.
            assert_true(out.column(2).is_valid(i))
            assert_equal(out.column(2).as_int64()[i].value(), 0)


# ---------------------------------------------------------------------------
# rejected queries — caught when the plan is built, not when it runs
# ---------------------------------------------------------------------------


def test_unknown_aggregate_is_rejected() raises:
    """The sugar (`.sum()`) cannot name an aggregate that does not exist; a
    frontend passing a runtime name through `.aggregate(name)` can, and is
    rejected when the plan is built."""
    with assert_raises(contains="unknown aggregate function"):
        _ = in_memory_table(_orders()).aggregate(
            keys=[col("region")],
            aggs=[col("amount").aggregate("median")],
        )


def test_aggregate_undefined_for_the_column_type_is_rejected() raises:
    """``sum`` and ``mean`` have no meaning over strings, and the plan says so
    before anything executes."""
    with assert_raises(contains="not defined for"):
        _ = in_memory_table(_orders()).aggregate(
            keys=[col("region")],
            aggs=[
                col("region").sum(),
            ],
        )
    with assert_raises(contains="not defined for"):
        _ = in_memory_table(_orders()).aggregate(
            keys=[col("region")],
            aggs=[
                col("region").aggregate("mean"),
            ],
        )


# ---------------------------------------------------------------------------
# a plan is a reusable template
# ---------------------------------------------------------------------------


def test_the_same_plan_can_be_executed_repeatedly() raises:
    var plan = in_memory_table(_orders()).aggregate(
        keys=[col("region")],
        aggs=[
            col("amount").sum().alias("total"),
        ],
    )
    var first = plan.execute()
    var second = plan.execute()
    assert_equal(first.num_rows(), second.num_rows())
    assert_equal(
        first.column(1).as_int64()[_row_for(first, "east")].value(),
        second.column(1).as_int64()[_row_for(second, "east")].value(),
    )


# ---------------------------------------------------------------------------
# AOT — the same queries with nothing left to resolve
#
# The fused form names its `Aggregation` outright, so the plan holds a direct
# pointer to `AggState[SumKernel, Int64Type]`: no function name, no dtype
# lookup, no other aggregate instantiated. This is the shape
# `benchmarks/binary_size/query_streaming_agg_fused.mojo` gates on, and it must
# produce exactly what the dynamic form produces.
# ---------------------------------------------------------------------------


def _fused_sum_max_by_region() raises -> DynRelation:
    """``SELECT region, sum(amount), max(amount) GROUP BY region``, fused."""
    return in_memory_table(_orders()).aggregate(
        keys=[BoxedValue(fused_col("region", string))],
        inputs=[
            BoxedValue(fused_col("amount", int64)),
            BoxedValue(fused_col("amount", int64)),
        ],
        aggs=[
            AggFunc.of[NumericAgg[SumKernel, Int64Type]](DynType(int64)),
            AggFunc.of[NumericAgg[MaxKernel, Int64Type]](DynType(int64)),
        ],
        names=["region", "total", "biggest"],
    )


def test_fused_aggregate_matches_the_dynamic_one() raises:
    """Naming the aggregation and naming the function must reach the same
    kernel — the fused path is the dynamic one with the resolution removed."""
    var fused = _fused_sum_max_by_region().execute()
    var dynamic = (
        in_memory_table(_orders())
        .aggregate(
            keys=[col("region")],
            aggs=[
                col("amount").sum().alias("total"),
                col("amount").aggregate("max").alias("biggest"),
            ],
        )
        .execute()
    )

    assert_equal(fused.num_rows(), dynamic.num_rows())
    assert_equal(fused.num_columns(), dynamic.num_columns())
    for i in range(fused.num_rows()):
        var region = fused.column(0).as_string()[i].to_string()
        var other = _row_for(dynamic, region)
        assert_equal(
            fused.column(1).as_int64()[i].value(),
            dynamic.column(1).as_int64()[other].value(),
        )
        assert_equal(
            fused.column(2).as_int64()[i].value(),
            dynamic.column(2).as_int64()[other].value(),
        )


def test_fused_aggregate_results() raises:
    var out = _fused_sum_max_by_region().execute()
    var east = _row_for(out, "east")
    assert_equal(out.column(1).as_int64()[east].value(), 90)
    assert_equal(out.column(2).as_int64()[east].value(), 50)


def test_fused_non_numeric_aggregation() raises:
    """A fused plan is not limited to the numeric folds: a bytewise string
    min is just a different `Aggregation` named at compile time."""
    var plan = in_memory_table(_orders()).aggregate(
        keys=[BoxedValue(fused_col("region", string))],
        inputs=[BoxedValue(fused_col("region", string))],
        aggs=[AggFunc.of[StringMinMax[MinOp, StringType]](DynType(string))],
        names=["region", "lo"],
    )
    var out = plan.execute()
    var east = _row_for(out, "east")
    assert_true(out.column(1).as_string()[east].to_string() == "east")


# ---------------------------------------------------------------------------
# A5 — aggregate parity across the two lanes
#
# The third axis of invariant 2, and the one `test_parity.mojo` cannot reach:
# an aggregate is not a `Value`, so there is no `assert_parity` for it.
# `col("amount").sum()` on a `DynValue` yields a `DynAgg` (a group-by spec),
# while `col("amount", int64).sum()` on a fused node yields a scalar
# `Reduction`. They converge only at `AggExpr`, so parity has to be observed
# where both are usable — through a plan.
# ---------------------------------------------------------------------------
def _assert_agg_parity(
    var fused: AggExpr, var dyn: AggExpr, label: String
) raises:
    """Run one aggregate through a keyless plan in each lane and compare.

    Keyless on purpose: with no GROUP BY there is one output row, so the
    comparison cannot be confused by group order, which follows the key hash.
    """
    var f = (
        in_memory_table(_orders())
        .aggregate(keys=List[BoxedValue](), aggs=[fused^])
        .execute()
    )
    var d = (
        in_memory_table(_orders())
        .aggregate(keys=List[BoxedValue](), aggs=[dyn^])
        .execute()
    )
    assert_equal(f.num_rows(), d.num_rows())
    assert_true(
        f.column(0) == d.column(0),
        String("aggregate `") + label + "` disagrees across lanes",
    )


def test_agg_parity_sum() raises:
    _assert_agg_parity(
        fused_col("amount", int64).sum(), col("amount").sum(), "sum"
    )


def test_agg_parity_mean() raises:
    _assert_agg_parity(
        fused_col("amount", int64).mean(), col("amount").mean(), "mean"
    )


def test_agg_parity_min() raises:
    _assert_agg_parity(
        fused_col("amount", int64).min(), col("amount").min(), "min"
    )


def test_agg_parity_max() raises:
    _assert_agg_parity(
        fused_col("amount", int64).max(), col("amount").max(), "max"
    )


def test_agg_parity_product() raises:
    _assert_agg_parity(
        fused_col("quantity", int32).product(),
        col("quantity").product(),
        "product",
    )


def test_agg_parity_count() raises:
    _assert_agg_parity(
        fused_col("quantity", int32).count(), col("quantity").count(), "count"
    )


def test_agg_parity_count_distinct() raises:
    """`count_distinct` is defaulted on `Value` itself, so both lanes reach the
    same `DistinctAgg` — this pins that they still agree on a column with
    repeats (region has 5 rows and 2 distinct values)."""
    _assert_agg_parity(
        fused_col("region", string).count_distinct(),
        col("region").count_distinct(),
        "count_distinct",
    )


def test_agg_parity_grouped_sum() raises:
    """The same aggregate under a GROUP BY, since the grouped path is a
    different implementation from the whole-table fold."""
    var f = (
        in_memory_table(_orders())
        .aggregate(
            keys=[BoxedValue(fused_col("region", string))],
            aggs=[fused_col("amount", int64).sum().alias("total")],
        )
        .execute()
    )
    var d = (
        in_memory_table(_orders())
        .aggregate(
            keys=[BoxedValue(col("region"))],
            aggs=[col("amount").sum().alias("total")],
        )
        .execute()
    )
    assert_equal(f.num_rows(), d.num_rows())
    for region in ["east", "west"]:
        assert_equal(
            f.column(1).as_int64()[_row_for(f, region)].value(),
            d.column(1).as_int64()[_row_for(d, region)].value(),
        )
