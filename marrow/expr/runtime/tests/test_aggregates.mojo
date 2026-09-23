"""The runtime lane's aggregates — a name and an erased operand.

`Aggregate` covers the case where the aggregate is written in Mojo and
its operand can stay fused. This covers the other one: the aggregate arrives as
a **string**, from a frontend that built the whole query after the program
started, so both the aggregate and its operands are erased.

The claim under test is that both roads reach the same `AggKernel`, and
that `RuntimeAggregate` refuses a name it cannot serve at the point it is
written rather than on the first morsel.
"""

from std.testing import assert_equal, assert_true

from ....arrays import StringArray
from ....builders import array
from ....dtypes import DynType, Int64Type, float64, int32, int64, string
from ....dtypes import StringType
from ....kernels.aggregate import (
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    MaxFold,
    MaxOp,
    MeanFold,
    MinFold,
    MinOp,
    ProductFold,
    LexicalExtremum,
    SumFold,
    ValidCount,
)
from ....tabular import RecordBatch, record_batch
from ...builders import col, lit, table
from ....scalars import DynScalar, Int64Scalar
from ...logical import DynValue, Shape
from ..aggregates import RuntimeAggregate, resolve_aggregate
from ..values import RuntimeValue, column, gt


def _batch() raises -> RecordBatch:
    var values: List[Optional[String]] = ["a", "b", "a", "c", "c"]
    return record_batch(
        [array([1, 1, 1, 2, 2], int64).to_dyn(), array(values).to_dyn()],
        names=["g", "s"],
    )


def test_named_aggregate_rejects_an_unknown_name_where_it_is_written() raises:
    """Validation in `__init__` is why the name is not just a `String` field:
    `"summ"` fails where the node is built, not on the first morsel of a long
    scan.

    Constructed directly rather than through the fluent surface, because the
    fluent surface cannot produce a bad name — which is the point of it.
    """
    var raised = False
    try:
        _ = RuntimeAggregate(column("s"), String("summ"))
    except:
        raised = True
    assert_true(raised)
    # And a real one does not.
    _ = RuntimeAggregate(column("s"), String("count_distinct"))


def _out_dtype(name: String, in_dtype: DynType) raises -> DynType:
    """The catalog's answer, through the one ladder both callers use."""

    def job[Agg: AggKernel]() raises {imm} -> DynType:
        return Agg.dtype(in_dtype)

    return resolve_aggregate(name, in_dtype, job)


def test_named_aggregate_resolution_answers_dtype_and_fold_together() raises:
    """The catalog's dtype and the kernel that will run cannot disagree.

    `agg_out_dtype` answers from each kernel's own `dtype` static rather than
    restating a constant, so `min` over a string column reports `string`
    because `LexicalExtremum` says so — not because an arm here spells it.
    """
    assert_true(_out_dtype("count_distinct", DynType(string)) == int64)
    assert_true(_out_dtype("count", DynType(string)) == int64)
    assert_true(_out_dtype("min", DynType(string)) == string)
    assert_true(_out_dtype("max", DynType(string)) == string)
    # `sum(int32)` widens; the widening rule is `SumFold`'s, not the
    # catalog's.
    assert_true(_out_dtype("sum", DynType(int32)) == int64)
    assert_true(_out_dtype("variance", DynType(int64)) == float64)


def test_named_aggregate_rejects_a_dtype_it_has_no_arm_for() raises:
    """The catalog is the domain gate: a `sum` over strings raises at plan
    time, where the query was written, not on the first morsel."""
    var raised = False
    try:
        _ = _out_dtype("sum", DynType(string))
    except:
        raised = True
    assert_true(raised)


def test_named_aggregate_is_scalar_shaped_and_named_by_its_function() raises:
    """Built through the fluent surface — `col("s")` with no dtype is the
    runtime lane, so the same spelling that fuses in the comptime lane lands
    here instead."""
    var agg = col("s").count_distinct()
    assert_equal(agg.name(), "count_distinct")
    assert_equal(agg.shape, Shape.scalar)
    assert_equal(String(agg), "count_distinct(s)")
    assert_equal(agg.columns()[0], "s")


def test_named_aggregate_alias_leaves_the_function_alone() raises:
    """Two name fields. `_alias` reaches the output schema; `_func` is what
    resolves, and `alias` must not touch it — one field would print `n(col(s))`
    and send the resolver looking for an aggregate called `n`."""
    var agg = col("s").count_distinct().alias("n")
    assert_equal(agg.name(), "n")
    assert_equal(String(agg), "count_distinct(s)")


def test_named_aggregate_runs_keyless() raises:
    """The erased operand resolves against the batch's schema, and the empty
    id array takes the whole-input branch."""
    var plan = table(_batch()).aggregate(
        [col("s").count_distinct()], List[DynValue]()
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].as_int64() == array([3], int64))
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_runs_grouped() raises:
    """`col("g")` is a runtime key too — the whole plan is built from names."""
    var plan = table(_batch()).aggregate(
        [col("s").min().alias("lo")], [col("g")]
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_string() == array(["a", "c"]))
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_covers_the_folds_the_comptime_lane_fuses() raises:
    """`sum`/`mean`/`count` are reachable by name as well as by type.

    The comptime lane fuses these into `Aggregate`; a frontend that only
    has a string reaches the same algebra through `Fold` and
    `ValidCount[StringArray]` instead. Same answers, one materialised column.
    """
    var plan = table(_batch()).aggregate(
        [
            col("g").sum().alias("total"),
            col("g").mean().alias("avg"),
            col("s").count().alias("n"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([7], int64))
    assert_equal(out.columns[1].as_float64()[0].value(), 1.4)
    assert_true(out.columns[2].as_int64() == array([5], int64))
    assert_true(plan.schema() == out.schema)


def test_named_variance_and_stddev_reach_the_composite_accumulator() raises:
    """`variance` is the first aggregate reachable by name whose state is not
    a scalar — Welford's (count, mean, M2). Nothing about the runtime lane
    changes for it: a name, one `AggKernel`, one materialised column.

    Group column `g` is [1,1,1,2,2]: population variance 0.24, stddev 0.4899.
    """
    var plan = table(_batch()).aggregate(
        [
            col("g").variance().alias("var_g"),
            col("g").stddev().alias("sd_g"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(abs(out.columns[0].as_float64()[0].value() - 0.24) < 1e-9)
    assert_true(
        abs(out.columns[1].as_float64()[0].value() - 0.4898979485566356) < 1e-9
    )
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_vocabulary_all_resolves() raises:
    """The two tables must agree: every name `__init__` accepts must be one
    `resolve` can serve.

    They are separate — a string list and a ladder that binds types — so
    nothing but this case connects them. It exists because they *did* drift:
    `variance` and `stddev` reached `resolve` one commit before they reached
    the accept list, and every query naming them raised "unknown aggregate"
    from the constructor.
    """
    for ref name in RuntimeAggregate.vocabulary():
        var node = RuntimeAggregate(column("g"), name.copy())
        # Raises if the ladder has no arm for it; int64 is in every domain.


# ---------------------------------------------------------------------------
# FILTER (WHERE ...) — the same verb, the other lane
#
# The comptime lane carries its predicate as a type parameter; here it is an
# `Optional[RuntimeValue]` field, because a lane that discovers its operand's
# dtype at run time gains nothing from discovering the predicate's presence at
# compile time. The *answers* must not differ, and these are DuckDB's over the
# golden corpus's `basic` fixture.
# ---------------------------------------------------------------------------


def _int(value: Int) -> RuntimeValue:
    """An int64 constant in the runtime lane, where a literal is an erased
    scalar rather than a typed node."""
    return lit(DynScalar(Int64Scalar(Int64(value))))


def _basic_rows() raises -> RecordBatch:
    """`golden/fixtures/basic.arrow`, row for row."""
    var k: List[Optional[String]] = ["a", "b", "a", "c", "b", "a", None]
    var v: List[Optional[Int]] = [1, 2, 3, 4, None, 6, 7]
    var w: List[Optional[Int]] = [10, None, 30, 40, 50, 60, 70]
    return record_batch(
        [
            array(k).to_dyn(),
            array(v, int64).to_dyn(),
            array(w, int64).to_dyn(),
        ],
        names=["k", "v", "w"],
    )


def test_named_aggregate_filter_restricts_what_it_sees() raises:
    """`sum(v) FILTER (WHERE v > 2)` built entirely from names — DuckDB
    answers 20."""
    var plan = table(_basic_rows()).aggregate(
        [col("v").sum().filter(gt(col("v"), _int(2))).alias("total")],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([20], int64))
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_filter_keeps_a_group_with_no_admitted_row() raises:
    """`FILTER` restricts the aggregate, never the query: group `b` survives
    with a null total, which is what tells it apart from a `WHERE`. DuckDB:
    a -> 9, b -> NULL, c -> 4, NULL -> 7."""
    var plan = table(_basic_rows()).aggregate(
        [col("v").sum().filter(gt(col("v"), _int(2))).alias("total")],
        [col("k")],
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 4)
    ref totals = out.columns[1].as_int64()
    assert_equal(totals[0].value(), 9)
    assert_true(totals.is_null(1))
    assert_equal(totals[2].value(), 4)
    assert_equal(totals[3].value(), 7)


def test_named_aggregate_filter_excludes_a_null_predicate() raises:
    """A predicate that is NULL is not TRUE. `count(v) FILTER (WHERE w > 20)`
    counts the non-null `v` of the rows where `w` is greater than 20, and the
    row whose `w` is null is not one of them: DuckDB answers 4."""
    var plan = table(_basic_rows()).aggregate(
        [col("v").count().filter(gt(col("w"), _int(20))).alias("n")],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([4], int64))


def test_named_aggregate_filter_is_declared_and_printed() raises:
    """Two things a field has to carry that a type parameter carries for free.

    `references` is written out here because an `Optional[RuntimeValue]` is not
    a field the reflected walk can see — and if it went unseen, `ColumnPruning`
    would drop the column only the predicate names.
    """
    var predicate = gt(col("w"), _int(20))
    var agg = col("v").sum().filter(predicate.copy()).alias("total")
    var cols = agg.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "v")
    assert_equal(cols[1], "w")
    assert_equal(agg.name(), "total")
    assert_equal(
        String(agg), String("sum(v) filter (") + String(predicate) + ")"
    )


def test_named_aggregate_filter_rejects_a_non_boolean_predicate() raises:
    """At plan time, from the same `reject_non_boolean_filter` the comptime
    lane calls, so one rule serves both lanes."""
    var plan = table(_basic_rows()).aggregate(
        [col("v").sum().filter(col("w")).alias("total")],
        List[DynValue](),
    )
    var raised = False
    try:
        _ = plan.execute()
    except:
        raised = True
    assert_true(raised)
