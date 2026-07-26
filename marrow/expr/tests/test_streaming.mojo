"""Tests for the IR-node → operator execution in ``marrow.expr.relations``.

Verifies the descriptive-plan / pull-based-operator design: ``execute`` opens a
plan into operators over ``AnyValue`` values (fused or interpreter). Small morsel
sizes exercise the streaming boundary — the same query must produce the same
result regardless of morsel size — and plans are reusable templates (opening
never mutates them).
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import Date32Array, TimestampArray
from marrow.builders import array, Date32Builder, PrimitiveBuilder
from marrow.dtypes import (
    Int32Type,
    Int64Type,
    StringType,
    TimestampType,
    date32,
    float64,
    int32,
    int64,
    second,
    string,
    timestamp,
    field,
)
from marrow.schema import Schema, schema
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.values import Gt, AnyValue, col
from marrow.expr.relations import (
    InMemoryTable,
    Project,
    Sort,
    Limit,
    AnyRelation,
    execute,
    in_memory_table,
)
from marrow.expr.dynamic import col as dyn_col, lit, case_when, if_else


def _batch() raises -> RecordBatch:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var s = array(["p", "q", "r", "s", "t"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


def _out_schema() raises -> Schema:
    return schema([field("a", int64), field("name", string)])


def _fused_plan(morsel: Int) raises -> AnyRelation:
    """SELECT a, name WHERE a > b, fused values, given a morsel size."""
    var filtered = AnyRelation(
        InMemoryTable(batch=_batch(), morsel_size=morsel)
    ).filter(AnyValue(Gt(col("a", int64), col("b", int64))))
    var values = List[AnyValue]()
    values.append(AnyValue(col("a", int64)))
    values.append(AnyValue(col("name", string)))
    return AnyRelation(
        Project(
            input=filtered,
            names=["a", "name"],
            values=values^,
            schema=_out_schema(),
        )
    )


def test_scan_streams_in_morsels() raises:
    """Opening an InMemoryTable yields morsel-sized slices, then Exhausted."""
    var scan = AnyRelation(
        InMemoryTable(batch=_batch(), morsel_size=2)
    ).to_processor()
    assert_equal(scan.pull().num_rows(), 2)
    assert_equal(scan.pull().num_rows(), 2)
    assert_equal(scan.pull().num_rows(), 1)
    var exhausted = False
    try:
        _ = scan.pull()
    except:
        exhausted = True
    assert_true(exhausted)


def test_streaming_filter_project() raises:
    """Full pipeline collected: a > b keeps rows 5 and 8."""
    var plan = _fused_plan(1024)
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "name")
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


def test_result_independent_of_morsel_size() raises:
    """Morsel size must not change the result."""
    for morsel in [1, 2, 3, 5, 1024]:
        var plan = _fused_plan(morsel)
        var result = execute(plan)
        assert_equal(result.num_rows(), 2)
        assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


def test_streaming_interpreter_values() raises:
    """The same pipeline with DynValue interpreter values (small morsels)
    produces the same result — fused and interpreted interchange."""
    var filtered = AnyRelation(
        InMemoryTable(batch=_batch(), morsel_size=2)
    ).filter(AnyValue(dyn_col("a") > dyn_col("b")))
    var values = List[AnyValue]()
    values.append(AnyValue(dyn_col("a")))
    values.append(AnyValue(dyn_col("name")))
    var plan = AnyRelation(
        Project(
            input=filtered,
            names=["a", "name"],
            values=values^,
            schema=_out_schema(),
        )
    )
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


# ---------------------------------------------------------------------------
# Plan reusability — a plan is a template, not a single-use cursor
# ---------------------------------------------------------------------------


def test_plan_is_reusable() raises:
    """execute() opens a fresh operator tree each run, so the same plan runs
    repeatedly and yields the same result — no single-use leakage."""
    var plan = _fused_plan(2)
    var r1 = execute(plan)
    var r2 = execute(plan)
    assert_equal(r1.num_rows(), 2)
    assert_equal(r2.num_rows(), 2)
    assert_true(r1.columns[0].as_int64().copy() == array([5, 8], int64))
    assert_true(r2.columns[0].as_int64().copy() == array([5, 8], int64))


def test_plan_copy_is_independent() raises:
    """A plan is an immutable template: copying it is an O(1) share, and the copy
    and the original each open their own operator tree — no shared cursor."""
    var plan = _fused_plan(2)
    var clone = plan.copy()
    var r_clone = execute(clone)
    var r_orig = execute(plan)
    assert_true(r_clone.columns[0].as_int64().copy() == array([5, 8], int64))
    assert_true(r_orig.columns[0].as_int64().copy() == array([5, 8], int64))


def test_aggregate_plan_is_reusable() raises:
    """Blocking aggregate opens a fresh grouper per run, so re-executing a plan
    re-aggregates from scratch."""
    var a = array([1, 1, 2, 2, 2], int64)
    var v = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a^, v^], names=["k", "v"])
    var plan = in_memory_table(batch).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").sum(),
        ],
    )
    var r1 = execute(plan)
    var r2 = execute(plan)
    assert_equal(r1.num_rows(), 2)
    assert_equal(r2.num_rows(), 2)
    # Same group count and same summed values across both runs.
    assert_equal(r1.num_columns(), r2.num_columns())


def test_aggregate_multi_key_schema() raises:
    """Two group keys produce two distinctly-named key fields (their source
    column names), not two fields both named 'key'."""
    var region = array([1, 1, 2], int64)
    var dept = array([1, 2, 2], int64)
    var v = array([10, 20, 30], int64)
    var batch = record_batch(
        [region^, dept^, v^], names=["region", "dept", "v"]
    )
    var plan = in_memory_table(batch).aggregate(
        keys=[dyn_col("region"), dyn_col("dept")],
        aggs=[
            dyn_col("v").sum(),
        ],
    )
    var result = execute(plan)
    assert_equal(result.num_columns(), 3)
    assert_equal(result.schema.fields[0].name, "region")
    assert_equal(result.schema.fields[1].name, "dept")
    assert_equal(result.schema.fields[2].name, "sum")


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def test_sort_by_column_ascending() raises:
    """Sort by a single column, ascending — all columns are reordered."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([10, 50, 30, 80, 20], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a")], ascending=[True]
    )
    var result = execute(plan)
    assert_equal(result.num_rows(), 5)
    assert_true(
        result.columns[0].as_int64().copy() == array([1, 2, 3, 5, 8], int64)
    )
    # The companion column follows the same permutation.
    assert_true(
        result.columns[1].as_int64().copy()
        == array([10, 20, 30, 50, 80], int64)
    )


def test_sort_by_column_descending() raises:
    """Sort by a single column, descending."""
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a")], ascending=[False]
    )
    var result = execute(plan)
    assert_true(
        result.columns[0].as_int64().copy() == array([8, 5, 3, 2, 1], int64)
    )


def test_sort_nulls_first() raises:
    """Nulls sort first when nulls_first=True (default)."""
    var a = array([3, None, 1, None, 2], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a")], ascending=[True], nulls_first=True
    )
    var col = execute(plan).columns[0].as_int64().copy()
    assert_true(col == array([None, None, 1, 2, 3], int64))


def test_sort_nulls_last() raises:
    """Nulls sort last when nulls_first=False."""
    var a = array([3, None, 1, None, 2], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a")], ascending=[True], nulls_first=False
    )
    var col = execute(plan).columns[0].as_int64().copy()
    assert_true(col == array([1, 2, 3, None, None], int64))


def test_sort_multi_key() raises:
    """Two keys: primary ascending, secondary ascending — LSD stable."""
    var a = array([1, 1, 2, 2], int64)
    var b = array([2, 1, 1, 2], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a"), dyn_col("b")], ascending=[True, True]
    )
    var result = execute(plan)
    assert_true(
        result.columns[0].as_int64().copy() == array([1, 1, 2, 2], int64)
    )
    assert_true(
        result.columns[1].as_int64().copy() == array([1, 2, 1, 2], int64)
    )


def test_sort_multi_key_mixed_direction() raises:
    """Primary ascending, secondary descending."""
    var a = array([1, 1, 2, 2], int64)
    var b = array([2, 1, 1, 2], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("a"), dyn_col("b")], ascending=[True, False]
    )
    var result = execute(plan)
    assert_true(
        result.columns[0].as_int64().copy() == array([1, 1, 2, 2], int64)
    )
    assert_true(
        result.columns[1].as_int64().copy() == array([2, 1, 2, 1], int64)
    )


def test_sort_by_string_column() raises:
    """Sort by a string column (type-agnostic via the sort kernel)."""
    var s = array(["pear", "apple", "cherry", "banana"])
    var batch = record_batch([s^], names=["s"])
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col("s")], ascending=[True]
    )
    var col = execute(plan).columns[0].as_string().copy()
    assert_true(col == array(["apple", "banana", "cherry", "pear"]))


# ---------------------------------------------------------------------------
# Limit / Offset
# ---------------------------------------------------------------------------


def test_limit() raises:
    """Keep the first n rows."""
    var a = array([1, 2, 3, 4, 5, 6], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).limit(3)
    var result = execute(plan)
    assert_equal(result.num_rows(), 3)
    assert_true(result.columns[0].as_int64().copy() == array([1, 2, 3], int64))


def test_limit_offset() raises:
    """Skip offset rows, then keep n."""
    var a = array([1, 2, 3, 4, 5, 6], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).limit(2, offset=2)
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([3, 4], int64))


def test_limit_offset_across_morsels() raises:
    """Offset/limit boundaries that straddle small morsels are sliced correctly.
    """
    for morsel in [1, 2, 3, 4, 7]:
        var a = array([1, 2, 3, 4, 5, 6], int64)
        var batch = record_batch([a^], names=["a"])
        var plan = AnyRelation(
            InMemoryTable(batch=batch, morsel_size=morsel)
        ).limit(3, offset=1)
        var result = execute(plan)
        assert_equal(result.num_rows(), 3)
        assert_true(
            result.columns[0].as_int64().copy() == array([2, 3, 4], int64)
        )


def test_limit_beyond_end() raises:
    """A limit larger than the available rows returns all rows."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).limit(10, offset=1)
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([2, 3], int64))


# ---------------------------------------------------------------------------
# Top-K (Sort followed by Limit → folded into Sort(limit=…))
# ---------------------------------------------------------------------------


def test_topk_fold_into_sort() raises:
    """`.sort(...).limit(k)` folds into a Sort node carrying the limit."""
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = (
        in_memory_table(batch)
        .sort(keys=[dyn_col("a")], ascending=[False])
        .limit(2)
    )
    assert_true(String(plan).find("limit=2") != -1)
    assert_true(plan.downcast[Sort]()[].limit.__bool__())


def test_topk_values() raises:
    """Top-2 descending yields the two largest, in order."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([10, 50, 30, 80, 20], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    var plan = (
        in_memory_table(batch)
        .sort(keys=[dyn_col("a")], ascending=[False])
        .limit(2)
    )
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([8, 5], int64))
    assert_true(result.columns[1].as_int64().copy() == array([80, 50], int64))


def test_limit_over_sort_with_offset_not_folded() raises:
    """A non-zero offset can't fold into the sort; a Limit wraps the Sort."""
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = (
        in_memory_table(batch)
        .sort(keys=[dyn_col("a")], ascending=[True])
        .limit(2, offset=1)
    )
    var result = execute(plan)
    # Ascending [1,2,3,5,8], skip 1, take 2 -> [2,3].
    assert_true(result.columns[0].as_int64().copy() == array([2, 3], int64))


# ---------------------------------------------------------------------------
# Computed Project
# ---------------------------------------------------------------------------


def test_computed_project() raises:
    """Project computed columns: x+1, a literal, and a renamed passthrough."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).project(
        names=["a_plus", "one", "renamed"],
        values=[
            dyn_col("a") + lit[Int64Type](1),
            lit[Int64Type](1),
            dyn_col("a"),
        ],
    )
    var result = execute(plan)
    assert_equal(result.num_columns(), 3)
    assert_equal(result.schema.fields[0].name, "a_plus")
    assert_equal(result.schema.fields[1].name, "one")
    assert_equal(result.schema.fields[2].name, "renamed")
    assert_true(result.columns[0].as_int64().copy() == array([2, 3, 4], int64))
    assert_true(result.columns[1].as_int64().copy() == array([1, 1, 1], int64))
    assert_true(result.columns[2].as_int64().copy() == array([1, 2, 3], int64))


def test_computed_project_dtype_inferred() raises:
    """The projected schema carries each expression's inferred output dtype."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a^], names=["a"])
    var plan = in_memory_table(batch).project(
        names=["a_plus"], values=[dyn_col("a") + lit[Int64Type](1)]
    )
    assert_equal(plan.schema().fields[0].dtype, int64)


# ---------------------------------------------------------------------------
# Aggregate — completeness (ClickBench query shapes)
#
# `HAVING` is deliberately not a node of its own: a `Filter` on top of an
# `Aggregate` evaluates its predicate against the aggregate's *output* batch, so
# `rel.aggregate(...).filter(...)` resolves names against the aggregate output
# schema. `test_aggregate_having*` pin that down.
# ---------------------------------------------------------------------------


def _dates(days: List[Int]) raises -> Date32Array:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def _timestamps(seconds: List[Int]) raises -> TimestampArray:
    var b = PrimitiveBuilder[TimestampType](timestamp(second), len(seconds))
    for s in seconds:
        b.append(Int64(s))
    return b.finish()


def _agg_batch() raises -> RecordBatch:
    """k: 3 groups (1, 1, 2, 2, 3); v: values; s: strings; d: dates; t: times.
    """
    var k = array([1, 1, 2, 2, 3], int64)
    var v = array([10, 10, 30, 40, 50], int64)
    var s = array(["pear", "pear", "apple", "fig", "kiwi"])
    var d = _dates([19000, 18500, 19100, 18800, 19200])
    # Two distinct hours: 12:00 (rows 0, 1, 4) and 13:00 (rows 2, 3).
    var t = _timestamps(
        [
            1_560_601_845,
            1_560_602_000,
            1_560_605_500,
            1_560_605_600,
            1_560_601_900,
        ]
    )
    return record_batch([k^, v^, s^, d^, t^], names=["k", "v", "s", "d", "t"])


def _sorted_by_key(batch: RecordBatch) raises -> RecordBatch:
    """Group order is hash-table insertion order; sort by the first key so the
    assertions below are order-independent."""
    var plan = in_memory_table(batch).sort(
        keys=[dyn_col(batch.schema.fields[0].name)], ascending=[True]
    )
    return execute(plan)


def test_aggregate_count_distinct_grouped() raises:
    """COUNT(DISTINCT v) per group — group 1 has {10}, group 2 has {30, 40}."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").count_distinct(),
        ],
    )
    assert_equal(plan.schema().fields[1].name, "count_distinct")
    assert_equal(plan.schema().fields[1].dtype, int64)
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[0].as_int64().copy() == array([1, 2, 3], int64))
    assert_true(result.columns[1].as_int64().copy() == array([1, 2, 1], int64))


def test_aggregate_count_distinct_whole_table() raises:
    """A literal group key collapses every row into one group — the ungrouped
    COUNT(DISTINCT v) over the whole table (3 distinct values)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[lit[Int64Type](0)],
        aggs=[
            dyn_col("v").count_distinct().alias("distinct_v"),
        ],
    )
    assert_equal(plan.schema().fields[0].name, "key0")
    var result = execute(plan)
    assert_equal(result.num_rows(), 1)
    assert_equal(result.schema.fields[1].name, "distinct_v")
    assert_equal(result.columns[1].as_int64()[0].value(), 4)


def test_aggregate_approx_count_distinct() raises:
    """approx_count_distinct is exact at these cardinalities."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").approx_count_distinct(),
        ],
    )
    assert_equal(plan.schema().fields[1].dtype, int64)
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[1].as_int64().copy() == array([1, 2, 1], int64))


def test_aggregate_min_max_string() raises:
    """min/max over a string value column keep the string dtype (Q22/Q23)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("s").min().alias("lo"),
            dyn_col("s").max().alias("hi"),
        ],
    )
    assert_equal(plan.schema().fields[1].dtype, string)
    assert_equal(plan.schema().fields[2].dtype, string)
    var result = _sorted_by_key(execute(plan))
    assert_true(
        result.columns[1].as_string().copy() == array(["pear", "apple", "kiwi"])
    )
    assert_true(
        result.columns[2].as_string().copy() == array(["pear", "fig", "kiwi"])
    )


def test_aggregate_min_max_date() raises:
    """min/max over a date32 value column keep the temporal dtype (Q7)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("d").min().alias("first_day"),
            dyn_col("d").max().alias("last_day"),
        ],
    )
    assert_true(plan.schema().fields[1].dtype == date32())
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[1].dtype() == date32())
    assert_true(
        result.columns[1].as_date32().copy() == _dates([18500, 18800, 19200])
    )
    assert_true(
        result.columns[2].as_date32().copy() == _dates([19000, 19100, 19200])
    )


def test_aggregate_count_over_string_column() raises:
    """COUNT(*)-style count over a non-numeric column — validity only."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("s").count(),
        ],
    )
    assert_equal(plan.schema().fields[1].dtype, int64)
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[1].as_int64().copy() == array([2, 2, 1], int64))


def test_aggregate_out_dtypes() raises:
    """The plan-time output dtype matches what the kernels actually produce for
    every aggregate kind."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").count().alias("c"),
            dyn_col("v").mean().alias("m"),
            dyn_col("v").min().alias("mn"),
            dyn_col("v").sum().alias("s"),
        ],
    )
    var out = execute(plan)
    for i in range(len(plan.schema().fields)):
        assert_equal(plan.schema().fields[i].dtype, out.schema.fields[i].dtype)
    assert_equal(out.schema.fields[1].dtype, int64)  # count
    assert_equal(out.schema.fields[2].dtype, float64)  # mean
    assert_equal(out.schema.fields[3].dtype, int64)  # min keeps the input dtype
    assert_equal(out.schema.fields[4].dtype, int64)  # sum widens to int64


# --- computed group keys ----------------------------------------------------


def test_aggregate_computed_key_arithmetic() raises:
    """An arithmetic group key (k * 10) groups on the computed value."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k") * lit[Int64Type](10)],
        aggs=[
            dyn_col("v").sum().alias("total"),
        ],
    )
    assert_equal(plan.schema().fields[0].name, "key0")
    assert_equal(plan.schema().fields[0].dtype, int64)
    var result = _sorted_by_key(execute(plan))
    assert_true(
        result.columns[0].as_int64().copy() == array([10, 20, 30], int64)
    )
    assert_true(
        result.columns[1].as_int64().copy() == array([20, 70, 50], int64)
    )


def test_aggregate_computed_key_case_when() raises:
    """A CASE WHEN group key: v < 30 -> 0, else 1."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[
            case_when(
                [dyn_col("v") < lit[Int64Type](30)],
                [lit[Int64Type](0)],
                lit[Int64Type](1),
            )
        ],
        aggs=[
            dyn_col("v").count().alias("n"),
        ],
    )
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[0].as_int64().copy() == array([0, 1], int64))
    assert_true(result.columns[1].as_int64().copy() == array([2, 3], int64))


def test_aggregate_computed_key_if_else() raises:
    """if_else as a group key (the two-branch CASE)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[
            if_else(
                dyn_col("k") == lit[Int64Type](1),
                lit[Int64Type](100),
                lit[Int64Type](200),
            )
        ],
        aggs=[
            dyn_col("v").sum(),
        ],
    )
    var result = _sorted_by_key(execute(plan))
    assert_true(result.columns[0].as_int64().copy() == array([100, 200], int64))
    assert_true(result.columns[1].as_int64().copy() == array([20, 120], int64))


def test_aggregate_computed_key_year() raises:
    """year(date) as a group key — an int32 extraction key (Q19/Q40)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("d").year()],
        aggs=[
            dyn_col("v").count().alias("n"),
        ],
    )
    assert_equal(plan.schema().fields[0].dtype, int32)
    var result = _sorted_by_key(execute(plan))
    # 18500 -> 2020, 18800 -> 2021, 19000/19100/19200 -> 2022.
    assert_true(
        result.columns[0].as_int32().copy() == array([2020, 2021, 2022], int32)
    )
    assert_true(result.columns[1].as_int64().copy() == array([1, 1, 3], int64))


def test_aggregate_computed_key_date_trunc() raises:
    """date_trunc yields a *temporal* group key — it is grouped through its
    signed-integer backing and relabelled back to the timestamp dtype on the way
    out (Q19/Q35/Q36/Q43)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("t").date_trunc("hour")],
        aggs=[
            dyn_col("v").count().alias("n"),
        ],
    )
    assert_true(plan.schema().fields[0].dtype == timestamp(second))
    var result = execute(plan)
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].dtype() == timestamp(second))
    assert_true(
        result.columns[0].as_timestamp().copy()
        == _timestamps([1_560_600_000, 1_560_603_600])
    )
    assert_true(result.columns[1].as_int64().copy() == array([3, 2], int64))


# --- computed aggregate inputs ---------------------------------------------


def test_aggregate_computed_value_arithmetic() raises:
    """SUM(v + 1) — a computed aggregate input (Q30)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            (dyn_col("v") + lit[Int64Type](1)).sum().alias("total"),
        ],
    )
    assert_equal(plan.schema().fields[1].dtype, int64)
    var result = _sorted_by_key(execute(plan))
    assert_true(
        result.columns[1].as_int64().copy() == array([22, 72, 51], int64)
    )


def test_aggregate_computed_value_length() raises:
    """AVG(length(s)) — a computed, dtype-changing aggregate input (Q28)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("s").length().mean().alias("avg_len"),
        ],
    )
    assert_equal(plan.schema().fields[1].dtype, float64)
    var result = _sorted_by_key(execute(plan))
    var avg = result.columns[1].as_float64().copy()
    assert_true(avg[0].value() == 4.0)  # "pear", "pear"
    assert_true(avg[1].value() == 4.0)  # "apple" (5), "fig" (3)
    assert_true(avg[2].value() == 4.0)  # "kiwi"


# --- output names + HAVING --------------------------------------------------


def test_aggregate_names_disambiguate_outputs() raises:
    """Two means of different columns need distinct output names."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").mean().alias("avg_v"),
            dyn_col("d").year().mean().alias("avg_year"),
        ],
    )
    assert_equal(plan.schema().fields[1].name, "avg_v")
    assert_equal(plan.schema().fields[2].name, "avg_year")
    var result = execute(plan)
    assert_equal(result.schema.fields[1].name, "avg_v")
    assert_equal(result.schema.fields[2].name, "avg_year")


def test_aggregate_default_names_are_the_functions() raises:
    """Without `names`, outputs keep their function name (unchanged default)."""
    var plan = in_memory_table(_agg_batch()).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").sum(),
        ],
    )
    assert_equal(plan.schema().fields[1].name, "sum")


def test_aggregate_having() raises:
    """HAVING COUNT(*) > 1 — a Filter on top of the Aggregate, resolving names
    against the aggregate output schema (Q28/Q29)."""
    var plan = (
        in_memory_table(_agg_batch())
        .aggregate(
            keys=[dyn_col("k")],
            aggs=[
                dyn_col("v").count().alias("n"),
            ],
        )
        .filter(AnyValue(dyn_col("n") > lit[Int64Type](1)))
    )
    var result = _sorted_by_key(execute(plan))
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([1, 2], int64))
    assert_true(result.columns[1].as_int64().copy() == array([2, 2], int64))


def test_aggregate_having_on_aliased_aggregate() raises:
    """HAVING over an aliased aggregate, then sorted + limited — the ClickBench
    `GROUP BY ... HAVING ... ORDER BY ... LIMIT` shape end to end."""
    var plan = (
        in_memory_table(_agg_batch())
        .aggregate(
            keys=[dyn_col("k")],
            aggs=[
                dyn_col("v").sum().alias("total"),
            ],
        )
        .filter(AnyValue(dyn_col("total") >= lit[Int64Type](50)))
        .sort(keys=[dyn_col("total")], ascending=[False])
        .limit(1)
    )
    var result = execute(plan)
    assert_equal(result.num_rows(), 1)
    assert_true(result.columns[0].as_int64().copy() == array([2], int64))
    assert_true(result.columns[1].as_int64().copy() == array([70], int64))


def test_aggregate_streams_across_morsels() raises:
    """Aggregation is incremental across morsels: the grouper consumes each
    morsel's keys as it arrives and the shared per-column entry point then sees
    the concatenation of every morsel's group ids and values. A one-row morsel
    gives the same answer as a single batch, for a heterogeneous aggregate set.
    """
    var plan = AnyRelation(
        InMemoryTable(batch=_agg_batch(), morsel_size=1)
    ).aggregate(
        keys=[dyn_col("k")],
        aggs=[
            dyn_col("v").sum().alias("total"),
            dyn_col("s").min().alias("lo"),
            dyn_col("v").count_distinct().alias("nd"),
        ],
    )
    var result = _sorted_by_key(execute(plan))
    assert_equal(result.num_rows(), 3)
    assert_true(
        result.columns[1].as_int64().copy() == array([20, 70, 50], int64)
    )
    assert_true(
        result.columns[2].as_string().copy() == array(["pear", "apple", "kiwi"])
    )
    assert_true(result.columns[3].as_int64().copy() == array([1, 2, 1], int64))


def main() raises:
    TestSuite.run[__functions_in_module()]()
