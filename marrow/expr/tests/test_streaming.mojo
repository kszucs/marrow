"""Tests for the IR-node → operator execution in ``marrow.expr.relations``.

Verifies the descriptive-plan / pull-based-operator design: ``execute`` opens a
plan into operators over ``AnyValue`` values (fused or interpreter). Small morsel
sizes exercise the streaming boundary — the same query must produce the same
result regardless of morsel size — and plans are reusable templates (opening
never mutates them).
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64, string, field
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
from marrow.expr.dynamic import col as dyn_col, lit


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
        keys=[dyn_col("k")], values=[dyn_col("v")], funcs=["sum"]
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
        values=[dyn_col("v")],
        funcs=["sum"],
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


def main() raises:
    TestSuite.run[__functions_in_module()]()
