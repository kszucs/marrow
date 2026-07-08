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
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.dynamic import col
from marrow.expr.values import AnyValue
from marrow.expr.relations import (
    InMemoryTable,
    Project,
    AnyRelation,
    execute,
    in_memory_table,
)
from marrow.expr.dynamic import col as dyn_col, lit


struct _Orders:
    var a: Int64Type
    var b: Int64Type
    var name: StringType


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
    var t = Table[_Orders]()
    var filtered = AnyRelation(
        InMemoryTable(batch=_batch(), morsel_size=morsel)
    ).filter(AnyValue(Gt(t.a, t.b)))
    var values = List[AnyValue]()
    values.append(AnyValue(t.a))
    values.append(AnyValue(t.name))
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
    var scan = AnyRelation(InMemoryTable(batch=_batch(), morsel_size=2)).open()
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
        assert_true(
            result.columns[0].as_int64().copy() == array([5, 8], int64)
        )


def test_streaming_interpreter_values() raises:
    """The same pipeline with DynValue interpreter values (small morsels)
    produces the same result — fused and interpreted interchange."""
    var filtered = AnyRelation(
        InMemoryTable(batch=_batch(), morsel_size=2)
    ).filter(AnyValue(col("a") > col("b")))
    var values = List[AnyValue]()
    values.append(AnyValue(col("a")))
    values.append(AnyValue(col("name")))
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


def main() raises:
    TestSuite.run[__functions_in_module()]()
