"""Tests for the pull-based streaming relational ops in ``marrow.expr.streaming``.

Verifies the fat-node design (``docs/expr-unification-plan.md``, Phase 2): each
op is its own processor, pulled morsel-at-a-time with no ``Planner``, over
``AnyValue`` values (fused or interpreter). Small morsel sizes exercise the
streaming boundary — the same query must produce the same result regardless of
morsel size, and fused and interpreter values must interchange.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.erased import AnyValue, DynValue
from marrow.expr.streaming import AnySource, Scan, Filter, Project


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


def _fused_plan(morsel: Int) raises -> AnySource:
    """SELECT a, name WHERE a > b, fused values, given a morsel size."""
    var t = Table[_Orders]()
    var scan = AnySource(Scan(_batch(), morsel))
    var filtered = AnySource(Filter(scan^, AnyValue(Gt(t.a, t.b))))
    var values = List[AnyValue]()
    values.append(AnyValue(t.a))
    values.append(AnyValue(t.name))
    return AnySource(Project(filtered^, values^))


def test_scan_streams_in_morsels() raises:
    """Scan yields the batch in morsel-sized slices, then Exhausted."""
    var scan = AnySource(Scan(_batch(), 2))
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
    var result = plan.collect()
    assert_equal(result.num_rows(), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "name")
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


def test_result_independent_of_morsel_size() raises:
    """Morsel size must not change the result — the streaming boundary is
    transparent."""
    for morsel in [1, 2, 3, 5, 1024]:
        var plan = _fused_plan(morsel)
        var result = plan.collect()
        assert_equal(result.num_rows(), 2)
        assert_true(
            result.columns[0].as_int64().copy() == array([5, 8], int64)
        )


def test_streaming_interpreter_values() raises:
    """The same pipeline with DynValue interpreter values (small morsels)
    produces the same result — fused and interpreted interchange."""
    var scan = AnySource(Scan(_batch(), 2))
    var pred = AnyValue(
        DynValue.gt(
            AnyValue(DynValue.load(0, "a")), AnyValue(DynValue.load(1, "b"))
        )
    )
    var filtered = AnySource(Filter(scan^, pred^))
    var values = List[AnyValue]()
    values.append(AnyValue(DynValue.load(0, "a")))
    values.append(AnyValue(DynValue.load(2, "name")))
    var plan = AnySource(Project(filtered^, values^))
    var result = plan.collect()
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([5, 8], int64))


def main() raises:
    TestSuite.run[__functions_in_module()]()
