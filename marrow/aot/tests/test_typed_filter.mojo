"""Tests for ``Filter[Input, Pred]`` (``marrow.aot.table``) -- the
fully-typed row filter over a typed relation, chained via
``Project.filter(...)``.

This is the milestone-4/5 definition-of-done round trip from
``docs/aot-relations-design.md``: a two-column ``Orders`` table projected and
filtered, producing a ``RecordBatch`` equal to the hand-written
``marrow.dyn.relations`` equivalent (``in_memory_table(batch).select(...).
filter(...)``).

The predicate is evaluated against the *original* input batch, not the
projected output -- this is exercised directly by projecting away column
``b`` while filtering on it (mirrors SQL's ``WHERE`` referencing a column
outside the ``SELECT`` list).
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import Int64Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.dyn import col, in_memory_table, execute
from marrow.aot.table import Column, StringColumn, Table, Project, Filter
from marrow.aot.values import Gt, Lt


struct _Orders(Table):
    var a: Column[_Orders, "a", Int64Type]
    var b: Column[_Orders, "b", Int64Type]
    var name: StringColumn[_Orders, "name"]

    def __init__(out self):
        self.a = {}
        self.b = {}
        self.name = {}


def _make_batch() raises -> RecordBatch:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var s = array(["p", "q", "r", "s", "t"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


def test_filter_keeps_matching_rows() raises:
    """Filter keeps only rows where the predicate is True."""
    var t = _Orders()
    var batch = _make_batch()

    var plan = Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b))
    var result = plan.execute(batch)

    assert_equal(result.num_rows(), 2)
    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([5, 8], int64))
    ref col_b = result.columns[1].as_int64()
    assert_true(col_b.copy() == array([4, 4], int64))


def test_filter_predicate_references_dropped_column() raises:
    """The predicate resolves against the original batch, so it can
    reference a column that was projected away (like SQL WHERE vs SELECT).
    """
    var t = _Orders()
    var batch = _make_batch()

    var plan = Project(Tuple(t.a, t.name)).filter(Gt(t.a, t.b))
    var result = plan.execute(batch)

    assert_equal(len(result.schema), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "name")

    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([5, 8], int64))
    ref col_name = result.columns[1].as_string()
    assert_equal(col_name[0].to_string(), "q")
    assert_equal(col_name[1].to_string(), "s")


def test_filter_no_matching_rows() raises:
    """Filter produces a zero-row (but still valid) batch when nothing
    matches.
    """
    var t = _Orders()
    var batch = _make_batch()

    var plan = Project(Tuple(t.a, t.b)).filter(Lt(t.a, t.a))
    var result = plan.execute(batch)
    assert_equal(result.num_rows(), 0)


def test_filter_matches_hand_written_relations_equivalent() raises:
    """The typed Project+Filter round trip matches the hand-written
    marrow.dyn.relations (AnyRelation) equivalent -- the milestone
    definition-of-done from docs/aot-relations-design.md.
    """
    var t = _Orders()
    var batch = _make_batch()

    var typed_plan = Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b))
    var typed_result = typed_plan.execute(batch)

    var erased_plan = (
        in_memory_table(batch).select("a", "b").filter(col("a") > col("b"))
    )
    var erased_result = execute(erased_plan)

    assert_true(erased_result.schema == typed_result.schema)
    assert_equal(erased_result.num_rows(), typed_result.num_rows())
    ref erased_a = erased_result.columns[0].as_int64()
    ref typed_a = typed_result.columns[0].as_int64()
    assert_true(erased_a.copy() == typed_a.copy())
    ref erased_b = erased_result.columns[1].as_int64()
    ref typed_b = typed_result.columns[1].as_int64()
    assert_true(erased_b.copy() == typed_b.copy())


def main() raises:
    TestSuite.run[__functions_in_module()]()
