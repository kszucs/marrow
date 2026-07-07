"""Tests for the named relational layer in ``marrow.expr.relations``.

Three surfaces, all fully monomorphized (``docs/aot-relations-design.md``):

- ``Table[Tbl]()`` — the column-access handle over a plain schema struct of
  dtype-tag fields. ``t.a`` reflects each field's dtype into a
  ``NumericColumn[T]`` / ``StringColumn`` leaf (columns carry only a runtime
  ``name``; position is resolved by name against the batch); string fields
  dispatch to ``StringColumn`` automatically.
- ``col(name, dtype)`` — the schema-struct-free, polars-style by-name factory
  producing the same leaves.
- ``Project[*Es: Column]`` — variadic projection assembling a ``RecordBatch``
  from a fixed heterogeneous column pack via ``Column.to_array()``. Takes a
  pre-built ``Tuple[*Es]`` (a ``VariadicPack`` can't be forwarded to another
  function's variadic parameter in current Mojo), so the call site is
  ``Project(Tuple(t.a, t.b))``.
- ``Filter[Input, Pred]`` — row filter chained via ``Project.filter(...)``; the
  predicate resolves against the *original* input batch, so it can reference a
  column projected away (like SQL ``WHERE`` vs ``SELECT``).

A field name that doesn't exist on ``Tbl`` is a compile error, not a runtime
exception — there is no ``compile_fail``-style harness in this project, so that
behavior is documented rather than asserted.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64, string
from marrow.tabular import RecordBatch, record_batch
from marrow.expr import col as dyn_col, in_memory_table, execute
from marrow.expr.relations import Table, Project, Filter
from marrow.expr.values import Add, Gt, Lt, col


struct _Orders:
    """Test fixture: a plain schema struct of dtype-tag fields — two int64
    columns and one string column, declared in that order.
    """

    var a: Int64Type
    var b: Int64Type
    var name: StringType


def _make_batch() raises -> RecordBatch:
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var s = array(["x", "y", "z"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


def _make_filter_batch() raises -> RecordBatch:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var s = array(["p", "q", "r", "s", "t"])
    return record_batch(
        [a.copy(), b.copy(), s.copy()], names=["a", "b", "name"]
    )


# ---------------------------------------------------------------------------
# Table[Tbl]() — column-access handle
# ---------------------------------------------------------------------------


def test_column_name_matches_declared_field() raises:
    """Table[Tbl]().<name> carries the field name; position is resolved by name
    against the batch schema (columns store no index)."""
    var t = Table[_Orders]()
    assert_equal(t.a.field_name(), "a")
    assert_equal(t.b.field_name(), "b")
    assert_equal(t.name.field_name(), "name")


def test_column_executes_without_runtime_schema() raises:
    """NumericColumn resolves data via its baked-in index."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    ref result = t.a.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([1, 2, 3], int64))

    ref result_b = t.b.execute(batch).to_any().as_int64()
    assert_true(result_b.copy() == array([10, 20, 30], int64))


def test_named_column_add_fuses() raises:
    """Fully-typed columns compose with the existing Add fusion node."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var added = Add(t.a, t.b)
    ref result = added.execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([11, 22, 33], int64))


def test_named_column_gt_fuses() raises:
    """Fully-typed columns compose with the Gt comparison node."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var pred = Gt(t.a, t.b)
    var result = pred.execute(batch)
    assert_true(result[0].value() == False)
    assert_true(result[1].value() == False)
    assert_true(result[2].value() == False)


def test_named_column_write_to() raises:
    """NumericColumn.write_to() displays the compile-time name."""
    var t = Table[_Orders]()
    assert_equal(String(t.a), "Col[a]")
    assert_equal(String(t.b), "Col[b]")


def test_named_string_column_executes() raises:
    """A string-typed field is dispatched to StringColumn and resolves via
    its baked-in index.
    """
    var t = Table[_Orders]()
    var batch = _make_batch()

    var result = t.name.execute(batch)
    assert_equal(result[0].to_string(), "x")
    assert_equal(result[1].to_string(), "y")
    assert_equal(result[2].to_string(), "z")


def test_named_string_column_write_to() raises:
    """StringColumn.write_to() displays the compile-time name."""
    var t = Table[_Orders]()
    assert_equal(String(t.name), "StrCol[name]")


# ---------------------------------------------------------------------------
# Project[*Es: Column] — variadic projection
# ---------------------------------------------------------------------------


def test_project_two_named_columns() raises:
    """Project over two plain named Columns preserves values and names."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var proj = Project(Tuple(t.a, t.b))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "b")

    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([1, 2, 3], int64))
    ref col_b = result.columns[1].as_int64()
    assert_true(col_b.copy() == array([10, 20, 30], int64))


def test_project_mixed_numeric_and_string() raises:
    """Project handles a heterogeneous NumericValue + StringValue mix."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var proj = Project(Tuple(t.a, t.name))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[0].dtype, int64)
    assert_equal(result.schema.fields[1].name, "name")
    assert_equal(result.schema.fields[1].dtype, string)

    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([1, 2, 3], int64))

    ref col_name = result.columns[1].as_string()
    assert_equal(col_name[0].to_string(), "x")
    assert_equal(col_name[1].to_string(), "y")
    assert_equal(col_name[2].to_string(), "z")


def test_project_single_column() raises:
    """Project over a single column still produces a valid one-field batch."""
    var t = Table[_Orders]()
    var batch = _make_batch()

    var proj = Project(Tuple(t.b))
    var result = proj.execute(batch)

    assert_equal(len(result.schema), 1)
    assert_equal(result.schema.fields[0].name, "b")
    ref col_b = result.columns[0].as_int64()
    assert_true(col_b.copy() == array([10, 20, 30], int64))


# ---------------------------------------------------------------------------
# Filter[Input, Pred] — row filter chained off Project
# ---------------------------------------------------------------------------


def test_filter_keeps_matching_rows() raises:
    """Filter keeps only rows where the predicate is True."""
    var t = Table[_Orders]()
    var batch = _make_filter_batch()

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
    var t = Table[_Orders]()
    var batch = _make_filter_batch()

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
    var t = Table[_Orders]()
    var batch = _make_filter_batch()

    var plan = Project(Tuple(t.a, t.b)).filter(Lt(t.a, t.a))
    var result = plan.execute(batch)
    assert_equal(result.num_rows(), 0)


def test_filter_matches_hand_written_relations_equivalent() raises:
    """The typed Project+Filter round trip matches the hand-written
    marrow.expr.plan (AnyRelation) equivalent -- the milestone
    definition-of-done from docs/aot-relations-design.md.
    """
    var t = Table[_Orders]()
    var batch = _make_filter_batch()

    var typed_plan = Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b))
    var typed_result = typed_plan.execute(batch)

    var erased_plan = (
        in_memory_table(batch)
        .select("a", "b")
        .filter(dyn_col("a") > dyn_col("b"))
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


# ---------------------------------------------------------------------------
# col(name, dtype) — polars-style by-name column factory
# ---------------------------------------------------------------------------


def test_col_resolves_numeric_by_name() raises:
    """A numeric col resolves its position by name against the batch."""
    var batch = _make_batch()
    ref result = col("a", int64).execute(batch).to_any().as_int64()
    assert_true(result.copy() == array([1, 2, 3], int64))


def test_col_resolves_string_by_name() raises:
    """A string col dispatches to StringColumn and resolves by name."""
    var batch = _make_batch()
    var result = col("name", string).execute(batch)
    assert_equal(result[0].to_string(), "x")
    assert_equal(result[2].to_string(), "z")


def test_col_composes_and_filters() raises:
    """``col`` columns compose into Add/Gt and drive a full Project+Filter — the
    same mixed numeric-predicate + string-projection query as Table[Tbl](),
    with no schema struct and no reflection.
    """
    var batch = _make_filter_batch()

    var added = Add(col("a", int64), col("b", int64))
    ref sum = added.execute(batch).to_any().as_int64()
    assert_true(sum.copy() == array([5, 9, 7, 12, 6], int64))

    var plan = Project(Tuple(col("a", int64), col("name", string))).filter(
        Gt(col("a", int64), col("b", int64))
    )
    var result = plan.execute(batch)
    assert_equal(result.num_rows(), 2)
    assert_equal(result.schema.fields[0].name, "a")
    assert_equal(result.schema.fields[1].name, "name")
    ref out_a = result.columns[0].as_int64()
    assert_true(out_a.copy() == array([5, 8], int64))


def main() raises:
    TestSuite.run[__functions_in_module()]()
