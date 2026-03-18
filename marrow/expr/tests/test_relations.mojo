from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import Field, int64, float64
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr import (
    AnyValue, col, lit, ADD, LT, in_memory_table,
    SCAN_NODE, FILTER_NODE, PROJECT_NODE, IN_MEMORY_TABLE_NODE,
)
from marrow.expr.relations import AnyRelation, Scan, Filter, Project, InMemoryTable
from marrow.expr.values import Column


fn _schema() -> Schema:
    var s = Schema()
    s.append(Field("x", int64))
    s.append(Field("y", float64))
    return s^


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_scan_schema() raises:
    """Scan.schema returns the declared schema."""
    var src = Scan(name="t", schema_=_schema())
    var schema = src.schema()
    assert_equal(len(schema), 2)
    assert_equal(schema.fields[0].name, "x")
    assert_equal(schema.fields[1].name, "y")


def test_scan_no_inputs() raises:
    """Scan is a leaf — no child plans."""
    var src = Scan(name="t", schema_=_schema())
    assert_equal(len(src.inputs()), 0)


def test_scan_no_exprs() raises:
    """Scan has no expressions."""
    var src = Scan(name="t", schema_=_schema())
    assert_equal(len(src.exprs()), 0)


def test_scan_write_to() raises:
    var src = AnyRelation(Scan(name="orders", schema_=_schema()))
    assert_equal(String(src), "Scan(orders)")


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_schema_passthrough() raises:
    """Filter output schema equals the input schema."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = col(0) > lit[int64](0)
    var filt = Filter(input=src, predicate=pred)
    var schema = filt.schema()
    assert_equal(len(schema), 2)
    assert_equal(schema.fields[0].name, "x")


def test_filter_one_input() raises:
    """Filter has exactly one child plan."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = col(0) < col(1)
    var filt = Filter(input=src, predicate=pred)
    assert_equal(len(filt.inputs()), 1)


def test_filter_one_expr() raises:
    """Filter exposes its predicate as an expression."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = col(0) < col(1)
    var filt = Filter(input=src, predicate=pred)
    var exprs = filt.exprs()
    assert_equal(len(exprs), 1)
    assert_equal(exprs[0].kind(), LT)


def test_filter_write_to() raises:
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = col(0) < col(1)
    var filt = AnyRelation(Filter(input=src, predicate=pred))
    assert_equal(String(filt), "Filter(predicate=less(input(0), input(1)))")


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def test_project_schema() raises:
    """Project output schema contains only the projected columns."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var out_schema = Schema()
    out_schema.append(Field("z", int64))
    var names = List[String]()
    names.append("z")
    var exprs = List[AnyValue]()
    exprs.append(col(0) + col(1))
    var proj = Project(
        input=src, names=names^, exprs_=exprs^, schema_=out_schema
    )
    var schema = proj.schema()
    assert_equal(len(schema), 1)
    assert_equal(schema.fields[0].name, "z")


def test_project_exprs() raises:
    """Project exposes its expressions."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var out_schema = Schema()
    out_schema.append(Field("z", int64))
    var names = List[String]()
    names.append("z")
    var exprs_list = List[AnyValue]()
    exprs_list.append(col(0) + col(1))
    var proj = Project(
        input=src, names=names^, exprs_=exprs_list^, schema_=out_schema
    )
    var exprs = proj.exprs()
    assert_equal(len(exprs), 1)
    assert_equal(exprs[0].kind(), ADD)


def test_project_write_to() raises:
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var out_schema = Schema()
    out_schema.append(Field("z", int64))
    var names = List[String]()
    names.append("z")
    var exprs = List[AnyValue]()
    exprs.append(col(0) + col(1))
    var proj = AnyRelation(
        Project(input=src, names=names^, exprs_=exprs^, schema_=out_schema)
    )
    assert_equal(String(proj), "Project([z=add(input(0), input(1))])")


# ---------------------------------------------------------------------------
# AnyRelation type erasure / downcast
# ---------------------------------------------------------------------------


def test_anyrelation_downcast_scan() raises:
    """AnyRelation wrapping a Scan can be downcast back."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    assert_equal(src.downcast[Scan]()[].name, "t")


def test_anyrelation_o1_copy() raises:
    """AnyRelation copies share the same underlying allocation (O(1))."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var copy = src  # O(1) ref-count bump
    assert_equal(copy.schema().fields[0].name, "x")


# ---------------------------------------------------------------------------
# kind() dispatch
# ---------------------------------------------------------------------------


def test_scan_kind() raises:
    """Scan reports SCAN_NODE kind."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    assert_equal(src.kind(), SCAN_NODE)


def test_filter_kind() raises:
    """Filter reports FILTER_NODE kind."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var filt = AnyRelation(Filter(input=src, predicate=col(0) > lit[int64](0)))
    assert_equal(filt.kind(), FILTER_NODE)


def test_project_kind() raises:
    """Project reports PROJECT_NODE kind."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var out_schema = Schema()
    out_schema.append(Field("z", int64))
    var names = List[String]()
    names.append("z")
    var exprs = List[AnyValue]()
    exprs.append(col(0))
    var proj = AnyRelation(
        Project(input=src, names=names^, exprs_=exprs^, schema_=out_schema)
    )
    assert_equal(proj.kind(), PROJECT_NODE)


# ---------------------------------------------------------------------------
# InMemoryTable
# ---------------------------------------------------------------------------


fn _test_batch() raises -> RecordBatch:
    var schema = Schema()
    schema.append(Field("a", int64))
    var a = array[int64]([1, 2, 3])
    var cols = List[Array]()
    cols.append(Array(a))
    return RecordBatch(schema=schema, columns=cols^)


def test_in_memory_table_kind() raises:
    """InMemoryTable reports IN_MEMORY_TABLE_NODE kind."""
    var t = in_memory_table(_test_batch())
    assert_equal(t.kind(), IN_MEMORY_TABLE_NODE)


def test_in_memory_table_schema() raises:
    """InMemoryTable schema matches the batch schema."""
    var t = in_memory_table(_test_batch())
    var schema = t.schema()
    assert_equal(len(schema), 1)
    assert_equal(schema.fields[0].name, "a")


def test_in_memory_table_leaf() raises:
    """InMemoryTable is a leaf node with no inputs or expressions."""
    var t = in_memory_table(_test_batch())
    assert_equal(len(t.inputs()), 0)
    assert_equal(len(t.exprs()), 0)


def test_in_memory_table_downcast() raises:
    """InMemoryTable can be downcast to access the batch."""
    var t = in_memory_table(_test_batch())
    var imt = t.downcast[InMemoryTable]()
    assert_equal(imt[].batch.num_rows(), 3)


# ---------------------------------------------------------------------------
# Scan + filter + select plan composition
# ---------------------------------------------------------------------------


def test_scan_filter_kind() raises:
    """Scan.filter() produces a FILTER_NODE wrapping the scan."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.filter(col("x") > lit[int64](0))
    assert_equal(plan.kind(), FILTER_NODE)


def test_scan_filter_schema_passthrough() raises:
    """Scan.filter() preserves the scan's output schema."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.filter(col("x") > lit[int64](0))
    var schema = plan.schema()
    assert_equal(len(schema), 2)
    assert_equal(schema.fields[0].name, "x")
    assert_equal(schema.fields[1].name, "y")


def test_scan_filter_resolves_column_name() raises:
    """col('x') inside filter is resolved to a positional col(0)."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.filter(col("x") > lit[int64](0))
    var filt = plan.downcast[Filter]()
    var pred_inputs = filt[].predicate.inputs()
    assert_equal(pred_inputs[0].downcast[Column]()[].index, 0)


def test_scan_select_kind() raises:
    """Scan.select() produces a PROJECT_NODE."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.select("x")
    assert_equal(plan.kind(), PROJECT_NODE)


def test_scan_select_schema() raises:
    """Scan.select('x') yields a single-field schema."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.select("x")
    var schema = plan.schema()
    assert_equal(len(schema), 1)
    assert_equal(schema.fields[0].name, "x")


def test_scan_filter_select_kinds() raises:
    """Scan.filter().select() chains FILTER_NODE under PROJECT_NODE."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.filter(col("x") > lit[int64](0)).select("x")
    assert_equal(plan.kind(), PROJECT_NODE)
    var proj = plan.downcast[Project]()
    assert_equal(proj[].input.kind(), FILTER_NODE)


def test_scan_filter_select_schema() raises:
    """Scan.filter().select('y') final schema has only 'y'."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.filter(col("x") > lit[int64](0)).select("y")
    var schema = plan.schema()
    assert_equal(len(schema), 1)
    assert_equal(schema.fields[0].name, "y")


def test_scan_select_filter_kinds() raises:
    """Scan.select().filter() chains PROJECT_NODE under FILTER_NODE."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var plan = src.select("x").filter(col("x") > lit[int64](0))
    assert_equal(plan.kind(), FILTER_NODE)
    var filt = plan.downcast[Filter]()
    assert_equal(filt[].input.kind(), PROJECT_NODE)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
