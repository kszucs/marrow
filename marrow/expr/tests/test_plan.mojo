from std.testing import assert_equal, TestSuite

from marrow.dtypes import Field, int64, float64
from marrow.schema import Schema
from marrow.expr import Expr, ADD, LT
from marrow.expr.plan import AnyRelation, Scan, Filter, Project


fn _schema() -> Schema:
    var s = Schema()
    s.append(Field("x", int64))
    s.append(Field("y", float64))
    return s^


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_scan_schema() raises:
    """Scan.output_schema returns the declared schema."""
    var src = Scan(name="t", schema_=_schema())
    var schema = src.output_schema()
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
    var pred = Expr.greater(Expr.input(0), Expr.literal[int64](0))
    var filt = Filter(input=src, predicate=pred)
    var schema = filt.output_schema()
    assert_equal(len(schema), 2)
    assert_equal(schema.fields[0].name, "x")


def test_filter_one_input() raises:
    """Filter has exactly one child plan."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = Expr.less(Expr.input(0), Expr.input(1))
    var filt = Filter(input=src, predicate=pred)
    assert_equal(len(filt.inputs()), 1)


def test_filter_one_expr() raises:
    """Filter exposes its predicate as an expression."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = Expr.less(Expr.input(0), Expr.input(1))
    var filt = Filter(input=src, predicate=pred)
    var exprs = filt.exprs()
    assert_equal(len(exprs), 1)
    assert_equal(exprs[0].kind(), LT)


def test_filter_write_to() raises:
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var pred = Expr.less(Expr.input(0), Expr.input(1))
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
    var exprs = List[Expr]()
    exprs.append(Expr.add(Expr.input(0), Expr.input(1)))
    var proj = Project(
        input=src, names=names^, exprs_=exprs^, schema_=out_schema
    )
    var schema = proj.output_schema()
    assert_equal(len(schema), 1)
    assert_equal(schema.fields[0].name, "z")


def test_project_exprs() raises:
    """Project exposes its expressions."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var out_schema = Schema()
    out_schema.append(Field("z", int64))
    var names = List[String]()
    names.append("z")
    var exprs_list = List[Expr]()
    exprs_list.append(Expr.add(Expr.input(0), Expr.input(1)))
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
    var exprs = List[Expr]()
    exprs.append(Expr.add(Expr.input(0), Expr.input(1)))
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
    assert_equal(src.as_scan()[].name, "t")


def test_anyrelation_o1_copy() raises:
    """AnyRelation copies share the same underlying allocation (O(1))."""
    var src = AnyRelation(Scan(name="t", schema_=_schema()))
    var copy = src  # O(1) ref-count bump
    assert_equal(copy.output_schema().fields[0].name, "x")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
