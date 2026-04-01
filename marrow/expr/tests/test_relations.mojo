from std.testing import assert_equal, TestSuite

from marrow.arrays import AnyArray
from marrow.builders import array
from marrow.dtypes import field, int64, float64
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr import (
    AnyValue,
    col,
    lit,
    ADD,
    LT,
    in_memory_table,
    SCAN_NODE,
    FILTER_NODE,
    PROJECT_NODE,
    IN_MEMORY_TABLE_NODE,
    PARQUET_SCAN_NODE,
)
from marrow.expr.relations import (
    AnyRelation,
    Scan,
    Filter,
    Project,
    InMemoryTable,
    ParquetScan,
)
from marrow.expr.values import Column


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_scan_schema() raises:
    """Scan.schema returns the declared schema."""
    var src = Scan(
        name="t", schema_=schema([field("x", int64), field("y", float64)])
    )
    var s = src.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")


def test_scan_no_inputs() raises:
    """Scan is a leaf — no child plans."""
    var src = Scan(
        name="t", schema_=schema([field("x", int64), field("y", float64)])
    )
    assert_equal(len(src.inputs()), 0)


def test_scan_no_exprs() raises:
    """Scan has no expressions."""
    var src = Scan(
        name="t", schema_=schema([field("x", int64), field("y", float64)])
    )
    assert_equal(len(src.exprs()), 0)


def test_scan_write_to() raises:
    var src = AnyRelation(
        Scan(
            name="orders",
            schema_=schema([field("x", int64), field("y", float64)]),
        )
    )
    assert_equal(String(src), "Scan(orders)")


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_schema_passthrough() raises:
    """Filter output schema equals the input schema."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var pred = col(0) > lit[Int64Type](0)
    var filt = Filter(input=src, predicate=pred)
    var s = filt.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")


def test_filter_one_input() raises:
    """Filter has exactly one child plan."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var filt = Filter(input=src, predicate=col(0) < col(1))
    assert_equal(len(filt.inputs()), 1)


def test_filter_one_expr() raises:
    """Filter exposes its predicate as an expression."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var filt = Filter(input=src, predicate=col(0) < col(1))
    var exprs = filt.exprs()
    assert_equal(len(exprs), 1)
    assert_equal(exprs[0].kind(), LT)


def test_filter_write_to() raises:
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var filt = AnyRelation(Filter(input=src, predicate=col(0) < col(1)))
    assert_equal(String(filt), "Filter(predicate=less(input(0), input(1)))")


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def test_project_schema() raises:
    """Project output schema contains only the projected columns."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var proj = Project(
        input=src,
        names=["z"],
        exprs_=[col(0) + col(1)],
        schema_=schema([field("z", int64)]),
    )
    var s = proj.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "z")


def test_project_exprs() raises:
    """Project exposes its expressions."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var proj = Project(
        input=src,
        names=["z"],
        exprs_=[col(0) + col(1)],
        schema_=schema([field("z", int64)]),
    )
    var exprs = proj.exprs()
    assert_equal(len(exprs), 1)
    assert_equal(exprs[0].kind(), ADD)


def test_project_write_to() raises:
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var proj = AnyRelation(
        Project(
            input=src,
            names=["z"],
            exprs_=[col(0) + col(1)],
            schema_=schema([field("z", int64)]),
        )
    )
    assert_equal(String(proj), "Project([z=add(input(0), input(1))])")


# ---------------------------------------------------------------------------
# AnyRelation type erasure / downcast
# ---------------------------------------------------------------------------


def test_anyrelation_downcast_scan() raises:
    """AnyRelation wrapping a Scan can be downcast back."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    assert_equal(src.downcast[Scan]()[].name, "t")


def test_anyrelation_o1_copy() raises:
    """AnyRelation copies share the same underlying allocation (O(1))."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var copy = src  # O(1) ref-count bump
    assert_equal(copy.schema().fields[0].name, "x")


# ---------------------------------------------------------------------------
# kind() dispatch
# ---------------------------------------------------------------------------


def test_scan_kind() raises:
    """Scan reports SCAN_NODE kind."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    assert_equal(src.kind(), SCAN_NODE)


def test_filter_kind() raises:
    """Filter reports FILTER_NODE kind."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var filt = AnyRelation(Filter(input=src, predicate=col(0) > lit[Int64Type](0)))
    assert_equal(filt.kind(), FILTER_NODE)


def test_project_kind() raises:
    """Project reports PROJECT_NODE kind."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var proj = AnyRelation(
        Project(
            input=src,
            names=["z"],
            exprs_=[col(0)],
            schema_=schema([field("z", int64)]),
        )
    )
    assert_equal(proj.kind(), PROJECT_NODE)


# ---------------------------------------------------------------------------
# InMemoryTable
# ---------------------------------------------------------------------------


def test_in_memory_table_kind() raises:
    """InMemoryTable reports IN_MEMORY_TABLE_NODE kind."""
    var a = array[Int64Type]([1, 2, 3])
    var t = in_memory_table(record_batch([a^], names=["a"]))
    assert_equal(t.kind(), IN_MEMORY_TABLE_NODE)


def test_in_memory_table_schema() raises:
    """InMemoryTable schema matches the batch schema."""
    var a = array[Int64Type]([1, 2, 3])
    var t = in_memory_table(record_batch([a^], names=["a"]))
    var s = t.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "a")


def test_in_memory_table_leaf() raises:
    """InMemoryTable is a leaf node with no inputs or expressions."""
    var a = array[Int64Type]([1, 2, 3])
    var t = in_memory_table(record_batch([a^], names=["a"]))
    assert_equal(len(t.inputs()), 0)
    assert_equal(len(t.exprs()), 0)


def test_in_memory_table_downcast() raises:
    """InMemoryTable can be downcast to access the batch."""
    var a = array[Int64Type]([1, 2, 3])
    var t = in_memory_table(record_batch([a^], names=["a"]))
    var imt = t.downcast[InMemoryTable]()
    assert_equal(imt[].batch.num_rows(), 3)


# ---------------------------------------------------------------------------
# Scan + filter + select plan composition
# ---------------------------------------------------------------------------


def test_scan_filter_kind() raises:
    """Scan.filter() produces a FILTER_NODE wrapping the scan."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.filter(col("x") > lit[Int64Type](0))
    assert_equal(plan.kind(), FILTER_NODE)


def test_scan_filter_schema_passthrough() raises:
    """Scan.filter() preserves the scan's output schema."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.filter(col("x") > lit[Int64Type](0))
    var s = plan.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")


def test_scan_filter_resolves_column_name() raises:
    """``col('x')`` inside filter is resolved to a positional col(0)."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.filter(col("x") > lit[Int64Type](0))
    var filt = plan.downcast[Filter]()
    var pred_inputs = filt[].predicate.inputs()
    assert_equal(pred_inputs[0].downcast[Column]()[].index, 0)


def test_scan_select_kind() raises:
    """Scan.select() produces a PROJECT_NODE."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.select("x")
    assert_equal(plan.kind(), PROJECT_NODE)


def test_scan_select_schema() raises:
    """Scan.select('x') yields a single-field schema."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.select("x")
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "x")


def test_scan_filter_select_kinds() raises:
    """Scan.filter().select() chains FILTER_NODE under PROJECT_NODE."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.filter(col("x") > lit[Int64Type](0)).select("x")
    assert_equal(plan.kind(), PROJECT_NODE)
    var proj = plan.downcast[Project]()
    assert_equal(proj[].input.kind(), FILTER_NODE)


def test_scan_filter_select_schema() raises:
    """Scan.filter().select('y') final schema has only 'y'."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.filter(col("x") > lit[Int64Type](0)).select("y")
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "y")


def test_scan_select_filter_kinds() raises:
    """Scan.select().filter() chains PROJECT_NODE under FILTER_NODE."""
    var src = AnyRelation(
        Scan(name="t", schema_=schema([field("x", int64), field("y", float64)]))
    )
    var plan = src.select("x").filter(col("x") > lit[Int64Type](0))
    assert_equal(plan.kind(), FILTER_NODE)
    var filt = plan.downcast[Filter]()
    assert_equal(filt[].input.kind(), PROJECT_NODE)


# ---------------------------------------------------------------------------
# ParquetScan — structural tests (no I/O)
# ---------------------------------------------------------------------------


def test_parquet_scan_kind() raises:
    """ParquetScan reports PARQUET_SCAN_NODE kind."""
    var node = AnyRelation(
        ParquetScan(
            path="/tmp/x.parquet",
            schema_=schema([field("id", int64), field("val", float64)]),
        )
    )
    assert_equal(node.kind(), PARQUET_SCAN_NODE)


def test_parquet_scan_schema() raises:
    """ParquetScan.schema returns the declared schema."""
    var node = ParquetScan(
        path="/tmp/x.parquet",
        schema_=schema([field("id", int64), field("val", float64)]),
    )
    var s = node.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "id")
    assert_equal(s.fields[1].name, "val")


def test_parquet_scan_leaf() raises:
    """ParquetScan is a leaf node with no inputs or expressions."""
    var node = ParquetScan(
        path="/tmp/x.parquet",
        schema_=schema([field("id", int64), field("val", float64)]),
    )
    assert_equal(len(node.inputs()), 0)
    assert_equal(len(node.exprs()), 0)


def test_parquet_scan_write_to() raises:
    """ParquetScan formats as ParquetScan(path)."""
    var node = AnyRelation(
        ParquetScan(
            path="/tmp/x.parquet",
            schema_=schema([field("id", int64), field("val", float64)]),
        )
    )
    assert_equal(String(node), "ParquetScan(/tmp/x.parquet)")


def test_parquet_scan_downcast() raises:
    """AnyRelation wrapping a ParquetScan can be downcast to access path."""
    var node = AnyRelation(
        ParquetScan(
            path="/tmp/x.parquet",
            schema_=schema([field("id", int64), field("val", float64)]),
        )
    )
    assert_equal(node.downcast[ParquetScan]()[].path, "/tmp/x.parquet")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
