from std.testing import assert_equal, assert_true
from ...expr.values import AnyValue

from ...arrays import AnyArray
from ...builders import array
from ...dtypes import field, int64, float64, Int64Type
from ...schema import schema
from ...tabular import record_batch
from ...expr import (
    col,
    lit,
    ADD,
    LT,
    in_memory_table,
)
from ...expr.relations import (
    AnyRelation,
    Filter,
    Project,
    Sort,
    Limit,
    InMemoryTable,
    ParquetScan,
)
from ...expr.dynamic import col as dyn_col


# ---------------------------------------------------------------------------
# Source nodes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_schema_passthrough() raises:
    """Filter output schema equals the input schema."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var pred = col(0) > lit[Int64Type](0)
    var filt = Filter(input=src, predicate=pred)
    var s = filt.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")


def test_filter_predicate() raises:
    """Filter exposes its predicate expression as a field."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var filt = Filter(input=src, predicate=col(0) < col(1))
    assert_true(String(filt.predicate).find("less") != -1)


def test_filter_write_to() raises:
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var filt = AnyRelation(Filter(input=src, predicate=col(0) < col(1)))
    assert_equal(String(filt), "Filter(predicate=less(input(0), input(1)))")


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def test_project_schema() raises:
    """Project output schema contains only the projected columns."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var proj = Project(
        input=src,
        names=["z"],
        values=[col(0) + col(1)],
        schema=schema([field("z", int64)]),
    )
    var s = proj.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "z")


def test_project_exprs() raises:
    """Project exposes its expressions as a field."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var proj = Project(
        input=src,
        names=["z"],
        values=[AnyValue(col(0) + col(1))],
        schema=schema([field("z", int64)]),
    )
    assert_equal(len(proj.values), 1)
    assert_true(String(proj.values[0]).find("add") != -1)


def test_project_write_to() raises:
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var proj = AnyRelation(
        Project(
            input=src,
            names=["z"],
            values=[col(0) + col(1)],
            schema=schema([field("z", int64)]),
        )
    )
    assert_equal(String(proj), "Project([z=add(input(0), input(1))])")


# ---------------------------------------------------------------------------
# AnyRelation type erasure / downcast
# ---------------------------------------------------------------------------


def test_anyrelation_o1_copy() raises:
    """AnyRelation copies share the same underlying allocation (O(1))."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var copy = src  # O(1) ref-count bump
    assert_equal(copy.schema().fields[0].name, "x")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# InMemoryTable
# ---------------------------------------------------------------------------


def test_in_memory_table_schema() raises:
    """InMemoryTable schema matches the batch schema."""
    var a = array([1, 2, 3], int64)
    var t = in_memory_table(record_batch([a^], names=["a"]))
    var s = t.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "a")


def test_in_memory_table_downcast() raises:
    """InMemoryTable can be downcast to access the batch."""
    var a = array([1, 2, 3], int64)
    var t = in_memory_table(record_batch([a^], names=["a"]))
    var imt = t.downcast[InMemoryTable]()
    assert_equal(imt[].batch.num_rows(), 3)


# ---------------------------------------------------------------------------
# filter + select plan composition
# ---------------------------------------------------------------------------


def test_scan_filter_schema_passthrough() raises:
    """`source.filter()` preserves the scan's output schema."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var plan = src.filter(col("x") > lit[Int64Type](0))
    var s = plan.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")


def test_scan_select_schema() raises:
    """`source.select('x')` yields a single-field schema."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var plan = src.select("x")
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "x")


def test_scan_filter_select_schema() raises:
    """`source.filter().select('y')` final schema has only 'y'."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var plan = src.filter(col("x") > lit[Int64Type](0)).select("y")
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "y")


# ---------------------------------------------------------------------------
# ParquetScan — structural tests (no I/O)
# ---------------------------------------------------------------------------


def test_parquet_scan_schema() raises:
    """ParquetScan.schema returns the declared schema."""
    var node = ParquetScan(
        path="/tmp/x.parquet",
        schema=schema([field("id", int64), field("val", float64)]),
    )
    var s = node.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "id")
    assert_equal(s.fields[1].name, "val")


def test_parquet_scan_write_to() raises:
    """ParquetScan formats as ParquetScan(path)."""
    var node = AnyRelation(
        ParquetScan(
            path="/tmp/x.parquet",
            schema=schema([field("id", int64), field("val", float64)]),
        )
    )
    assert_equal(String(node), "ParquetScan(/tmp/x.parquet)")


def test_parquet_scan_downcast() raises:
    """AnyRelation wrapping a ParquetScan can be downcast to access path."""
    var node = AnyRelation(
        ParquetScan(
            path="/tmp/x.parquet",
            schema=schema([field("id", int64), field("val", float64)]),
        )
    )
    assert_equal(node.downcast[ParquetScan]()[].path, "/tmp/x.parquet")


# ---------------------------------------------------------------------------
# Sort / Limit — structural
# ---------------------------------------------------------------------------


def test_sort_schema_passthrough() raises:
    """Sort leaves the schema unchanged."""
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var plan = src.sort(keys=[dyn_col("x")], ascending=[True])
    var s = plan.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")
    assert_equal(plan.kind(), 2)  # RELATION_SORT


def test_sort_write_to() raises:
    var src = AnyRelation(
        ParquetScan(path="t", schema=schema([field("x", int64)]))
    )
    var plan = src.sort(keys=[dyn_col("x")], ascending=[True])
    assert_true(String(plan).find("Sort(keys=[") != -1)


def test_limit_write_to() raises:
    var src = AnyRelation(
        ParquetScan(path="t", schema=schema([field("x", int64)]))
    )
    var plan = src.limit(5, offset=2)
    assert_equal(String(plan), "Limit(length=5, offset=2)")


def test_limit_schema_passthrough() raises:
    var src = AnyRelation(
        ParquetScan(
            path="t", schema=schema([field("x", int64), field("y", float64)])
        )
    )
    var plan = src.limit(5)
    assert_equal(len(plan.schema()), 2)


def test_sort_limit_folds_to_topk() raises:
    """.sort().limit(k) with offset=0 folds into a single Sort(limit=k) node."""
    var src = AnyRelation(
        ParquetScan(path="t", schema=schema([field("x", int64)]))
    )
    var plan = src.sort(keys=[dyn_col("x")], ascending=[True]).limit(3)
    assert_equal(plan.kind(), 2)  # still a Sort, not a Limit
    assert_true(plan.downcast[Sort]()[].limit.__bool__())


def test_sort_limit_offset_does_not_fold() raises:
    """A non-zero offset keeps a distinct Limit node above the Sort."""
    var src = AnyRelation(
        ParquetScan(path="t", schema=schema([field("x", int64)]))
    )
    var plan = src.sort(keys=[dyn_col("x")], ascending=[True]).limit(
        3, offset=1
    )
    assert_equal(plan.kind(), 0)  # RELATION_GENERIC (Limit)


def test_computed_project_schema() raises:
    """.project() infers each computed column's output dtype into the schema."""
    var src = AnyRelation(
        ParquetScan(path="t", schema=schema([field("x", int64)]))
    )
    var plan = src.project(names=["x2"], values=[dyn_col("x") + dyn_col("x")])
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "x2")
    assert_equal(s.fields[0].dtype, int64)
