"""Structural tests for the relational IR, written through the plan-building API.

Every plan here is built the way a caller builds one — ``parquet_scan(...)``
/``in_memory_table(...)`` followed by ``.filter()``/``.select()``/``.project()``
/``.sort()``/``.limit()`` — never by constructing ``Filter``/``Project``/``Sort``
nodes and hand-writing their output schema. That is the point: the node
constructors take a schema, so a test that supplies one asserts against its own
arithmetic rather than against the plan builder's. The concrete node types appear
only as ``downcast`` targets, which is the read-side API.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import field, int64, float64, Int64Type
from ...parquet import LeafSet
from ...schema import schema
from ...tabular import record_batch
from ...expr import col, lit, in_memory_table
from ...expr.relations import (
    DynRelation,
    Filter,
    Project,
    Sort,
    InMemoryTable,
    ParquetScan,
    parquet_scan,
)


def _scan() raises -> DynRelation:
    """The two-column source every structural test builds on."""
    return parquet_scan("t", schema([field("x", int64), field("y", float64)]))


# ---------------------------------------------------------------------------
# Source nodes
# ---------------------------------------------------------------------------


def test_parquet_scan_schema() raises:
    """ParquetScan.schema returns the declared schema."""
    var s = _scan().schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")


def test_parquet_scan_write_to() raises:
    """ParquetScan formats as ParquetScan(path)."""
    assert_equal(String(_scan()), "ParquetScan(t)")


def test_parquet_scan_downcast() raises:
    """DynRelation wrapping a ParquetScan can be downcast to access path."""
    assert_equal(_scan().downcast[ParquetScan[LeafSet.all()]]()[].path, "t")


def test_in_memory_table_schema() raises:
    """InMemoryTable schema matches the batch schema."""
    var a = array([1, 2, 3], int64)
    var s = in_memory_table(record_batch([a^], names=["a"])).schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "a")


def test_in_memory_table_downcast() raises:
    """InMemoryTable can be downcast to access the batch."""
    var a = array([1, 2, 3], int64)
    var t = in_memory_table(record_batch([a^], names=["a"]))
    assert_equal(t.downcast[InMemoryTable]()[].batch.num_rows(), 3)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_schema_passthrough() raises:
    """`.filter()` preserves the input schema exactly."""
    var s = _scan().filter(col("x") > lit[Int64Type](0)).schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")


def test_filter_predicate() raises:
    """Filter exposes its predicate expression as a field."""
    var plan = _scan().filter(col("x") < col("y"))
    ref pred = plan.downcast[Filter]()[].predicate
    assert_true(String(pred).find("less") != -1)


def test_filter_write_to() raises:
    """A filter renders its predicate, naming the columns it references."""
    var plan = _scan().filter(col("x") < col("y"))
    assert_equal(String(plan), "Filter(predicate=less(x, y))")


# ---------------------------------------------------------------------------
# Select / project
# ---------------------------------------------------------------------------


def test_select_schema() raises:
    """`.select('x')` yields a single-field schema."""
    var s = _scan().select("x").schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "x")


def test_filter_select_schema() raises:
    """`.filter().select('y')` final schema has only 'y'."""
    var s = _scan().filter(col("x") > lit[Int64Type](0)).select("y").schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "y")


def test_project_infers_computed_dtypes() raises:
    """`.project()` infers each computed column's dtype rather than being told
    one: it probes the expression against a 0-row batch of the input schema."""
    var plan = _scan().project(names=["x2"], values=[col("x") + col("x")])
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].name, "x2")
    assert_equal(s.fields[0].dtype, int64)


def test_project_mixed_dtype_arithmetic_promotes() raises:
    """`int64 + float64` projects as float64 — the wider operand's type.

    This used to assert the *divergence*: the fused algebra promoted
    (`promote[L, R]`) while the interpreted lane the plan builder uses demanded
    identical dtypes and raised `add: dtype mismatch: int64 vs float64`, which
    `.project()` surfaced at plan-build time because it probes the expression's
    dtype against a 0-row batch. Both lanes promote now (Q0.4)."""
    var plan = _scan().project(names=["z"], values=[col("x") + col("y")])
    var s = plan.schema()
    assert_equal(len(s), 1)
    assert_equal(s.fields[0].dtype, float64)


def test_project_exprs() raises:
    """Project exposes its expressions as a field."""
    var plan = _scan().project(names=["z"], values=[col("x") + col("x")])
    ref values = plan.downcast[Project]()[].values
    assert_equal(len(values), 1)
    assert_true(String(values[0]).find("add") != -1)


def test_project_write_to() raises:
    """A projection renders each output column as `name=expression`.

    Column references print by name: `project` binds them when the boxed value
    executes, exactly as `filter` does, rather than rewriting them to positions
    at plan-build time. `select` is the one that renders positionally, because
    it builds positional references directly."""
    var plan = _scan().project(names=["z"], values=[col("y") + col("y")])
    assert_equal(String(plan), "Project([z=add(y, y)])")


def test_select_write_to() raises:
    """`select` resolves names to positions itself, so it renders positionally —
    `y` is index 1, which an unresolved reference could not report."""
    assert_equal(String(_scan().select("y")), "Project([y=input(1)])")


# ---------------------------------------------------------------------------
# Sort / limit
# ---------------------------------------------------------------------------


def test_sort_schema_passthrough() raises:
    """Sort leaves the schema unchanged."""
    var plan = _scan().sort(keys=[col("x")], ascending=[True])
    var s = plan.schema()
    assert_equal(len(s), 2)
    assert_equal(s.fields[0].name, "x")
    assert_equal(s.fields[1].name, "y")
    assert_equal(plan.kind(), 2)  # RELATION_SORT


def test_sort_write_to() raises:
    var plan = _scan().sort(keys=[col("x")], ascending=[True])
    assert_true(String(plan).find("Sort(keys=[") != -1)


def test_limit_schema_passthrough() raises:
    assert_equal(len(_scan().limit(5).schema()), 2)


def test_limit_write_to() raises:
    assert_equal(
        String(_scan().limit(5, offset=2)), "Limit(length=5, offset=2)"
    )


def test_sort_limit_folds_to_topk() raises:
    """.sort().limit(k) with offset=0 folds into a single Sort(limit=k) node."""
    var plan = _scan().sort(keys=[col("x")], ascending=[True]).limit(3)
    assert_equal(plan.kind(), 2)  # still a Sort, not a Limit
    assert_true(plan.downcast[Sort]()[].limit.__bool__())


def test_sort_limit_offset_does_not_fold() raises:
    """A non-zero offset keeps a distinct Limit node above the Sort."""
    var plan = (
        _scan().sort(keys=[col("x")], ascending=[True]).limit(3, offset=1)
    )
    assert_equal(plan.kind(), 0)  # RELATION_GENERIC (Limit)


# ---------------------------------------------------------------------------
# DynRelation type erasure
# ---------------------------------------------------------------------------


def test_anyrelation_o1_copy() raises:
    """DynRelation copies share the same underlying allocation (O(1))."""
    var src = _scan()
    var copy = src  # O(1) ref-count bump
    assert_equal(copy.schema().fields[0].name, "x")
