"""Structural tests for the relational IR, written through the plan-building API.

Every plan here is built the way a caller builds one — ``parquet_scan(...)``
/``in_memory_table(...)`` followed by ``.filter()``/``.select()``/``.project()``
/``.sort()``/``.limit()`` — never by constructing ``Filter``/``Project``/``Sort``
nodes and hand-writing their output schema. That is the point: the node
constructors take a schema, so a test that supplies one asserts against its own
arithmetic rather than against the plan builder's. The concrete node types appear
only as ``downcast`` targets, which is the read-side API.
"""

from std.testing import assert_equal, assert_false, assert_true

from ...builders import array
from ...dtypes import Field, field, int64, float64, Float64Type, Int64Type
from ...execution import ExecContext
from ...parquet import LeafSet
from ...schema import Schema, schema
from ...tabular import RecordBatch, record_batch
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
    assert_equal(
        _scan().downcast[ParquetScan[LeafSet.all()]]()[].path.resolve(), "t"
    )


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
    """`select` renders by name. It used to resolve to a position first and print
    `input(1)`; positional references were an interpreter artefact and the lane
    now resolves names at execution, like the fused one always has."""
    assert_equal(String(_scan().select("y")), "Project([y=y])")


def test_select_list_preserves_field_nullable_and_metadata() raises:
    """`select` carries each surviving `Field` over whole, both spellings.

    A pass-through column *is* its input field, so `nullable` and metadata must
    survive. `project` cannot promise this — it probes the expression's dtype
    and builds a fresh `Field` around it, which widens a non-nullable column to
    nullable and drops its metadata — and that is exactly why the Python
    bindings needed the `List[String]` overload rather than routing `select`
    through `project`."""
    var meta = Dict[String, String]()
    meta["unit"] = "usd"
    var src = parquet_scan(
        "t",
        Schema(
            fields=[
                Field("x", int64, nullable=False, metadata=meta.copy()),
                Field("y", float64),
            ]
        ),
    )

    # The list overload — what a runtime frontend calls.
    var wanted: List[String] = ["x"]
    var listed = src.select(wanted).schema()
    assert_equal(len(listed), 1)
    assert_equal(listed.fields[0].name, "x")
    assert_equal(listed.fields[0].dtype, int64)
    assert_false(listed.fields[0].nullable)
    assert_equal(listed.fields[0].metadata["unit"], "usd")

    # The variadic overload delegates to it, so it must agree field-for-field.
    assert_true(src.select("x").schema().fields[0] == listed.fields[0])

    # `project` of the same column is the lossy alternative this exists to
    # avoid: right dtype, wrong nullability, no metadata.
    var projected = src.project(names=["x"], values=[col("x")]).schema()
    assert_equal(projected.fields[0].dtype, int64)
    assert_true(projected.fields[0].nullable)
    assert_equal(len(projected.fields[0].metadata), 0)


# ---------------------------------------------------------------------------
# with_columns / drop / rename
#
# These three lower to `Project` like `select` and `project` do, so most of what
# there is to check is schema-level and belongs here. The handful of `.execute()`
# cases below are the exception: "replaces in place" and "every expression sees
# the input" are claims about *values*, and a schema assertion cannot tell a
# correct plan from one that merely names its columns correctly.
# ---------------------------------------------------------------------------


def _abc_batch() raises -> RecordBatch:
    """Three int64 columns, distinct values, so a mix-up is visible."""
    var a = array([1, 2], int64)
    var b = array([10, 20], int64)
    var c = array([100, 200], int64)
    return record_batch([a.copy(), b.copy(), c.copy()], names=["a", "b", "c"])


def _abc() raises -> DynRelation:
    return in_memory_table(_abc_batch())


def _names(s: Schema) -> String:
    """The schema's column names, comma-joined — one assertion covers both the
    set of columns and their order, and a failure prints both sides readably."""
    var out = String()
    for i in range(len(s.fields)):
        if i > 0:
            out += ","
        out += s.fields[i].name
    return out^


def test_with_columns_appends_new_column() raises:
    """A name absent from the input schema lands after the existing columns."""
    var s = (
        _abc().with_columns(names=["d"], values=[col("a") + col("b")]).schema()
    )
    assert_equal(_names(s), "a,b,c,d")
    assert_equal(s.fields[3].dtype, int64)


def test_with_columns_replaces_in_place() raises:
    """A name already in the schema overwrites that column *at its position* —
    polars `with_columns` and ibis `mutate` both keep ['a','b','c'] here, rather
    than moving the rebuilt column to the end."""
    var plan = _abc().with_columns(names=["b"], values=[col("b") + col("b")])
    assert_equal(_names(plan.schema()), "a,b,c")

    var result = plan.execute()
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([1, 2], int64))
    assert_true(result.columns[1].as_int64().copy() == array([20, 40], int64))
    assert_true(result.columns[2].as_int64().copy() == array([100, 200], int64))


def test_with_columns_probes_replacement_dtype() raises:
    """The replacement's dtype is the expression's, not the column it displaces:
    overwriting int64 `b` with an int64+float64 sum makes the field float64."""
    var plan = _abc().with_columns(
        names=["b"], values=[col("b") + lit[Float64Type](0.5)]
    )
    assert_equal(plan.schema().fields[1].dtype, float64)


def test_with_columns_keeps_untouched_columns_nullability() raises:
    """A pass-through column carries its whole input `Field` across, not just a
    re-probed dtype — `nullable=False` survives."""
    var src = parquet_scan(
        "t", schema([field("x", int64, nullable=False), field("y", float64)])
    )
    var s = src.with_columns(names=["z"], values=[col("x") + col("x")]).schema()
    assert_equal(_names(s), "x,y,z")
    assert_false(s.fields[0].nullable)


def test_with_columns_values_see_the_input_batch() raises:
    """All values in one call are evaluated against the same input morsel, so a
    replacement is invisible to its siblings: `c` reads the original `b` (10, 20)
    even though `b` is being overwritten in the same call. Chaining two calls is
    how you get sequential semantics — this matches polars and ibis."""
    var result = (
        _abc()
        .with_columns(
            names=["b", "c"], values=[col("b") + col("b"), col("b") + col("b")]
        )
        .execute()
    )
    assert_true(result.columns[1].as_int64().copy() == array([20, 40], int64))
    assert_true(result.columns[2].as_int64().copy() == array([20, 40], int64))


def test_with_columns_chained_is_sequential() raises:
    """Two calls *do* compose: the second sees the first's output."""
    var result = (
        _abc()
        .with_columns(names=["b"], values=[col("b") + col("b")])
        .with_columns(names=["d"], values=[col("b") + col("b")])
        .execute()
    )
    assert_equal(_names(result.schema), "a,b,c,d")
    assert_true(result.columns[3].as_int64().copy() == array([40, 80], int64))


def test_with_columns_rejects_duplicate_output_name() raises:
    var raised = False
    try:
        _ = _abc().with_columns(names=["d", "d"], values=[col("a"), col("b")])
    except e:
        raised = True
        assert_true(String(e).find("duplicate output column 'd'") != -1)
    assert_true(raised)


def test_with_columns_rejects_length_mismatch() raises:
    var raised = False
    try:
        _ = _abc().with_columns(names=["d", "e"], values=[col("a")])
    except e:
        raised = True
        assert_true(String(e).find("len(names) != len(values)") != -1)
    assert_true(raised)


def test_with_columns_unknown_column_fails_at_plan_build() raises:
    """The dtype probe executes the expression, so a reference to a column that
    does not exist fails when the plan is built, not when it runs."""
    var raised = False
    try:
        _ = _abc().with_columns(names=["d"], values=[col("nope") + col("a")])
    except:
        raised = True
    assert_true(raised)


def test_drop_removes_named_columns() raises:
    """`drop` keeps the survivors in their original order."""
    var plan = _abc().drop(["b"])
    assert_equal(_names(plan.schema()), "a,c")

    var result = plan.execute()
    assert_equal(result.num_rows(), 2)
    assert_true(result.columns[0].as_int64().copy() == array([1, 2], int64))
    assert_true(result.columns[1].as_int64().copy() == array([100, 200], int64))


def test_drop_repeated_name_is_not_an_error() raises:
    """Naming the same column twice drops it once — the column is gone either
    way, so there is nothing to warn about."""
    assert_equal(_names(_abc().drop(["b", "b"]).schema()), "a,c")


def test_drop_unknown_column_raises() raises:
    var raised = False
    try:
        _ = _abc().drop(["nope"])
    except e:
        raised = True
        assert_true(String(e).find("drop: column 'nope' not found") != -1)
    assert_true(raised)


def test_rename_renames_in_place() raises:
    """Only the named columns change name; order, dtypes and data are untouched.
    """
    var plan = _abc().rename(names=["b"], new_names=["B"])
    assert_equal(_names(plan.schema()), "a,B,c")
    assert_equal(plan.schema().fields[1].dtype, int64)

    var result = plan.execute()
    assert_equal(_names(result.schema), "a,B,c")
    assert_true(result.columns[1].as_int64().copy() == array([10, 20], int64))


def test_rename_swaps_two_columns() raises:
    """Renames are simultaneous, so swapping two names is legal and does not
    collide with itself."""
    var s = _abc().rename(names=["a", "b"], new_names=["b", "a"]).schema()
    assert_equal(_names(s), "b,a,c")


def test_rename_unknown_column_raises() raises:
    var raised = False
    try:
        _ = _abc().rename(names=["nope"], new_names=["x"])
    except e:
        raised = True
        assert_true(String(e).find("rename: column 'nope' not found") != -1)
    assert_true(raised)


def test_rename_same_column_twice_raises() raises:
    var raised = False
    try:
        _ = _abc().rename(names=["a", "a"], new_names=["x", "y"])
    except e:
        raised = True
        assert_true(String(e).find("renamed twice") != -1)
    assert_true(raised)


def test_rename_collision_with_untouched_column_raises() raises:
    """Renaming `a` onto the existing `c` would leave two columns called `c`,
    which no later `select`/`col` could tell apart."""
    var raised = False
    try:
        _ = _abc().rename(names=["a"], new_names=["c"])
    except e:
        raised = True
        assert_true(String(e).find("duplicate output column 'c'") != -1)
    assert_true(raised)


def test_rename_length_mismatch_raises() raises:
    var raised = False
    try:
        _ = _abc().rename(names=["a", "b"], new_names=["x"])
    except e:
        raised = True
        assert_true(String(e).find("len(names) != len(new_names)") != -1)
    assert_true(raised)


def test_with_columns_drop_rename_compose() raises:
    """The three verbs chain, and each lowers to a Project so the chain is just
    stacked projections."""
    var result = (
        _abc()
        .with_columns(names=["total"], values=[col("a") + col("c")])
        .drop(["b"])
        .rename(names=["total"], new_names=["sum_ac"])
        .execute()
    )
    assert_equal(_names(result.schema), "a,c,sum_ac")
    assert_true(result.columns[2].as_int64().copy() == array([101, 202], int64))


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


def test_execute_defaults_to_auto_not_serial() raises:
    """`execute()` with no context must not force serial execution.

    It used to default to the bare `ExecContext()` — `num_threads=1` — so every
    plan ran single-threaded and the kernels' parallel strategies were
    unreachable from the relational API. The default is now `auto`, and the
    answer is the same either way."""
    var k = array([1, 2, 1, 2, 1], int64)
    var v = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([k^, v^], names=["k", "v"])
    var plan = in_memory_table(batch).aggregate(
        keys=[col("k")], aggs=[col("v").sum()]
    )
    var auto = plan.execute()
    var serial = plan.execute(ExecContext.serial())
    var forced = plan.execute(ExecContext.parallel(8))
    assert_equal(auto.num_rows(), 2)
    assert_true(auto.column(1).as_int64() == serial.column(1).as_int64())
    assert_true(auto.column(1).as_int64() == forced.column(1).as_int64())
