"""Plan nodes and the operators they become.

`logical.mojo` and `physical.mojo` are covered together because neither is
observable alone: a `Relation` is a description, so the only way to ask whether
it described the right thing is to run the plan it builds. Each test
therefore states a claim about the plan and checks it against the rows.

The recurring failure these guard is a **schema that disagrees with the data** —
a `Project` whose declared fields do not match the columns it emits, or an empty
result whose schema names fields it has no columns for. Both run fine and
corrupt whatever reads them by index.
"""

from std.testing import assert_equal, assert_true

from ...arrays import StructArray, DynArray
from ...builders import array
from ...dtypes import DynType, Int64Type, float64, int64
from ...execution import ExecContext
from ...kernels.join import JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from ...dtypes import Field, field
from ...schema import Schema, schema
from ...parquet.writer import write_table
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import Datum
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..builders import col, lit, scan, table
from ..runtime.values import (
    column as runtime_column,
    gt as runtime_gt,
    literal as runtime_literal,
)
from ...scalars import Int64Scalar
from ..logical import (
    Aggregate,
    Join,
    ParquetScan,
    Limit,
    Sort,
    DynRelation,
    Filter,
    InMemoryTable,
    Project,
)


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
def test_a_dynamic_plan_runs_a_fused_predicate() raises:
    """The configuration the whole two-lane design exists to allow.

    The plan is erased and composed at run time; the predicate is a comptime
    type fused into one loop. That combination is what measures 1.46 MB against
    4.91 MB for the same plan with runtime expressions — and it only works
    because `DynValue` lets a `Filter` hold either lane without knowing which.
    """
    var plan = table(_batch()).filter((col("a", int64) > lit(2, int64)))
    var out = plan.execute()

    # a = [1, 2, None, 4] -> only 4 > 2
    assert_equal(out.num_rows(), 1)
    ref a = out.columns[0].as_int64()
    assert_equal(a[0].value(), 4)


def test_a_null_predicate_does_not_select() raises:
    """SQL's rule, and the reason `Filter` must not read the data bit alone.

    `None > 2` is NULL, not false — but the SIMD lane still produced a bit for
    that row. If the filter selected on data bits, the null row's payload would
    decide whether it survives.
    """
    # every non-null row passes, so only the null's treatment shows
    var plan = table(_batch()).filter(col("a", int64) > lit(-100, int64))
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 1, 2, 4 — the null is not selected


def test_filter_preserves_its_input_schema() raises:
    """A filter changes which rows survive, never which columns exist."""
    var b = _batch()
    var plan = table(b.copy()).filter((col("a", int64) > lit(2, int64)))
    assert_true(plan.schema() == b.schema)
    assert_equal(plan.execute().num_columns(), b.num_columns())


def test_an_empty_result_is_a_well_formed_batch() raises:
    """Zero rows still means one zero-length column per field.

    A schema naming fields beside an empty column list leaves `num_columns()`
    at 0, so anything walking columns by schema index runs off the end — and
    exporting it over the C Data interface returns NULL without setting an
    exception.
    """
    var b = _batch()
    var plan = table(b.copy()).filter((col("a", int64) > lit(999, int64)))
    var out = plan.execute()
    assert_equal(out.num_rows(), 0)
    assert_equal(out.num_columns(), len(b.schema.fields))


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def test_project_carries_a_bare_column_field_whole() raises:
    """A projected pass-through keeps its source `Field`, not just its dtype.

    Rebuilding the field from `dtype()` alone loses `nullable`, so projecting
    a column would produce a *different* schema for it than selecting the same
    column does. the previous expression package records that divergence with
    `nullable` False
    becoming True.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    # `a` is non-nullable here; the projection must not widen it.
    var src = b.schema.fields[0].nullable

    var p = table(b.copy()).project(["out"], [col("a", int64)])
    assert_equal(p.schema().fields[0].nullable, src)
    assert_true(p.schema().fields[0].dtype == DynType(int64))


def test_project_names_a_computed_column_from_its_dtype() raises:
    """A computed value has no `Field` to carry, so `dtype()` answers instead.

    This is `dtype()`'s reason to exist: the schema must be known *before*
    anything runs, and the previous expression package got it by evaluating
    against a zero-row batch.
    """
    var b = _batch()
    var p = table(b.copy()).project(
        ["sum"], [(col("a", int64) + col("b", int64))]
    )
    assert_equal(p.schema().fields[0].name, "sum")
    assert_true(p.schema().fields[0].dtype == DynType(int64))


def test_project_schema_matches_what_it_produces() raises:
    """The soundness property: the declared schema and the executed batch agree.

    A schema computed statically can disagree with the batch; one derived by
    evaluating cannot. Trading the probe for `dtype()` is what makes this
    worth asserting.
    """
    var b = _batch()
    var plan = table(b.copy()).project(
        ["sum", "orig"],
        [
            (col("a", int64) + col("b", int64)),
            col("a", int64),
        ],
    )
    var out = plan.execute()
    assert_true(out.schema == plan.schema())
    assert_equal(out.num_columns(), 2)
    assert_equal(out.num_rows(), b.num_rows())


def test_project_rejects_mismatched_names_and_values() raises:
    """Two parallel lists that must agree, checked where they are supplied."""
    var raised = False
    try:
        _ = table(_batch()).project(
            ["only_one"],
            [
                col("a", int64),
                col("b", int64),
            ],
        )
    except e:
        raised = True
        assert_true("project" in String(e))
    assert_true(raised)


# ---------------------------------------------------------------------------
# builders — `col` and `lit` select a lane by what the caller knows


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _keyed() raises -> RecordBatch:
    """Two groups, interleaved, so first-appearance ordering is observable."""
    return record_batch(
        [
            array([1, 2, 1, 2], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["g", "a"],
    )


def test_an_aggregate_with_no_keys_folds_into_one_row() raises:
    """`SELECT sum(a) FROM t` — one implicit group, and a column of one row.

    An empty key list is not a different node. It selects the *ungrouped* fold
    at plan-build time, which is the whole reason `to_state` takes `grouped`.
    """
    var plan = table(_batch()).aggregate(
        [col("a", int64).sum().alias("total")], List[DynValue]()
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_equal(out.num_columns(), 1)
    # a = [1, 2, None, 4] — the null contributes nothing, it is not a zero.
    assert_true(out.columns[0].as_int64() == array([7], int64))


def test_an_aggregate_groups_by_its_key() raises:
    """Group ids are dense and assigned in first-appearance order, so the keys
    come back in the order they were first seen, not sorted."""
    var plan = table(_keyed()).aggregate(
        [col("a", int64).sum().alias("total")], [col("g", int64)]
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[0].as_int64() == array([1, 2], int64))
    assert_true(out.columns[1].as_int64() == array([40, 60], int64))


def test_aggregate_schema_is_keys_then_aggregates() raises:
    """The ordering every consumer depends on, asserted where it is decided.

    The operator reads its key fields straight off the front of this schema,
    so a change that appended keys last would mis-type the grouper rather than
    fail loudly.
    """
    var plan = table(_keyed()).aggregate(
        [
            col("a", int64).sum().alias("total"),
            col("a", int64).min().alias("smallest"),
        ],
        [col("g", int64)],
    )
    var s = plan.schema()
    assert_equal(len(s.fields), 3)
    assert_equal(s.fields[0].name, "g")
    assert_equal(s.fields[1].name, "total")
    assert_equal(s.fields[2].name, "smallest")
    assert_true(s.fields[0].dtype == DynType(int64))
    assert_true(plan.schema() == plan.execute().schema)


def test_a_computed_key_is_named_by_position() raises:
    """A bare column keeps its name; anything computed has none.

    the previous expression package shipped a defect where one lane answered
    `d` and the other `key0`
    for the same `GROUP BY d`, giving one query two output schemas.
    """
    var plan = table(_keyed()).aggregate(
        [col("a", int64).sum().alias("total")],
        [(col("g", int64) + col("a", int64))],
    )
    assert_equal(plan.schema().fields[0].name, "key0")


def test_an_aggregate_over_no_rows_answers_one_null() raises:
    """`sum` of nothing is NULL, not 0 — and the input here yields *no morsel*.

    This is the case that made `AggState.finish` grow before reading: only
    `update` ever extended the builders, so a fold that saw zero batches read
    unallocated slots. It aborts under `ASSERT=all` and is a silent bad read in
    release.
    """
    var plan = (
        table(_batch())
        .filter((col("a", int64) > lit(999, int64)))
        .aggregate([col("a", int64).sum().alias("total")], List[DynValue]())
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].is_null(0))


def test_an_aggregate_folds_a_fused_subtree() raises:
    """`sum(a + b)` never materialises `a + b`.

    The state binds the subtree and reads lanes straight out of the morsel, so
    there is no intermediate column to buffer — the one thing DataFusion,
    ClickHouse and Polars cannot express, because all three hand an aggregate
    an already-computed array.
    """
    var plan = table(_batch()).aggregate(
        [(col("a", int64) + col("b", int64)).sum().alias("total")],
        List[DynValue](),
    )
    # a = [1, 2, None, 4], b = [10, 20, 30, 40] -> 11 + 22 + 44, the null row
    # propagating through the addition rather than contributing b alone.
    assert_true(plan.execute().columns[0].as_int64() == array([77], int64))


def test_having_is_a_filter_above_an_aggregate() raises:
    """`HAVING` needs no node of its own.

    A `Filter` above the aggregate sees the aggregate's *output* batch, so the
    predicate reads the aggregate's output column by name.
    """
    var plan = (
        table(_keyed())
        .aggregate([col("a", int64).sum().alias("total")], [col("g", int64)])
        .filter((col("total", int64) > lit(50, int64)))
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)  # group 1 totals 40, group 2 totals 60
    assert_true(out.columns[0].as_int64() == array([2], int64))


# ---------------------------------------------------------------------------
# The push engine
# ---------------------------------------------------------------------------


def test_the_flush_cascade_feeds_the_stages_above() raises:
    """An aggregate's result must still pass through everything above it.

    `finish` on stage *i* produces a batch no later stage has ever seen, so it
    has to be pushed through *i+1..* before stage *i+1* is itself finished. A
    projection over an aggregate is the smallest query that returns nothing at
    all if the flush is a plain loop of independent `finish` calls.
    """
    var plan = (
        table(_keyed())
        .aggregate([col("a", int64).sum().alias("total")], [col("g", int64)])
        .project(
            ["doubled"],
            [(col("total", int64) + col("total", int64))],
        )
    )
    var out = plan.execute()
    # groups total 40 and 60, doubled by a projection above the aggregate
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[0].as_int64() == array([80, 120], int64))


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


def test_limit_takes_a_prefix() raises:
    var plan = table(_batch()).limit(length=2, offset=0)
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_int64() == array([10, 20], int64))


def test_limit_skips_the_offset() raises:
    var plan = table(_batch()).limit(length=2, offset=2)
    var out = plan.execute()
    assert_true(out.columns[1].as_int64() == array([30, 40], int64))


def test_limit_preserves_its_input_schema() raises:
    var b = _batch()
    var plan = table(b.copy()).limit(length=1, offset=0)
    assert_true(plan.schema() == b.schema)


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def test_sort_orders_by_one_key() raises:
    var b = record_batch([array([3, 1, 2], int64).copy()], names=["a"])
    var plan = table(b^).sort_by([col("a", int64)], [True])
    assert_true(plan.execute().columns[0].as_int64() == array([1, 2, 3], int64))


def test_sort_descending() raises:
    var b = record_batch([array([3, 1, 2], int64).copy()], names=["a"])
    var plan = table(b^).sort_by([col("a", int64)], [False])
    assert_true(plan.execute().columns[0].as_int64() == array([3, 2, 1], int64))


def test_sort_composes_multiple_keys() raises:
    """Keys are applied stably last-first, and each pass **permutes** the
    previous order rather than replacing it.

    Dropping the composition is the classic multi-key sort bug: the last key
    wins and every earlier one is silently discarded. Here `a` alone would give
    [1,1,2,2] in some order — only a correct composition also orders `b`
    within each `a`.
    """
    var b = record_batch(
        [
            array([2, 1, 2, 1], int64).copy(),
            array([20, 30, 10, 40], int64).copy(),
        ],
        names=["a", "b"],
    )
    var plan = table(b^).sort_by(
        [
            col("a", int64),
            col("b", int64),
        ],
        [True, True],
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([1, 1, 2, 2], int64))
    assert_true(out.columns[1].as_int64() == array([30, 40, 10, 20], int64))


def test_sort_rejects_mismatched_keys_and_directions() raises:
    var raised = False
    try:
        _ = table(_batch()).sort_by([col("a", int64)], [True, False])
    except e:
        raised = True
        assert_true("sort" in String(e))
    assert_true(raised)


def test_an_inner_join_streams_the_probe_side() raises:
    """Keys 2 and 3 match; 1 and 4 do not."""
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_INNER)
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_equal(out.num_columns(), 4)  # left k, lv + right k, rv
    assert_true(out.columns[1].as_int64() == array([20, 30], int64))
    assert_true(out.columns[3].as_int64() == array([200, 300], int64))


def test_join_schema_is_left_then_right() raises:
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_INNER)
    var s = plan.schema()
    assert_equal(len(s.fields), 4)
    assert_equal(s.fields[1].name, "lv")
    assert_equal(s.fields[3].name, "rv")
    assert_true(plan.schema() == plan.execute().schema)


def test_a_semi_join_emits_only_the_left_side() raises:
    """`SEMI` answers "which left rows matched", so the right side contributes
    no columns — the schema rule and the kernel must agree on that."""
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_SEMI)
    assert_equal(len(plan.schema().fields), 2)
    var out = plan.execute()
    assert_equal(out.num_columns(), 2)
    assert_equal(out.num_rows(), 2)  # left keys 2 and 3 matched


def test_a_left_join_keeps_unmatched_build_rows_once() raises:
    """The reason LEFT buffers the probe side instead of streaming it.

    Its tail of unmatched build rows is a property of *every* probe row taken
    together. Probing morsel-by-morsel would re-emit that tail once per morsel;
    key 1 must appear exactly once.
    """
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_LEFT)
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 2 and 3 matched, 1 null-widened once


def test_join_rejects_mismatched_key_counts() raises:
    var raised = False
    try:
        _ = table(_left()).join(table(_right()), [0], [0, 1], JOIN_INNER)
    except e:
        raised = True
        assert_true("join" in String(e))
    assert_true(raised)


def test_a_join_composes_with_a_filter_above_it() raises:
    """The build side is a whole sub-plan and the probe side is a pipeline, so
    a join has to sit in a chain like any other stage."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER)
        .filter((col("lv", int64) > lit(25, int64)))
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)  # lv = 30


# ---------------------------------------------------------------------------
# ParquetScan
# ---------------------------------------------------------------------------


def test_a_parquet_scan_feeds_the_pipeline() raises:
    """A scan is a source like any other, so everything composes above it.

    Written and read back rather than mocked: the point is that the operator
    really decodes a file and that its batches flow through the same stages an
    in-memory table's do.
    """
    var path = String("/tmp/marrow_expr2_scan.parquet")
    var b = record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )
    write_table(Table.from_batches(b.schema.copy(), [b.copy()]), path)

    var plan = scan(path.copy(), b.schema.copy()).filter(
        (col("a", int64) > lit(2, int64))
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_int64() == array([30, 40], int64))


def test_a_parquet_scan_schema_is_the_projection() raises:
    """Narrowing the scan's schema is how a projection is pushed into it —
    only the named columns are read out of the file."""
    var path = String("/tmp/marrow_expr2_proj.parquet")
    var b = record_batch(
        [
            array([1, 2], int64).copy(),
            array([10, 20], int64).copy(),
        ],
        names=["a", "b"],
    )
    write_table(Table.from_batches(b.schema.copy(), [b.copy()]), path)

    var only_b = schema([field("b", int64)])
    var plan = scan(path.copy(), only_b.copy())
    var out = plan.execute()
    assert_equal(out.num_columns(), 1)
    assert_true(out.columns[0].as_int64() == array([10, 20], int64))


def _left() raises -> RecordBatch:
    return record_batch(
        [array([1, 2, 3], int64).copy(), array([10, 20, 30], int64).copy()],
        names=["k", "lv"],
    )


def _right() raises -> RecordBatch:
    return record_batch(
        [array([2, 3, 4], int64).copy(), array([200, 300, 400], int64).copy()],
        names=["k", "rv"],
    )


# ---------------------------------------------------------------------------
# An aggregate is a `Value`, but not one every relation can take
# ---------------------------------------------------------------------------
def test_projecting_an_aggregate_raises_rather_than_aborting() raises:
    """It used to **abort the process**, not raise.

    An aggregate answers from `drain`, so its operator's `push` returns `None`
    — and `ProjectOperator.push` called `.value()` on that. Under `ASSERT=all`
    an abort takes down the whole runner, so this was one bad query away from
    failing every case in a file. `Value.aggregates` is what lets `Project` say
    no at plan time; the node used to conform to `Evaluable` and raise from an
    `evaluate` that was never reached.
    """
    var raised = False
    try:
        _ = table(_batch()).project(["s"], [col("a", int64).sum()])
    except e:
        raised = True
        assert_true("is an aggregate" in String(e))
    assert_true(raised, "projecting an aggregate must raise")


def test_filtering_on_an_aggregate_raises() raises:
    """The same rule on the other verb, and it points at `HAVING`: filtering
    an aggregate is legal *above* an `.aggregate()`, never beside it."""
    var raised = False
    try:
        _ = table(_batch()).filter(col("a", int64).sum())
    except e:
        raised = True
        assert_true("HAVING" in String(e))
    assert_true(raised, "filtering on an aggregate must raise")


def test_sorting_on_an_aggregate_raises() raises:
    """The third per-row position, and one of the two that kept aborting.

    `Filter` and `Project` grew the guard; `Sort` and `Aggregate`'s keys did
    not, so this reached `SortOperator`, which calls `.value()` on the `None`
    an aggregate answers from `push`. Four positions need the check and two
    had it — which is why it now lives in one `reject_aggregate` rather than
    being copied per node.
    """
    var raised = False
    try:
        _ = table(_batch()).sort_by([col("a", int64).sum()], [True])
    except e:
        raised = True
        assert_true("is an aggregate" in String(e))
    assert_true(raised, "sorting on an aggregate must raise")


def test_grouping_by_an_aggregate_raises() raises:
    """The fourth, and the one where the asymmetry is the whole point: an
    aggregate in `aggs` is what the node is *for*, and the same expression in
    `keys` is the abort."""
    var raised = False
    try:
        _ = table(_batch()).aggregate(
            [col("b", int64).count().alias("n")], [col("a", int64).sum()]
        )
    except e:
        raised = True
        assert_true("is an aggregate" in String(e))
    assert_true(raised, "grouping by an aggregate must raise")


def test_a_non_aggregate_value_is_still_projectable() raises:
    """The gate reads `Value.aggregates`, so an ordinary fused subtree — which
    is also `Shape.scalar` when it is a literal — is untouched."""
    var out = (
        table(_batch())
        .project(["s"], [(col("a", int64) + col("b", int64))])
        .execute()
    )
    assert_equal(out.num_rows(), 4)


# ---------------------------------------------------------------------------
# with_columns / drop / rename — the verbs that say what changes
# ---------------------------------------------------------------------------
# All three are sugar over `Project`, exactly as `select` is: the surviving
# columns are runtime column reads, so no caller has to supply their dtypes.
# What each one owns is a *rule about the output schema*, and that is what
# these cases pin — the row values follow from `Project`, which is already
# covered above.


def test_with_columns_appends_and_keeps_the_input_order() raises:
    """A new name goes on the end; the existing columns keep their positions
    and their fields. `select` cannot express this without the caller writing
    out the complement, which is wrong the moment a column is added
    upstream."""
    var out = (
        table(_batch())
        .with_columns(["s"], [(col("a", int64) + col("b", int64))])
        .execute()
    )
    assert_equal(out.num_columns(), 3)
    assert_equal(out.schema.fields[0].name, String("a"))
    assert_equal(out.schema.fields[1].name, String("b"))
    assert_equal(out.schema.fields[2].name, String("s"))
    assert_equal(out.column(2).as_int64()[1].value(), Int64(22))


def test_with_columns_replaces_an_existing_name_in_place() raises:
    """Polars' rule, and the only one that keeps the output free of
    duplicates: `b` is overwritten where it already sits rather than appended
    a second time."""
    var out = (
        table(_batch())
        .with_columns(["b"], [(col("b", int64) * lit(2, int64))])
        .execute()
    )
    assert_equal(out.num_columns(), 2)
    assert_equal(out.schema.fields[1].name, String("b"))
    assert_equal(out.column(1).as_int64()[0].value(), Int64(20))


def test_drop_keeps_the_survivors_in_input_order() raises:
    """`drop` says what goes; everything else stays where it was."""
    var out = table(_batch()).drop(["a"]).execute()
    assert_equal(out.num_columns(), 1)
    assert_equal(out.schema.fields[0].name, String("b"))


def test_drop_rejects_a_name_that_is_not_there() raises:
    """A typo in a `drop` list is otherwise silent — the column it meant to
    remove survives — which is the failure this verb exists to avoid."""
    var raised = False
    try:
        _ = table(_batch()).drop(["nope"])
    except e:
        raised = True
        assert_true("not found" in String(e))
    assert_true(raised, "dropping an unknown column must raise")


def test_rename_carries_the_source_field_over() raises:
    """Not just the name: dtype, `nullable` and metadata come from the source
    field, because `Project._output_schema` recognises a bare column. Rebuilding
    from the dtype alone turns `nullable=False` into `True`, which is the
    divergence that method exists to fix."""
    var fields = List[Field](capacity=1)
    fields.append(field("a", int64, nullable=False))
    var b = RecordBatch(Schema(fields=fields^), [array([1, 2], int64).to_dyn()])
    var renamed = table(b^).rename(["a"], ["z"]).schema()
    assert_equal(renamed.fields[0].name, String("z"))
    assert_true(renamed.fields[0].dtype.is_int64())
    assert_true(not renamed.fields[0].nullable)


def test_rename_leaves_untouched_columns_alone() raises:
    """Two parallel lists rather than a mapping, because Mojo has no dict
    literal in argument position. Columns not named keep their own names and
    their positions."""
    var out = table(_batch()).rename(["b"], ["total"]).execute()
    assert_equal(out.num_columns(), 2)
    assert_equal(out.schema.fields[0].name, String("a"))
    assert_equal(out.schema.fields[1].name, String("total"))
    assert_equal(out.column(1).as_int64()[3].value(), Int64(40))


def test_rename_rejects_mismatched_list_lengths() raises:
    var raised = False
    try:
        _ = table(_batch()).rename(["a", "b"], ["z"])
    except e:
        raised = True
        assert_true("new names" in String(e))
    assert_true(raised, "rename with unequal lists must raise")


def test_variadic_select_matches_the_list_form() raises:
    """`select("a")` and `select(["a"])` are one verb. The variadic spelling is
    what the golden corpus's Python twin uses, so the two lanes cannot be one
    text without it."""
    var one = table(_batch()).select("b", "a").execute()
    var two = table(_batch()).select(["b", "a"]).execute()
    assert_equal(one.schema.fields[0].name, two.schema.fields[0].name)
    assert_equal(one.schema.fields[1].name, String("a"))
    assert_equal(one.num_columns(), 2)


def test_filter_above_limit_with_offset_reads_the_limited_rows() raises:
    """A filter *above* an offset limit, in **both** lanes.

    `LimitOperator` emits `batch.slice(start, wanted)`, and a struct slice is
    zero-copy: it moves the struct's offset and shares its children whole. So
    `StructArray.field` had to learn to carry that slice, and until it did the
    two lanes failed differently on the same plan:

    - the runtime lane materialised the parent's full child and `to_array`
      caught the length mismatch, raising rather than answering;
    - the comptime lane read elements `[0, len)` of an unsliced child, which
      is the *wrong window* the moment the offset is non-zero -- a silent wrong
      answer, and the reason this case uses `offset=2` rather than a bare
      `limit`. With offset 0 both lanes were right by coincidence.

    `sort_by` first so the four kept rows are determined by value rather than
    by input order.

    **Asserted element by element, not with `==`.** `DynArray.__eq__` goes
    through `ArrayData.__eq__`, which compares the layout — offset and whole
    buffers included — and is documented as such: "two layouts holding the same
    values at different offsets are not equal here". A filtered result
    over-allocates its buffer, so `got.column("v") == array([5, 7], int64)`
    answers False on the *right* values. `PrimitiveArray[T].__eq__` does loop
    and would have worked; the erased one does not.
    """
    var t = table(
        record_batch(
            [array([5, 3, 7, 1, 9, 2, 4], int64).to_dyn()], names=["v"]
        )
    )
    # Sorted: 1 2 3 4 5 7 9. Skip 2, keep 4 -> 3 4 5 7. Then keep > 4 -> 5 7.
    var limited = t.sort_by([DynValue(Column[Int64Type]("v"))], [True]).limit(
        4, offset=2
    )

    var fused = limited.filter(
        Gt(Column[Int64Type]("v"), Literal[Int64Type](Int64(4)))
    )
    var got = fused.execute()
    assert_equal(got.num_rows(), 2)
    ref fused_v = got.column("v").as_int64()
    assert_equal(Int(fused_v[0].value()), 5)
    assert_equal(Int(fused_v[1].value()), 7)

    var erased = limited.filter(
        DynValue(
            runtime_gt(runtime_column("v"), runtime_literal(Int64Scalar(4)))
        )
    )
    var got2 = erased.execute()
    assert_equal(got2.num_rows(), 2)
    ref erased_v = got2.column("v").as_int64()
    assert_equal(Int(erased_v[0].value()), 5)
    assert_equal(Int(erased_v[1].value()), 7)
