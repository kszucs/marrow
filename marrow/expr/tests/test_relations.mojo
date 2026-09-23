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
from ...kernels.join import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_RIGHT_ANTI,
    JoinKind,
    BUILD_LEFT,
    BUILD_RIGHT,
)
from ...kernels.sort import sort
from ..optimizer import AllRules, PushFilterBelowJoin
from ...dtypes import Field, field
from ...schema import Schema, schema
from ...parquet.writer import write_table
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import Datum
from ..`comptime`.leaves import NumericColumn, NumericLiteral
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
    """`limit` is zero-copy, so it hands back a *window* on its input rather
    than a fresh array -- and array equality is structural, so the expected
    side has to be the same window. Comparing against a standalone
    `array([30, 40])` asserted the values and the layout at once, and only
    passed while equality ignored `offset`."""
    var plan = table(_batch()).limit(length=2, offset=2)
    var out = plan.execute()
    assert_true(
        out.columns[1].as_int64() == array([10, 20, 30, 40], int64).slice(2, 2)
    )


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
    var limited = t.sort_by(
        [DynValue(NumericColumn[Int64Type]("v"))], [True]
    ).limit(4, offset=2)

    var fused = limited.filter(
        Gt(NumericColumn[Int64Type]("v"), NumericLiteral[Int64Type](Int64(4)))
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


# ---------------------------------------------------------------------------
# Join.build_side — a physical choice carried through a logical plan
#
# `left` and `right` say what the answer is; `build_side` says what it costs.
# The three claims below are what make that true at the plan layer: the schema
# does not move, the rows do not move, and a rewrite does not lose the choice.
# ---------------------------------------------------------------------------


def _canonical_rows(batch: RecordBatch) raises -> StructArray:
    """`batch`'s rows in a canonical order — sorted on every column.

    Row order follows the probe side, so it genuinely differs between the two
    build sides; the multiset of rows is what must not.
    """
    var sa = batch.to_struct_array()
    var keys = List[Int](capacity=len(sa.children))
    var asc = List[Bool](capacity=len(sa.children))
    for i in range(len(sa.children)):
        keys.append(i)
        asc.append(True)
    return sort(sa, keys, asc)


def _assert_plan_build_sides_agree(kind: JoinKind) raises:
    var built_left = table(_left()).join(
        table(_right()), [0], [0], kind, BUILD_LEFT
    )
    var built_right = table(_left()).join(
        table(_right()), [0], [0], kind, BUILD_RIGHT
    )
    assert_true(
        built_left.schema() == built_right.schema(),
        String("kind ", kind, ": the declared schema moved"),
    )
    var a = built_left.execute()
    var b = built_right.execute()
    # The relabel in `JoinOperator._probe` is only honest if what it relabels
    # already matched, so the executed schema is checked against the declared
    # one on both sides rather than against each other.
    assert_true(built_left.schema() == a.schema)
    assert_true(built_right.schema() == b.schema)
    assert_equal(
        a.num_rows(), b.num_rows(), String("kind ", kind, ": row count")
    )
    assert_true(
        _canonical_rows(a) == _canonical_rows(b),
        String("kind ", kind, ": the rows differ"),
    )


def test_plan_build_side_agrees_for_every_kind() raises:
    """Through a plan, either build side returns the declared schema and the
    same rows, for every kind."""
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        _assert_plan_build_sides_agree(k)


# ---------------------------------------------------------------------------
# The two mirrored existence filters, at the plan layer
#
# `JOIN_RIGHT_SEMI` and `JOIN_RIGHT_ANTI` are what made `JoinKind.mirror`
# total, which is what made a build side free to choose — so the kernel tests
# them and `Estimate.joined` tests them, and until now this file did not
# mention them at all. They are reachable as logical kinds
# (`JoinKind.parse("right semi")`, `Join(kind=...)`), so every claim the other
# six carry here has to hold for them too: the schema, the rows, survival
# through `traverse`, and survival through a rule that rebuilds the node.
# ---------------------------------------------------------------------------
def test_a_right_semi_join_emits_only_the_right_side() raises:
    """The mirror of `test_a_semi_join_emits_only_the_left_side`.

    `_output_schema` asks `emits_left_columns` / `emits_right_columns`, so a
    kind whose arm was missing there would come back with four columns rather
    than two — and the declared schema is what everything above the join reads.
    """
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_RIGHT_SEMI)
    var s = plan.schema()
    assert_equal(len(s.fields), 2)
    assert_equal(s.fields[0].name, "k")
    assert_equal(s.fields[1].name, "rv")

    var out = plan.execute()
    assert_true(plan.schema() == out.schema)
    # right keys 2 and 3 match a left key; 4 does not
    assert_equal(out.num_rows(), 2)
    assert_equal(out.num_columns(), 2)
    var canonical = _canonical_rows(out)
    assert_true(canonical.children[0].as_int64() == array([2, 3], int64))
    assert_true(canonical.children[1].as_int64() == array([200, 300], int64))


def test_a_right_anti_join_emits_the_unmatched_right_rows() raises:
    """The complement of the case above, and the reason a boolean-shaped
    reading of the kind is not enough: SEMI and ANTI agree on both
    `emits_*_columns` predicates and differ only in which rows they keep."""
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_RIGHT_ANTI)
    assert_equal(len(plan.schema().fields), 2)

    var out = plan.execute()
    assert_true(plan.schema() == out.schema)
    assert_equal(out.num_rows(), 1)
    var canonical = _canonical_rows(out)
    assert_true(canonical.children[0].as_int64() == array([4], int64))
    assert_true(canonical.children[1].as_int64() == array([400], int64))


def test_a_right_sided_existence_filter_survives_traverse() raises:
    """`traverse` rebuilds a join through the by-name constructor, which takes
    the kind as an argument like any other field.

    Paired with the build side because the two fail the same way: a rebuild
    that dropped either produces a plan that still runs, and only a kind that
    changed would change the answer."""

    def identity(node: DynRelation) raises {imm} -> DynRelation:
        return node.copy()

    var kinds: List[JoinKind] = [JOIN_RIGHT_SEMI, JOIN_RIGHT_ANTI]
    for ref k in kinds:
        var j = Join(
            table(_left()),
            table(_right()),
            [0],
            [0],
            k,
            build_side=BUILD_RIGHT,
        )
        var again = j.traverse(identity)
        assert_true(again.isa[Join]())
        assert_true(
            again.get[Join]().kind == k,
            String("traverse dropped the kind: ", again),
        )
        assert_true(
            again.get[Join]().build_side == BUILD_RIGHT,
            String("traverse dropped the build side: ", again),
        )
        assert_true(
            j.schema() == again.schema(), String("the schema moved: ", again)
        )


def test_a_right_sided_existence_filter_survives_the_optimizer() raises:
    """`optimize[AllRules]()` may re-take the build side and must not touch
    anything else.

    The kind decides *which* rows come back, so a rule that lost it is a wrong
    answer rather than a slow one — and `SelectBuildSide` reaches these two
    precisely because `commutes` admits them.
    """
    var kinds: List[JoinKind] = [JOIN_RIGHT_SEMI, JOIN_RIGHT_ANTI]
    for ref k in kinds:
        var plan = table(_left()).join(table(_right()), [0], [0], k)
        var optimized = plan.optimize[AllRules]()
        assert_true(optimized.isa[Join](), String(optimized))
        assert_true(
            optimized.get[Join]().kind == k,
            String("kind ", k, ": the optimizer changed it — ", optimized),
        )
        assert_true(
            plan.schema() == optimized.schema(),
            String("kind ", k, ": the schema moved — ", optimized),
        )
        assert_true(
            _canonical_rows(plan.execute())
            == _canonical_rows(optimized.execute()),
            String("kind ", k, ": the rows moved — ", optimized),
        )


def test_a_right_sided_existence_filter_is_reachable_by_name() raises:
    """`JoinKind.parse` is how a frontend names a kind, and the plan layer
    takes whatever it answers — so the two spellings must reach the two
    constants rather than raising."""
    assert_true(JoinKind.parse(String("right semi")) == JOIN_RIGHT_SEMI)
    assert_true(JoinKind.parse(String("right anti")) == JOIN_RIGHT_ANTI)
    var plan = table(_left()).join(
        table(_right()), [0], [0], JoinKind.parse(String("right semi"))
    )
    assert_equal(plan.execute().num_rows(), 2)


def test_a_right_built_left_join_still_streams_its_morsels() raises:
    """A LEFT join built on its *right* input is a physical RIGHT join.

    Its extra rows are then unmatched *probe* rows, each of which belongs to
    exactly one morsel, so it streams — and key 1 must still appear exactly
    once rather than once per morsel. `_blocks_on_probe_side` asking the
    logical kind would buffer here for nothing; asking the physical kind and
    getting the mirror wrong would duplicate the tail.
    """
    var plan = table(_left()).join(
        table(_right()), [0], [0], JOIN_LEFT, BUILD_RIGHT
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 2 and 3 matched, 1 null-widened once
    assert_equal(out.num_columns(), 4)


def test_join_build_side_survives_traverse() raises:
    """`traverse` rebuilds a join through the by-name constructor.

    Losing `build_side` there costs an optimization rather than an answer, so
    nothing else in this file would fail — the same argument `ParquetScan`'s
    pruners carry, and the reason they are tested the same way.
    """
    var j = Join(
        table(_left()),
        table(_right()),
        [0],
        [0],
        JOIN_INNER,
        build_side=BUILD_RIGHT,
    )

    def identity(node: DynRelation) raises {imm} -> DynRelation:
        return node.copy()

    var again = j.traverse(identity)
    assert_true(again.isa[Join]())
    assert_true(
        again.get[Join]().build_side == BUILD_RIGHT,
        "traverse dropped the build side",
    )


def test_join_build_side_survives_a_rule_that_rebuilds_the_node() raises:
    """The rewrite that actually happens: a filter pushed below the join.

    `PushFilterBelowJoin` rebuilds the node, so this is `traverse`'s claim one
    level up, through a rule that has its own `Join(...)` call. Asked of that
    rule *alone*, because `AllRules` also contains a rule whose whole job is to
    change this field — see the case below, which is the one that would fail if
    the two were run together and could not be told apart.
    """
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT)
        .filter(col("lv", int64) > lit(15, int64))
    )
    var rewritten = PushFilterBelowJoin.apply(plan)
    assert_true(
        rewritten.isa[Join](),
        String("expected the filter below the join, got ", rewritten),
    )
    assert_true(
        rewritten.get[Join]().build_side == BUILD_RIGHT,
        "the rule dropped the build side",
    )
    # A right-built join prints its build side, so a plan that lost it is
    # visible in the rendering as well as in the field.
    assert_true("build=right" in String(rewritten), String(rewritten))


def test_the_optimizer_may_overrule_a_hand_written_build_side() raises:
    """And here it does, which is the field doing its job rather than losing
    it.

    `build_side` is a physical choice, not part of what the query means, so a
    cost-based rule is entitled to re-take it. Pushing the filter below the
    join is what changes the arithmetic: the left input drops to an estimated
    one row against the right's three, so indexing the left becomes the cheaper
    arrangement and `SelectBuildSide` says so — even though the author asked
    for `BUILD_RIGHT` when both sides still had three rows.

    The distinction this pins is between *chosen* and *lost*. A rule that
    silently dropped the field would also leave `BUILD_LEFT` here, so the
    assertion is paired with the case above, which proves the rebuild carries
    it when nothing decides otherwise.
    """
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT)
        .filter(col("lv", int64) > lit(15, int64))
    )
    var rewritten = plan.optimize[AllRules]()
    assert_true(rewritten.isa[Join](), String(rewritten))
    assert_true(
        rewritten.get[Join]().build_side == BUILD_LEFT,
        String("expected the cheaper side to win, got ", rewritten),
    )

    # Nothing shrank the right side, so the author's choice stands there.
    var other = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT)
        .filter(col("rv", int64) > lit(150, int64))
    )
    var kept = other.optimize[AllRules]()
    assert_true(kept.isa[Join](), String(kept))
    assert_true(
        kept.get[Join]().build_side == BUILD_RIGHT,
        String(
            "the filter went to the right side, so should the build: ", kept
        ),
    )


def test_join_schema_ignores_the_build_side() raises:
    """The field `_output_schema` is not allowed to see.

    A schema that moved with the build side would make choosing one a change
    of meaning, which is the coupling this field exists to break.
    """
    var built_left = table(_left()).join(
        table(_right()), [0], [0], JOIN_SEMI, BUILD_LEFT
    )
    var built_right = table(_left()).join(
        table(_right()), [0], [0], JOIN_SEMI, BUILD_RIGHT
    )
    var s = built_right.schema()
    assert_equal(len(s.fields), 2)
    assert_equal(s.fields[0].name, "k")
    assert_equal(s.fields[1].name, "lv")
    assert_true(s == built_left.schema())
