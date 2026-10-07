# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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

from std.testing import assert_equal, assert_raises, assert_true
from std.os.path import join

from ...utils.testing import ScratchDir
from ...arrays import StructArray, DynArray, StringArray
from ...builders import array
from ...dtypes import (
    DynType,
    Int64Type,
    bool_,
    float64,
    int64,
    string,
    string_view,
)
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
    JOIN_ALL,
    JOIN_ANY,
)
from ...kernels.sort import sort
from ..optimizer import (
    AllRules,
    ColumnPruning,
    MergeJoinChains,
    MergeProjectIntoJoin,
    PushFilterIntoJoin,
)
from ...dtypes import Field, field
from ...schema import Schema, schema
from ...parquet.writer import write_table
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..bindings import Bindings
from ..physical import (
    Datum,
    JoinOrder,
    Morsel,
    PlannedJoin,
    RenamedPredicate,
    SelectOperator,
)
from ..`comptime`.leaves import NumericColumn, NumericLiteral
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..builders import col, lit, param, scan, table
from ..runtime.values import (
    column as runtime_column,
    gt as runtime_gt,
    literal as runtime_literal,
)
from ...scalars import BoolScalar, Int64Scalar
from ..logical import (
    Aggregate,
    JoinChain,
    JoinLink,
    JoinFilter,
    JoinRef,
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
    assert_equal(out.num_columns(), 3)  # k once, lv, rv
    assert_true(out.columns[1].as_int64() == array([20, 30], int64))
    assert_true(out.columns[2].as_int64() == array([200, 300], int64))


def test_join_schema_is_left_then_right() raises:
    """An inner key both sides call `k` is one column, as in ibis: the join
    made the two equal, so the second copy says nothing."""
    var plan = table(_left()).join(table(_right()), [0], [0], JOIN_INNER)
    var s = plan.schema()
    assert_equal(len(s.fields), 3)
    assert_equal(s.fields[0].name, "k")
    assert_equal(s.fields[1].name, "lv")
    assert_equal(s.fields[2].name, "rv")
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
    # An outer join's right key is NULL where the left one is not, so it is
    # its own column, renamed rather than merged.
    assert_equal(out.schema.fields[2].name, "k_right")


def test_join_rejects_mismatched_key_counts() raises:
    var raised = False
    try:
        _ = table(_left()).join(table(_right()), [0], [0, 1], JOIN_INNER)
    except e:
        raised = True
        assert_true("join" in String(e))
    assert_true(raised)


def test_join_rejects_a_missing_or_out_of_range_key() raises:
    var none = List[Int]()
    with assert_raises(contains="at least one key"):
        _ = table(_left()).join(table(_right()), none.copy(), none.copy())
    with assert_raises(contains="left key 5"):
        _ = table(_left()).join(table(_right()), [5], [0])
    with assert_raises(contains="right key -1"):
        _ = table(_left()).join(table(_right()), [0], [-1])


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_expr2_scan.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_expr2_proj.parquet")
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
    had it — which is why it now lives in one `require_per_row` rather than
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
# JoinLink.build_side — a physical choice carried through a logical plan
#
# The participants and keys say what the answer is; `build_side` says what it
# costs.
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
# (`JoinKind.parse("right semi")`, `.join(..., kind)`), so every claim the other
# six carry here has to hold for them too: the schema, the rows, survival
# through `traverse`, and survival through a rule that rebuilds the node.
# ---------------------------------------------------------------------------
def test_a_right_semi_join_emits_only_the_right_side() raises:
    """The mirror of `test_a_semi_join_emits_only_the_left_side`.

    `JoinChain.joined` asks the kind which side it emits, so a
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
    """`traverse` rebuilds a chain over rewritten participants and keeps its
    links as they are.

    Paired with the build side because the two fail the same way: a rebuild
    that dropped either produces a plan that still runs, and only a kind that
    changed would change the answer."""

    def identity(node: DynRelation) raises {imm} -> DynRelation:
        return node.copy()

    var kinds: List[JoinKind] = [JOIN_RIGHT_SEMI, JOIN_RIGHT_ANTI]
    for ref k in kinds:
        var j = table(_left()).join(table(_right()), [0], [0], k, BUILD_RIGHT)
        var again = j.traverse(identity)
        assert_true(again.isa[JoinChain]())
        ref step = again.get[JoinChain]().links[0]
        assert_true(
            step.kind == k, String("traverse dropped the kind: ", again)
        )
        assert_true(
            step.build_side == BUILD_RIGHT,
            String("traverse dropped the build side: ", again),
        )
        assert_true(
            j.schema() == again.schema(), String("the schema moved: ", again)
        )


def test_a_right_sided_existence_filter_survives_the_optimizer() raises:
    """`optimize[AllRules]()` may re-take the build side and must not touch
    anything else.

    The kind decides *which* rows come back, so a rule that lost it is a wrong
    answer rather than a slow one — and `JoinOrdering` reaches these two
    precisely because `commutes` admits them.
    """
    var kinds: List[JoinKind] = [JOIN_RIGHT_SEMI, JOIN_RIGHT_ANTI]
    for ref k in kinds:
        var plan = table(_left()).join(table(_right()), [0], [0], k)
        var optimized = plan.optimize[AllRules]()
        assert_true(optimized.isa[JoinChain](), String(optimized))
        assert_true(
            optimized.get[JoinChain]().links[0].kind == k,
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
    """`traverse` rebuilds a chain over rewritten participants.

    Losing `build_side` there costs an optimization rather than an answer, so
    nothing else in this file would fail — the same argument `ParquetScan`'s
    pruners carry, and the reason they are tested the same way.
    """
    var j = table(_left()).join(
        table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT
    )

    def identity(node: DynRelation) raises {imm} -> DynRelation:
        return node.copy()

    var again = j.traverse(identity)
    assert_true(again.isa[JoinChain]())
    assert_true(
        again.get[JoinChain]().links[0].build_side == BUILD_RIGHT,
        "traverse dropped the build side",
    )


def test_join_build_side_survives_a_rule_that_rebuilds_the_node() raises:
    """The rewrite that actually happens: a filter pushed into a participant.

    `PushFilterIntoJoin` rebuilds the chain, so this is `traverse`'s claim one
    level up. Asked of that rule *alone*, because `AllRules` ends with
    `JoinOrdering`, whose planner may hash the other side — see the case
    below, which is the one that would fail if the two were run together and
    could not be told apart.
    """
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT)
        .filter(col("lv", int64) > lit(15, int64))
    )
    var rewritten = PushFilterIntoJoin.apply(plan)
    assert_true(
        rewritten.isa[JoinChain](),
        String("expected the filter inside the join, got ", rewritten),
    )
    ref chain = rewritten.get[JoinChain]()
    assert_true(chain.inputs[0][].isa[Filter](), String(rewritten))
    assert_true(
        chain.links[0].build_side == BUILD_RIGHT,
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
    arrangement and `JoinOrdering` says so — even though the author asked
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
    assert_true(rewritten.isa[JoinChain](), String(rewritten))
    assert_true(
        rewritten.get[JoinChain]().planned_order().joins[0].link.build_side
        == BUILD_LEFT,
        String("expected the cheaper side to win, got ", rewritten),
    )

    # Nothing shrank the right side, so the author's choice stands there.
    var other = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT)
        .filter(col("rv", int64) > lit(150, int64))
    )
    var kept = other.optimize[AllRules]()
    assert_true(kept.isa[JoinChain](), String(kept))
    assert_true(
        kept.get[JoinChain]().planned_order().joins[0].link.build_side
        == BUILD_RIGHT,
        String(
            "the filter went to the right side, so should the build: ", kept
        ),
    )


def test_join_schema_ignores_the_build_side() raises:
    """The field the chain's schema is not allowed to see.

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


def test_a_parquet_scan_schema_picks_the_string_layout() raises:
    """A string column declared `string_view` in the scan's schema is decoded
    as views; one declared `string` beside it keeps the offsets layout."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_scan_views.parquet")
        var s: StringArray = ["pear", "a value longer than twelve", "plum"]
        var t: StringArray = ["x", "y", "z"]
        var b = record_batch([s^.to_dyn(), t^.to_dyn()], names=["s", "t"])
        write_table(Table.from_batches(b.schema.copy(), [b.copy()]), path)

        var views = schema([field("s", string_view), field("t", string)])
        var out = (
            scan(path.copy(), views^)
            .filter(col("s", string_view) != lit("plum", string))
            .execute()
        )
        assert_equal(out.num_rows(), 2)
        assert_true(out.columns[0].dtype() == DynType(string_view))
        assert_true(out.columns[1].dtype() == DynType(string))
        assert_equal(
            out.columns[0].as_string_view()[1].value(),
            "a value longer than twelve",
        )


# ---------------------------------------------------------------------------
# JoinChain — every join as ibis represents one
#
# One node holds the participants, a link per join and the columns they
# answer with. Names follow ibis's `disambiguate_fields`: an inner key both
# sides name alike is one column, any other clash is renamed by `lname` /
# `rname`, and a clash those leave raises at the join.
# ---------------------------------------------------------------------------
def _third() raises -> RecordBatch:
    return record_batch(
        [array([2, 3, 5], int64).copy(), array([7, 8, 9], int64).copy()],
        names=["k", "tv"],
    )


def _names(plan: DynRelation) -> List[String]:
    var out = List[String]()
    for ref f in plan.schema().fields:
        out.append(f.name.copy())
    return out^


def test_a_chain_emits_a_key_every_join_equates_once() raises:
    """Three inputs joined on `k` are one node with one `k`: each join made
    the new `k` equal to the one already emitted."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0])
        .join(table(_third()), [0], [0])
    )
    assert_true(plan.isa[JoinChain](), String(plan))
    assert_equal(len(plan.get[JoinChain]().inputs), 3)
    assert_equal(_names(plan), ["k", "lv", "rv", "tv"])
    var out = plan.execute()
    assert_true(plan.schema() == out.schema)
    var rows = _canonical_rows(out)
    assert_true(rows.children[0].as_int64() == array([2, 3], int64))
    assert_true(rows.children[3].as_int64() == array([7, 8], int64))


def test_a_chain_on_the_right_is_spliced_into_one_chain() raises:
    """`a.join(b.join(c))` takes `b.join(c)` as one participant, as ibis
    does; `MergeJoinChains` splices it in, so the chain written bushy is one
    of three inputs and two links, numbered as the left-deep spelling numbers
    them, with the same names and the same rows."""
    var plan = table(_left()).join(
        table(_right()).join(table(_third()), [0], [0]), [0], [0]
    )
    assert_equal(len(plan.get[JoinChain]().inputs), 2)
    var spliced = MergeJoinChains.apply(plan)
    assert_true(spliced.isa[JoinChain](), String(spliced))
    ref chain = spliced.get[JoinChain]()
    assert_equal(len(chain.inputs), 3)
    assert_equal(len(chain.links), 2)
    assert_true(spliced.schema() == plan.schema())
    assert_equal(_names(spliced), ["k", "lv", "rv", "tv"])
    assert_true(
        _canonical_rows(spliced.execute()) == _canonical_rows(plan.execute())
    )

    # Under an outer join the nested chain stays one participant: its joins
    # may not run before the LEFT join's.
    var outer = table(_left()).join(
        table(_right()).join(table(_third()), [0], [0]), [0], [0], JOIN_LEFT
    )
    var spliced_outer = MergeJoinChains.apply(outer)
    assert_equal(len(spliced_outer.get[JoinChain]().inputs), 2)
    assert_true(spliced_outer.schema() == outer.schema())
    assert_equal(spliced_outer.execute().num_rows(), 3)
    assert_true(
        _canonical_rows(spliced_outer.execute())
        == _canonical_rows(outer.execute())
    )


def test_a_join_renames_a_clash_it_cannot_merge() raises:
    """`lv` is not a key, so the right one is renamed; under LEFT even the key
    is, since a padded row holds NULL on one side only."""
    var inner = table(_left()).join(table(_left()), [0], [0])
    assert_equal(_names(inner), ["k", "lv", "lv_right"])
    var custom = table(_left()).join(
        table(_left()), [0], [0], lname="{name}_l", rname="{name}_r"
    )
    assert_equal(_names(custom), ["k", "lv_l", "lv_r"])
    var outer = table(_left()).join(table(_left()), [0], [0], JOIN_LEFT)
    assert_equal(_names(outer), ["k", "lv", "k_right", "lv_right"])

    # Two participants call a column `lv`, and the chain lowers by position,
    # so each output reads its own: a self-join on a unique key pairs every
    # row with itself.
    var out = inner.execute()
    assert_true(inner.schema() == out.schema)
    assert_equal(out.num_rows(), 3)
    assert_true(
        out.column("lv").as_int64() == out.column("lv_right").as_int64()
    )


def test_a_clash_lname_makes_raises_at_the_join() raises:
    """Renaming the left `v` to `v_l` lands on a column the left side already
    has: two output columns would share a name, so the join refuses."""
    var left = record_batch(
        [
            array([1, 2], int64).copy(),
            array([10, 20], int64).copy(),
            array([5, 6], int64).copy(),
        ],
        names=["k", "v", "v_l"],
    )
    var right = record_batch(
        [array([1, 2], int64).copy(), array([7, 8], int64).copy()],
        names=["k", "v"],
    )
    with assert_raises(contains="two columns would be named 'v_l'"):
        _ = table(left^).join(table(right^), [0], [0], lname="{name}_l")


def test_a_join_input_may_not_repeat_a_column_name() raises:
    """A participant's columns are read by name, so a repeated one would be
    ambiguous inside the chain as well as out of it."""
    var dup = record_batch(
        [array([1], int64).copy(), array([2], int64).copy()],
        names=["k", "k"],
    )
    with assert_raises(contains="twice"):
        _ = table(dup^).join(table(_right()), [0], [0])


def test_select_rename_and_drop_fold_into_the_chain() raises:
    """Each says which participant columns the chain answers with, so
    `MergeProjectIntoJoin` makes each the chain's own output rather than a
    node above it — with the schema and rows the verb declared."""
    var plan = table(_left()).join(table(_right()), [0], [0])
    var picked = plan.select(["rv", "k"])
    var renamed = plan.rename(["lv"], ["left_value"])
    var dropped = plan.drop(["lv"])
    var shapes: List[DynRelation] = [
        picked.copy(),
        renamed.copy(),
        dropped.copy(),
    ]
    for ref written in shapes:
        assert_true(written.isa[Project](), String(written))
        var folded = MergeProjectIntoJoin.apply(written)
        assert_true(folded.isa[JoinChain](), String(folded))
        assert_true(folded.schema() == written.schema(), String(folded))
        assert_true(folded.schema() == folded.execute().schema)
        assert_true(
            _canonical_rows(folded.execute())
            == _canonical_rows(written.execute())
        )
    assert_equal(_names(picked), ["rv", "k"])
    assert_equal(_names(renamed), ["k", "left_value", "rv"])
    assert_equal(_names(dropped), ["k", "rv"])


def test_a_project_folds_only_when_it_reads() raises:
    """A read folds; a computation sits above; and an aggregate aliased to the
    very column it reads is no read at all — `project` refuses it, so it
    never reaches the rule."""
    var plan = table(_left()).join(table(_right()), [0], [0])
    var reads: List[DynValue] = [col("rv", int64), col("k", int64)]
    var folded = MergeProjectIntoJoin.apply(
        plan.project(["value", "k"], reads^)
    )
    assert_true(folded.isa[JoinChain](), String(folded))
    assert_equal(_names(folded), ["value", "k"])

    var computes: List[DynValue] = [col("rv", int64) + col("lv", int64)]
    var above = MergeProjectIntoJoin.apply(plan.project(["total"], computes^))
    assert_true(above.isa[Project](), String(above))
    assert_true(above.get[Project]().input[].isa[JoinChain]())

    var aggregate: List[DynValue] = [col("rv", int64).sum().alias("rv")]
    with assert_raises(contains=".aggregate()"):
        _ = plan.project(["rv"], aggregate^)


def test_a_filter_on_a_renamed_column_runs_on_its_participant() raises:
    """`lv_right` is participant 1's `lv`. The filter cannot move into the
    participant under a name it does not have there, so the chain keeps it —
    and evaluates it straight after that participant, under the name it was
    written with."""
    var plan = (
        table(_left())
        .join(table(_left()), [0], [0])
        .filter(col("lv_right", int64) > lit(15, int64))
    )
    var pushed = PushFilterIntoJoin.apply(plan)
    assert_true(pushed.isa[JoinChain](), String(pushed))
    ref chain = pushed.get[JoinChain]()
    assert_equal(len(chain.filters), 1)
    assert_equal(_landing(chain), 1)
    var want = _canonical_rows(plan.execute())
    assert_equal(len(want), 2)
    assert_true(_canonical_rows(pushed.execute()) == want)

    # The same after a folded rename: the filter reads a name no input has.
    var renamed = (
        table(_left())
        .join(table(_right()), [0], [0])
        .rename(["rv"], ["value"])
        .filter(col("value", int64) > lit(250, int64))
    )
    var absorbed = renamed.optimize[AllRules]()
    assert_true(absorbed.isa[JoinChain](), String(absorbed))
    assert_equal(_landing(absorbed.get[JoinChain]()), 1)
    assert_equal(absorbed.execute().num_rows(), 1)


def test_a_filter_reading_no_column_still_runs() raises:
    """A parameter test reads no participant. It still has to run exactly
    once, and a plan's parameters still have to include it."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0])
        .filter(param("keep", bool_))
    )
    var optimized = plan.optimize[AllRules]()
    assert_true(optimized.isa[JoinChain](), String(optimized))
    assert_equal(len(optimized.get[JoinChain]().filters), 1)
    assert_equal(len(optimized.params()), 1)
    var keep: Bindings = {"keep": BoolScalar(True).to_dyn()}
    var drop: Bindings = {"keep": BoolScalar(False).to_dyn()}
    assert_equal(optimized.execute(bindings=keep).num_rows(), 2)
    assert_equal(optimized.execute(bindings=drop).num_rows(), 0)


def test_a_join_appended_later_leaves_a_filter_below_it() raises:
    """A filter absorbed into an inner chain is bounded by the participants
    it was absorbed over. A RIGHT join appended afterwards pads rows the filter never
    saw; evaluating it above that join would drop them."""
    var filtered = (
        table(_left())
        .join(table(_right()), [0], [0])
        .filter(col("lv", int64) + col("rv", int64) > lit(250, int64))
    )
    var optimized = filtered.optimize[AllRules]()
    assert_true(optimized.isa[JoinChain](), String(optimized))
    var written = filtered.join(table(_third()), [0], [0], JOIN_RIGHT)
    var appended = optimized.join(table(_third()), [0], [0], JOIN_RIGHT)
    assert_true(appended.isa[JoinChain](), String(appended))
    assert_equal(len(appended.get[JoinChain]().inputs), 3)
    # k = 2 and 5 padded, k = 3 matched
    assert_equal(written.execute().num_rows(), 3)
    assert_true(
        _canonical_rows(appended.execute())
        == _canonical_rows(written.execute())
    )


def test_an_any_join_keeps_its_side_and_its_filters_above() raises:
    """`JOIN_ANY` keeps one match per probe row, so the build side decides
    which — the optimizer may not flip it, and a filter on the build side
    changes which match is picked, so it stays above the join."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT, JOIN_ANY)
        .filter(col("rv", int64) > lit(250, int64))
    )
    var optimized = plan.optimize[AllRules]()
    ref chain = optimized.get[JoinChain]()
    var order = chain.planned_order()
    assert_true(
        order.joins[0].link.build_side == BUILD_RIGHT, String(optimized)
    )
    assert_true(order.joins[0].link.strictness == JOIN_ANY)
    assert_equal(len(chain.filters), 1, String(optimized))
    assert_equal(_landing(chain), order.root())

    # The probe side picks nothing, so a filter there moves in.
    var probe = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT, JOIN_ANY)
        .filter(col("lv", int64) > lit(25, int64))
    )
    var pushed = probe.optimize[AllRules]()
    assert_true(pushed.isa[JoinChain](), String(pushed))
    assert_true(pushed.get[JoinChain]().inputs[0][].isa[Filter]())
    assert_equal(pushed.execute().num_rows(), 1)


def _landing(chain: JoinChain) raises -> Int:
    """The node of `chain`'s planned tree its first filter lands on."""
    var order = chain.planned_order()
    return order.landing(order.participants(), chain.rules(), chain.filters[0])


def _join(
    left: Int, right: Int, kind: JoinKind, lkey: JoinRef, rkey: JoinRef
) -> PlannedJoin:
    return PlannedJoin(
        left,
        right,
        JoinLink(kind, JOIN_ALL, BUILD_LEFT, [lkey.copy()], [rkey.copy()]),
    )


def _tree(chain: JoinChain, var joins: List[PlannedJoin]) -> JoinOrder:
    var out = JoinOrder(len(chain.inputs))
    for ref j in joins:
        _ = out.add(j.copy())
    return out^


def _run(chain: JoinChain, order: JoinOrder) raises -> StructArray:
    """`chain` computed by `order`, its rows in canonical order."""
    var pipe = chain.with_order(order.copy()).to_operator(
        ExecContext(), Bindings()
    )
    return _canonical_rows(
        RecordBatch.from_struct_array(pipe.collect(chain.schema()))
    )


def test_a_join_order_is_verified_against_the_links() raises:
    """`JoinOrder.verify` is what a planner's tree must pass, so it is where
    one that got the equalities, a participant or a link's place wrong is
    caught."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0])
        .join(table(_third()), [0], [0])
    )
    ref chain = plan.get[JoinChain]()
    var k0 = JoinRef(0, "k")
    var k1 = JoinRef(1, "k")
    var k2 = JoinRef(2, "k")

    # Another order over the same equalities is the same answer.
    var other = _tree(
        chain,
        [_join(0, 2, JOIN_INNER, k0, k2), _join(3, 1, JOIN_INNER, k2, k1)],
    )
    other.verify(chain)
    assert_true(_canonical_rows(plan.execute()) == _run(chain, other))
    # Equating `lv` with `#2.k` instead of `k` is not.
    with assert_raises(contains="other equalities"):
        _tree(
            chain,
            [
                _join(0, 1, JOIN_INNER, k0, k1),
                _join(3, 2, JOIN_INNER, JoinRef(0, "lv"), k2),
            ],
        ).verify(chain)
    # A participant joined twice is not a tree.
    with assert_raises(contains="joined twice"):
        _tree(
            chain,
            [_join(0, 1, JOIN_INNER, k0, k1), _join(3, 1, JOIN_INNER, k0, k1)],
        ).verify(chain)
    # A tree must join every participant.
    with assert_raises(contains="every input"):
        _tree(chain, [_join(0, 1, JOIN_INNER, k0, k1)]).verify(chain)
    # A join compares only columns its own two inputs carry: `#0 ⋈ #1`
    # equating `#0.k` with `#2.k` makes the same classes, but `#2` is not
    # there to compare.
    with assert_raises(contains="wrong side"):
        _tree(
            chain,
            [_join(0, 1, JOIN_INNER, k0, k2), _join(3, 2, JOIN_INNER, k1, k2)],
        ).verify(chain)

    # A LEFT join attaching `#1` may not attach `#1 ⋈ #2` instead, nor join on
    # other columns: its keys are part of what it answers.
    var outer = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_LEFT)
        .join(table(_third()), [0], [0])
    )
    ref outer_chain = outer.get[JoinChain]()
    with assert_raises(contains="not joined as"):
        _tree(
            outer_chain,
            [_join(1, 2, JOIN_INNER, k1, k2), _join(0, 3, JOIN_LEFT, k0, k1)],
        ).verify(outer_chain)
    with assert_raises(contains="not joined as"):
        _tree(
            outer_chain,
            [
                _join(0, 1, JOIN_LEFT, JoinRef(0, "lv"), JoinRef(1, "rv")),
                _join(3, 2, JOIN_INNER, k0, k2),
            ],
        ).verify(outer_chain)

    # `JOIN_ANY` keeps one match per probe row, so its build side is part of
    # the answer.
    var any = table(_left()).join(
        table(_right()), [0], [0], JOIN_INNER, BUILD_RIGHT, JOIN_ANY
    )
    ref any_chain = any.get[JoinChain]()
    with assert_raises(contains="not joined as"):
        _tree(
            any_chain,
            [
                PlannedJoin.of(any_chain.links[0], 0, 1, BUILD_LEFT),
            ],
        ).verify(any_chain)


def test_a_join_filter_reads_only_participants_before_its_bound() raises:
    """A filter bounded by `#1` is evaluated before `#2` joins, so it may not
    read `#2`."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0])
        .join(table(_third()), [0], [0])
    )
    var reads_third: List[JoinFilter] = [
        JoinFilter(col("tv", int64) > lit(7, int64), [JoinRef(2, "tv")], 1)
    ]
    with assert_raises(contains="bounded by"):
        _ = plan.get[JoinChain]().with_filters(reads_third^)


def test_a_join_order_keeps_equalities_multi_join_by_multi_join() raises:
    """`A ⋈ B` on `k, m` sits on the probe side of a `JOIN_ANY` join to `C`,
    with `D` joined on `m` above it. Moving the `B.m` equality from below the
    ANY join to above it equates the same columns chain-wide — and changes
    which `(A, B)` pair the ANY join can pick, so it is refused."""
    var a = record_batch(
        [
            array([1], int64).copy(),
            array([1], int64).copy(),
            array([1], int64).copy(),
        ],
        names=["k", "m", "j"],
    )
    var b = record_batch(
        [array([1], int64).copy(), array([1], int64).copy()],
        names=["k", "m"],
    )
    var c = record_batch([array([1], int64).copy()], names=["j"])
    var d = record_batch([array([1], int64).copy()], names=["m"])
    var plan = (
        table(a^)
        .join(table(b^), [0, 1], [0, 1])
        .join(table(c^), [2], [0], JOIN_INNER, BUILD_LEFT, JOIN_ANY)
        .join(table(d^), [1], [0])
    )
    ref chain = plan.get[JoinChain]()
    var moved = _tree(
        chain,
        [
            _join(0, 1, JOIN_INNER, JoinRef(0, "k"), JoinRef(1, "k")),
            PlannedJoin.of(chain.links[1], 4, 2, BUILD_LEFT),
            PlannedJoin(
                5,
                3,
                JoinLink(
                    JOIN_INNER,
                    JOIN_ALL,
                    BUILD_LEFT,
                    [JoinRef(0, "m"), JoinRef(1, "m")],
                    [JoinRef(3, "m"), JoinRef(3, "m")],
                ),
            ),
        ],
    )
    with assert_raises(contains="other equalities"):
        moved.verify(chain)


def test_a_join_order_moves_an_inner_join_across_a_left_join() raises:
    """`(#0 ⋈ #2) ⟕ #1` and `(#0 ⟕ #1) ⋈ #2` answer alike: `#2`'s key reads
    nothing the LEFT join pads. Reading `#1` instead, `#2` may not join below
    it — the LEFT join would then pad what the inner join drops."""
    var plan = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_LEFT)
        .join(table(_third()), [0], [0])
    )
    ref chain = plan.get[JoinChain]()
    var k0 = JoinRef(0, "k")
    var crossed = _tree(
        chain,
        [
            _join(0, 2, JOIN_INNER, k0, JoinRef(2, "k")),
            _join(3, 1, JOIN_LEFT, k0, JoinRef(1, "k")),
        ],
    )
    crossed.verify(chain)
    assert_true(_canonical_rows(plan.execute()) == _run(chain, crossed))

    var reads_padded = (
        table(_left())
        .join(table(_right()), [0], [0], JOIN_LEFT)
        .join(table(_third()), [2], [0], rname="{name}_third")
    )
    ref padded = reads_padded.get[JoinChain]()
    with assert_raises(contains="not joined as"):
        _tree(
            padded,
            [
                _join(1, 2, JOIN_INNER, JoinRef(1, "k"), JoinRef(2, "k")),
                _join(0, 3, JOIN_LEFT, k0, JoinRef(1, "k")),
            ],
        ).verify(padded)


def test_a_filter_over_a_spine_lands_below_the_left_join() raises:
    """`lv > tv + 15` reads two participants on the preserved side of a LEFT
    join, so it is evaluated where both are joined, below the LEFT join — and
    the rows are the ones the filter above the chain returns."""
    var plan = (
        table(_left())
        .join(table(_third()), [0], [0])
        .join(table(_right()), [0], [0], JOIN_LEFT)
        .filter(col("lv", int64) > col("tv", int64) + lit(15, int64))
    )
    var absorbed = PushFilterIntoJoin.apply(plan)
    ref chain = absorbed.get[JoinChain]()
    assert_equal(len(chain.filters), 1, String(absorbed))
    assert_equal(_landing(chain), len(chain.inputs), String(absorbed))
    assert_equal(plan.execute().num_rows(), 1)
    assert_true(
        _canonical_rows(plan.execute()) == _canonical_rows(absorbed.execute())
    )


def test_a_chain_answering_with_no_column_emits_none() raises:
    """An empty output is a valid projection — every row, no column — and the
    root join must emit exactly that rather than everything it joined."""
    var plan = table(_left()).join(table(_right()), [0], [0])
    var empty = plan.with_chain(plan.get[JoinChain]().with_output([], []))
    assert_equal(len(empty.schema().fields), 0)
    var out = empty.execute()
    assert_true(empty.schema() == out.schema, String(out.schema))
    assert_equal(out.num_columns(), 0)


# ---------------------------------------------------------------------------
# The rest of a chain's contract, one claim each
# ---------------------------------------------------------------------------
def test_a_clash_rname_leaves_raises_at_the_join() raises:
    """A self-join with `rname=""` renames nothing, so both `lv`s would keep
    their name: refused at the join, before anything can read either."""
    with assert_raises(contains="two columns would be named 'lv'"):
        _ = table(_left()).join(table(_left()), [0], [0], rname="")


def test_an_existence_join_has_no_clash_to_refuse() raises:
    """SEMI and ANTI emit only the left side and RIGHT_SEMI and RIGHT_ANTI
    only the right, so two sides sharing every name join without renaming."""
    var semi = table(_left()).join(
        table(_left()), [0], [0], JOIN_SEMI, rname=""
    )
    assert_equal(_names(semi), ["k", "lv"])
    var anti = table(_left()).join(
        table(_third()), [0], [0], JOIN_ANTI, rname=""
    )
    assert_equal(_names(anti), ["k", "lv"])
    var right_semi = table(_left()).join(
        table(_third()), [0], [0], JOIN_RIGHT_SEMI, rname=""
    )
    assert_equal(_names(right_semi), ["k", "tv"])
    var right_anti = table(_left()).join(
        table(_third()), [0], [0], JOIN_RIGHT_ANTI, rname=""
    )
    assert_equal(_names(right_anti), ["k", "tv"])


def test_column_pruning_narrows_each_participant_to_what_the_chain_reads() raises:
    """A self-join: both participants call their columns `k` and `lv`. Asked
    for `k` alone, the first keeps its key; the second keeps its key and the
    `lv` an absorbed filter reads — by participant, though the names are the
    same."""
    var plan = PushFilterIntoJoin.apply(
        table(_left())
        .join(table(_left()), [0], [0])
        .filter(col("lv_right", int64) > lit(15, int64))
    )
    var pruned = ColumnPruning.apply(plan, ["k"])
    ref chain = pruned.get[JoinChain]()
    assert_equal(_names(pruned), ["k"])
    assert_equal(chain.inputs[0][].schema().names(), ["k"])
    assert_equal(chain.inputs[1][].schema().names(), ["k", "lv"])
    assert_equal(pruned.execute().num_rows(), 2)

    # Asked for nothing, a chain still answers with one column: a batch
    # carries its row count in its columns.
    var none = ColumnPruning.apply(
        table(_left()).join(table(_right()), [0], [0]), ["x"]
    )
    assert_equal(_names(none), ["k"])
    assert_equal(none.execute().num_rows(), 2)


def _absorbed(plan: DynRelation) raises -> Optional[JoinChain]:
    """What `plan`, a filter over a chain, absorbs into the chain, or `None`
    when the filter stays above."""
    var out = PushFilterIntoJoin.apply(plan)
    if not out.isa[JoinChain]():
        return None
    return out.get[JoinChain]().copy()


def test_a_filter_moves_into_a_participant_only_where_no_join_pads_it() raises:
    """Per kind: the side a join pads with NULLs keeps its filter above the
    join — absorbed by the chain (`-1`), landing no lower than the join — and
    the other side takes it."""

    def pushed(k: JoinKind, on_left: Bool) raises -> Optional[Int]:
        var predicate: DynValue
        if on_left:
            predicate = DynValue(col("lv", int64) > lit(15, int64))
        else:
            predicate = DynValue(col("rv", int64) > lit(250, int64))
        var plan = table(_left()).join(table(_right()), [0], [0], k)
        var got = _absorbed(plan.filter(predicate^))
        if not got:
            return None
        for p in range(2):
            if got.value().inputs[p][].isa[Filter]():
                return p
        return -1

    # RIGHT pads the left side; FULL pads both; LEFT pads the right.
    assert_equal(pushed(JOIN_RIGHT, True).value(), -1)
    assert_equal(pushed(JOIN_RIGHT, False).value(), 1)
    assert_equal(pushed(JOIN_FULL, True).value(), -1)
    assert_equal(pushed(JOIN_FULL, False).value(), -1)
    assert_equal(pushed(JOIN_LEFT, True).value(), 0)
    assert_equal(pushed(JOIN_LEFT, False).value(), -1)
    # SEMI emits only the left side, which it never pads.
    assert_equal(pushed(JOIN_SEMI, True).value(), 0)

    # Under a LEFT join an inner participant still takes its own filter, and
    # one spanning two on its spine lands on the inner join below it.
    var mixed = (
        table(_left())
        .join(table(_right()), [0], [0])
        .join(table(_third()), [0], [0], JOIN_LEFT)
    )
    var own = _absorbed(mixed.filter(col("rv", int64) > lit(250, int64)))
    assert_true(own.value().inputs[1][].isa[Filter]())
    var spanning = _absorbed(
        mixed.filter(col("lv", int64) + col("rv", int64) > lit(250, int64))
    )
    assert_equal(_landing(spanning.value()), 3)


def test_a_spliced_chain_keeps_its_filters_and_names() raises:
    """`MergeJoinChains` over a nested chain that has absorbed a filter, and
    one answering with a renamed column — each the same rows and schema
    spliced as nested. One with an outer join stays nested."""
    var filtered = PushFilterIntoJoin.apply(
        table(_right())
        .join(table(_third()), [0], [0])
        .filter(col("rv", int64) + col("tv", int64) > lit(300, int64))
    )
    var nested_filter = table(_left()).join(filtered^, [0], [0])
    var renamed = table(_left()).join(table(_left()), [0], [0])
    var nested_renamed = table(_right()).join(renamed^, [0], [0])

    var plans: List[DynRelation] = [nested_filter^, nested_renamed^]
    for ref plan in plans:
        var spliced = MergeJoinChains.apply(plan)
        ref chain = spliced.get[JoinChain]()
        assert_equal(len(chain.inputs), 3, String(spliced))
        assert_true(spliced.schema() == plan.schema(), String(spliced))
        assert_true(
            _canonical_rows(spliced.execute())
            == _canonical_rows(plan.execute()),
            String(spliced),
        )
    # The filter came along, reading the participants it was written over.
    ref first = MergeJoinChains.apply(plans[0]).get[JoinChain]()
    assert_equal(len(first.filters), 1)
    assert_true(first.filters[0].participants() == {1, 2})

    var outer = table(_right()).join(table(_third()), [0], [0], JOIN_LEFT)
    var nested_outer = table(_left()).join(outer^, [0], [0])
    var kept = MergeJoinChains.apply(nested_outer)
    assert_equal(len(kept.get[JoinChain]().inputs), 2, String(kept))


def test_select_operator_picks_a_sliced_batch_by_position() raises:
    """The stage picking a chain's output after a filter on its root. A slice
    must select the slice's rows; compared by value, since array equality is
    structural and a slice keeps its offset."""
    var sliced = _batch().to_struct_array().slice(1, 2)
    var op = SelectOperator(
        [1, 0], schema([field("x", int64), field("a", int64)])
    )
    var got = op.push(Morsel.ungrouped(sliced^)).value().to_array(2)
    ref out = got.as_struct()
    assert_equal(out.dtype.as_struct().fields[0].name, "x")
    var x = out.field(0)
    assert_equal(Int(x.as_int64()[0].value()), 20)
    assert_equal(Int(x.as_int64()[1].value()), 30)
    var a = out.field(1)
    assert_equal(Int(a.as_int64()[0].value()), 2)
    assert_true(a.as_int64().is_null(1))


def test_renamed_predicate_reads_its_columns_by_position() raises:
    """A filter between joins reads columns by position under the names it
    was written with: `b` at index 1 is read as `x`."""
    var sliced = _batch().to_struct_array().slice(1, 2)
    var view = schema([field("x", int64)])
    var predicate: DynValue = col("x", int64) > lit(25, int64)
    var op = RenamedPredicate(
        predicate.to_operator(view, False, Bindings()), [1], view.copy()
    )
    var mask = op.push(Morsel.ungrouped(sliced^)).value().to_array(2)
    ref bits = mask.as_bool()
    assert_true(not bits[0].value())
    assert_true(bits[1].value())


def test_a_filter_never_moves_below_an_outer_join() raises:
    """`tv_right` is the nested chain's `tv`, from the right side of its LEFT
    join, read under a renamed name — so the outer chain absorbs the filter
    rather than pushing it, and the nested chain stays one participant, so
    the filter cannot land below the join that pads `tv`. Above the join, a
    padded row's NULL fails the filter; below, it would survive."""
    var nested = table(_right()).join(table(_third()), [0], [0], JOIN_LEFT)
    var plan = (
        table(_third())
        .join(nested^, [0], [0])
        .filter(col("tv_right", int64) > lit(7, int64))
    )
    var absorbed = PushFilterIntoJoin.apply(plan)
    assert_equal(len(absorbed.get[JoinChain]().filters), 1, String(absorbed))
    var spliced = MergeJoinChains.apply(absorbed)
    ref chain = spliced.get[JoinChain]()
    assert_equal(len(chain.inputs), 2, String(spliced))
    assert_equal(_landing(chain), 1, String(spliced))
    assert_equal(plan.execute().num_rows(), 1)
    assert_equal(spliced.execute().num_rows(), 1)
