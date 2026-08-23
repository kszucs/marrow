"""Plan nodes and the processors they become.

`logical.mojo` and `physical.mojo` are covered together because neither is
observable alone: a `Relation` is a description, so the only way to ask whether
it described the right thing is to run the `Processor` it builds. Each test
therefore states a claim about the plan and checks it against the rows.

The recurring failure these guard is a **schema that disagrees with the data** —
a `Project` whose declared fields do not match the columns it emits, or an empty
result whose schema names fields it has no columns for. Both run fine and
corrupt whatever reads them by index.
"""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import DynType, Int64Type, float64, int64
from ...execution import ExecContext
from ...kernels.join import JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from ...dtypes import field
from ...schema import schema
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import Datum
from ..physical import (
    AggregateOperator,
    BatchSourceOperator,
    Pipeline,
    DynOperator,
    FilterOperator,
    LimitOperator,
    Morsel,
)
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..logical import (
    Aggregate,
    Join,
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
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(_batch())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](2))),
        )
    )
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
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(_batch())),
            # every non-null row passes, so only the null's treatment shows
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](-100))),
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 1, 2, 4 — the null is not selected


def test_filter_preserves_its_input_schema() raises:
    """A filter changes which rows survive, never which columns exist."""
    var b = _batch()
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(b.copy())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](2))),
        )
    )
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
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(b.copy())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](999))),
        )
    )
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
    column does. `expr/` records that divergence with `nullable` False
    becoming True.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    # `a` is non-nullable here; the projection must not widen it.
    var src = b.schema.fields[0].nullable

    var p = Project(
        DynRelation(InMemoryTable(b.copy())),
        ["out"],
        [DynValue(Column[Int64Type]("a"))],
    )
    assert_equal(p.schema().fields[0].nullable, src)
    assert_true(p.schema().fields[0].dtype == DynType(int64))


def test_project_names_a_computed_column_from_its_dtype() raises:
    """A computed value has no `Field` to carry, so `dtype()` answers instead.

    This is `dtype()`'s reason to exist: the schema must be known *before*
    anything runs, and `expr/` got it by evaluating against a zero-row batch.
    """
    var b = _batch()
    var p = Project(
        DynRelation(InMemoryTable(b.copy())),
        ["sum"],
        [DynValue(Add(Column[Int64Type]("a"), Column[Int64Type]("b")))],
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
    var plan = DynRelation(
        Project(
            DynRelation(InMemoryTable(b.copy())),
            ["sum", "orig"],
            [
                DynValue(Add(Column[Int64Type]("a"), Column[Int64Type]("b"))),
                DynValue(Column[Int64Type]("a")),
            ],
        )
    )
    var out = plan.execute()
    assert_true(out.schema == plan.schema())
    assert_equal(out.num_columns(), 2)
    assert_equal(out.num_rows(), b.num_rows())


def test_project_rejects_mismatched_names_and_values() raises:
    """Two parallel lists that must agree, checked where they are supplied."""
    var raised = False
    try:
        _ = Project(
            DynRelation(InMemoryTable(_batch())),
            ["only_one"],
            [
                DynValue(Column[Int64Type]("a")),
                DynValue(Column[Int64Type]("b")),
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
    var plan = DynRelation(
        Aggregate(
            DynRelation(InMemoryTable(_batch())),
            List[DynValue](),
            [DynValue(Sum(Column[Int64Type]("a"), "total"))],
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_equal(out.num_columns(), 1)
    # a = [1, 2, None, 4] — the null contributes nothing, it is not a zero.
    assert_true(out.columns[0].as_int64() == array([7], int64))


def test_an_aggregate_groups_by_its_key() raises:
    """Group ids are dense and assigned in first-appearance order, so the keys
    come back in the order they were first seen, not sorted."""
    var plan = DynRelation(
        Aggregate(
            DynRelation(InMemoryTable(_keyed())),
            [DynValue(Column[Int64Type]("g"))],
            [DynValue(Sum(Column[Int64Type]("a"), "total"))],
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[0].as_int64() == array([1, 2], int64))
    assert_true(out.columns[1].as_int64() == array([40, 60], int64))


def test_aggregate_schema_is_keys_then_aggregates() raises:
    """The ordering every consumer depends on, asserted where it is decided.

    The processor reads its key fields straight off the front of this schema,
    so a change that appended keys last would mis-type the grouper rather than
    fail loudly.
    """
    var plan = Aggregate(
        DynRelation(InMemoryTable(_keyed())),
        [DynValue(Column[Int64Type]("g"))],
        [
            DynValue(Sum(Column[Int64Type]("a"), "total")),
            DynValue(Min(Column[Int64Type]("a"), "smallest")),
        ],
    )
    var s = plan.schema()
    assert_equal(len(s.fields), 3)
    assert_equal(s.fields[0].name, "g")
    assert_equal(s.fields[1].name, "total")
    assert_equal(s.fields[2].name, "smallest")
    assert_true(s.fields[0].dtype == DynType(int64))
    assert_true(plan.schema() == DynRelation(plan^).execute().schema)


def test_a_computed_key_is_named_by_position() raises:
    """A bare column keeps its name; anything computed has none.

    `expr/` shipped a defect where one lane answered `d` and the other `key0`
    for the same `GROUP BY d`, giving one query two output schemas.
    """
    var plan = Aggregate(
        DynRelation(InMemoryTable(_keyed())),
        [DynValue(Add(Column[Int64Type]("g"), Column[Int64Type]("a")))],
        [DynValue(Sum(Column[Int64Type]("a"), "total"))],
    )
    assert_equal(plan.schema().fields[0].name, "key0")


def test_an_aggregate_over_no_rows_answers_one_null() raises:
    """`sum` of nothing is NULL, not 0 — and the input here yields *no morsel*.

    This is the case that made `AggState.finish` grow before reading: only
    `update` ever extended the builders, so a fold that saw zero batches read
    unallocated slots. It aborts under `ASSERT=all` and is a silent bad read in
    release.
    """
    var plan = DynRelation(
        Aggregate(
            DynRelation(
                Filter(
                    DynRelation(InMemoryTable(_batch())),
                    DynValue(
                        Gt(Column[Int64Type]("a"), Literal[Int64Type](999))
                    ),
                )
            ),
            List[DynValue](),
            [DynValue(Sum(Column[Int64Type]("a"), "total"))],
        )
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
    var plan = DynRelation(
        Aggregate(
            DynRelation(InMemoryTable(_batch())),
            List[DynValue](),
            [
                DynValue(
                    Sum(
                        Add(Column[Int64Type]("a"), Column[Int64Type]("b")),
                        "total",
                    )
                )
            ],
        )
    )
    # a = [1, 2, None, 4], b = [10, 20, 30, 40] -> 11 + 22 + 44, the null row
    # propagating through the addition rather than contributing b alone.
    assert_true(plan.execute().columns[0].as_int64() == array([77], int64))


def test_having_is_a_filter_above_an_aggregate() raises:
    """`HAVING` needs no node of its own.

    A `Filter` above the aggregate sees the aggregate's *output* batch, so the
    predicate reads the aggregate's output column by name.
    """
    var plan = DynRelation(
        Filter(
            DynRelation(
                Aggregate(
                    DynRelation(InMemoryTable(_keyed())),
                    [DynValue(Column[Int64Type]("g"))],
                    [DynValue(Sum(Column[Int64Type]("a"), "total"))],
                )
            ),
            DynValue(Gt(Column[Int64Type]("total"), Literal[Int64Type](50))),
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)  # group 1 totals 40, group 2 totals 60
    assert_true(out.columns[0].as_int64() == array([2], int64))


# ---------------------------------------------------------------------------
# The push engine
# ---------------------------------------------------------------------------
def test_a_streaming_operator_answers_from_push() raises:
    """`Filter` produces per morsel and has nothing to flush."""
    var op = FilterOperator(
        DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](2))).to_operator(
            False
        ),
        ExecContext.serial(),
    )
    var out = op.push(Morsel.ungrouped(_batch()))
    assert_true(Bool(out))
    assert_equal(out.value().num_rows(), 1)
    assert_true(not Bool(op.drain()))


def test_a_blocking_operator_answers_nothing_until_finish() raises:
    """The distinction the push interface replaces a *type* with.

    An aggregate accumulates through every `push` and answers `None`; its whole
    result arrives from `finish`. Under the pull design this needed a different
    trait from `Filter` and therefore a second erased box — collapsing that is
    the point of the engine.
    """
    var folds = List[DynOperator[Datum]]()
    folds.append(Sum(Column[Int64Type]("a"), "total").to_operator(False))
    var op = AggregateOperator(
        List[DynOperator[Datum]](),
        folds^,
        schema([field("total", int64)]),
        ExecContext.serial(),
    )
    assert_true(not Bool(op.push(Morsel.ungrouped(_batch()))))

    var out = op.drain()
    assert_true(Bool(out))
    assert_true(out.value().columns[0].as_int64() == array([7], int64))
    # Emitting the fold twice would double every grouped result downstream.
    assert_true(not Bool(op.drain()))


def test_the_flush_cascade_feeds_the_stages_above() raises:
    """An aggregate's result must still pass through everything above it.

    `finish` on stage *i* produces a batch no later stage has ever seen, so it
    has to be pushed through *i+1..* before stage *i+1* is itself finished. A
    projection over an aggregate is the smallest query that returns nothing at
    all if the flush is a plain loop of independent `finish` calls.
    """
    var plan = DynRelation(
        Project(
            DynRelation(
                Aggregate(
                    DynRelation(InMemoryTable(_keyed())),
                    [DynValue(Column[Int64Type]("g"))],
                    [DynValue(Sum(Column[Int64Type]("a"), "total"))],
                )
            ),
            ["doubled"],
            [
                DynValue(
                    Add(
                        Column[Int64Type]("total"),
                        Column[Int64Type]("total"),
                    )
                )
            ],
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
    var plan = DynRelation(
        Limit(DynRelation(InMemoryTable(_batch())), offset=0, length=2)
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_int64() == array([10, 20], int64))


def test_limit_skips_the_offset() raises:
    var plan = DynRelation(
        Limit(DynRelation(InMemoryTable(_batch())), offset=2, length=2)
    )
    var out = plan.execute()
    assert_true(out.columns[1].as_int64() == array([30, 40], int64))


def test_limit_reports_done_so_the_source_can_stop() raises:
    """The signal a push engine needs and a pull engine gets for free.

    Without it a `LIMIT 10` over a billion-row scan reads a billion rows: the
    source drives, so nothing downstream could otherwise halt it.
    """
    var op = LimitOperator(offset=0, length=2)
    assert_true(not op.done())
    var out = op.push(Morsel.ungrouped(_batch()))
    assert_true(Bool(out))
    assert_equal(out.value().num_rows(), 2)
    assert_true(op.done())


def test_limit_preserves_its_input_schema() raises:
    var b = _batch()
    var plan = DynRelation(
        Limit(DynRelation(InMemoryTable(b.copy())), offset=0, length=1)
    )
    assert_true(plan.schema() == b.schema)


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------
def test_sort_orders_by_one_key() raises:
    var b = record_batch([array([3, 1, 2], int64).copy()], names=["a"])
    var plan = DynRelation(
        Sort(
            DynRelation(InMemoryTable(b^)),
            [DynValue(Column[Int64Type]("a"))],
            [True],
        )
    )
    assert_true(plan.execute().columns[0].as_int64() == array([1, 2, 3], int64))


def test_sort_descending() raises:
    var b = record_batch([array([3, 1, 2], int64).copy()], names=["a"])
    var plan = DynRelation(
        Sort(
            DynRelation(InMemoryTable(b^)),
            [DynValue(Column[Int64Type]("a"))],
            [False],
        )
    )
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
    var plan = DynRelation(
        Sort(
            DynRelation(InMemoryTable(b^)),
            [
                DynValue(Column[Int64Type]("a")),
                DynValue(Column[Int64Type]("b")),
            ],
            [True, True],
        )
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([1, 1, 2, 2], int64))
    assert_true(out.columns[1].as_int64() == array([30, 40, 10, 20], int64))


def test_sort_rejects_mismatched_keys_and_directions() raises:
    var raised = False
    try:
        _ = Sort(
            DynRelation(InMemoryTable(_batch())),
            [DynValue(Column[Int64Type]("a"))],
            [True, False],
        )
    except e:
        raised = True
        assert_true("sort" in String(e))
    assert_true(raised)


def test_sort_then_limit_is_top_n() raises:
    """The composition every engine needs, and the one that proves a blocking
    stage and a bounded stage cooperate: `Sort` emits only from `finish`, and
    the `Limit` above it must still see that batch through the flush cascade.
    """
    var b = record_batch([array([5, 3, 9, 1], int64).copy()], names=["a"])
    var plan = DynRelation(
        Limit(
            DynRelation(
                Sort(
                    DynRelation(InMemoryTable(b^)),
                    [DynValue(Column[Int64Type]("a"))],
                    [True],
                )
            ),
            offset=0,
            length=2,
        )
    )
    assert_true(plan.execute().columns[0].as_int64() == array([1, 3], int64))


# ---------------------------------------------------------------------------
# Pipeline — a composite Operator
# ---------------------------------------------------------------------------
def test_a_pipeline_drains_until_spent() raises:
    """`drain` is resumable and must eventually answer `None`.

    The driver loops on it, so a pipeline that never reported spent would hang.
    Asserted directly because `collect` hides it: `collect` stops at the first
    `None` and would look correct even if the second call answered `Some`
    again.
    """
    var pipe = Pipeline(BatchSourceOperator(_batch()))
    var first = pipe.drain()
    assert_true(Bool(first))
    assert_equal(first.value().num_rows(), 4)
    assert_true(not Bool(pipe.drain()))
    assert_true(not Bool(pipe.drain()))


def test_a_pipeline_cascades_through_its_stages() raises:
    """A batch leaving stage `i` must be seen by `i+1..` before `i+1` drains.

    Here the source's batch has to reach the filter, which is the same cascade
    an aggregate under a projection depends on — checked at the `Pipeline`
    level rather than only through `execute`.
    """
    var pipe = Pipeline(BatchSourceOperator(_batch()))
    pipe.append(
        FilterOperator(
            DynValue(
                Gt(Column[Int64Type]("a"), Literal[Int64Type](2))
            ).to_operator(False),
            ExecContext.serial(),
        )
    )
    var got = pipe.drain()
    assert_true(Bool(got))
    assert_equal(got.value().num_rows(), 1)  # a = [1, 2, None, 4] -> only 4
    assert_true(not Bool(pipe.drain()))


def test_a_pipeline_is_an_operator() raises:
    """It is a composite, not a second abstraction — so it boxes like any
    other stage. This is what lets `Join` hold two whole sub-plans."""
    var pipe = Pipeline(BatchSourceOperator(_batch()))
    var boxed = DynOperator[RecordBatch](pipe^)
    var got = boxed.drain()
    assert_true(Bool(got))
    assert_equal(got.value().num_rows(), 4)


# ---------------------------------------------------------------------------
# Join — the operator with two inputs
# ---------------------------------------------------------------------------
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


def test_an_inner_join_streams_the_probe_side() raises:
    """Keys 2 and 3 match; 1 and 4 do not."""
    var plan = DynRelation(
        Join(
            DynRelation(InMemoryTable(_left())),
            DynRelation(InMemoryTable(_right())),
            [0],
            [0],
            JOIN_INNER,
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_equal(out.num_columns(), 4)  # left k, lv + right k, rv
    assert_true(out.columns[1].as_int64() == array([20, 30], int64))
    assert_true(out.columns[3].as_int64() == array([200, 300], int64))


def test_join_schema_is_left_then_right() raises:
    var plan = Join(
        DynRelation(InMemoryTable(_left())),
        DynRelation(InMemoryTable(_right())),
        [0],
        [0],
        JOIN_INNER,
    )
    var s = plan.schema()
    assert_equal(len(s.fields), 4)
    assert_equal(s.fields[1].name, "lv")
    assert_equal(s.fields[3].name, "rv")
    assert_true(plan.schema() == DynRelation(plan^).execute().schema)


def test_a_semi_join_emits_only_the_left_side() raises:
    """`SEMI` answers "which left rows matched", so the right side contributes
    no columns — the schema rule and the kernel must agree on that."""
    var plan = Join(
        DynRelation(InMemoryTable(_left())),
        DynRelation(InMemoryTable(_right())),
        [0],
        [0],
        JOIN_SEMI,
    )
    assert_equal(len(plan.schema().fields), 2)
    var out = DynRelation(plan^).execute()
    assert_equal(out.num_columns(), 2)
    assert_equal(out.num_rows(), 2)  # left keys 2 and 3 matched


def test_a_left_join_keeps_unmatched_build_rows_once() raises:
    """The reason LEFT buffers the probe side instead of streaming it.

    Its tail of unmatched build rows is a property of *every* probe row taken
    together. Probing morsel-by-morsel would re-emit that tail once per morsel;
    key 1 must appear exactly once.
    """
    var plan = DynRelation(
        Join(
            DynRelation(InMemoryTable(_left())),
            DynRelation(InMemoryTable(_right())),
            [0],
            [0],
            JOIN_LEFT,
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 2 and 3 matched, 1 null-widened once


def test_join_rejects_mismatched_key_counts() raises:
    var raised = False
    try:
        _ = Join(
            DynRelation(InMemoryTable(_left())),
            DynRelation(InMemoryTable(_right())),
            [0],
            [0, 1],
            JOIN_INNER,
        )
    except e:
        raised = True
        assert_true("join" in String(e))
    assert_true(raised)


def test_a_join_composes_with_a_filter_above_it() raises:
    """The build side is a whole sub-plan and the probe side is a pipeline, so
    a join has to sit in a chain like any other stage."""
    var plan = DynRelation(
        Filter(
            DynRelation(
                Join(
                    DynRelation(InMemoryTable(_left())),
                    DynRelation(InMemoryTable(_right())),
                    [0],
                    [0],
                    JOIN_INNER,
                )
            ),
            DynValue(Gt(Column[Int64Type]("lv"), Literal[Int64Type](25))),
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)  # lv = 30
