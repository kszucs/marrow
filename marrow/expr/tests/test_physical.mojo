# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The physical contract — `Operator`, `Morsel`, `Pipeline`.

These are the tests the plan-building API deliberately *cannot* express.
Everything reachable through `table(...).filter(...)` lives in
`test_relations.mojo`; what lives here is the executor contract itself: that a
streaming stage answers from `push` and a blocking one from `drain`, that
`drain` is repeatable, that `done` lets a source stop early, and that a
`Pipeline` holds what a blocking stage absorbed until it is drained.

Written against the operators on purpose. `b128b3e` added the `Pipeline`
coverage precisely because it was reachable only indirectly, and a fold that
could not report "spent" was a latent infinite loop that no plan-level test
would have caught.
"""

from std.os.path import join
from std.testing import assert_equal, assert_raises, assert_true
from ...arrays import StructArray, DynArray
from ...builders import array
from ...dtypes import DynType, Int64Type, float64, int64
from ...execution import ExecContext
from ...kernels.join import JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from ...dtypes import field
from ...schema import schema
from ...io import FileSink
from ...formats.parquet.writer import Compression, FileWriter, write_table
from ...utils.stats import STATS_ENABLED, counter
from ...utils.testing import ScratchDir
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import (
    Datum,
    DynOperator,
    Morsel,
    Pipeline,
    SortOperator,
)
from ...engine import Driver, Phase, PlanReport
from ..builders import scan
from ..`comptime`.leaves import NumericColumn, NumericLiteral
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..builders import col, count_star, lit, row_number, table
from ..logical import DynValue, DynRelation


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def _feed(plan: DynRelation, mut out: List[StructArray]) raises -> List[Phase]:
    """`plan`'s one phase, its pipeline fed the phase's one morsel, and in
    `out` what it answered: nothing drained yet."""
    var driver = Driver.serial(ExecContext.serial())
    var phases = plan.lower(driver)
    assert_equal(len(phases), 1)
    var morsel = phases[0].source.read(0, ExecContext.serial())
    phases[0].pipelines[0].feed(morsel.pop(0), out)
    return phases^


def test_a_streaming_operator_answers_from_push() raises:
    """`Filter` produces per morsel and has nothing to flush."""
    var out = List[StructArray]()
    var phases = _feed(
        table(_batch()).filter(col("a", int64) > lit(2, int64)), out
    )
    assert_equal(len(out), 1)
    assert_equal(len(out[0]), 1)
    assert_true(not Bool(phases[0].pipelines[0].drain()))


def test_a_blocking_operator_answers_nothing_until_finish() raises:
    """The distinction the push interface replaces a *type* with.

    An aggregate accumulates through every `push` and answers `None`; its whole
    result arrives from `drain`. Under the pull design this needed a different
    trait from `Filter` and therefore a second erased box — collapsing that is
    the point of the engine.
    """
    var fed = List[StructArray]()
    var phases = _feed(
        table(_batch()).aggregate([col("a", int64).sum().alias("total")]), fed
    )
    assert_equal(len(fed), 0)
    var out = phases[0].pipelines[0].drain()
    assert_true(Bool(out))
    assert_true(
        out.value().struct_array().children[0].as_int64() == array([7], int64)
    )
    # Emitting the fold twice would double every grouped result downstream.
    assert_true(not Bool(phases[0].pipelines[0].drain()))


def test_limit_reports_done_so_the_source_can_stop() raises:
    """The signal a push engine needs and a pull engine gets for free.

    Without it a `LIMIT 10` over a billion-row scan reads a billion rows: the
    driver reads morsels until a stage reports it has all it needs.
    """
    var out = List[StructArray]()
    var phases = _feed(table(_batch()).limit(length=2, offset=0), out)
    assert_equal(len(out), 1)
    assert_equal(len(out[0]), 2)
    assert_true(phases[0].pipelines[0].done())


def test_sort_then_limit_is_top_n() raises:
    """The composition every engine needs, and the one that proves a blocking
    stage and a bounded stage cooperate: `Sort` emits only from `finish`, and
    the `Limit` above it must still see that batch through the flush cascade.
    """
    var b = record_batch([array([5, 3, 9, 1], int64).copy()], names=["a"])
    var plan = (
        table(b^).sort_by([col("a", int64)], [True]).limit(length=2, offset=0)
    )
    assert_true(plan.execute().columns[0].as_int64() == array([1, 3], int64))


# ---------------------------------------------------------------------------
# Pipeline — one worker's chain of operators
# ---------------------------------------------------------------------------


def test_a_pipeline_drains_until_spent() raises:
    """`drain` is resumable and must eventually answer `None`.

    The driver loops on it, so a pipeline that never reported spent would hang.
    Asserted directly because `collect` hides it: `collect` stops at the first
    `None` and would look correct even if the second call answered `Some`
    again.
    """
    var pipe = Pipeline()
    pipe.append(
        SortOperator(
            List[DynOperator](), List[Bool](), True, None, ExecContext.serial()
        )
    )
    var fed = List[StructArray]()
    pipe.feed(_batch().to_struct_array(), fed)
    assert_equal(len(fed), 0, "a blocking stage keeps what it is fed")
    var first = pipe.drain()
    assert_true(Bool(first))
    assert_equal(len(first.value().struct_array()), 4)
    assert_true(not Bool(pipe.drain()))
    assert_true(not Bool(pipe.drain()))


def test_a_pipeline_cascades_through_its_stages() raises:
    """A batch leaving stage `i` must be seen by `i+1..` before `i+1` drains.

    Here the source's batch has to reach the filter, which is the same cascade
    an aggregate under a projection depends on.
    """
    var driver = Driver.serial(ExecContext.serial())
    var plan = table(_batch()).filter(col("a", int64) > lit(2, int64))
    var got = driver.run(plan.lower(driver))
    assert_equal(len(got), 1)
    assert_equal(len(got[0]), 1)  # a = [1, 2, None, 4] -> only 4


# ---------------------------------------------------------------------------
# Inspecting a plan — explain and analyze
# ---------------------------------------------------------------------------


def test_explain_reports_the_phase_and_runs_nothing() raises:
    var plan = (
        table(_batch())
        .filter(col("a", int64) > lit(2, int64))
        .aggregate([col("b", int64).sum().alias("total")])
    )
    var report = plan.explain()
    assert_equal(len(report.phases), 1)
    ref phase = report.phases[0]
    assert_equal(phase.role, "result")
    assert_true(phase.plan().startswith("TableSource(4 rows) -> Filter"))
    assert_true(phase.plan().endswith(" -> Aggregate(sum(col(b)) as total)"))
    assert_equal(phase.morsels, 1)
    assert_equal(phase.rows, 4)
    assert_equal(phase.ns, -1, "nothing ran")


def test_explain_shows_a_join_build_before_its_probe() raises:
    var right = record_batch(
        [array([2, 4], int64).copy(), array([20, 40], int64).copy()],
        names=["k", "w"],
    )
    var plan = table(_batch()).join(table(right^), [0], [0])
    var report = plan.explain()
    assert_equal(len(report.phases), 2)
    assert_equal(report.phases[0].role, "join build")
    assert_equal(report.phases[1].role, "result")
    assert_true("-> Join" in report.phases[1].plan())


def test_explain_counts_row_groups_from_the_footer() raises:
    """A Parquet scan's morsels are its row groups, counted from the footer
    while the plan lowers; explaining decodes none of them."""
    with ScratchDir() as dir:
        var path = join(dir, "rows.parquet")
        var batch = _batch()
        var writer = FileWriter(
            FileSink(path), batch.schema, Compression.SNAPPY, row_group_size=2
        )
        writer.write(Table.from_batches(batch.schema, [batch.copy()]))
        writer.close()
        var plan = scan(path, batch.schema).filter(
            col("a", int64) > lit(2, int64)
        )
        var report = plan.explain()
        assert_equal(len(report.phases), 1)
        assert_true(report.phases[0].plan().startswith("ParquetScan("))
        assert_equal(report.phases[0].morsels, 2)
        assert_equal(report.phases[0].rows, 4)
        assert_equal(report.phases[0].ns, -1, "nothing ran")


def test_explain_names_a_missing_file() raises:
    var plan = scan(
        "/nonexistent/marrow-explain.parquet", schema([field("a", int64)])
    )
    with assert_raises(contains="marrow-explain.parquet"):
        _ = plan.explain()


def test_analyze_reports_what_ran() raises:
    var plan = table(_batch()).filter(col("a", int64) > lit(1, int64))
    var report = plan.explain[analyze=True]()
    assert_equal(len(report.phases), 1)
    assert_equal(report.phases[0].output, plan.execute().num_rows())
    assert_equal(report.phases[0].morsels, 1)
    assert_true(report.phases[0].ns >= 0)


def _six() raises -> RecordBatch:
    """Six rows, so two-row morsels give three of them."""
    return record_batch(
        [
            array([1, 2, None, 4, 5, 6], int64).copy(),
            array([10, 20, 30, 40, 50, 60], int64).copy(),
        ],
        names=["a", "b"],
    )


def _report(plan: DynRelation, analyze: Bool = False) raises -> PlanReport:
    """`plan`'s report on four workers and two-row morsels, so a few rows
    show what many would."""
    var ctx = ExecContext.parallel(4)
    var driver = Driver.analyze(ctx, 2) if analyze else Driver.explain(ctx, 2)
    return driver.inspect(plan)


def test_explain_says_why_a_phase_has_its_workers() raises:
    var plan = table(_six()).filter(col("a", int64) > lit(2, int64)).select("b")
    assert_equal(
        String(_report(plan)),
        """Phase 1 [result] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    -> Filter(greater(col(a), lit(2)))
    -> Project(b)""",
    )
    var serial = Driver.serial(ExecContext.parallel(4))
    var phases = plan.lower(serial)
    assert_equal(phases[0].why, "serial driver")


def test_explain_shows_the_exchange_an_aggregate_meets_at() raises:
    """The partial aggregates run on every worker; their states meet in an
    exchange the dry run did not fill, so it shows the estimate."""
    var plan = table(_six()).aggregate(
        [col("b", int64).sum().alias("total"), count_star().alias("n")],
        [col("a", int64)],
    )
    assert_equal(
        String(_report(plan)),
        """Phase 1 [aggregate] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    -> Aggregate(partial: sum(col(b)) as total, count(lit(1)) as n by col(a))
    ═══ read by Phase 2
Phase 2 [result] Exchange(~6 groups est)
    workers=4 morsels=16 rows=? est=6
    -> Aggregate(combine: sum(col(b)) as total, count(lit(1)) as n by col(a))""",
    )


def test_explain_shows_a_join_tail_reading_its_probe_phase() raises:
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    assert_equal(
        String(_report(plan)),
        """Phase 1 [join build] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    ═══ read by Phase 2
Phase 2 [result] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    -> Project(a)
    -> Join(left outer, built on the left)
    ═══ read by Phase 3
Phase 3 [result] Unmatched
    workers=1 (one morsel) morsels=1 rows=?""",
    )


def test_explain_estimates_the_rows_behind_a_barrier() raises:
    var plan = table(_six()).with_columns(
        ["rn"], [row_number().over(order_by=[col("b", int64)])]
    )
    var report = _report(plan)
    assert_equal(len(report.phases), 2)
    assert_equal(report.phases[1].source, "Barrier(~6 rows est)")
    assert_equal(report.phases[1].estimate, 6)
    assert_equal(len(report.phases[1].inputs), 1)
    assert_equal(report.phases[1].inputs[0], 1)


def test_analyze_counts_the_morsels_of_a_phase_that_emits_nothing() raises:
    var plan = table(_six()).filter(col("a", int64) > lit(100, int64))
    var report = _report(plan, analyze=True)
    assert_equal(report.phases[0].morsels, 3)
    assert_equal(report.phases[0].output, 0)
    assert_true(report.phases[0].ns >= 0)


def test_a_report_renders_as_json_and_as_a_batch() raises:
    var report = _report(table(_six()).filter(col("a", int64) > lit(2, int64)))
    var json = report.to_json()
    assert_true(json.startswith('{"ns":-1,"rows":-1,"estimate":'))
    assert_true(
        '"stages":[{"label":"Filter(greater(col(a), lit(2)))"}],"workers":3,'
        '"why":"3 morsels","inputs":[],"morsels":3,"rows":6'
        in json
    )
    assert_true(
        json.endswith('"busy":[],"read_ns":-1,"counters":{}}],"spans":[]}')
    )
    var batch = report.to_batch()
    assert_equal(batch.num_rows(), 1)
    assert_equal(batch.schema.fields[3].name, "stages")
    assert_true(batch.schema.fields[3].dtype.is_list())


def test_analyze_counts_what_each_stage_did() raises:
    """Three two-row morsels of `a = [1, 2, None, 4, 5, 6]`: the filter sees
    all six rows and keeps three, and each worker reports its busy time."""
    var plan = table(_six()).filter(col("a", int64) > lit(2, int64))
    var report = _report(plan, analyze=True)
    assert_equal(report.rows, 3)
    ref phase = report.phases[0]
    assert_equal(len(phase.stats), 1)
    assert_equal(phase.stats[0].rows_in, 6)
    assert_equal(phase.stats[0].rows_out, 3)
    assert_equal(phase.stats[0].batches_in, 3)
    assert_equal(len(phase.busy), phase.workers)
    assert_true(phase.read_ns >= 0)
    assert_true(report.ns >= phase.ns)


def test_analyze_times_the_work_between_phases() raises:
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    var report = _report(plan, analyze=True)
    var names = List[String]()
    for ref s in report.spans:
        names.append(s.name)
    assert_true("hash build for Join(left outer)" in names)
    assert_true("concatenate the result" in names)
    var costs = report.costs()
    for i in range(1, len(costs)):
        assert_true(costs[i - 1][1] >= costs[i][1], "largest first")


def test_analyze_counts_the_work_between_phases() raises:
    """A join's hash build runs between phases: with `-D MARROW_STATS=true`
    its span carries what the build counted; without it, nothing."""
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    var report = _report(plan, analyze=True)
    var found = False
    for ref s in report.spans:
        if s.name == "hash build for Join(left outer)":
            found = True
            comptime if STATS_ENABLED:
                var inserts = counter(s.counters, "hash.inserts")
                assert_true(inserts >= 6, "every build row was encoded")
            else:
                assert_equal(len(s.counters), 0)
    assert_true(found)


def test_a_traced_run_is_a_timeline_per_thread() raises:
    """Three two-row morsels: each read and each push of the filter is one
    slice, each worker's drain one more, and the phase one more, on the
    calling thread."""
    var driver = Driver.analyze(ExecContext.parallel(4), 2, trace=True)
    var report = driver.inspect(
        table(_six()).filter(col("a", int64) > lit(2, int64))
    )
    var reads = 0
    var pushes = 0
    var phases = 0
    for ref e in report.events:
        if e.name.startswith("read morsel"):
            reads += 1
        elif e.name.startswith("Filter(") and not e.name.endswith(" drain"):
            pushes += 1
        elif e.name.startswith("Phase "):
            phases += 1
    assert_equal(reads, 3)
    assert_equal(pushes, 3)
    assert_equal(phases, 1)
    var trace = report.to_trace_json()
    assert_true(
        trace.startswith(
            '{"displayTimeUnit":"ms","traceEvents":[{"name":"thread_name"'
        )
    )
    assert_true('"name":"read morsel 0"' in trace)


def test_analyze_reports_the_kernel_counters() raises:
    """A grouped aggregate encodes its keys: with `-D MARROW_STATS=true` the
    phases that ran count the rows and the new keys; without it nothing is
    counted at all."""
    var report = _report(
        table(_six()).aggregate([count_star().alias("n")], [col("a", int64)]),
        analyze=True,
    )
    comptime if STATS_ENABLED:
        var inserts = 0
        var keys = 0
        for ref phase in report.phases:
            inserts += counter(phase.counters, "hash.inserts")
            keys += counter(phase.counters, "hash.new_keys")
        assert_true(inserts >= 6, "every input row was encoded")
        assert_true(keys >= 6, "six distinct keys, null among them")
        assert_true("counters:" in String(report))
    else:
        for ref phase in report.phases:
            assert_equal(len(phase.counters), 0)
