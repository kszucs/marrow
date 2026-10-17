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
from ...dtypes import DynType, Int64Type, float64, int64, string
from ...execution import ExecContext
from ...kernels.join import JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from ...dtypes import field
from ...schema import schema
from ...io import FileSink
from ...formats.parquet.writer import Compression, FileWriter, write_table
from ...utils.stats import STATS_ENABLED
from ...utils.testing import ScratchDir
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import SortOperator
from ...engine.core import Datum, DynOperator, Morsel, Pipeline
from ...engine import Driver, Phase, PlanProfile, PlanReport
from ..builders import scan
from ..`comptime`.leaves import NumericColumn, NumericLiteral
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..builders import col, count_star, lit, row_number, table
from ..logical import DynValue, DynRelation
from ..runtime.values import cast
from ...builders import Int64Builder, StringBuilder
from ...errors import DynError, InvalidError


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
    assert_true(phases[0].step(0, ExecContext.serial(), out))
    return phases^


def test_a_streaming_operator_answers_from_push() raises:
    """`Filter` produces per morsel and has nothing to flush."""
    var out = List[StructArray]()
    var phases = _feed(
        table(_batch()).filter(col("a", int64) > lit(2, int64)), out
    )
    assert_equal(len(out), 1)
    assert_equal(len(out[0]), 1)
    var rest = List[StructArray]()
    assert_true(not phases[0].step(1, ExecContext.serial(), rest))
    assert_equal(len(rest), 0)


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
    var out = List[StructArray]()
    assert_true(not phases[0].step(1, ExecContext.serial(), out))
    assert_equal(len(out), 1)
    assert_true(out[0].children[0].as_int64() == array([7], int64))
    # Emitting the fold twice would double every grouped result downstream.
    _ = phases[0].step(1, ExecContext.serial(), out)
    assert_equal(len(out), 1)


def test_limit_reports_done_so_the_source_can_stop() raises:
    """The signal a push engine needs and a pull engine gets for free.

    Without it a `LIMIT 10` over a billion-row scan reads a billion rows: the
    driver reads morsels until a stage reports it has all it needs.
    """
    var driver = Driver.dynamic(ExecContext.serial(), morsel_rows=1)
    var phases = table(_batch()).limit(length=2, offset=0).lower(driver)
    var out = List[StructArray]()
    assert_true(phases[0].step(0, ExecContext.serial(), out))
    assert_true(phases[0].step(1, ExecContext.serial(), out))
    assert_true(
        not phases[0].step(2, ExecContext.serial(), out),
        "two of four one-row morsels read",
    )
    assert_equal(len(out), 2)


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
    pipe.feed([_batch().to_struct_array()], fed)
    assert_equal(len(fed), 0, "a blocking stage keeps what it is fed")
    var out = List[StructArray]()
    pipe.drain(out)
    assert_equal(len(out), 1)
    assert_equal(len(out[0]), 4)
    pipe.drain(out)
    assert_equal(len(out), 1, "a drained stage has nothing left")


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
    assert_equal(phase.source, "TableSource(4 rows)")
    assert_true(phase.stages[0].startswith("Filter"))
    assert_equal(
        phase.stages[len(phase.stages) - 1], "Aggregate(sum(col(b)) as total)"
    )
    assert_equal(phase.morsels, 1)
    assert_equal(phase.rows, 4)


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
    assert_true(report.phases[1].stages[0].startswith("Join("))


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
        assert_true(report.phases[0].source.startswith("ParquetScan("))
        assert_equal(report.phases[0].morsels, 2)
        assert_equal(report.phases[0].rows, 4)


def test_explain_names_a_missing_file() raises:
    var plan = scan(
        "/nonexistent/marrow-explain.parquet", schema([field("a", int64)])
    )
    with assert_raises(contains="marrow-explain.parquet"):
        _ = plan.explain()


def test_profile_reports_what_ran() raises:
    var plan = table(_batch()).filter(col("a", int64) > lit(1, int64))
    var profile = plan.profile()
    assert_equal(len(profile.runs), 1)
    assert_equal(profile.runs[0].output, plan.execute().num_rows())
    assert_equal(profile.runs[0].morsels_read, 1)
    assert_true(profile.runs[0].ns >= 0)


def _six() raises -> RecordBatch:
    """Six rows, so two-row morsels give three of them."""
    return record_batch(
        [
            array([1, 2, None, 4, 5, 6], int64).copy(),
            array([10, 20, 30, 40, 50, 60], int64).copy(),
        ],
        names=["a", "b"],
    )


def _report(plan: DynRelation) raises -> PlanReport:
    """`plan`'s report on four workers and two-row morsels, so a few rows
    show what many would."""
    return plan.explain(ExecContext.parallel(4), morsel_rows=2)


def _profile(plan: DynRelation) raises -> PlanProfile:
    """`plan`'s profile on four workers and two-row morsels."""
    return plan.profile(ExecContext.parallel(4), morsel_rows=2)


def test_explain_prints_the_plan_estimate() raises:
    var report = _report(table(_six()).filter(col("a", int64) > lit(2, int64)))
    assert_true(report.estimate >= 0)
    assert_true(
        String(report).startswith(
            String("Estimated rows: ", report.estimate, "\n")
        )
    )


def test_profile_of_an_aggregate_over_an_empty_input() raises:
    """Every phase that ran says what it produced, and the plan's rows are
    the result's, though the aggregate's input is empty."""
    var plan = (
        table(_six())
        .filter(col("a", int64) > lit(99, int64))
        .aggregate([col("b", int64).sum().alias("s")], [col("a", int64)])
    )
    var profile = _profile(plan)
    assert_equal(profile.rows, plan.execute().num_rows())
    _assert_outputs_are_the_last_stages(profile)


def _assert_outputs_are_the_last_stages(profile: PlanProfile) raises:
    """Every phase that ran produced what its last stage emitted."""
    assert_equal(len(profile.runs), len(profile.report.phases))
    for ref run in profile.runs:
        if len(run.stages) > 0:
            assert_equal(run.output, run.stages[len(run.stages) - 1].rows_out)


def test_profile_of_a_whole_table_aggregate_over_an_empty_input() raises:
    """Every worker of the partial phase owes one row of nothing, and the
    combine one more: the plan answers one row."""
    var plan = (
        table(_six())
        .filter(col("a", int64) > lit(99, int64))
        .aggregate([col("b", int64).sum().alias("s")])
    )
    var profile = _profile(plan)
    assert_equal(profile.rows, 1)
    assert_equal(len(profile.runs), 2)
    assert_equal(profile.runs[0].output, profile.report.phases[0].workers)
    _assert_outputs_are_the_last_stages(profile)


def test_explain_says_why_a_phase_has_its_workers() raises:
    var plan = table(_six()).filter(col("a", int64) > lit(2, int64)).select("b")
    assert_equal(
        String(_report(plan)),
        """Estimated rows: 1
Phase 1 [result] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    -> Filter(greater(col(a), lit(2)))
    -> Project(b)""",
    )
    var serial = Driver.serial(ExecContext.parallel(4))
    var phases = plan.lower(serial)
    assert_equal(phases[0].report("result").why, "serial driver")


def test_explain_shows_the_exchange_an_aggregate_meets_at() raises:
    """The partial aggregates run on every worker; their states meet in an
    exchange the dry run did not fill, so it shows the estimate."""
    var plan = table(_six()).aggregate(
        [col("b", int64).sum().alias("total"), count_star().alias("n")],
        [col("a", int64)],
    )
    assert_equal(
        String(_report(plan)),
        """Estimated rows: 6
Phase 1 [aggregate] TableSource(6 rows)
    workers=3 (3 morsels) morsels=3 rows=6
    -> Aggregate(partial: sum(col(b)) as total, count(lit(1)) as n by col(a))
    ═══ read by Phase 2
Phase 2 [result] Exchange
    workers=4 morsels=16 rows=? est=6
    -> Aggregate(combine: sum(col(b)) as total, count(lit(1)) as n by col(a))""",
    )


def test_explain_shows_a_join_tail_reading_its_probe_phase() raises:
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    assert_equal(
        String(_report(plan)),
        """Estimated rows: 6
Phase 1 [join build] TableSource(6 rows)
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
    assert_equal(report.phases[1].source, "Barrier")
    assert_equal(report.phases[1].estimate, 6)
    assert_equal(len(report.phases[1].inputs), 1)
    assert_equal(report.phases[1].inputs[0], 1)


def test_profile_counts_the_morsels_of_a_phase_that_emits_nothing() raises:
    var plan = table(_six()).filter(col("a", int64) > lit(100, int64))
    var profile = _profile(plan)
    assert_equal(profile.runs[0].morsels_read, 3)
    assert_equal(profile.runs[0].output, 0)
    assert_true(profile.runs[0].ns >= 0)


def test_a_report_renders_as_json() raises:
    var report = _report(table(_six()).filter(col("a", int64) > lit(2, int64)))
    var json = report.to_json()
    assert_true(json.startswith('{"estimate":'))
    assert_true(
        '"workers":3,"why":"3 morsels","inputs":[],"morsels":3,"rows":6,'
        in json
    )
    assert_true(
        json.endswith(
            '"stages":[{"label":"Filter(greater(col(a), lit(2)))"}]}]}'
        )
    )


def test_a_profile_renders_as_json() raises:
    var profile = _profile(
        table(_six()).filter(col("a", int64) > lit(2, int64))
    )
    var json = profile.to_json()
    assert_true(json.startswith('{"ns":'))
    assert_true('"morsels":3,"rows":6,' in json)
    assert_true('"morsels_read":3,"output":3,' in json)
    assert_true(
        '"stages":[{"label":"Filter(greater(col(a), lit(2)))","rows_in":6,'
        '"rows_out":3,'
        in json
    )
    assert_true('"spans":[{"name":"concatenate the result","ns":' in json)


def test_profile_counts_what_each_stage_did() raises:
    """Three two-row morsels of `a = [1, 2, None, 4, 5, 6]`: the filter sees
    all six rows and keeps three, and each worker reports its busy time."""
    var plan = table(_six()).filter(col("a", int64) > lit(2, int64))
    var profile = _profile(plan)
    assert_equal(profile.rows, 3)
    ref run = profile.runs[0]
    assert_equal(len(run.stages), 1)
    assert_equal(run.stages[0].rows_in, 6)
    assert_equal(run.stages[0].rows_out, 3)
    assert_equal(run.stages[0].batches_in, 3)
    assert_equal(len(run.busy), profile.report.phases[0].workers)
    assert_true(run.read_ns >= 0)
    assert_true(profile.ns >= run.ns)


def test_profile_times_the_work_between_phases() raises:
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    var report = _profile(plan)
    var names = List[String]()
    for ref s in report.spans:
        names.append(s.name)
    assert_true("hash build for Join(left outer)" in names)
    assert_true("concatenate the result" in names)
    var costs = report.costs()
    for i in range(1, len(costs)):
        assert_true(costs[i - 1][1] >= costs[i][1], "largest first")


def test_profile_counts_the_work_between_phases() raises:
    """A join's hash build runs between phases: with `-D MARROW_STATS=true`
    its span carries what the build counted; without it, nothing."""
    var plan = table(_six()).join(
        table(_six()).select("a"), [0], [0], JOIN_LEFT
    )
    var report = _profile(plan)
    var found = False
    for ref s in report.spans:
        if s.name == "hash build for Join(left outer)":
            found = True
            comptime if STATS_ENABLED:
                var inserts = s.counters["hash.inserts"]
                assert_true(inserts >= 6, "every build row was encoded")
            else:
                assert_true(not s.counters.moved())
    assert_true(found)


def test_a_traced_run_is_a_timeline_per_thread() raises:
    """Three two-row morsels: each read and each push of the filter is one
    slice, each worker's drain one more, and the phase one more, on the
    calling thread."""
    var plan = table(_six()).filter(col("a", int64) > lit(2, int64))
    var report = plan.profile(
        ExecContext.parallel(4), trace=True, morsel_rows=2
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


def test_profile_reports_the_kernel_counters() raises:
    """A grouped aggregate encodes its keys: with `-D MARROW_STATS=true` the
    phases that ran count the rows and the new keys; without it nothing is
    counted at all."""
    var report = _profile(
        table(_six()).aggregate([count_star().alias("n")], [col("a", int64)])
    )
    comptime if STATS_ENABLED:
        var inserts = 0
        var keys = 0
        for ref run in report.runs:
            inserts += run.counters["hash.inserts"]
            keys += run.counters["hash.new_keys"]
        assert_true(inserts >= 6, "every input row was encoded")
        assert_true(keys >= 6, "six distinct keys, null among them")
        assert_true("counters:" in String(report))
    else:
        for ref run in report.runs:
            assert_true(not run.counters.moved())


def test_limit_refuses_a_negative_offset_or_length() raises:
    with assert_raises(contains="limit"):
        _ = table(_six()).limit(5, offset=-3)
    with assert_raises(contains="limit"):
        _ = table(_six()).limit(-1)


def _seven(relation: OpaquePointer[MutUntrackedOrigin]) -> Int:
    return 7


def test_a_lowering_that_raises_leaves_nothing_entered() raises:
    """A relation that fails to lower leaves the driver as it was: nothing
    it would estimate after the relation is gone."""
    var plan = scan(
        "/nonexistent/marrow-entered.parquet", schema([field("a", int64)])
    )
    var driver = Driver.explain()
    driver.estimate_with(_seven)
    with assert_raises(contains="marrow-entered.parquet"):
        _ = plan.lower(driver)
    assert_equal(driver.entered(), -1)


def test_profile_limit_reads_fewer_morsels_on_several_workers() raises:
    var rows = Int64Builder(1000)
    for i in range(1000):
        rows.append(Int64(i))
    var batch = record_batch([rows.finish()], names=["a"])
    var profile = (
        table(batch^).limit(3).profile(ExecContext.parallel(4), morsel_rows=10)
    )
    assert_equal(profile.rows, 3)
    assert_true(
        profile.runs[0].morsels_read < profile.report.phases[0].morsels,
        String(profile),
    )


def test_a_workers_error_reaches_the_caller_with_its_kind() raises:
    """A row that fails on one worker fails the query, with the kernel's
    error kind and message, whichever worker read it."""
    var s = StringBuilder()
    for i in range(200):
        s.append(String(i) if i != 137 else String("x"))
    var batch = record_batch([s.finish()], names=["s"])
    var plan = table(batch^).with_columns(
        ["n"], [cast(col("s"), DynType(int64))]
    )
    try:
        _ = plan.execute(ExecContext.parallel(4))
        raise Error("expected the cast to fail")
    except e:
        var err = DynError(e)
        assert_true(err.isa[InvalidError](), String(err))
        assert_true("cannot parse 'x'" in String(err), String(err))
