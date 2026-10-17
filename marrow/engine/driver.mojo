# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Runs a plan: which worker reads which morsel, and on which thread.

A plan lowers into `Phase`s, each a source and one pipeline per worker. A
node that needs another input whole — a join's build side, a sort's input —
runs that input's phases through the driver while it lowers, so the phases left
at the root are the ones `collect` runs.

`Driver.dynamic()` runs a phase's morsels on its workers in parallel, each
worker taking the next unread morsel as it finishes one, and answers what they
produced in morsel order. `Driver.serial()` runs every phase on the calling
thread. `Driver.explain()` runs nothing: it records each phase it is handed and
answers no rows, so lowering a plan through it is a dry run.
`Driver.profile()` runs the plan as `dynamic()` does and records what each
phase did.
"""

from std.time import perf_counter_ns

from ..arrays import StructArray
from ..execution import ExecContext
from ..kernels.concat import concat
from ..kernels.partition import RadixPartitioner
from ..schema import Schema
from ..utils.stats import STATS_ENABLED, Counters
from ..utils.threads import Thread
from .core import DynOperator, Pipeline
from .observe import Watch
from .report import (
    PhaseReport,
    PhaseRun,
    PlanProfile,
    PlanReport,
    SpanReport,
    TraceEvent,
)
from .sources import (
    DynSource,
    PlannedSource,
    Stream,
    StreamSource,
    TableSource,
    rows_of,
)


comptime MORSEL_ROWS = 131_072
"""Rows per morsel of an in-memory table: enough that a morsel's dispatch is
noise beside its work, few enough that a table of a few million rows keeps
every worker busy."""


struct Phase(Movable):
    """A source, and one pipeline per worker that pushes the source's morsels
    through its operators.

    Lowering appends stages to it; the driver numbers it, records what it
    reads, and runs it."""

    var _source: DynSource
    var _pipelines: List[Pipeline]
    var _stages: List[String]
    """A label per stage of the pipelines, for a plan report."""
    var _ctx: ExecContext
    """The context the phase's operators run their kernels on: the driver's
    with one worker, single-threaded with several, since the workers are
    the parallelism then."""
    var _id: Int
    """Its place in the order phases ran, from 1; 0 until it runs."""
    var _inputs: List[Int]
    """The phases whose rows its source reads, for a plan report."""
    var _why: String
    """Why it has the workers it has, for a plan report."""
    var _estimate: Int
    """The rows its source is estimated to read, for a source that reads
    other phases' rows; -1 when nobody estimated them."""
    var _follows: Int
    """How many phases just before it, in the same run, its source reads
    once they have run — a join's probe phases, for the build rows they
    owe."""
    var _blocking: Bool
    """Whether its stages emit only when drained. A stage that blocks sets
    it, and nothing clears it: every stage after one that blocks is drained
    too."""
    var _grows: Bool
    """Whether a stage may emit more rows than it is pushed, so its source's
    rows do not bound its output."""

    def __init__(
        out self,
        var source: DynSource,
        var ctx: ExecContext,
        workers: Int = 1,
        var why: String = "",
        follows: Int = 0,
    ):
        self._source = source^
        self._pipelines = List[Pipeline](capacity=workers)
        for _ in range(workers):
            self._pipelines.append(Pipeline())
        self._stages = List[String]()
        self._ctx = ctx^ if workers == 1 else ctx.with_threads(1)
        self._id = 0
        self._inputs = List[Int]()
        self._why = why^
        self._estimate = -1
        self._follows = follows
        self._blocking = False
        self._grows = False

    def workers(self) -> Int:
        """How many pipelines read this phase's source."""
        return len(self._pipelines)

    def append_each[
        F: def(ExecContext) raises -> DynOperator
    ](
        mut self,
        make: F,
        label: String,
        blocks: Bool = False,
        grows: Bool = False,
    ) raises:
        """`make(ctx)` as the next stage of every worker's pipeline, `ctx`
        being the phase's. `blocks` says the stage emits only when drained —
        an aggregate, a sort, a window — and `grows` that it may emit more
        rows than it is pushed, a join's probe."""
        for w in range(self.workers()):
            self._pipelines[w].append(make(self._ctx))
        self._stages.append(label)
        self._blocking = self._blocking or blocks
        self._grows = self._grows or grows

    def append_lowered[
        F: def(mut Pipeline, ExecContext) raises -> None
    ](mut self, lower: F, label: String, blocks: Bool = False) raises:
        """`lower(pipeline, ctx)` appending one stage to every worker's
        pipeline: a stage appended through a lowering function rather than
        built as one operator. `blocks` as for `append_each`."""
        for w in range(self.workers()):
            lower(self._pipelines[w], self._ctx)
        self._stages.append(label)
        self._blocking = self._blocking or blocks

    def describe(self) -> String:
        """The source and its stages, in order."""
        var out = self._source.describe()
        for ref s in self._stages:
            out += " -> " + s
        return out^

    def emits_on_drain(self) -> Bool:
        """Whether its stages emit only when drained, so running it to
        completion first costs nothing."""
        return self._blocking

    def rows_bound(self) -> Int:
        """At most how many rows it produces: its source's, when the source
        knows them and no stage adds rows; -1 otherwise."""
        return -1 if self._grows else self._source.rows()

    def starts(mut self, id: Int, before: List[Int]):
        """Number it `id`, run after the phases `before` in the same run: it
        reads the ones it follows."""
        self._id = id
        for i in range(max(0, len(before) - self._follows), len(before)):
            self._inputs.append(before[i])

    def reads(mut self, ids: List[Int]):
        """Record that its source reads what phases `ids` produced."""
        self._inputs.extend(ids.copy())

    def expects(mut self, rows: Int):
        """Record that its source is estimated to read `rows` rows; -1 when
        nobody estimated them."""
        self._estimate = rows

    def report(self, role: String) -> PhaseReport:
        """It as lowering made it, `role` naming what its rows are for."""
        return PhaseReport(
            self._id,
            role,
            self._source.describe(),
            self._stages.copy(),
            self.workers(),
            self._why.copy(),
            self._inputs.copy(),
            self._source.count(),
            self._source.rows(),
            self._estimate,
        )

    def observed(mut self, trace: Bool) -> Watch:
        """Observe it from now on: its stages and source are wrapped, so what
        they do is counted and timed — with `trace`, recorded as timeline
        slices too — in the answered `Watch`."""
        var watch = Watch(trace)
        for w in range(self.workers()):
            watch.pipeline(self._pipelines[w], self._stages)
        var source: DynSource = TableSource(List[List[StructArray]]())
        swap(source, self._source)
        self._source = watch.source(source^)
        return watch^

    def step(
        mut self, morsel: Int, ctx: ExecContext, mut out: List[StructArray]
    ) raises -> Bool:
        """Push morsel `morsel` through the first worker's pipeline, read on
        `ctx`, into `out`, and answer True; once there is nothing left to
        read, drain the pipeline into `out` and answer False."""
        var count = self._source.count()
        # A stage with all it needs, a `Limit` with its rows, stops the
        # reading; the stages above it still drain.
        if not self._pipelines[0].done() and (count < 0 or morsel < count):
            var batches = self._source.read(morsel, ctx)
            if count >= 0 or len(batches) > 0:
                self._pipelines[0].feed(batches^, out)
                return True
        self._pipelines[0].drain(out)
        return False

    def run(
        var self, ctx: ExecContext, mut out: List[StructArray]
    ) raises -> Int:
        """Run into `out`, in morsel order and then what the workers drain, in
        worker order, on `ctx`'s threads; answer how many morsels were read.

        One worker, or a source that cannot say how many morsels it has, runs
        on the calling thread. Otherwise every worker takes the next unread
        morsel as it finishes one, and stops taking them once a stage of its
        pipeline has all it needs."""
        var phase = self^
        var count = phase._source.count()
        var workers = phase.workers()
        if workers == 1 or count < 0:
            var read = 0
            while phase.step(read, ctx, out):
                read += 1
            return read

        # A worker reads its morsel with its share of the threads: all of them
        # split evenly, so a source with fewer morsels than threads still
        # decodes each one in parallel.
        var read_ctx = ctx.with_threads(
            max(1, ctx.resolved_num_threads() // workers)
        )
        var slots = List[List[StructArray]](capacity=count)
        for _ in range(count):
            slots.append(List[StructArray]())
        var read = List[Int](length=workers, fill=0)

        def feed(w: Int, i: Int) raises {mut phase, mut slots, mut read, imm}:
            ref pipeline = phase._pipelines[w]
            if pipeline.done():
                return
            read[w] += 1
            pipeline.feed(phase._source.read(i, read_ctx), slots[i])

        ctx.fan_out(count, feed, workers)

        var drained = List[List[StructArray]](capacity=workers)
        for _ in range(workers):
            drained.append(List[StructArray]())

        def drain(w: Int, i: Int) raises {mut phase, mut drained, imm}:
            phase._pipelines[i].drain(drained[i])

        ctx.fan_out(workers, drain, workers)

        for i in range(count):
            var produced = List[StructArray]()
            swap(produced, slots[i])
            out.extend(produced^)
        for i in range(workers):
            var rest = List[StructArray]()
            swap(rest, drained[i])
            out.extend(rest^)
        var morsels = 0
        for w in range(workers):
            morsels += read[w]
        return morsels


def append_each[
    F: def(ExecContext) raises -> DynOperator
](
    mut phases: List[Phase],
    make: F,
    label: String,
    blocks: Bool = False,
    grows: Bool = False,
) raises:
    """`make(ctx)` as the next stage of every worker of every phase: an
    operator that sees each row on its own, so it may run once per phase.
    `blocks` and `grows` as for `Phase.append_each`."""
    for p in range(len(phases)):
        phases[p].append_each(make, label, blocks, grows)


def runs_in_parallel(phases: List[Phase]) -> Bool:
    """Whether any of `phases` has more than one worker."""
    for i in range(len(phases)):
        if phases[i].workers() > 1:
            return True
    return False


struct PhaseCursor(Stream):
    """Runs phases one after another, a morsel at a time: what each morsel
    produced, then what the pipelines drain once their source is spent."""

    var _phases: List[Phase]
    """The phases left, the current one last: each is dropped once spent."""
    var _ctx: ExecContext
    var _morsel: Int

    def __init__(out self, var phases: List[Phase], var ctx: ExecContext):
        self._phases = List[Phase](capacity=len(phases))
        while len(phases) > 0:
            self._phases.append(phases.pop())
        self._ctx = ctx^
        self._morsel = 0

    def next(mut self) raises -> List[StructArray]:
        """The next batches the phases produce, never empty until they are
        spent; nothing once they are."""
        var out = List[StructArray]()
        while len(self._phases) > 0 and len(out) == 0:
            ref phase = self._phases[len(self._phases) - 1]
            if phase.step(self._morsel, self._ctx, out):
                self._morsel += 1
            else:
                _ = self._phases.pop()
                self._morsel = 0
        return out^


struct BuildSide(Movable):
    """The side a hash join builds, once the driver has run it: its rows,
    and whether it is the side that was asked to probe."""

    var swapped: Bool
    var _rows: List[StructArray]

    def __init__(out self, swapped: Bool, var rows: List[StructArray]):
        self.swapped = swapped
        self._rows = rows^

    def rows(deinit self) -> List[StructArray]:
        """The built side's rows."""
        return self._rows^


struct DriverMode(Copyable, ImplicitlyCopyable, Movable):
    """What a driver does besides running phases: nothing, run none of them,
    or time and count what they do — with `traced()`, every call too."""

    var _kind: UInt8

    def __init__(out self, kind: UInt8):
        self._kind = kind

    @staticmethod
    def running() -> Self:
        return Self(0)

    @staticmethod
    def dry() -> Self:
        return Self(1)

    @staticmethod
    def profiling() -> Self:
        return Self(2)

    @staticmethod
    def traced() -> Self:
        return Self(3)

    def is_dry(self) -> Bool:
        """Whether nothing runs, so a barrier's input is estimated, not
        read."""
        return self._kind == 1

    def observes(self) -> Bool:
        """Whether what runs is timed and counted, for a profile."""
        return self._kind >= 2

    def traces(self) -> Bool:
        """Whether what runs is also recorded as a timeline."""
        return self._kind == 3


comptime RunFn = def(mut Driver, var List[Phase], String) thin raises -> List[
    StructArray
]
"""How a driver runs phases, wired by its constructor: a strategy's code is
linked only into binaries that construct it."""


comptime EstimateFn = def(OpaquePointer[MutUntrackedOrigin]) thin -> Int
"""A relation's estimated rows, the relation behind an erased pointer: the
driver runs phases, and knows no relation. A pointer rather than a copy,
since copying a relation links its every member's copy."""


def _no_estimate(input: OpaquePointer[MutUntrackedOrigin]) -> Int:
    return -1


struct _Recording(Movable):
    """What a driver records for a report: every phase it is handed, what
    each did when it ran, and the work between them."""

    var phases: List[PhaseReport]
    var runs: List[PhaseRun]
    """One per entry of `phases` when the driver profiles; empty otherwise."""
    var spans: List[SpanReport]
    var events: List[TraceEvent]
    var counted: Counters
    """The kernel counters at the latest `Driver.clock()`."""

    def __init__(out self):
        self.phases = List[PhaseReport]()
        self.runs = List[PhaseRun]()
        self.spans = List[SpanReport]()
        self.events = List[TraceEvent]()
        self.counted = Counters()

    def report(deinit self, estimate: Int) -> PlanReport:
        """The phases recorded, of a plan estimated at `estimate` rows."""
        return PlanReport(self.phases^, estimate)

    def profile(deinit self, estimate: Int, ns: Int, rows: Int) -> PlanProfile:
        """The phases recorded and what each did, of a run that took `ns`
        and answered `rows` rows."""
        return PlanProfile(
            PlanReport(self.phases^, estimate),
            self.runs^,
            self.spans^,
            self.events^,
            ns,
            rows,
        )


struct Driver(Movable):
    """How a plan's phases run: a strategy, picked by constructor.

    `dynamic()` runs each phase's morsels on as many workers as the context
    has threads. `serial()` runs every phase on the calling thread, with one
    pipeline, in morsel order. `explain()` runs nothing and records the phases
    `dynamic()` would run; `profile()` runs them as `dynamic()` does and
    records what each did."""

    var ctx: ExecContext
    """The context operators run their kernels on."""
    var _recorded: _Recording
    """What `explain` and `profile` recorded; empty otherwise."""
    var _morsel_rows: Int
    """Rows per morsel of an in-memory table."""
    var _workers: Int
    var _why: String
    """Why the driver has the workers it has, when that is one."""
    var _run: RunFn
    var _mode: DriverMode
    var _estimate: EstimateFn
    """A relation's estimated rows, for a report; wired by whoever asks for
    the report, so no other binary links the estimates."""
    var _ran: Int
    var _last: List[Int]
    """The ids of the phases the latest `run` was handed."""
    var _lowering: List[OpaquePointer[MutUntrackedOrigin]]
    """The relations being lowered, the innermost last."""

    def __init__(
        out self,
        var ctx: ExecContext,
        run: RunFn,
        mode: DriverMode,
        morsel_rows: Int,
        serial: Bool,
    ):
        """A driver running phases with `run`, doing what `mode` says beside.
        It has one worker per thread of `ctx` on the host, or one when
        `serial` or on a device, whose kernels use the device instead."""
        if serial:
            self._workers = 1
            self._why = "serial driver"
        elif ctx.is_gpu():
            self._workers = 1
            self._why = "device context"
        else:
            self._workers = max(1, ctx.resolved_num_threads())
            self._why = "one thread"
        self.ctx = ctx^
        self._recorded = _Recording()
        self._morsel_rows = max(1, morsel_rows)
        self._run = run
        self._mode = mode
        self._estimate = _no_estimate
        self._ran = 0
        self._last = List[Int]()
        self._lowering = List[OpaquePointer[MutUntrackedOrigin]]()
        comptime if STATS_ENABLED:
            # The counters are created on first touch, which several workers
            # must not race to: they are created here, on the calling thread.
            _ = Counters.now()

    @staticmethod
    def serial(ctx: ExecContext = ExecContext.auto()) -> Self:
        return Self(
            ctx.copy(), Self._run_all, DriverMode.running(), MORSEL_ROWS, True
        )

    @staticmethod
    def dynamic(
        ctx: ExecContext = ExecContext.auto(), morsel_rows: Int = MORSEL_ROWS
    ) -> Self:
        return Self(
            ctx.copy(), Self._run_all, DriverMode.running(), morsel_rows, False
        )

    @staticmethod
    def explain(
        ctx: ExecContext = ExecContext.auto(), morsel_rows: Int = MORSEL_ROWS
    ) -> Self:
        return Self(
            ctx.copy(), Self._run_dry, DriverMode.dry(), morsel_rows, False
        )

    @staticmethod
    def profile(
        ctx: ExecContext = ExecContext.auto(),
        morsel_rows: Int = MORSEL_ROWS,
        trace: Bool = False,
    ) -> Self:
        """`dynamic()`, recording what each phase did; with `trace`, also
        every stage call and morsel read as a timeline slice."""
        return Self(
            ctx.copy(),
            Self._run_observed,
            DriverMode.traced() if trace else DriverMode.profiling(),
            morsel_rows,
            False,
        )

    @staticmethod
    def _run_all(
        mut driver: Self, var phases: List[Phase], role: String
    ) raises -> List[StructArray]:
        """`dynamic()` and `serial()`: every phase, one after another."""
        var out = List[StructArray]()
        while len(phases) > 0:
            _ = phases.pop(0).run(driver.ctx, out)
        return out^

    @staticmethod
    def _run_dry(
        mut driver: Self, var phases: List[Phase], role: String
    ) raises -> List[StructArray]:
        """`explain()`: every phase recorded, none run."""
        for i in range(len(phases)):
            driver._recorded.phases.append(phases[i].report(role))
        return List[StructArray]()

    @staticmethod
    def _run_observed(
        mut driver: Self, var phases: List[Phase], role: String
    ) raises -> List[StructArray]:
        """`profile()`: every phase run, and what it did recorded."""
        var out = List[StructArray]()
        while len(phases) > 0:
            var phase = phases.pop(0)
            var report = phase.report(role)
            var name = String("Phase ", report.id, " [", role, "]")
            driver._recorded.phases.append(report^)
            var watch = phase.observed(driver._mode.traces())
            var counted = Counters.now()
            var start = Int(perf_counter_ns())
            var produced = List[StructArray]()
            var morsels = phase^.run(driver.ctx, produced)
            var ns = Int(perf_counter_ns()) - start
            driver._recorded.runs.append(
                watch.record(
                    morsels, rows_of(produced), ns, Counters.since(counted)
                )
            )
            watch.trace(driver._recorded.events, name^, start, ns)
            out.extend(produced^)
        return out^

    def phase(self, var source: DynSource) -> Phase:
        """A phase reading `source`: one worker per morsel up to the driver's
        count, and one for a source that cannot say how many it has."""
        var count = source.count()
        var workers = min(self._workers, count) if count > 1 else 1
        var why: String
        if self._workers == 1:
            why = self._why.copy()
        elif count < 0:
            why = "read in order"
        elif count <= 1:
            why = "one morsel"
        elif count < self._workers:
            why = String(count, " morsels")
        else:
            why = ""
        return Phase(source^, self.ctx.copy(), workers, why^)

    def table(self, batch: StructArray) -> Phase:
        """A phase reading `batch`, in morsels of the driver's size."""
        return self.phase(TableSource.sliced(batch, self._morsel_rows))

    def owed(self, var source: DynSource, after: Int) -> Phase:
        """A phase of one worker reading `source`, which holds what the
        `after` phases just before it owe once they have all run: the build
        rows a join's probe phases did not match."""
        return Phase(source^, self.ctx.copy(), why="one morsel", follows=after)

    def run(
        mut self, var phases: List[Phase], role: String = "result"
    ) raises -> List[StructArray]:
        """Every batch the phases produce, phase by phase, in morsel order;
        none under `explain`. `role` names what they are for, in a report."""
        self._last = List[Int](capacity=len(phases))
        for i in range(len(phases)):
            self._ran += 1
            phases[i].starts(self._ran, self._last)
            self._last.append(self._ran)
        return self._run(self, phases^, role)

    def build_side(
        mut self, var build: List[Phase], mut probe: List[Phase], may_swap: Bool
    ) raises -> BuildSide:
        """Run `build`, the side a hash join indexes, to completion; answer
        its rows, and leave in `probe` the phases that probe them, each
        reading them.

        With `may_swap`, the join's answer not depending on the side built,
        the side asked for is a hint: the side with at most half the rows is
        built, once both are known. A probe side that emits only when
        drained runs first, which costs nothing; any other is bounded by its
        sources' rows. A dry run knows no rows, so it builds the side asked
        for."""
        var probing = List[Phase]()
        swap(probing, probe)
        var rows = self.run(build^, "join build")
        var built = self._last.copy()
        var swapped = False
        var build_rows = rows_of(rows)
        if build_rows > 0 and may_swap:
            var drained = len(probing) > 0
            var bound = 0
            for p in range(len(probing)):
                drained = drained and probing[p].emits_on_drain()
                var n = probing[p].rows_bound()
                bound = -1 if bound < 0 or n < 0 else bound + n
            if drained:
                var probed = self.run(probing^, "join input")
                var probed_from = self._last.copy()
                swapped = 2 * rows_of(probed) < build_rows
                if swapped:
                    probing = [self._materialised(rows^, built^)]
                    rows = probed^
                    built = probed_from^
                else:
                    probing = [self._materialised(probed^, probed_from^)]
            else:
                swapped = bound >= 0 and 2 * bound < build_rows
                if swapped:
                    var flipped = self.run(probing^, "join build")
                    probing = [self._materialised(rows^, built^)]
                    rows = flipped^
                    built = self._last.copy()
        for p in range(len(probing)):
            probing[p].reads(built)
        swap(probe, probing)
        return BuildSide(swapped, rows^)

    def clock(mut self) -> Int:
        """Now, for a span, when the driver observes; 0 otherwise. The kernel
        counters are read too, so the span that follows reports how much
        they moved."""
        if self._mode.observes():
            # One snapshot: a span reports the counters since the latest
            # `clock()`, so spans do not nest.
            self._recorded.counted = Counters.now()
            return Int(perf_counter_ns())
        return 0

    def span(mut self, name: String, since: Int):
        """Record the work done since `since` (a `clock()`) as `name`, when
        the driver observes."""
        if self._mode.observes():
            var ns = Int(perf_counter_ns()) - since
            self._recorded.spans.append(
                SpanReport(
                    name, ns, self._ran, Counters.since(self._recorded.counted)
                )
            )
            if self._mode.traces():
                self._recorded.events.append(
                    TraceEvent(name, Thread.current_id(), since, ns, -1)
                )

    def plan_report(mut self, estimate: Int) -> PlanReport:
        """The phases recorded, of a plan estimated at `estimate` rows."""
        var recorded = _Recording()
        swap(recorded, self._recorded)
        return recorded^.report(estimate)

    def plan_profile(
        mut self, since: Int, rows: Int, estimate: Int
    ) -> PlanProfile:
        """The profile of the run that began at `since` (a `clock()`) and
        answered `rows` rows, of a plan estimated at `estimate`."""
        var ns = self.clock() - since
        var recorded = _Recording()
        swap(recorded, self._recorded)
        return recorded^.profile(estimate, ns, rows)

    def estimate_with(mut self, estimate: EstimateFn):
        """Estimate relations with `estimate`, for a report: wiring it links
        the estimates of every node type, which a size-critical binary may
        decline."""
        self._estimate = estimate

    def estimate(self, relation: OpaquePointer[MutUntrackedOrigin]) -> Int:
        """The estimated rows of the relation `relation` points at, when
        this driver reports estimates; -1 otherwise."""
        return self._estimate(relation)

    def entering(mut self, relation: OpaquePointer[MutUntrackedOrigin]):
        """The relation `relation` points at starts lowering itself: until it
        `leaving`s, it is what `entered` estimates. Only a pointer is kept, so
        a driver that reports estimates estimates only the relations asked
        about."""
        self._lowering.append(relation)

    def leaving(mut self):
        """The relation that entered last has lowered itself."""
        _ = self._lowering.pop()

    def entered(self) -> Int:
        """The estimated rows of the relation lowering itself, its inputs
        excluded; -1 when this driver reports no estimates."""
        if len(self._lowering) == 0:
            return -1
        return self._estimate(self._lowering[len(self._lowering) - 1])

    def _materialised(
        self, var batches: List[StructArray], var inputs: List[Int]
    ) -> Phase:
        """A phase over batches already produced — a morsel each — by the
        phases `inputs`."""
        var phase = self.phase(TableSource.of(batches^))
        phase.reads(inputs)
        return phase^

    def _after_run(self, var phase: Phase, estimate: Int) -> Phase:
        """`phase`, reading what the latest `run` produced, estimated at
        `estimate` rows."""
        phase.reads(self._last)
        phase.expects(estimate)
        return phase^

    def _alone(self, var source: DynSource, var why: String) -> Phase:
        """A phase of one worker reading `source`, for `why`."""
        return Phase(source^, self.ctx.copy(), why=why^)

    def barrier(
        mut self, var phases: List[Phase], role: String, input_rows: Int
    ) raises -> Phase:
        """`phases` — a lowered input estimated at `input_rows` — as one
        phase of one worker, for an operator that must see every row of them:
        a sort, a window, a limit's last cut.

        Phases of one worker each are read in order, as they run, as a
        `PhaseCursor` stream. Phases with several run first, in parallel, and
        the phase answered reads their batches in morsel order. A dry run
        reads nothing, where they would be, estimated."""
        if not runs_in_parallel(phases):
            if len(phases) == 1:
                return phases.pop()
            var names = List[String](capacity=len(phases))
            for i in range(len(phases)):
                names.append(phases[i].describe())
            var name = String("Phases(", String("; ").join(names), ")")
            return self._alone(
                StreamSource(PhaseCursor(phases^, self.ctx.copy()), name^),
                "barrier",
            )
        var batches = self.run(phases^, role)
        if self._mode.is_dry():
            return self._after_run(
                self._alone(PlannedSource("Barrier", 1), "barrier"), input_rows
            )
        return self._after_run(
            self._alone(TableSource.of(batches^), "barrier"), input_rows
        )

    def barrier[
        F: def(ExecContext) raises -> DynOperator
    ](
        mut self,
        var phases: List[Phase],
        role: String,
        input_rows: Int,
        local: F,
        label: String,
    ) raises -> Phase:
        """`barrier`, with `local` first on every worker when the phases run
        on several: a cut each worker can take of what it reads — a limit's
        first rows, a top-N's — before the rows meet."""
        if runs_in_parallel(phases):
            append_each(phases, local, label)
        return self.barrier(phases^, role, input_rows)

    def exchange(
        mut self,
        var phases: List[Phase],
        keys: Int,
        role: String,
        output_rows: Int,
    ) raises -> Phase:
        """Run `phases` and answer a phase over their batches with every row
        of one key — the first `keys` columns — in one morsel, so a worker
        holds all of a key. Few rows are one morsel, and no keys means one
        morsel of everything. `output_rows` is what they are estimated to
        become, for a report; -1 when nobody estimates."""
        var batches = self.run(phases^, role)
        var bits = RadixPartitioner.bits_for(self._workers)
        var one = String("no keys") if keys == 0 else String("few rows")
        if self._mode.is_dry():
            if keys > 0 and output_rows > self._morsel_rows:
                return self._after_run(
                    self.phase(PlannedSource("Exchange", 1 << bits)),
                    output_rows,
                )
            return self._after_run(
                self._alone(PlannedSource("Exchange", 1), one^), output_rows
            )
        if keys == 0 or rows_of(batches) <= self._morsel_rows:
            return self._after_run(
                self._alone(TableSource([batches^]), one^), output_rows
            )
        var key_indices = List[Int](capacity=keys)
        for i in range(keys):
            key_indices.append(i)
        var since = self.clock()
        var parts = RadixPartitioner(bits, self.ctx.copy()).repartition(
            batches, key_indices
        )
        self.span(String("repartition into ", 1 << bits), since)
        return self._after_run(self.phase(TableSource(parts^)), output_rows)

    def collect(
        mut self,
        var phases: List[Phase],
        schema: Schema,
        role: String = "result",
    ) raises -> StructArray:
        """Run the phases into one batch of `schema`; under `explain`, an
        empty one, with a zero-length column per field."""
        var batches = self.run(phases^, role)
        var since = self.clock()
        var result = concat(batches, schema, self.ctx)
        # Several batches were concatenated, and that compacts sparse view
        # columns already; a single one is returned as it is, so its view
        # columns are compacted here -- a result must not keep the pages it
        # was filtered out of alive. An explicit ladder rather than
        # `dispatch_binaryview` with a closure: every plan collects, and the
        # closure form measured +13.8 KB of `__text` on each fused size gate.
        for i in range(len(result.children)):
            ref child = result.children[i]
            if child.dtype().is_string_view():
                result.children[i] = child.as_string_view().compact().to_dyn()
            elif child.dtype().is_binary_view():
                result.children[i] = child.as_binary_view().compact().to_dyn()
        self.span("concatenate the result", since)
        return result^
