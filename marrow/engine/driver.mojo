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
`Driver.analyze()` runs the plan as `dynamic()` does and records what each
phase did.
"""

from std.atomic import Atomic
from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..arrays import StructArray
from ..execution import PARALLEL_ENABLED, ExecContext
from ..expr.logical import Bindings, DynRelation
from ..expr.physical import (
    DynOperator,
    Observation,
    Pipeline,
    TraceEvent,
    Trail,
    finish_batches,
)
from ..utils.stats import (
    STATS_ENABLED,
    since as stats_since,
    snapshot as stats_snapshot,
)
from ..utils.threads import Thread
from ..kernels.partition import partition_bits, repartition
from ..schema import Schema
from .report import PhaseReport, PlanReport, SpanReport, StageReport
from .sources import (
    DynSource,
    ObservedSource,
    PlannedSource,
    Source,
    TableSource,
)


comptime MORSEL_ROWS = 131_072
"""Rows per morsel of an in-memory table: enough that a morsel's dispatch is
noise beside its work, few enough that a table of a few million rows keeps
every worker busy."""


struct Phase(Movable):
    """A source, and one pipeline per worker that pushes the source's morsels
    through its operators."""

    var source: DynSource
    var pipelines: List[Pipeline]
    var stages: List[String]
    """A label per stage of the pipelines, for a plan report."""
    var ctx: ExecContext
    """The context the phase's operators run their kernels on: the driver's
    with one worker, single-threaded with several, since the workers are
    the parallelism then."""
    var id: Int
    """Its place in the order phases ran, from 1; 0 until it runs."""
    var inputs: List[Int]
    """The phases whose rows its source reads, for a plan report."""
    var why: String
    """Why it has the workers it has, for a plan report."""
    var estimate: Int
    """The rows its source is estimated to read, where the phases it reads
    did not run; -1 otherwise."""
    var follows: Int
    """How many phases just before it, in the same run, its source reads
    once they have run — a join's probe phases, for the build rows they
    owe."""
    var blocking: Bool
    """Whether its stages emit only when drained — an aggregate, a sort, a
    window — so running it to completion first costs nothing. A stage that
    blocks sets it, and nothing clears it: every stage after one that blocks
    is drained too."""
    var grows: Bool
    """Whether a stage may emit more rows than it is pushed — a join's
    probe — so its source's rows do not bound its output."""

    def __init__(
        out self,
        var source: DynSource,
        var ctx: ExecContext,
        workers: Int = 1,
        var why: String = "",
    ):
        self.source = source^
        self.pipelines = List[Pipeline](capacity=workers)
        for _ in range(workers):
            self.pipelines.append(Pipeline())
        self.stages = List[String]()
        self.ctx = ctx^ if workers == 1 else ctx.with_threads(1)
        self.id = 0
        self.inputs = List[Int]()
        self.why = why^
        self.estimate = -1
        self.follows = 0
        self.blocking = False
        self.grows = False

    def workers(self) -> Int:
        """How many pipelines read this phase's source."""
        return len(self.pipelines)

    def append(mut self, worker: Int, var op: DynOperator, label: String):
        """`op` as the next stage of worker `worker`'s pipeline. Every worker
        gets the same stages, so the label is kept once."""
        self.pipelines[worker].append(op^)
        if worker == 0:
            self.stages.append(label)

    def append_each[
        F: def(ExecContext) raises -> DynOperator
    ](mut self, make: F, label: String) raises:
        """`make(ctx)` as the next stage of every worker's pipeline, `ctx`
        being the phase's."""
        for w in range(self.workers()):
            self.append(w, make(self.ctx), label)

    def describe(self) -> String:
        """The source and its stages, in order."""
        var out = self.source.describe()
        for ref s in self.stages:
            out += " -> " + s
        return out^


def append_each[
    F: def(ExecContext) raises -> DynOperator
](mut phases: List[Phase], make: F, label: String) raises:
    """`make(ctx)` as the next stage of every worker of every phase: an
    operator that sees each row on its own, so it may run once per phase."""
    for p in range(len(phases)):
        phases[p].append_each(make, label)


def rows_of(batches: List[StructArray]) -> Int:
    """The rows of `batches`, together."""
    var n = 0
    for ref b in batches:
        n += len(b)
    return n


def rows_bound(phases: List[Phase]) -> Int:
    """At most how many rows `phases` produce, when every source knows its
    rows and no stage adds rows; -1 otherwise."""
    var total = 0
    for i in range(len(phases)):
        var n = phases[i].source.rows()
        if phases[i].grows or n < 0:
            return -1
        total += n
    return total


def all_blocking(phases: List[Phase]) -> Bool:
    """Whether every one of `phases` emits only when drained."""
    if len(phases) == 0:
        return False
    for i in range(len(phases)):
        if not phases[i].blocking:
            return False
    return True


def runs_in_parallel(phases: List[Phase]) -> Bool:
    """Whether any of `phases` has more than one worker."""
    for i in range(len(phases)):
        if phases[i].workers() > 1:
            return True
    return False


struct PhaseCursor(Movable):
    """Runs phases one after another, a morsel at a time: what each morsel
    produced, then what the pipelines drain once their source is spent."""

    var _phases: List[Phase]
    """The phases left, the current one last: each is dropped once spent."""
    var _ctx: ExecContext
    var _morsel: Int
    var _read: Int
    var _reading: Bool

    def __init__(out self, var phases: List[Phase], var ctx: ExecContext):
        self._phases = List[Phase](capacity=len(phases))
        while len(phases) > 0:
            self._phases.append(phases.pop())
        self._ctx = ctx^
        self._morsel = 0
        self._read = 0
        self._reading = True

    def read(self) -> Int:
        """How many morsels have been read, of every phase."""
        return self._read

    def next(mut self) raises -> List[StructArray]:
        """The next batches the phases produce, never empty until they are
        spent; nothing once they are."""
        while len(self._phases) > 0:
            ref phase = self._phases[len(self._phases) - 1]
            ref pipeline = phase.pipelines[0]
            var out = List[StructArray]()
            if self._reading:
                var count = phase.source.count()
                # A stage with all it needs, a `Limit` with its rows, stops
                # the reading; the stages above it still drain.
                if pipeline.done() or (count >= 0 and self._morsel >= count):
                    self._reading = False
                    continue
                var batches = phase.source.read(self._morsel, self._ctx)
                if count < 0 and len(batches) == 0:
                    self._reading = False
                    continue
                self._morsel += 1
                self._read += 1
                while len(batches) > 0:
                    pipeline.feed(batches.pop(0), out)
            else:
                var rest = pipeline.drain()
                if rest:
                    out.append(rest.value().struct_array())
                else:
                    _ = self._phases.pop()
                    self._morsel = 0
                    self._reading = True
            if len(out) > 0:
                return out^
        return List[StructArray]()


struct PhasesSource(Source):
    """Phases read as one source: each morsel is the next batches they
    produce. Its count is unknown, so it is read in order."""

    var _cursor: ArcPointer[PhaseCursor]
    var _inner: List[String]

    def __init__(out self, var phases: List[Phase], var ctx: ExecContext):
        self._inner = List[String](capacity=len(phases))
        # Indexed: iterating a `List` needs copyable elements.
        for i in range(len(phases)):
            self._inner.append(phases[i].describe())
        self._cursor = ArcPointer(PhaseCursor(phases^, ctx^))

    def count(self) -> Int:
        return -1

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._cursor[].next()

    def describe(self) -> String:
        var out = String("Phases(")
        for i in range(len(self._inner)):
            if i > 0:
                out += "; "
            out += self._inner[i]
        return out + ")"


comptime RunFn = def(mut Driver, var List[Phase], String) thin raises -> List[
    StructArray
]
"""How a driver runs phases, wired by its constructor: a strategy's code is
linked only into binaries that construct it."""


def _run_phase(
    driver: Driver, var phase: Phase, mut out: List[StructArray]
) raises -> Int:
    """Run one phase into `out`, in morsel order and then what its workers
    drain, in worker order; answer how many morsels were read.

    A phase of one worker, or of a source that cannot say how many morsels
    it has, runs on the calling thread. Otherwise every worker takes the next
    unread morsel as it finishes one, and stops taking them once a stage of
    its pipeline has all it needs."""
    var count = phase.source.count()
    var workers = phase.workers()
    comptime if not PARALLEL_ENABLED:
        workers = 1
    if workers == 1 or count < 0:
        var one = List[Phase]()
        one.append(phase^)
        var cursor = PhaseCursor(one^, driver.ctx.copy())
        while True:
            var batches = cursor.next()
            if len(batches) == 0:
                break
            out.extend(batches^)
        return cursor.read()

    # A worker reads its morsel with its share of the driver's threads: all of
    # them split evenly, so a source with fewer morsels than threads still
    # decodes each one in parallel.
    var read_ctx = driver.ctx.with_threads(
        max(1, driver.ctx.resolved_num_threads() // workers)
    )
    var slots = List[List[StructArray]](capacity=count)
    for _ in range(count):
        slots.append(List[StructArray]())
    var read = List[Int](length=workers, fill=0)

    def feed(w: Int, i: Int) raises {mut phase, mut slots, mut read, imm}:
        ref pipeline = phase.pipelines[w]
        if pipeline.done():
            return
        var batches = phase.source.read(i, read_ctx)
        read[w] += 1
        var produced = List[StructArray]()
        while len(batches) > 0:
            pipeline.feed(batches.pop(0), produced)
        slots[i] = produced^

    driver.ctx.fan_out(count, feed, workers)

    var drained = List[List[StructArray]](capacity=workers)
    for _ in range(workers):
        drained.append(List[StructArray]())

    def drain(w: Int, i: Int) raises {mut phase, mut drained, imm}:
        ref pipeline = phase.pipelines[i]
        while True:
            var rest = pipeline.drain()
            if not rest:
                break
            drained[i].append(rest.value().struct_array())

    driver.ctx.fan_out(workers, drain, workers)

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


def _run_phases(
    mut driver: Driver, var phases: List[Phase], role: String
) raises -> List[StructArray]:
    var out = List[StructArray]()
    while len(phases) > 0:
        _ = _run_phase(driver, phases.pop(0), out)
    return out^


def _report(
    phase: Phase, role: String, morsels: Int, output: Int, ns: Int
) -> PhaseReport:
    return PhaseReport(
        phase.id,
        role,
        phase.source.describe(),
        phase.stages.copy(),
        phase.workers(),
        phase.why.copy(),
        phase.inputs.copy(),
        morsels,
        phase.source.rows(),
        phase.estimate,
        output,
        ns,
        List[StageReport](),
        List[Int](),
        -1,
        List[Int](),
    )


def _run_explain(
    mut driver: Driver, var phases: List[Phase], role: String
) raises -> List[StructArray]:
    # Indexed: iterating a `List` needs copyable elements.
    for i in range(len(phases)):
        ref p = phases[i]
        driver.report.phases.append(_report(p, role, p.source.count(), -1, -1))
    return List[StructArray]()


def _run_analyze(
    mut driver: Driver, var phases: List[Phase], role: String
) raises -> List[StructArray]:
    var out = List[StructArray]()
    while len(phases) > 0:
        var phase = phases.pop(0)
        var report = _report(phase, role, -1, -1, -1)
        # Observed by wrapping: each stage in an `ObservedOperator`, the
        # source in an `ObservedSource`, so a run nobody observes pays
        # nothing.
        var watched = List[Observation](capacity=phase.workers())
        var trails = List[Trail]()
        var labels = phase.stages.copy()
        for w in range(phase.workers()):
            var trail = Optional[Trail](None)
            if driver._trace:
                trails.append(Trail(List[TraceEvent]()))
                trail = trails[len(trails) - 1]
            watched.append(phase.pipelines[w].observe(trail, labels))
        var reading = ArcPointer(Atomic[Int64](0))
        var source = DynSource(PlannedSource("", "", 0, -1))
        swap(source, phase.source)
        var source_trail = Optional[Trail](None)
        if driver._trace:
            trails.append(Trail(List[TraceEvent]()))
            source_trail = trails[len(trails) - 1]
        phase.source = ObservedSource(source^, reading, source_trail)
        var counted = stats_snapshot()
        var start = Int(perf_counter_ns())
        var before = len(out)
        report.morsels = _run_phase(driver, phase^, out)
        report.ns = Int(perf_counter_ns()) - start
        report.counters = stats_since(counted)
        if driver._trace:
            driver.report.events.append(
                TraceEvent(
                    String("Phase ", report.id, " [", role, "]"),
                    Thread.current_id(),
                    start,
                    report.ns,
                    -1,
                )
            )
            for ref trail in trails:
                for ref e in trail[]:
                    if e.ns >= 0:
                        driver.report.events.append(e.copy())
        report.read_ns = Int(reading[].load())
        for ref w in watched:
            var busy = 0
            for ref s in w[]:
                busy += s.ns
            report.busy.append(busy)
        report.output = 0
        for i in range(before, len(out)):
            report.output += len(out[i])
        for stage in range(len(report.stages)):
            var sum = StageReport(0, 0, 0, 0, 0, 0)
            for ref w in watched:
                ref s = w[][stage]
                sum.rows_in += s.rows_in
                sum.rows_out += s.rows_out
                sum.batches_in += s.batches_in
                sum.batches_out += s.batches_out
                sum.ns += s.ns
                sum.ns_max = max(sum.ns_max, s.ns)
            report.stats.append(sum^)
        driver.report.phases.append(report^)
    return out^


comptime EstimateFn = def(OpaquePointer[MutUntrackedOrigin]) thin -> Int
"""A relation's estimated rows, the relation behind an erased pointer: a
`Driver` field whose type named `DynRelation` would make the two types spell
each other in every symbol that mentions either, past what the linker
accepts. A pointer rather than a copy, since copying a `DynRelation` links
its every member's copy."""


def _no_estimate(input: OpaquePointer[MutUntrackedOrigin]) -> Int:
    return -1


def _estimate_rows(input: OpaquePointer[MutUntrackedOrigin]) -> Int:
    """The estimated row count of the `DynRelation` `input` points at, or -1
    when nothing bounds it."""
    try:
        var rows = input.unsafe_bitcast[DynRelation]()[].estimate().rows
        var known = rows.known()
        return known.value() if known else -1
    except:
        return -1


struct Driver(Movable):
    """How a plan's phases run: a strategy, picked by constructor.

    `dynamic()` runs each phase's morsels on as many workers as the context
    has threads. `serial()` runs every phase on the calling thread, with one
    pipeline, in morsel order. `explain()` runs nothing and records the phases
    `dynamic()` would run; `analyze()` runs them as `dynamic()` does and
    records what each did."""

    var ctx: ExecContext
    """The context operators run their kernels on."""
    var report: PlanReport
    """What `explain` and `analyze` recorded; empty otherwise."""
    var morsel_rows: Int
    """Rows per morsel of an in-memory table."""
    var _workers: Int
    var _serial: String
    """Why the driver has one worker, when it does."""
    var _run: RunFn
    var _estimate: EstimateFn
    """A relation's estimated rows, for a report; wired only by the
    strategies that report, so no other binary links the estimates."""
    var _dry: Bool
    """Whether nothing runs, so a barrier's input is estimated, not read."""
    var _observe: Bool
    """Whether what runs is timed and counted, for a report."""
    var _trace: Bool
    """Whether what runs is also recorded as a timeline, for `trace`."""
    var _ran: Int
    var _last: List[Int]
    """The ids of the phases the latest `run` was handed."""
    var _counted: List[Int]
    """The kernel counters at the latest `clock()`."""
    var _entered: Int
    """The estimated rows of the relation lowering last entered, when this
    driver reports estimates; -1 otherwise."""

    def __init__(
        out self,
        var ctx: ExecContext,
        run: RunFn,
        workers: Int,
        morsel_rows: Int,
        var serial: String,
        estimate: EstimateFn = _no_estimate,
        dry: Bool = False,
        observe: Bool = False,
        trace: Bool = False,
    ):
        self.ctx = ctx^
        self.report = PlanReport()
        self.morsel_rows = max(1, morsel_rows)
        self._workers = max(1, workers)
        self._serial = serial^
        self._run = run
        self._estimate = estimate
        self._dry = dry
        self._observe = observe
        self._trace = trace
        self._ran = 0
        self._last = List[Int]()
        self._counted = List[Int]()
        self._entered = -1
        comptime if STATS_ENABLED:
            # The counters are created on first touch, which several workers
            # must not race to: they are created here, on the calling thread.
            _ = stats_snapshot()

    @staticmethod
    def _parallelism(ctx: ExecContext) -> Int:
        """One worker per thread, on the host only: a device context runs one
        pipeline and lets its kernels use the device."""
        comptime if not PARALLEL_ENABLED:
            return 1
        if ctx.is_gpu():
            return 1
        return ctx.resolved_num_threads()

    @staticmethod
    def _serial_because(ctx: ExecContext) -> String:
        """Why `_parallelism` answers one, when it does."""
        comptime if not PARALLEL_ENABLED:
            return "MARROW_DRIVER=serial"
        if ctx.is_gpu():
            return "device context"
        return "one thread"

    @staticmethod
    def serial(ctx: ExecContext = ExecContext.auto()) -> Self:
        return Self(ctx.copy(), _run_phases, 1, MORSEL_ROWS, "serial driver")

    @staticmethod
    def dynamic(
        ctx: ExecContext = ExecContext.auto(), morsel_rows: Int = MORSEL_ROWS
    ) -> Self:
        return Self(
            ctx.copy(),
            _run_phases,
            Self._parallelism(ctx),
            morsel_rows,
            Self._serial_because(ctx),
        )

    @staticmethod
    def _estimator[estimates: Bool]() -> EstimateFn:
        """`_estimate_rows` when `estimates`: wiring it links the optimizer's
        estimates for every node type, which a size-critical binary may
        decline."""
        comptime if estimates:
            return _estimate_rows
        else:
            return _no_estimate

    @staticmethod
    def explain[
        estimates: Bool = True
    ](
        ctx: ExecContext = ExecContext.auto(), morsel_rows: Int = MORSEL_ROWS
    ) -> Self:
        return Self(
            ctx.copy(),
            _run_explain,
            Self._parallelism(ctx),
            morsel_rows,
            Self._serial_because(ctx),
            Self._estimator[estimates](),
            dry=True,
        )

    @staticmethod
    def analyze[
        estimates: Bool = True
    ](
        ctx: ExecContext = ExecContext.auto(),
        morsel_rows: Int = MORSEL_ROWS,
        trace: Bool = False,
    ) -> Self:
        """`dynamic()`, recording what each phase did; with `trace`, also
        every stage call and morsel read as a timeline slice."""
        return Self(
            ctx.copy(),
            _run_analyze,
            Self._parallelism(ctx),
            morsel_rows,
            Self._serial_because(ctx),
            Self._estimator[estimates](),
            observe=True,
            trace=trace,
        )

    def workers(self) -> Int:
        """How many pipelines a phase lowered for this driver may get."""
        return self._workers

    def phase(self, var source: DynSource) -> Phase:
        """A phase reading `source`: one worker per morsel up to the driver's
        count, and one for a source that cannot say how many it has."""
        var count = source.count()
        var workers = min(self._workers, count) if count > 1 else 1
        var why: String
        if self._workers == 1:
            why = self._serial.copy()
        elif count < 0:
            why = "read in order"
        elif count <= 1:
            why = "one morsel"
        elif count < self._workers:
            why = String(count, " morsels")
        else:
            why = ""
        return Phase(source^, self.ctx.copy(), workers, why^)

    def run(
        mut self, var phases: List[Phase], role: String = "result"
    ) raises -> List[StructArray]:
        """Every batch the phases produce, phase by phase, in morsel order;
        none under `explain`. `role` names what they are for, in a report."""
        self._last = List[Int](capacity=len(phases))
        for i in range(len(phases)):
            self._ran += 1
            phases[i].id = self._ran
            self._last.append(self._ran)
            for j in range(max(0, i - phases[i].follows), i):
                phases[i].inputs.append(phases[j].id)
        return self._run(self, phases^, role)

    def clock(mut self) -> Int:
        """Now, for a span, when the driver observes; 0 otherwise. The kernel
        counters are read too, so the span that follows reports how much
        they moved."""
        if self._observe:
            # One snapshot: a span reports the counters since the latest
            # `clock()`, so spans do not nest.
            self._counted = stats_snapshot()
            return Int(perf_counter_ns())
        return 0

    def span(mut self, name: String, since: Int):
        """Record the work done since `since` (a `clock()`) as `name`, when
        the driver observes."""
        if self._observe:
            var ns = Int(perf_counter_ns()) - since
            self.report.spans.append(
                SpanReport(name, ns, self._ran, stats_since(self._counted))
            )
            if self._trace:
                self.report.events.append(
                    TraceEvent(name, Thread.current_id(), since, ns, -1)
                )

    def estimate(self, input: DynRelation) -> Int:
        """`input`'s estimated rows, when this driver reports them; -1
        otherwise."""
        # Erased for the call only: `input` outlives it.
        return self._estimate(
            Pointer(to=input)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[NoneType]()
        )

    def materialised(
        self, var batches: List[StructArray], var inputs: List[Int]
    ) -> Phase:
        """A phase over batches already produced — a morsel each — by the
        phases `inputs`."""
        var phase = self.phase(TableSource.of(batches^))
        phase.inputs = inputs^
        return phase^

    def entering(mut self, relation: DynRelation):
        """`relation` is about to lower itself: its estimate is what
        `entered` answers until the next relation enters."""
        self._entered = self.estimate(relation)

    def entered(self) -> Int:
        """The estimated rows of the relation lowering last entered — read
        before lowering an input, which enters in turn; -1 when this driver
        reports no estimates."""
        return self._entered

    def last(self) -> List[Int]:
        """The ids of the phases the latest `run` was handed."""
        return self._last.copy()

    def _placeholder(
        self,
        name: String,
        unit: String,
        morsels: Int,
        rows: Int,
        var why: String,
    ) -> Phase:
        """A phase standing in for rows a dry run did not produce: `rows`
        estimated, of `unit`, read by `morsels` morsels."""
        var phase: Phase
        if morsels > 1:
            phase = self.phase(PlannedSource(name, unit, morsels, rows))
        else:
            phase = Phase(
                PlannedSource(name, unit, 1, rows), self.ctx.copy(), why=why^
            )
        phase.estimate = rows
        phase.inputs = self._last.copy()
        return phase^

    def barrier(
        mut self, var phases: List[Phase], role: String, input: DynRelation
    ) raises -> Phase:
        """`phases` — the lowered `input` — as one phase of one worker, for an
        operator that must see every row of them: a sort, a window, a limit's
        last cut.

        Phases of one worker each are read in order, as they run, through a
        `PhasesSource`. Phases with several run first, in parallel, and the
        phase answered reads their batches in morsel order."""
        if not runs_in_parallel(phases):
            if len(phases) == 1:
                return phases.pop()
            return Phase(
                PhasesSource(phases^, self.ctx.copy()),
                self.ctx.copy(),
                why="barrier",
            )
        var batches = self.run(phases^, role)
        if self._dry:
            return self._placeholder(
                "Barrier", "rows", 1, self.estimate(input), "barrier"
            )
        var phase = Phase(
            TableSource.of(batches^), self.ctx.copy(), why="barrier"
        )
        phase.inputs = self._last.copy()
        phase.estimate = self.estimate(input)
        return phase^

    def barrier[
        F: def(ExecContext) raises -> DynOperator
    ](
        mut self,
        var phases: List[Phase],
        role: String,
        input: DynRelation,
        local: F,
        label: String,
    ) raises -> Phase:
        """`barrier`, with `local` first on every worker when the phases run
        on several: a cut each worker can take of what it reads — a limit's
        first rows, a top-N's — before the rows meet."""
        if runs_in_parallel(phases):
            append_each(phases, local, label)
        return self.barrier(phases^, role, input)

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
        var bits = partition_bits(self._workers)
        if self._dry:
            var many = keys > 0 and output_rows > self.morsel_rows
            return self._placeholder(
                "Exchange",
                "groups",
                (1 << bits) if many else 1,
                output_rows,
                "no keys" if keys == 0 else "few rows",
            )
        var rows = 0
        for ref b in batches:
            rows += len(b)
        if keys == 0 or rows <= self.morsel_rows:
            var phase = Phase(
                TableSource([batches^]),
                self.ctx.copy(),
                why="no keys" if keys == 0 else "few rows",
            )
            phase.inputs = self._last.copy()
            phase.estimate = output_rows
            return phase^
        var key_indices = List[Int](capacity=keys)
        for i in range(keys):
            key_indices.append(i)
        var since = self.clock()
        var parts = repartition(batches, key_indices, bits, self.ctx)
        self.span(String("repartition into ", 1 << bits), since)
        var phase = self.phase(TableSource(parts^))
        phase.inputs = self._last.copy()
        phase.estimate = output_rows
        return phase^

    def inspect(
        mut self, plan: DynRelation, bindings: Bindings = Bindings()
    ) raises -> PlanReport:
        """Lower and run `plan` through this driver, and answer its report:
        the phases, and when they ran, the whole run's time and rows."""
        var since = self.clock()
        var result = self.collect(plan.lower(self, bindings), plan.schema())
        if self._observe:
            self.report.ns = self.clock() - since
            self.report.rows = len(result)
        self.report.estimate = self.estimate(plan)
        return self.report.copy()

    def collect(
        mut self,
        var phases: List[Phase],
        schema: Schema,
        role: String = "result",
    ) raises -> StructArray:
        """Run the phases into one batch of `schema`; under `explain`, an
        empty one."""
        var batches = self.run(phases^, role)
        var since = self.clock()
        var result = finish_batches(batches, schema, self.ctx)
        self.span("concatenate the result", since)
        return result^
