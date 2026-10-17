# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a plan's phases are, and what they did: the values `explain()` and
`profile()` answer.

A plan runs as phases, and a phase's source may read what earlier phases
produced — a barrier's batches, an exchange's partitions, a join's build side.
`PlanReport` is that graph: one `PhaseReport` per phase, in the order the
phases run, each naming the phases it reads (`inputs`). `PlanProfile` is a
`PlanReport` that ran: one `PhaseRun` per phase, saying what each stage and
worker did, the work done between phases (`SpanReport`), and the whole run's
time. Both render as text for a reader and as JSON for a tool.
"""

from emberjson.utils import write_escaped_string
from std.memory import ArcPointer

from ..utils.stats import Counters


@fieldwise_init
struct StageReport(Copyable, Movable):
    """What one stage did: one worker's, while its pipeline is observed, or
    every worker's, `add`ed."""

    var rows_in: Int
    var rows_out: Int
    var batches_in: Int
    var batches_out: Int
    var ns: Int
    """Time in the stage's own `push` and `drain`, the stages above it
    excluded."""
    var ns_max: Int
    """The most any one worker spent in it."""

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0, 0, 0, 0)

    def write_json(self, mut out: String):
        """Its fields as JSON object members."""
        out += String(
            '"rows_in":',
            self.rows_in,
            ',"rows_out":',
            self.rows_out,
            ',"batches_in":',
            self.batches_in,
            ',"batches_out":',
            self.batches_out,
            ',"ns":',
            self.ns,
            ',"ns_max":',
            self.ns_max,
        )

    def add(mut self, worker: Self):
        """Count one more worker's stage into this one."""
        self.rows_in += worker.rows_in
        self.rows_out += worker.rows_out
        self.batches_in += worker.batches_in
        self.batches_out += worker.batches_out
        self.ns += worker.ns
        self.ns_max = max(self.ns_max, worker.ns)


@fieldwise_init
struct PhaseReport(Copyable, Movable, Writable):
    """One phase as lowering made it."""

    var id: Int
    """Its place in the order the phases run, from 1."""
    var role: String
    """What its rows are for: the result, a join's build side, a barrier's
    input."""
    var source: String
    """Where its morsels come from."""
    var stages: List[String]
    """What each worker's pipeline runs over them, in order."""
    var workers: Int
    var why: String
    """Why it has that many workers, when that is not one per morsel up to
    the driver's."""
    var inputs: List[Int]
    """The phases whose rows its source reads."""
    var morsels: Int
    """Its source's morsel count; -1 when the source cannot say."""
    var rows: Int
    """The rows its source declares; -1 when the source cannot say."""
    var estimate: Int
    """The rows its source is estimated to read, for a source that reads
    other phases' rows; -1 when nobody estimated them."""

    def write_head[W: Writer](self, mut writer: W):
        """Its first two lines: what it reads and how many workers read it."""
        writer.write("Phase ", self.id, " [", self.role, "] ", self.source)
        writer.write("\n    workers=", self.workers)
        if self.why:
            writer.write(" (", self.why, ")")
        writer.write(" morsels=", _known(self.morsels))
        writer.write(" rows=", _known(self.rows))
        if self.estimate >= 0:
            writer.write(" est=", self.estimate)
            if self.rows > 0 and self.estimate > 0:
                var hi = Float64(max(self.rows, self.estimate))
                var lo = Float64(min(self.rows, self.estimate))
                writer.write(" (x", _round2(hi / lo), ")")

    def write_to[W: Writer](self, mut writer: W):
        self.write_head(writer)
        for ref s in self.stages:
            writer.write("\n    -> ", s)

    def write_json(self, mut out: String):
        """Its fields as JSON object members, but for `stages`."""
        out += String('"id":', self.id, ',"role":')
        write_escaped_string(self.role, out)
        out += ',"source":'
        write_escaped_string(self.source, out)
        out += String(',"workers":', self.workers, ',"why":')
        write_escaped_string(self.why, out)
        out += String(
            ',"inputs":',
            _ints(self.inputs),
            ',"morsels":',
            self.morsels,
            ',"rows":',
            self.rows,
            ',"estimate":',
            self.estimate,
        )


@fieldwise_init
struct PhaseRun(Copyable, Movable, Writable):
    """What one phase did when it ran."""

    var morsels_read: Int
    """How many morsels its workers read: fewer than its source holds when a
    stage had all it needed first."""
    var output: Int
    """The rows it produced."""
    var ns: Int
    """How long it ran."""
    var read_ns: Int
    """Time spent reading the source, over every worker."""
    var busy: List[Int]
    """How long each worker spent in its stages."""
    var stages: List[StageReport]
    """What each stage did, every worker's added, one per stage."""
    var counters: Counters
    """How much each kernel counter moved while it ran; empty unless built
    with `-D MARROW_STATS=true`."""

    def skew(self) -> Float64:
        """The busiest worker's time over the mean: 1 is even, the worker
        count is one worker doing everything."""
        var total = 0
        var most = 0
        for t in self.busy:
            total += t
            most = max(most, t)
        if total == 0:
            return 1.0
        return Float64(most) * Float64(len(self.busy)) / Float64(total)

    def write_to[W: Writer](self, mut writer: W):
        """Its own lines: what it read, produced and spent, and the counters
        that moved."""
        writer.write("\n    ran morsels=", self.morsels_read)
        writer.write(" out=", self.output, " time=", _ms(self.ns))
        var most = 0
        var total = 0
        for t in self.busy:
            most = max(most, t)
            total += t
        writer.write("\n    busy max=", _ms(most))
        writer.write(" mean=", _ms(total // max(1, len(self.busy))))
        writer.write(" skew=", _round2(self.skew()))
        writer.write(" read=", _ms(self.read_ns))
        if self.counters.moved():
            writer.write("\n    counters: ", self.counters)

    def write_json(self, mut out: String):
        """Its fields as JSON object members, but for `stages`."""
        out += String(
            '"morsels_read":',
            self.morsels_read,
            ',"output":',
            self.output,
            ',"ns":',
            self.ns,
            ',"busy":',
            _ints(self.busy),
            ',"read_ns":',
            self.read_ns,
            ',"counters":',
            self.counters.to_json(),
        )


@fieldwise_init
struct TraceEvent(Copyable, Movable):
    """One slice of a timeline: what ran, on which thread, from when and for
    how long, over how many rows (-1 when none)."""

    var name: String
    var thread: Int
    """The OS thread it ran on (`Thread.current_id`)."""
    var start: Int
    """`perf_counter_ns` when it began."""
    var ns: Int
    var rows: Int


comptime Trail = ArcPointer[List[TraceEvent]]
"""One pipeline's timeline slices, appended as it runs."""


@fieldwise_init
struct SpanReport(Copyable, Movable, Writable):
    """Work done between phases: a hash build, a repartition, the result's
    concatenation."""

    var name: String
    var ns: Int
    var after: Int
    """The phase that ran last before it; 0 before any."""
    var counters: Counters
    """How much each kernel counter moved during it; empty when the build
    does not count."""

    def write_to[W: Writer](self, mut writer: W):
        writer.write("┄┄ ", self.name, " ", _ms(self.ns))
        if self.counters.moved():
            writer.write("\n    counters: ", self.counters)


def _known(n: Int) -> String:
    return String(n) if n >= 0 else "?"


def _ms(ns: Int) -> String:
    return String(_round2(Float64(ns) / 1e6), "ms")


def _round2(x: Float64) -> String:
    """`x` with two decimals."""
    var hundredths = Int(x * 100 + 0.5)
    var frac = hundredths % 100
    return String(hundredths // 100, ".", "0" if frac < 10 else "", frac)


def _micros(ns: Int) -> String:
    """`ns` in microseconds, with three decimals."""
    var thousandths = String(ns % 1000)
    while thousandths.byte_length() < 3:
        thousandths = "0" + thousandths
    return String(ns // 1000, ".", thousandths)


def _ints(values: List[Int]) -> String:
    var out = String("[")
    for i in range(len(values)):
        out += String("," if i > 0 else "", values[i])
    return out + "]"


struct PlanReport(Copyable, Movable, Writable):
    """The phases a plan lowers into, in the order they run, and the rows the
    plan is estimated to answer."""

    var phases: List[PhaseReport]
    var estimate: Int
    """The rows the plan is estimated to answer; -1 when not estimated."""

    def __init__(out self, var phases: List[PhaseReport], estimate: Int):
        self.phases = phases^
        self.estimate = estimate

    def readers(self, id: Int) -> List[Int]:
        """The phases that read what phase `id` produced."""
        var out = List[Int]()
        for ref p in self.phases:
            for i in p.inputs:
                if i == id:
                    out.append(p.id)
        return out^

    def write_readers[W: Writer](self, mut writer: W, id: Int):
        """The line naming the phases that read phase `id`, when any do."""
        var readers = self.readers(id)
        if len(readers) > 0:
            writer.write("\n    ═══ read by Phase")
            for j in range(len(readers)):
                writer.write(", " if j > 0 else " ", readers[j])

    def write_to[W: Writer](self, mut writer: W):
        if self.estimate >= 0:
            writer.write("Estimated rows: ", self.estimate, "\n")
        for i in range(len(self.phases)):
            if i > 0:
                writer.write("\n")
            writer.write(self.phases[i])
            self.write_readers(writer, self.phases[i].id)

    def to_json(self) -> String:
        """The report as a JSON object: the plan's `estimate` and its
        `phases`."""
        var out = String('{"estimate":', self.estimate, ',"phases":[')
        for i in range(len(self.phases)):
            ref p = self.phases[i]
            out += "{" if i == 0 else ",{"
            p.write_json(out)
            out += ',"stages":['
            for j in range(len(p.stages)):
                out += '{"label":' if j == 0 else ',{"label":'
                write_escaped_string(p.stages[j], out)
                out += "}"
            out += "]}"
        return out + "]}"


struct PlanProfile(Copyable, Movable, Writable):
    """A plan that ran: its report, what each phase did, the work between
    them, and the whole run."""

    var report: PlanReport
    var runs: List[PhaseRun]
    """What each phase did, one per `report.phases` entry, in order."""
    var spans: List[SpanReport]
    var events: List[TraceEvent]
    """Every stage call, morsel read, phase and span as a timeline slice,
    when the run was traced; empty otherwise."""
    var ns: Int
    """The whole run, lowering included."""
    var rows: Int
    """The rows the plan answered."""

    def __init__(
        out self,
        var report: PlanReport,
        var runs: List[PhaseRun],
        var spans: List[SpanReport],
        var events: List[TraceEvent],
        ns: Int,
        rows: Int,
    ):
        debug_assert(len(runs) == len(report.phases), "a run per phase")
        self.report = report^
        self.runs = runs^
        self.spans = spans^
        self.events = events^
        self.ns = ns
        self.rows = rows

    def costs(self) -> List[Tuple[String, Int]]:
        """Where the run's time went, largest first: each phase, each span,
        and what neither accounts for — lowering, building operators."""
        var out = List[Tuple[String, Int]]()
        var accounted = 0
        for i in range(len(self.runs)):
            out.append(
                (String("Phase ", self.report.phases[i].id), self.runs[i].ns)
            )
            accounted += self.runs[i].ns
        for ref s in self.spans:
            out.append((s.name.copy(), s.ns))
            accounted += s.ns
        out.append((String("lowering and the rest"), self.ns - accounted))
        # Insertion sort: a profile has a handful of entries.
        for i in range(1, len(out)):
            var j = i
            while j > 0 and out[j][1] > out[j - 1][1]:
                out.swap_elements(j, j - 1)
                j -= 1
        return out^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Total ", _ms(self.ns), ", ", self.rows, " rows")
        if self.report.estimate >= 0:
            writer.write(" (est ", self.report.estimate, ")")
        var costs = self.costs()
        for i in range(min(5, len(costs))):
            writer.write("\n  ", _ms(costs[i][1]))
            writer.write("  ", costs[i][1] * 100 // max(1, self.ns), "%")
            writer.write("  ", costs[i][0])
        writer.write("\n")
        var span = 0
        for i in range(len(self.runs)):
            ref p = self.report.phases[i]
            ref run = self.runs[i]
            if i > 0:
                writer.write("\n")
            while span < len(self.spans) and self.spans[span].after < p.id:
                writer.write(self.spans[span], "\n")
                span += 1
            p.write_head(writer)
            writer.write(run)
            for j in range(len(p.stages)):
                ref s = run.stages[j]
                writer.write("\n    -> ", p.stages[j])
                writer.write("  [in=", s.rows_in, " out=", s.rows_out)
                writer.write(" ", _ms(s.ns))
                if p.workers > 1:
                    writer.write(" max=", _ms(s.ns_max))
                writer.write("]")
            self.report.write_readers(writer, p.id)
        while span < len(self.spans):
            writer.write("\n", self.spans[span])
            span += 1

    def to_json(self) -> String:
        """The profile as a JSON object: the run's totals, its `phases` —
        each what lowering made and what it did — and its `spans`."""
        var out = String(
            '{"ns":',
            self.ns,
            ',"rows":',
            self.rows,
            ',"estimate":',
            self.report.estimate,
            ',"phases":[',
        )
        for i in range(len(self.runs)):
            ref p = self.report.phases[i]
            ref run = self.runs[i]
            out += "{" if i == 0 else ",{"
            p.write_json(out)
            out += ","
            run.write_json(out)
            out += ',"stages":['
            for j in range(len(p.stages)):
                out += '{"label":' if j == 0 else ',{"label":'
                write_escaped_string(p.stages[j], out)
                out += ","
                run.stages[j].write_json(out)
                out += "}"
            out += "]}"
        out += '],"spans":['
        for i in range(len(self.spans)):
            ref s = self.spans[i]
            out += '{"name":' if i == 0 else ',{"name":'
            write_escaped_string(s.name, out)
            out += String(',"ns":', s.ns, ',"after":', s.after)
            out += String(',"counters":', s.counters.to_json(), "}")
        return out + "]}"

    def to_trace_json(self) -> String:
        """The traced run as Chrome trace events — one track per thread, the
        calling thread first — which Perfetto (ui.perfetto.dev) and
        `chrome://tracing` open."""
        var origin = Int.MAX
        for ref e in self.events:
            origin = min(origin, e.start)
        var threads = List[Int]()
        for ref e in self.events:
            if e.thread not in threads:
                threads.append(e.thread)
        var out = String('{"displayTimeUnit":"ms","traceEvents":[')
        for t in range(len(threads)):
            if t > 0:
                out += ","
            out += String(
                '{"name":"thread_name","ph":"M","pid":1,"tid":',
                t,
                ',"args":{"name":"',
                "driver" if t == 0 else String("worker ", t),
                '"}}',
            )
        for ref e in self.events:
            var tid = 0
            while threads[tid] != e.thread:
                tid += 1
            out += ',{"name":'
            write_escaped_string(e.name, out)
            out += String(
                ',"ph":"X","pid":1,"tid":',
                tid,
                ',"ts":',
                _micros(e.start - origin),
                ',"dur":',
                _micros(e.ns),
            )
            if e.rows >= 0:
                out += String(',"args":{"rows":', e.rows, "}")
            out += "}"
        return out + "]}"
