# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a plan's phases are, and what they did: the value `explain()` and
`analyze()` answer.

A plan runs as phases, and a phase's source may read what earlier phases
produced — a barrier's batches, an exchange's partitions, a join's build side.
`PlanReport` is that graph: one `PhaseReport` per phase, in the order the
phases run, each naming the phases it reads (`inputs`). Under `analyze()` it
also holds what each stage and worker did, the work done between phases
(`SpanReport`), and the whole run's time. It renders as text for a reader, as
JSON for a tool, and as an Arrow batch for a query over it.
"""

from emberjson.utils import write_escaped_string

from ..arrays import ListArray
from ..expr.physical import TraceEvent
from ..utils.stats import names as stat_names
from ..builders import Int32Builder, Int64Builder, StringBuilder
from ..tabular import RecordBatch, record_batch


@fieldwise_init
struct StageReport(Copyable, Movable):
    """What one stage did, summed over the phase's workers."""

    var rows_in: Int
    var rows_out: Int
    var batches_in: Int
    var batches_out: Int
    var ns: Int
    """Time in the stage itself, over every worker."""
    var ns_max: Int
    """The most any one worker spent in it."""


@fieldwise_init
struct PhaseReport(Copyable, Movable, Writable):
    """One phase as a plan report shows it."""

    var id: Int
    """Its place in the order the phases ran, from 1."""
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
    """Its source's morsel count, or how many were read; -1 when unknown."""
    var rows: Int
    """The rows its source declares; -1 when unknown."""
    var estimate: Int
    """The rows its source was estimated to read; -1 when not estimated."""
    var output: Int
    """The rows it produced; -1 when it did not run."""
    var ns: Int
    """How long it ran; -1 when it did not run."""
    var stats: List[StageReport]
    """What each stage did, one per entry of `stages`; empty when it did not
    run."""
    var busy: List[Int]
    """How long each worker spent in its stages; empty when it did not
    run."""
    var read_ns: Int
    """Time spent reading the source, over every worker; -1 when it did not
    run."""
    var counters: List[Int]
    """How much each kernel counter moved while it ran, in
    `utils.stats.names()` order; empty unless built with
    `-D MARROW_STATS=true`."""

    def plan(self) -> String:
        """The source and its stages, in order."""
        var out = self.source.copy()
        for ref s in self.stages:
            out += " -> " + s
        return out^

    def skew(self) -> Float64:
        """The busiest worker's time over the mean: 1 is even, `workers`
        is one worker doing everything."""
        var total = 0
        var most = 0
        for t in self.busy:
            total += t
            most = max(most, t)
        if total == 0:
            return 1.0
        return Float64(most) * Float64(len(self.busy)) / Float64(total)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Phase ", self.id, " [", self.role, "] ", self.source)
        writer.write("\n    workers=", self.workers)
        if self.why:
            writer.write(" (", self.why, ")")
        writer.write(" morsels=", _known(self.morsels))
        writer.write(" rows=", _known(self.rows))
        if self.estimate >= 0:
            writer.write(" est=", self.estimate)
            if self.rows > 0 and self.estimate > 0:
                writer.write(" (x", _ratio(self.rows, self.estimate), ")")
        if self.output >= 0:
            writer.write(" out=", self.output)
        if self.ns >= 0:
            writer.write(" time=", _ms(self.ns))
        if len(self.busy) > 0:
            var most = 0
            var total = 0
            for t in self.busy:
                most = max(most, t)
                total += t
            writer.write("\n    busy max=", _ms(most))
            writer.write(" mean=", _ms(total // len(self.busy)))
            writer.write(" skew=", _round2(self.skew()))
            writer.write(" read=", _ms(self.read_ns))
        _write_counters(self.counters, writer)
        for i in range(len(self.stages)):
            writer.write("\n    -> ", self.stages[i])
            if i < len(self.stats):
                ref s = self.stats[i]
                writer.write("  [in=", s.rows_in, " out=", s.rows_out)
                writer.write(" ", _ms(s.ns))
                if self.workers > 1:
                    writer.write(" max=", _ms(s.ns_max))
                writer.write("]")


@fieldwise_init
struct SpanReport(Copyable, Movable):
    """Work done between phases: a hash build, a repartition, the result's
    concatenation."""

    var name: String
    var ns: Int
    var after: Int
    """The phase that ran last before it; 0 before any."""
    var counters: List[Int]
    """How much each kernel counter moved during it, in `stats.STATS`
    order; empty when the build does not count."""


def _write_counters[W: Writer](counters: List[Int], mut writer: W):
    """The counters that moved, on a `counters:` line of their own."""
    var names = stat_names()
    var first = True
    for i in range(len(counters)):
        if counters[i] != 0:
            writer.write("\n    counters:" if first else "")
            writer.write(" ", names[i], "=", counters[i])
            first = False


def _counters_json(counters: List[Int]) -> String:
    """The counters that moved, as a JSON object."""
    var names = stat_names()
    var out = String("{")
    var first = True
    for i in range(len(counters)):
        if counters[i] != 0:
            if not first:
                out += ","
            out += String('"', names[i], '":', counters[i])
            first = False
    return out + "}"


def _known(n: Int) -> String:
    return String(n) if n >= 0 else "?"


def _ms(ns: Int) -> String:
    return String(_round2(Float64(ns) / 1e6), "ms")


def _round2(x: Float64) -> String:
    """`x` with two decimals."""
    var hundredths = Int(x * 100 + 0.5)
    var frac = hundredths % 100
    return String(hundredths // 100, ".", "0" if frac < 10 else "", frac)


def _ratio(a: Int, b: Int) -> String:
    """How many times the larger of `a` and `b` is the smaller."""
    var hi = max(a, b)
    var lo = min(a, b)
    return _round2(Float64(hi) / Float64(lo))


def _position(values: List[Int], value: Int) -> Int:
    for i in range(len(values)):
        if values[i] == value:
            return i
    return -1


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
    """The phases of one run, in the order they ran, the work between them,
    and the whole run."""

    var phases: List[PhaseReport]
    var spans: List[SpanReport]
    var events: List[TraceEvent]
    """Every stage call, morsel read, phase and span as a timeline slice,
    when the run was traced; empty otherwise."""
    var ns: Int
    """The whole run, lowering included; -1 when it did not run."""
    var rows: Int
    """The rows the plan answered; -1 when it did not run."""
    var estimate: Int
    """The rows the plan was estimated to answer; -1 when not estimated."""

    def __init__(out self):
        self.phases = List[PhaseReport]()
        self.spans = List[SpanReport]()
        self.events = List[TraceEvent]()
        self.ns = -1
        self.rows = -1
        self.estimate = -1

    def readers(self, id: Int) -> List[Int]:
        """The phases that read what phase `id` produced."""
        var out = List[Int]()
        for ref p in self.phases:
            for i in p.inputs:
                if i == id:
                    out.append(p.id)
        return out^

    def costs(self) -> List[Tuple[String, Int]]:
        """Where the run's time went, largest first: each phase, each span,
        and what neither accounts for — lowering, building operators."""
        var out = List[Tuple[String, Int]]()
        var accounted = 0
        for ref p in self.phases:
            if p.ns >= 0:
                out.append((String("Phase ", p.id), p.ns))
                accounted += p.ns
        for ref s in self.spans:
            out.append((s.name.copy(), s.ns))
            accounted += s.ns
        if self.ns >= 0:
            out.append((String("lowering and the rest"), self.ns - accounted))
        # Insertion sort: a report has a handful of entries.
        for i in range(1, len(out)):
            var j = i
            while j > 0 and out[j][1] > out[j - 1][1]:
                out.swap_elements(j, j - 1)
                j -= 1
        return out^

    def write_to[W: Writer](self, mut writer: W):
        if self.ns >= 0:
            writer.write("Total ", _ms(self.ns), ", ", self.rows, " rows")
            if self.estimate >= 0:
                writer.write(" (est ", self.estimate, ")")
            var costs = self.costs()
            for i in range(min(5, len(costs))):
                writer.write("\n  ", _ms(costs[i][1]))
                writer.write("  ", costs[i][1] * 100 // max(1, self.ns), "%")
                writer.write("  ", costs[i][0])
            writer.write("\n")
        var span = 0
        for i in range(len(self.phases)):
            ref p = self.phases[i]
            if i > 0:
                writer.write("\n")
            while span < len(self.spans) and self.spans[span].after < p.id:
                writer.write("┄┄ ", self.spans[span].name, " ")
                writer.write(_ms(self.spans[span].ns))
                _write_counters(self.spans[span].counters, writer)
                writer.write("\n")
                span += 1
            writer.write(p)
            var readers = self.readers(p.id)
            if len(readers) > 0:
                writer.write("\n    ═══ read by Phase")
                for j in range(len(readers)):
                    writer.write(", " if j > 0 else " ", readers[j])
        while span < len(self.spans):
            writer.write("\n┄┄ ", self.spans[span].name, " ")
            writer.write(_ms(self.spans[span].ns))
            _write_counters(self.spans[span].counters, writer)
            span += 1

    def to_json(self) -> String:
        """The report as a JSON object: the run's totals, its `phases` and
        its `spans`."""
        var out = String(
            '{"ns":',
            self.ns,
            ',"rows":',
            self.rows,
            ',"estimate":',
            self.estimate,
            ',"phases":[',
        )
        for i in range(len(self.phases)):
            ref p = self.phases[i]
            if i > 0:
                out += ","
            out += String('{"id":', p.id, ',"role":')
            write_escaped_string(p.role, out)
            out += ',"source":'
            write_escaped_string(p.source, out)
            out += ',"stages":['
            for j in range(len(p.stages)):
                if j > 0:
                    out += ","
                out += '{"label":'
                write_escaped_string(p.stages[j], out)
                if j < len(p.stats):
                    ref s = p.stats[j]
                    out += String(
                        ',"rows_in":',
                        s.rows_in,
                        ',"rows_out":',
                        s.rows_out,
                        ',"batches_in":',
                        s.batches_in,
                        ',"batches_out":',
                        s.batches_out,
                        ',"ns":',
                        s.ns,
                        ',"ns_max":',
                        s.ns_max,
                    )
                out += "}"
            out += String('],"workers":', p.workers, ',"why":')
            write_escaped_string(p.why, out)
            out += ',"inputs":' + _ints(p.inputs)
            out += String(
                ',"morsels":',
                p.morsels,
                ',"rows":',
                p.rows,
                ',"estimate":',
                p.estimate,
                ',"output":',
                p.output,
                ',"ns":',
                p.ns,
                ',"busy":',
                _ints(p.busy),
                ',"read_ns":',
                p.read_ns,
                ',"counters":',
                _counters_json(p.counters),
                "}",
            )
        out += '],"spans":['
        for i in range(len(self.spans)):
            ref s = self.spans[i]
            if i > 0:
                out += ","
            out += '{"name":'
            write_escaped_string(s.name, out)
            out += String(',"ns":', s.ns, ',"after":', s.after)
            out += String(',"counters":', _counters_json(s.counters), "}")
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
            out += ',{"name":'
            write_escaped_string(e.name, out)
            out += String(
                ',"ph":"X","pid":1,"tid":',
                _position(threads, e.thread),
                ',"ts":',
                _micros(e.start - origin),
                ',"dur":',
                _micros(e.ns),
            )
            if e.rows >= 0:
                out += String(',"args":{"rows":', e.rows, "}")
            out += "}"
        return out + "]}"

    def to_batch(self) raises -> RecordBatch:
        """The report's phases as a batch, one row per phase; a count nobody
        knows is -1, as in the report. Stage counts are lists aligned with
        `stages`, empty when the phase did not run."""
        var n = len(self.phases)
        var id = Int64Builder(n)
        var role = StringBuilder()
        var source = StringBuilder()
        var stage = StringBuilder()
        var stage_rows = Int64Builder()
        var stage_ns = Int64Builder()
        var stage_offsets = Int32Builder(n + 1)
        var count_offsets = Int32Builder(n + 1)
        stage_offsets.append(0)
        count_offsets.append(0)
        var workers = Int64Builder(n)
        var why = StringBuilder()
        var input = Int64Builder()
        var input_offsets = Int32Builder(n + 1)
        input_offsets.append(0)
        var busy = Int64Builder()
        var busy_offsets = Int32Builder(n + 1)
        busy_offsets.append(0)
        var morsels = Int64Builder(n)
        var rows = Int64Builder(n)
        var estimate = Int64Builder(n)
        var output = Int64Builder(n)
        var ns = Int64Builder(n)
        for ref p in self.phases:
            id.append(Int64(p.id))
            role.append(p.role)
            source.append(p.source)
            for ref s in p.stages:
                stage.append(s)
            stage_offsets.append(Int32(stage.length()))
            for ref s in p.stats:
                stage_rows.append(Int64(s.rows_out))
                stage_ns.append(Int64(s.ns))
            count_offsets.append(Int32(stage_ns.length()))
            workers.append(Int64(p.workers))
            why.append(p.why)
            for i in p.inputs:
                input.append(Int64(i))
            input_offsets.append(Int32(input.length()))
            for t in p.busy:
                busy.append(Int64(t))
            busy_offsets.append(Int32(busy.length()))
            morsels.append(Int64(p.morsels))
            rows.append(Int64(p.rows))
            estimate.append(Int64(p.estimate))
            output.append(Int64(p.output))
            ns.append(Int64(p.ns))
        var counts = count_offsets.finish()
        return record_batch(
            [
                id.finish(),
                role.finish(),
                source.finish(),
                ListArray.from_arrays(stage_offsets.finish(), stage.finish()),
                ListArray.from_arrays(counts.copy(), stage_rows.finish()),
                ListArray.from_arrays(counts, stage_ns.finish()),
                workers.finish(),
                why.finish(),
                ListArray.from_arrays(input_offsets.finish(), input.finish()),
                morsels.finish(),
                rows.finish(),
                estimate.finish(),
                output.finish(),
                ns.finish(),
                ListArray.from_arrays(busy_offsets.finish(), busy.finish()),
            ],
            names=[
                "id",
                "role",
                "source",
                "stages",
                "stage_rows_out",
                "stage_ns",
                "workers",
                "why",
                "inputs",
                "morsels",
                "rows",
                "estimate",
                "output",
                "ns",
                "busy_ns",
            ],
        )
