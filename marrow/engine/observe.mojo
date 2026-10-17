# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What `analyze` records while a phase runs.

A phase is observed by wrapping: each stage in an `ObservedOperator`, the
source in an `ObservedSource`, so a run nobody observes executes no timing
code. A `Watch` holds what one phase's wrappers record, and answers what the
phase did once it has run.
"""

from std.atomic import Atomic
from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..arrays import StructArray
from ..execution import ExecContext
from ..utils.threads import Thread
from .core import Datum, DynOperator, Morsel, Operator, Pipeline
from ..utils.stats import Counters
from .report import PhaseRun, StageReport, TraceEvent, Trail
from .sources import DynSource, Source, rows_of


comptime Reads = ArcPointer[List[Optional[TraceEvent]]]
"""A source's reads as timeline slices: one slot per morsel, none for a
morsel no worker read."""

comptime Observation = ArcPointer[List[StageReport]]
"""A pipeline's per-stage counts, shared with whoever asked to observe it."""


struct ObservedOperator(Operator):
    """A stage of an observed pipeline: the operator it wraps, and the
    stage's entry of the pipeline's counts, updated on every call. One worker
    owns a pipeline, so nothing else writes them while it runs."""

    var _op: DynOperator
    var _stats: Observation
    var _stage: Int
    var _trail: Optional[Trail]
    """Where each call goes as a timeline slice, when traced."""
    var _label: String

    def __init__(
        out self,
        var op: DynOperator,
        stats: Observation,
        stage: Int,
        trail: Optional[Trail],
        var label: String,
    ):
        self._op = op^
        self._stats = stats
        self._stage = stage
        self._trail = trail
        self._label = label^

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        var since = Int(perf_counter_ns())
        var produced = self._op.push(morsel)
        var ns = Int(perf_counter_ns()) - since
        ref s = self._stats[][self._stage]
        s.rows_in += len(morsel.batch)
        s.batches_in += 1
        s.ns += ns
        if produced:
            s.rows_out += produced.value().length(len(morsel.batch))
            s.batches_out += 1
        if self._trail:
            self._trail.value()[].append(
                TraceEvent(
                    self._label,
                    Thread.current_id(),
                    since,
                    ns,
                    len(morsel.batch),
                )
            )
        return produced^

    def drain(mut self) raises -> Optional[Datum]:
        var since = Int(perf_counter_ns())
        var produced = self._op.drain()
        var ns = Int(perf_counter_ns()) - since
        ref s = self._stats[][self._stage]
        s.ns += ns
        if produced:
            s.rows_out += produced.value().length(1)
            s.batches_out += 1
        if self._trail:
            self._trail.value()[].append(
                TraceEvent(
                    self._label + " drain", Thread.current_id(), since, ns, -1
                )
            )
        return produced^

    def done(self) -> Bool:
        return self._op.done()


struct ObservedSource(Source):
    """A source whose reads are timed, into a total several workers add to
    at once, and with a `trail` recorded as timeline slices: one slot per
    morsel, so the workers reading at once never write the same one."""

    var _source: DynSource
    var _ns: ArcPointer[Atomic[Int64]]
    var _reads: Optional[Reads]

    def __init__(
        out self,
        var source: DynSource,
        ns: ArcPointer[Atomic[Int64]],
        reads: Optional[Reads] = None,
    ):
        self._source = source^
        self._ns = ns
        self._reads = reads
        if self._reads:
            var count = self._source.count()
            for _ in range(max(0, count)):
                self._reads.value()[].append(None)

    def count(self) -> Int:
        return self._source.count()

    def rows(self) -> Int:
        return self._source.rows()

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var since = Int(perf_counter_ns())
        var batches = self._source.read(i, ctx)
        var ns = Int(perf_counter_ns()) - since
        _ = self._ns[].fetch_add(Int64(ns))
        if self._reads:
            var event = TraceEvent(
                String("read morsel ", i),
                Thread.current_id(),
                since,
                ns,
                rows_of(batches),
            )
            ref reads = self._reads.value()[]
            if i < len(reads) and self._source.count() >= 0:
                reads[i] = event^
            else:
                # A source of unknown count is read on one thread, in order.
                reads.append(event^)
        return batches^

    def describe(self) -> String:
        return self._source.describe()


struct Watch(Movable):
    """What one phase's observed stages and source record as it runs: each
    worker's per-stage counts, the source's read time, and with a trace every
    call as a timeline slice."""

    var _stages: List[Observation]
    """One per worker."""
    var _trails: List[Trail]
    """Each worker's stage calls, when traced."""
    var _reads: List[Reads]
    """The source's reads, when traced."""
    var _reading: ArcPointer[Atomic[Int64]]
    var _trace: Bool

    def __init__(out self, trace: Bool):
        self._stages = List[Observation]()
        self._trails = List[Trail]()
        self._reads = List[Reads]()
        self._reading = ArcPointer(Atomic[Int64](0))
        self._trace = trace

    def _trail(mut self) -> Optional[Trail]:
        """A timeline of its own, for a wrapper to record into, when
        traced."""
        if not self._trace:
            return None
        self._trails.append(Trail(List[TraceEvent]()))
        return self._trails[len(self._trails) - 1]

    def pipeline(mut self, mut pipeline: Pipeline, labels: List[String]):
        """Observe `pipeline`, one worker's, its stages named by `labels`."""
        var counts = Observation(List[StageReport](capacity=len(labels)))
        for _ in range(len(labels)):
            counts[].append(StageReport.zero())
        var trail = self._trail()

        def observe(var op: DynOperator, stage: Int) {imm} -> DynOperator:
            return ObservedOperator(
                op^, counts, stage, trail, labels[stage].copy()
            )

        pipeline.wrap(observe)
        self._stages.append(counts^)

    def source(mut self, var source: DynSource) -> DynSource:
        """`source`, its reads timed and, when traced, recorded."""
        var reads = Optional[Reads](None)
        if self._trace:
            self._reads.append(Reads(List[Optional[TraceEvent]]()))
            reads = self._reads[len(self._reads) - 1]
        return ObservedSource(source^, self._reading, reads)

    def record(
        self, morsels_read: Int, output: Int, ns: Int, var counters: Counters
    ) -> PhaseRun:
        """What the phase did: its workers' stage counts and busy time and its
        source's read time, beside what the driver measured — `morsels_read`,
        `output` rows, `ns` and the `counters` that moved. Read once the phase
        has run."""
        var busy = List[Int](capacity=len(self._stages))
        for ref w in self._stages:
            var total = 0
            for ref s in w[]:
                total += s.ns
            busy.append(total)
        var stages = List[StageReport]()
        if len(self._stages) > 0:
            for stage in range(len(self._stages[0][])):
                var sum = StageReport.zero()
                for ref w in self._stages:
                    sum.add(w[][stage])
                stages.append(sum^)
        return PhaseRun(
            morsels_read,
            output,
            ns,
            Int(self._reading[].load()),
            busy^,
            stages^,
            counters^,
        )

    def trace(
        self,
        mut events: List[TraceEvent],
        var name: String,
        start: Int,
        ns: Int,
    ):
        """With a trace, the phase's own slice, `name` from `start` for `ns`,
        and every slice its wrappers recorded, into `events`."""
        if self._trace:
            events.append(TraceEvent(name^, Thread.current_id(), start, ns, -1))
            for ref trail in self._trails:
                for ref e in trail[]:
                    events.append(e.copy())
            for ref reads in self._reads:
                for ref e in reads[]:
                    if e:
                        events.append(e.value().copy())
