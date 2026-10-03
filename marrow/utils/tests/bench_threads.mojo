# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The thread pool's benchmarks: against MAX's `sync_parallelize` where MAX
has a counterpart, and alone where it does not.

Each compared operation has a `marrow` and a `max` row:

- **dispatch** — an empty job on `k` threads, so the time is scheduling cost
  alone: publishing the work, waking or reaching the workers, and the join.
- **stripe** — a vectorised 1M-element `int32` add split into `k` stripes,
  the shape every `ExecContext.stripe` kernel has.
- **wake** — a 4-way job after an idle gap longer than the pool's spin
  budget, so the workers have gone to sleep and have to be woken.

The `bench_threads_*` rows have no MAX counterpart, and each pins one property
a change to the pool can regress:

- **scope_spawn_1k** — spawning cost, one coroutine frame per task.
- **fan_out_raising_1k** — `fan_out`'s failure bookkeeping and claim counter
  over `run`, on empty items.
- **fan_out_skewed** — load balance: every fourth item is 50x the rest, so a
  pool that deals each lane a fixed share runs ~4x slower here.
- **run_100k** — the per-index cost of `run` on indices too small to be worth
  a thread.
- **nested_run** — a `run` inside every index of a `run`.
- **submitters_4** — four threads submitting jobs to one pool at once.
- **io_fan_out** — one 100 us blocking wait per thread of the I/O pool, whose
  threads sleep when idle, so every job wakes the pool.

Lanes, each a pixi task in the `dev` environment:

- `bench-threads` — every row, with the MAX comparison table.
- `bench-threads-history A B` — every row at commits `A` and `B`, each built
  from its own sources. The `max` rows are untouched by any change to the
  pool, so they are the control to normalise machine drift against.

The `max` rows go when MAX leaves the CPU build; until then they are the
baseline the pool has to beat.
"""

from std.algorithm.backend.vectorize import vectorize
from std.benchmark import BenchMetric, keep
from std.math import align_up, ceildiv
from std.time import sleep

from max.algorithm.functional import sync_parallelize

from ..testing import Benchmark, busy_wait_us
from ..threads import TaskScope, ThreadPool

comptime N = 1_000_000


def _widths() -> Int:
    """Every thread the shared pool can put on one job."""
    return ThreadPool.shared()[].concurrency()


# ---------------------------------------------------------------------------
# dispatch: an empty job
# ---------------------------------------------------------------------------


def _bench_dispatch[
    lib: StaticString, gap_us: Int = 0
](mut b: Benchmark, k: Int) raises:
    """An empty job on `k` threads — after `gap_us` of idling first, when
    given, so the workers have gone to sleep. The bodies write nothing shared,
    so the time is the scheduling alone."""
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, k)
    b.extra_info("lib", String(lib))

    @always_inline
    def task(w: Int):
        keep(w)

    @always_inline
    def call() {imm pool, imm k}:
        comptime if gap_us > 0:
            busy_wait_us(gap_us)
        comptime if lib == "marrow":
            pool[].run(k, task, k)
        else:
            sync_parallelize(task, k)

    b.iter(call)
    keep(pool)


def bench_marrow_dispatch_2(mut b: Benchmark) raises:
    _bench_dispatch["marrow"](b, 2)


def bench_max_dispatch_2(mut b: Benchmark) raises:
    _bench_dispatch["max"](b, 2)


def bench_marrow_dispatch_4(mut b: Benchmark) raises:
    _bench_dispatch["marrow"](b, 4)


def bench_max_dispatch_4(mut b: Benchmark) raises:
    _bench_dispatch["max"](b, 4)


def bench_marrow_dispatch_8(mut b: Benchmark) raises:
    _bench_dispatch["marrow"](b, 8)


def bench_max_dispatch_8(mut b: Benchmark) raises:
    _bench_dispatch["max"](b, 8)


def bench_marrow_dispatch_all(mut b: Benchmark) raises:
    _bench_dispatch["marrow"](b, _widths())


def bench_max_dispatch_all(mut b: Benchmark) raises:
    _bench_dispatch["max"](b, _widths())


# ---------------------------------------------------------------------------
# stripe: a vectorised 1M-element add
# ---------------------------------------------------------------------------


def _bench_stripe[lib: StaticString](mut b: Benchmark, k: Int) raises:
    var a = List[Int32](length=N, fill=1)
    var x = List[Int32](length=N, fill=2)
    var c = List[Int32](length=N, fill=0)
    var pa = a.unsafe_ptr()
    var px = x.unsafe_ptr()
    var pc = c.unsafe_ptr()
    var chunk = align_up(ceildiv(N, k), 16)
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", String(lib))

    @always_inline
    def task(w: Int) {imm pa, imm px, imm pc, imm chunk}:
        var start = w * chunk
        var end = min(start + chunk, N)

        @always_inline
        def lane[W: Int](i: Int) {imm pa, imm px, imm pc, imm start}:
            pc.unsafe_offset(start + i).unsafe_store(
                pa.unsafe_offset(start + i).unsafe_load[width=W]()
                + px.unsafe_offset(start + i).unsafe_load[width=W]()
            )

        if end > start:
            vectorize[16](end - start, lane)

    @always_inline
    def call() {imm pool, imm task, imm k}:
        comptime if lib == "marrow":
            pool[].run(k, task, k)
        else:
            sync_parallelize(task, k)

    b.iter(call)
    keep(pool)
    keep(a)
    keep(x)
    keep(c)


def bench_marrow_stripe_2(mut b: Benchmark) raises:
    _bench_stripe["marrow"](b, 2)


def bench_max_stripe_2(mut b: Benchmark) raises:
    _bench_stripe["max"](b, 2)


def bench_marrow_stripe_4(mut b: Benchmark) raises:
    _bench_stripe["marrow"](b, 4)


def bench_max_stripe_4(mut b: Benchmark) raises:
    _bench_stripe["max"](b, 4)


def bench_marrow_stripe_8(mut b: Benchmark) raises:
    _bench_stripe["marrow"](b, 8)


def bench_max_stripe_8(mut b: Benchmark) raises:
    _bench_stripe["max"](b, 8)


def bench_marrow_stripe_all(mut b: Benchmark) raises:
    _bench_stripe["marrow"](b, _widths())


def bench_max_stripe_all(mut b: Benchmark) raises:
    _bench_stripe["max"](b, _widths())


# ---------------------------------------------------------------------------
# wake: a job after the workers have gone to sleep
# ---------------------------------------------------------------------------


# The 200 µs idle gap is part of every iteration on both rows, so the
# difference between them is what waking the workers costs.


def bench_marrow_wake_4(mut b: Benchmark) raises:
    _bench_dispatch["marrow", 200](b, 4)


def bench_max_wake_4(mut b: Benchmark) raises:
    _bench_dispatch["max", 200](b, 4)


# ---------------------------------------------------------------------------
# marrow only: scope and fan_out
# ---------------------------------------------------------------------------


def bench_threads_scope_spawn_1k(mut b: Benchmark) raises:
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, 1_000)

    def body(scope: TaskScope):
        for i in range(1_000):

            def task() {var i}:
                keep(i)

            scope.spawn(task)

    @always_inline
    def call() raises {imm pool}:
        pool[].scope(body)

    b.iter(call)
    keep(pool)


def bench_threads_fan_out_raising_1k(mut b: Benchmark) raises:
    # `fan_out` pays for its failure bookkeeping and a raising body; this is
    # the cost of that over `run`, on 1,000 empty items.
    var pool = ThreadPool.shared()
    var lanes = pool[].concurrency()
    b.throughput(BenchMetric.elements, 1_000)

    def visit(wid: Int, i: Int) raises:
        if i < 0:
            raise Error("unreachable")

    @always_inline
    def call() raises {imm pool, imm lanes}:
        pool[].fan_out(1_000, visit, lanes)

    b.iter(call)
    keep(pool)


def bench_threads_fan_out_skewed(mut b: Benchmark) raises:
    # 64 items on 4 lanes, every fourth 50 us and the rest 1 us: dealt a fixed
    # share per lane, lane 0 would draw all sixteen heavy items (800 us);
    # taken on demand they spread over the lanes (~210 us).
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, 64)

    def visit(wid: Int, i: Int) raises:
        busy_wait_us(50 if i % 4 == 0 else 1)

    @always_inline
    def call() raises {imm pool}:
        pool[].fan_out(64, visit, 4)

    b.iter(call)
    keep(pool)


def bench_threads_run_100k(mut b: Benchmark) raises:
    var pool = ThreadPool.shared()
    var k = pool[].concurrency()
    b.throughput(BenchMetric.elements, 100_000)

    @always_inline
    def task(i: Int):
        keep(i)

    @always_inline
    def call() {imm pool, imm k}:
        pool[].run(100_000, task, k)

    b.iter(call)
    keep(pool)


def bench_threads_nested_run(mut b: Benchmark) raises:
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, 16)

    @always_inline
    def outer(i: Int) {imm pool}:
        @always_inline
        def inner(j: Int):
            keep(j)

        pool[].run(4, inner, 4)

    @always_inline
    def call() {imm pool, imm outer}:
        pool[].run(4, outer, 4)

    b.iter(call)
    keep(pool)


def bench_threads_submitters_4(mut b: Benchmark) raises:
    # Four tasks, each submitting 20 small jobs, so up to four threads push to
    # and pop from the one queue at once.
    var pool = ThreadPool.shared()
    b.throughput(BenchMetric.elements, 4 * 20 * 4)

    def body(scope: TaskScope) {imm pool}:
        for _ in range(4):

            def submitter() {imm pool}:
                for _ in range(20):

                    def task(i: Int):
                        keep(i)

                    pool[].run(4, task, 4)

            scope.spawn(submitter)

    @always_inline
    def call() raises {imm pool, imm body}:
        pool[].scope(body)

    b.iter(call)
    keep(pool)


def bench_threads_io_fan_out(mut b: Benchmark) raises:
    # One 100 us blocking wait per thread of the I/O pool — a fast local read,
    # so every thread must take part and each job wakes the pool.
    var io = ThreadPool.shared_io()
    var lanes = io[].concurrency()
    b.throughput(BenchMetric.elements, lanes)

    def visit(wid: Int, i: Int) raises:
        sleep(0.0001)

    @always_inline
    def call() raises {imm io, imm lanes}:
        io[].fan_out(lanes, visit, lanes)

    b.iter(call)
    keep(io)
