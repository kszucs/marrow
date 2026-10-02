# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The thread pool's tests: the POSIX primitives, `run`, `scope`, `fan_out`,
pool construction and lifetime, and `ExecContext` dispatching through a pool.

Lanes, each a pixi task in the `dev` environment:

- `test_threads` — this file and `test_execution.mojo`, once.
- `test_threads_stress` — this file built once and run in 30 fresh processes
  (`--repeat`), every soak loop 50x longer (`MARROW_THREADS_STRESS`): a few
  minutes, for races that show once in a hundred runs.
- `test_tsan` — this file and `test_execution.mojo` under ThreadSanitizer.
- `test_threads_asan` (in the `asan` environment) — the same under
  AddressSanitizer and LeakSanitizer: anything still allocated at exit fails.

A race found here is almost never deterministic: reproduce it with the stress
lane, and under TSAN with `pixi run -e dev test_tsan --repeat 10`.

Values a test shares with a bare `Thread` are borrowed and kept alive with
`keep` after the join, never shared through `ArcPointer`: the last drop of an
`ArcPointer` synchronises on a fence, which TSAN does not model, so dropping it
on another thread reads as a race.
"""

from std.atomic import Atomic
from std.collections import Set
from std.benchmark import keep
from std.ffi import c_int, external_call
from std.memory import ArcPointer
from std.memory.alloc import unsafe_alloc
from std.os import makedirs, setenv, unsetenv
from std.sys import CompilationTarget, get_defined_bool, get_defined_int
from std.tempfile import TemporaryDirectory
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.time import perf_counter_ns, sleep

from ..threads import (
    Condition,
    Mutex,
    TaskScope,
    Thread,
    ThreadPool,
    _SharedPool,
    _cgroup_cpu_limit,
    _start_with_fewer,
)
from ...errors import DynError, IndexError, IOError, KeyError
from ...execution import ExecContext


comptime STRESS = get_defined_int["MARROW_THREADS_STRESS", 1]()
"""How many times longer every soak loop runs; `test_threads_stress` sets it."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _visits(pool: ThreadPool, n: Int, max_threads: Int) -> List[Int]:
    var counts = List[Int](length=n, fill=0)

    def body(i: Int) {mut counts}:
        counts[i] += 1

    pool.run(n, body, max_threads)
    return counts^


def _all_once(counts: List[Int]) -> Bool:
    return counts == List[Int](length=len(counts), fill=1)


def _distinct(ids: List[Int]) -> Int:
    return len(Set[Int](ids))


def _assert_ran_on(pool: ThreadPool, ids: List[Int]) raises:
    """Every id is one of `pool`'s workers or the calling thread."""
    var allowed = pool.thread_ids()
    allowed.append(Thread.current_id())
    for id in ids:
        assert_true(id in allowed)


def _pause_us(us: Int):
    var end = perf_counter_ns() + us * 1000
    while perf_counter_ns() < end:
        pass


def _eventually[P: def() -> Bool](ready: P) -> Bool:
    """Poll `ready` for up to two seconds."""
    var end = perf_counter_ns() + 2_000_000_000
    while perf_counter_ns() < end:
        if ready():
            return True
    return False


def _await(flag: Pointer[Atomic[Int64], _], value: Int) -> Bool:
    """Poll `flag` for `value` for up to two seconds."""

    def reached() {imm flag, imm value} -> Bool:
        return flag[].load() == Int64(value)

    return _eventually(reached)


def _all_asleep(pool: ThreadPool) -> Bool:
    """Poll for up to two seconds for every worker of `pool` to be asleep."""

    def asleep() {imm pool} -> Bool:
        return pool.sleeping() == pool.size()

    return _eventually(asleep)


def _noop():
    pass


def _thread_of_each(
    pool: ThreadPool, n: Int, max_threads: Int, pause_us: Int = 0
) -> List[Int]:
    """Run `n` indices on `pool`, each taking `pause_us`; which thread ran
    each."""
    var ids = List[Int](length=n, fill=0)

    def body(i: Int) {mut ids, imm pause_us}:
        ids[i] = Thread.current_id()
        _pause_us(pause_us)

    pool.run(n, body, max_threads)
    return ids^


def _spawned(pool: ThreadPool, n: Int) raises -> Int:
    """Spawn `n` tasks on a scope of `pool`; how many ran."""
    var hits = Atomic[Int64](0)
    var hp = Pointer(to=hits)

    def body(scope: TaskScope) {imm hp, imm n}:
        for _ in range(n):

            def task() {imm hp}:
                _ = hp[].fetch_add(1)

            scope.spawn(task)

    pool.scope(body)
    return Int(hits.load())


comptime _Count = Pointer[Atomic[Int64], MutUntrackedOrigin]


struct _Tally(Movable):
    """Counts its own destructions, to prove an erased closure is dropped."""

    var drops: _Count

    def __init__(out self, drops: _Count):
        self.drops = drops

    def __deinit__(deinit self):
        _ = self.drops[].fetch_add(1)


def _spawn_tallies(scope: TaskScope, n: Int, drops: _Count, runs: _Count):
    """Spawn `n` tasks, each owning a `_Tally` and counting its run. The
    pointers are captured by value: the tasks outlive this call."""
    for _ in range(n):
        var tally = _Tally(drops)

        def task() {var tally^, var runs}:
            keep(tally)
            _ = runs[].fetch_add(1)

        scope.spawn(task^)


def _names_on_workers(pool: ThreadPool) -> List[String]:
    """Run indices on `pool` until they have spread; per worker, the name it
    reported, or empty if it ran none."""
    var ids = List[Int](length=64, fill=0)
    var names = List[String](length=64, fill=String())

    def body(i: Int) {mut ids, mut names}:
        ids[i] = Thread.current_id()
        names[i] = Thread.current_name()
        _pause_us(200)

    pool.run(64, body, pool.concurrency())
    var workers = pool.thread_ids()
    var seen = List[String](length=len(workers), fill=String())
    for i in range(64):
        for k in range(len(workers)):
            if workers[k] == ids[i]:
                seen[k] = names[i]
    return seen^


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


comptime TSAN = get_defined_bool["MARROW_TSAN", False]()
"""The fork tests do not run under TSAN, which kills a forked child of a
threaded process once it starts a thread."""


def _fork() -> Int:
    return Int(external_call["fork", c_int]())


def _exit_child(code: Int):
    """End a forked child at once — no exit handlers, no flushing — so nothing
    of the parent's run happens twice."""
    external_call["_exit", NoneType](c_int(code))


def _child_exit_code(pid: Int) -> Int:
    """Wait up to a minute for forked child `pid`. Its exit code, or -1 if it
    crashed, or hung and was killed — a join on a thread the child does not
    have would hang."""
    var status = c_int(0)
    var deadline = perf_counter_ns() + 60_000_000_000
    var code = -1
    while True:
        var done = external_call["waitpid", c_int](
            c_int(pid), Pointer(to=status), c_int(1)  # WNOHANG
        )
        if done == c_int(pid):
            # Exited normally: the low seven bits are 0, the code above them.
            if Int(status) & 0x7F == 0:
                code = (Int(status) >> 8) & 0xFF
            break
        if done < 0 or perf_counter_ns() > deadline:
            _ = external_call["kill", c_int](c_int(pid), c_int(9))
            _ = external_call["waitpid", c_int](
                c_int(pid), Pointer(to=status), c_int(0)
            )
            break
        sleep(0.005)
    return code


def _concurrency_in_child[io: Bool](value: String) -> Int:
    """In a forked child, set the shared pool's size variable to `value` —
    unset it for an empty one — and exit with the concurrency of the pool
    the child's first `shared()` starts."""
    comptime variable = "MARROW_IO_THREADS" if io else "MARROW_NUM_THREADS"
    var pid = _fork()
    if pid == 0:
        if value:
            _ = setenv(variable, value)
        else:
            _ = unsetenv(variable)
        comptime if io:
            _exit_child(ThreadPool.shared_io()[].concurrency())
        else:
            _exit_child(ThreadPool.shared()[].concurrency())
    return _child_exit_code(pid)


# ---------------------------------------------------------------------------
# Mutex, Condition, Thread
# ---------------------------------------------------------------------------


def test_threads_mutex_serialises_increments() raises:
    # A plain Int, incremented by eight threads released together, so they
    # overlap: only mutual exclusion keeps every increment. Not scaled by
    # STRESS — the lock is libc's, and overlap, not length, is what catches a
    # broken one.
    var mutex = Mutex()
    var total = List[Int](length=1, fill=0)
    var tp = Pointer(to=total).unsafe_origin_cast[MutUntrackedOrigin]()
    var go = Atomic[Int64](0)
    var gp = Pointer(to=go)
    var threads = List[Thread]()
    for _ in range(8):

        def bump() {imm mutex, imm tp, imm gp}:
            while gp[].load() == 0:
                pass
            for _ in range(20_000):
                mutex.lock()
                tp[][0] += 1
                mutex.unlock()

        threads.append(Thread(bump))
    go.store(1)
    while threads:
        threads.pop().join()
    keep(mutex)
    keep(tp)
    keep(gp)
    assert_equal(total[0], 8 * 20_000)


def test_threads_condition_signal_hands_off() raises:
    var mutex = Mutex()
    var cond = Condition()
    var slot = List[Int](length=1, fill=0)
    var sp = Pointer(to=slot).unsafe_origin_cast[MutUntrackedOrigin]()

    def producer() {imm mutex, imm cond, imm sp}:
        with mutex.locked():
            sp[][0] = 42
            cond.signal()

    with mutex.locked() as guard:
        # Started under the lock, so the producer can only signal once this
        # thread waits — the hand-off, not a lucky early store.
        var t = Thread(producer)
        while sp[][0] == 0:
            cond.wait(guard)
        t^.join()
    keep(mutex)
    keep(cond)
    assert_equal(slot[0], 42)


def test_threads_condition_broadcast_wakes_every_waiter() raises:
    var mutex = Mutex()
    var cond = Condition()
    var go = Atomic[Int64](0)
    var woken = Atomic[Int64](0)
    var ready = Atomic[Int64](0)
    var gp = Pointer(to=go)
    var wp = Pointer(to=woken)
    var rp = Pointer(to=ready)
    var threads = List[Thread]()
    for _ in range(6):

        def waiter() {imm mutex, imm cond, imm gp, imm wp, imm rp}:
            with mutex.locked() as guard:
                _ = rp[].fetch_add(1)
                while gp[].load() == 0:
                    cond.wait(guard)
            _ = wp[].fetch_add(1)

        threads.append(Thread(waiter))
    assert_true(_await(rp, 6))
    with mutex.locked():
        go.store(1)
        cond.broadcast()
    while threads:
        threads.pop().join()
    keep(mutex)
    keep(cond)
    keep(gp)
    keep(wp)
    keep(rp)
    assert_equal(woken.load(), 6)


def _locked_return(mutex: Mutex) -> Int:
    with mutex.locked():
        return 1


def test_threads_mutex_guard_releases_on_every_exit() raises:
    var mutex = Mutex()
    # Taken on entering the block, not when the hold is made.
    var pending = mutex.locked()
    assert_true(mutex.try_lock())
    mutex.unlock()
    keep(pending)
    with mutex.locked():
        assert_false(mutex.try_lock())
    assert_true(mutex.try_lock())
    mutex.unlock()
    assert_equal(_locked_return(mutex), 1)
    assert_true(mutex.try_lock())
    mutex.unlock()
    with assert_raises(contains="inside"):
        with mutex.locked():
            raise Error("inside")
    assert_true(mutex.try_lock())
    mutex.unlock()


def test_threads_join_waits_for_the_body() raises:
    var done = Atomic[Int64](0)
    var dp = Pointer(to=done)

    def slow() {imm dp}:
        _pause_us(1_000)
        dp[].store(1)

    var t = Thread(slow)
    t^.join()
    keep(dp)
    assert_equal(done.load(), 1)


def test_threads_detached_thread_still_runs() raises:
    var ran = Atomic[Int64](0)
    var rp = Pointer(to=ran)

    def body() {imm rp}:
        rp[].store(1)

    _ = Thread(body)  # destroyed at once, so detached
    assert_true(_await(rp, 1))
    keep(rp)


def test_threads_thread_ids_are_distinct_and_current() raises:
    var seen = Atomic[Int64](0)
    var sp = Pointer(to=seen)

    def record() {imm sp}:
        sp[].store(Int64(Thread.current_id()))

    var t = Thread(record)
    var id = t.id()
    t^.join()
    keep(sp)
    assert_equal(Int(seen.load()), id)
    assert_true(id != Thread.current_id())


def test_threads_thread_rejects_a_bad_stack_size() raises:
    # The body must be destroyed, not leaked and not run.
    var drops = Atomic[Int64](0)
    var runs = Atomic[Int64](0)
    var drops_ptr = Pointer(to=drops).unsafe_origin_cast[MutUntrackedOrigin]()
    var runs_ptr = Pointer(to=runs).unsafe_origin_cast[MutUntrackedOrigin]()
    var tally = _Tally(drops_ptr)

    def body() {var tally^, imm runs_ptr}:
        keep(tally)
        _ = runs_ptr[].fetch_add(1)

    with assert_raises(contains="InvalidError: Thread: stack size"):
        _ = Thread(body^, stack_size=-1)
    assert_equal(runs.load(), 0)
    assert_equal(drops.load(), 1)


def test_threads_thread_pin() raises:
    # Pinning is Linux-only, and says so elsewhere rather than doing nothing.
    var t = Thread(_noop)
    comptime if CompilationTarget.is_linux():
        t.pin(0)
        with assert_raises(contains="IndexError: Thread.pin"):
            t.pin(-1)
        with assert_raises(contains="IndexError: Thread.pin"):
            t.pin(1 << 20)
    else:
        with assert_raises(contains="NotImplementedError: Thread.pin"):
            t.pin(0)
    t^.join()


# ---------------------------------------------------------------------------
# ThreadPool.run
# ---------------------------------------------------------------------------


def test_threads_run_visits_every_index_once() raises:
    var pool = ThreadPool(4)
    for n in [0, 1, 2, pool.size(), pool.size() + 1, 10 * pool.size(), 100_000]:
        var counts = _visits(pool, n, pool.concurrency())
        assert_equal(len(counts), n)
        assert_true(_all_once(counts))


def test_threads_run_degenerate_arguments() raises:
    # Zero, negative and oversized budgets all run every index; a budget of
    # at most one keeps it on the calling thread; `n <= 0` runs nothing.
    var pool = ThreadPool(3)
    var me = Thread.current_id()
    for cap in [-5, 0, 1]:
        var ids = _thread_of_each(pool, 50, cap)
        assert_equal(_distinct(ids), 1)
        assert_equal(ids[0], me)
    assert_true(_all_once(_visits(pool, 1_000, 1_000)))
    var calls = Atomic[Int64](0)
    var cp = Pointer(to=calls)

    def count(i: Int) {imm cp}:
        _ = cp[].fetch_add(1)

    for n in [0, -1, -100]:
        pool.run(n, count, 4)
    assert_equal(calls.load(), 0)


def test_threads_run_honours_max_threads() raises:
    var pool = ThreadPool(4)
    for cap in [2, 3, 4]:
        assert_true(_distinct(_thread_of_each(pool, 4_000, cap, 2)) <= cap)


def test_threads_run_uses_the_workers() raises:
    # Slow enough per index that a worker must join in: the job is not simply
    # finished by the caller.
    var pool = ThreadPool(3)
    var ids = _thread_of_each(pool, 64, 4, 200)
    assert_true(_distinct(ids) >= 2)
    _assert_ran_on(pool, ids)


def test_threads_run_body_capturing_only_a_pointer() raises:
    # A closure capturing nothing but a pointer is passed in registers, so a
    # pointer to the *argument* dangles once the callee that took it returns;
    # `run` once stored exactly that and the workers crashed calling it.
    var pool = ThreadPool(4)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)
    var expect = 0
    for rep in range(200 * STRESS):
        var n = 1 + (rep * 7) % 40

        def add(i: Int) {imm tp}:
            _ = tp[].fetch_add(1)

        pool.run(n, add, 1 + rep % 6)
        expect += n
    assert_equal(Int(total.load()), expect)


def test_threads_run_nested_does_not_deadlock() raises:
    # Two workers and three levels of `run`, each asking for every thread:
    # without help-while-waiting the inner calls would park every worker on a
    # join whose helpers nobody is free to run.
    var pool = ThreadPool(2)
    var counts = List[Int](length=64, fill=0)

    def outer(i: Int) {mut counts, imm pool}:
        def middle(j: Int) {mut counts, imm pool, imm i}:
            def inner(k: Int) {mut counts, imm i, imm j}:
                counts[i * 16 + j * 4 + k] += 1

            pool.run(4, inner, 3)

        pool.run(4, middle, 3)

    pool.run(4, outer, 3)
    assert_true(_all_once(counts))


def test_threads_run_across_two_pools() raises:
    # A body on one pool running a job on another: waiting on the inner pool
    # must not depend on the outer pool's threads.
    var outer_pool = ThreadPool(2)
    var inner_pool = ThreadPool(2)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)

    def outer(i: Int) {imm inner_pool, imm tp}:
        def inner(j: Int) {imm tp}:
            _ = tp[].fetch_add(1)

        inner_pool.run(10, inner, 3)

    outer_pool.run(8, outer, 3)
    assert_equal(total.load(), 80)


def test_threads_run_soak() raises:
    # Thousands of jobs of every width, back to back and with gaps longer than
    # the spin budget, so both the hot and the sleeping paths are exercised
    # over many hand-offs rather than a few.
    var pool = ThreadPool(4)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)
    var expect = 0
    for rep in range(3_000 * STRESS):
        var n = rep % 23

        def add(i: Int) {imm tp}:
            _ = tp[].fetch_add(1)

        pool.run(n, add, 1 + rep % 7)
        expect += n
        if rep % 500 == 0:
            _pause_us(150)
    assert_equal(Int(total.load()), expect)


def test_threads_sleep_and_wake_across_nested_jobs() raises:
    # Gaps longer than the spin budget send the workers to sleep between jobs,
    # which is the path where a lost wake-up would hang rather than fail.
    var pool = ThreadPool(4)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)
    var expect = 0
    for rep in range(120 * STRESS):
        var k = 1 + rep % 6
        var n = 1 + (rep * 7) % 20

        def outer(i: Int) {imm tp, imm pool, imm k}:
            def inner(j: Int) {imm tp}:
                _ = tp[].fetch_add(1)

            pool.run(3, inner, k)

        pool.run(n, outer, k)
        expect += n * 3
        var spawns = rep % 4

        def body(scope: TaskScope) {imm tp, imm spawns}:
            for _ in range(spawns):

                def task() {imm tp}:
                    _ = tp[].fetch_add(1)

                scope.spawn(task)

        pool.scope(body)
        expect += spawns
        if rep % 3 == 0:
            _pause_us(120)
    assert_equal(Int(total.load()), expect)


# ---------------------------------------------------------------------------
# ThreadPool.scope
# ---------------------------------------------------------------------------


def test_threads_scope_runs_every_task() raises:
    # 10,000 is more tasks than the ring has slots: those that do not fit run
    # on the caller instead of being dropped, so every one is still counted.
    var pool = ThreadPool(2)
    for n in [1, 1_000, 10_000]:
        assert_equal(_spawned(pool, n), n)


def test_threads_scope_with_nothing_spawned_returns() raises:
    var pool = ThreadPool(2)
    var ran = Atomic[Int64](0)
    var rp = Pointer(to=ran)

    def body(scope: TaskScope) {imm rp}:
        rp[].store(1)

    pool.scope(body)
    assert_equal(ran.load(), 1)


def test_threads_scope_keeps_borrowed_values_alive() raises:
    # Nothing here is used after the scope, and no `keep` is needed: `body`
    # borrows the same values the tasks do, and it is in use until every task
    # has finished.
    var pool = ThreadPool(3)
    var mutex = Mutex()
    var total = List[Int](length=1, fill=0)
    var tp = Pointer(to=total).unsafe_origin_cast[MutUntrackedOrigin]()

    def body(scope: TaskScope) {imm mutex, imm tp}:
        for _ in range(64):

            def task() {imm mutex, imm tp}:
                for _ in range(50):
                    mutex.lock()
                    tp[][0] += 1
                    mutex.unlock()

            scope.spawn(task)

    pool.scope(body)
    assert_equal(total[0], 64 * 50)


def _spawn_tree(
    scope: TaskScope,
    nodes: Pointer[Atomic[Int64], MutUntrackedOrigin],
    depth: Int,
):
    """Count this node, then spawn both children onto the same scope."""
    _ = nodes[].fetch_add(1)
    if depth < 9:

        def child() {imm scope, var nodes, var depth}:
            _spawn_tree(scope, nodes, depth + 1)

        scope.spawn(child)
        scope.spawn(child)


def test_threads_scope_tasks_spawn_more_tasks() raises:
    # A binary tree of tasks, each spawning its children onto the same scope:
    # the scope must wait for tasks spawned after it began waiting.
    var pool = ThreadPool(3)
    var nodes = Atomic[Int64](0)
    var np = Pointer(to=nodes).unsafe_origin_cast[MutUntrackedOrigin]()

    def body(scope: TaskScope) {imm np}:
        _spawn_tree(scope, np, 0)

    pool.scope(body)
    assert_equal(nodes.load(), 1023)


def test_threads_scope_nests_and_runs_inside_tasks() raises:
    var pool = ThreadPool(2)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)

    def outer(scope: TaskScope) {imm pool, imm tp}:
        for _ in range(6):

            def task() raises {imm pool, imm tp}:
                def inner(scope: TaskScope) {imm tp}:
                    _ = tp[].fetch_add(1)

                pool.scope(inner)

                def each(i: Int) {imm tp}:
                    _ = tp[].fetch_add(1)

                pool.run(4, each, 3)

            scope.spawn(task)

    pool.scope(outer)
    assert_equal(total.load(), 6 * 5)


def test_threads_scope_across_pools() raises:
    # Tasks on one pool opening scopes on another. Each scope is bound to its
    # own pool, so no wait can end up on the wrong one.
    var a = ThreadPool(2)
    var b = ThreadPool(2)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)

    def outer(scope: TaskScope) {imm b, imm tp}:
        for _ in range(8):

            def task() raises {imm b, imm tp}:
                def inner(scope: TaskScope) {imm tp}:
                    for _ in range(4):

                        def leaf() {imm tp}:
                            _ = tp[].fetch_add(1)

                        scope.spawn(leaf)

                b.scope(inner)

            scope.spawn(task)

    a.scope(outer)
    assert_equal(total.load(), 32)


def test_threads_scope_destroys_each_task_once() raises:
    var drops = Atomic[Int64](0)
    var runs = Atomic[Int64](0)
    var drops_ptr = Pointer(to=drops).unsafe_origin_cast[MutUntrackedOrigin]()
    var runs_ptr = Pointer(to=runs).unsafe_origin_cast[MutUntrackedOrigin]()
    var pool = ThreadPool(3)

    def body(scope: TaskScope) {imm drops_ptr, imm runs_ptr}:
        _spawn_tallies(scope, 100, drops_ptr, runs_ptr)

    pool.scope(body)
    assert_equal(runs.load(), 100)
    assert_equal(drops.load(), 100)


def test_threads_scope_and_run_from_many_submitters() raises:
    # Eight threads share one pool, each mixing scopes and `run`: many
    # producers and consumers on the ring at once, wrapping it several times
    # over (8 x 50 x 20 tasks against 4096 slots).
    var pool = ThreadPool(3)
    var total = Atomic[Int64](0)
    var failures = Atomic[Int64](0)
    var tp = Pointer(to=total)
    var fp = Pointer(to=failures)
    var threads = List[Thread]()
    for _ in range(8):

        def submit() {imm pool, imm tp, imm fp}:
            for _ in range(50 * STRESS):

                def body(scope: TaskScope) {imm tp}:
                    for _ in range(20):

                        def task() {imm tp}:
                            _ = tp[].fetch_add(1)

                        scope.spawn(task)

                try:
                    pool.scope(body)
                except:
                    _ = fp[].fetch_add(1)

                def each(i: Int) {imm tp}:
                    _ = tp[].fetch_add(1)

                pool.run(10, each, 4)

        threads.append(Thread(submit))
    while threads:
        threads.pop().join()
    keep(pool)
    keep(tp)
    keep(fp)
    assert_equal(failures.load(), 0)
    assert_equal(Int(total.load()), 8 * 50 * STRESS * 30)


def test_threads_scope_from_run_and_run_from_scope() raises:
    var pool = ThreadPool(3)
    var total = Atomic[Int64](0)
    var tp = Pointer(to=total)

    def body(scope: TaskScope) {imm pool, imm tp}:
        def each(i: Int) {imm scope, imm tp}:
            def task() {imm tp}:
                _ = tp[].fetch_add(1)

            scope.spawn(task)

        pool.run(40, each, 4)

    pool.scope(body)
    assert_equal(total.load(), 40)


def test_threads_scope_raises_the_earliest_spawned_failure() raises:
    # Later-spawned tasks fail at once and the earliest only after a pause,
    # so in time it fails last; it is still the one raised, every time, and
    # every task spawned before it has run.
    var pool = ThreadPool(3)
    for _ in range(20 * STRESS):
        var ran = List[Int](length=200, fill=0)
        var rp = Pointer(to=ran).unsafe_origin_cast[MutUntrackedOrigin]()

        def body(scope: TaskScope) {imm rp}:
            for i in range(200):

                def task() raises {imm rp, var i}:
                    rp[][i] = 1
                    if i == 40:
                        _pause_us(300)
                        raise Error("task ", i)
                    if i == 41 or i == 150 or i == 199:
                        raise Error("task ", i)

                scope.spawn(task)

        var msg = String()
        try:
            pool.scope(body)
        except e:
            msg = String(e)
        assert_equal(msg, "task 40")
        for i in range(41):
            assert_equal(ran[i], 1)


def test_threads_scope_failure_cancels_and_skips_later_tasks() raises:
    # Once the first task has failed, nothing spawned after it runs — and
    # each skipped task is still destroyed, once.
    var pool = ThreadPool(3)
    var drops = Atomic[Int64](0)
    var runs = Atomic[Int64](0)
    var drops_ptr = Pointer(to=drops).unsafe_origin_cast[MutUntrackedOrigin]()
    var runs_ptr = Pointer(to=runs).unsafe_origin_cast[MutUntrackedOrigin]()

    def body(scope: TaskScope) raises {imm drops_ptr, imm runs_ptr}:
        def first() raises:
            raise Error("first")

        def failed() {imm scope} -> Bool:
            return scope.cancelled()

        scope.spawn(first)
        if not _eventually(failed):
            raise Error("the failure never cancelled the scope")
        _spawn_tallies(scope, 100, drops_ptr, runs_ptr)

    with assert_raises(contains="first"):
        pool.scope(body)
    assert_equal(runs.load(), 0)
    assert_equal(drops.load(), 100)


def test_threads_scope_cancel_skips_what_has_not_started() raises:
    # A cancelled scope is not a failed one: it returns normally.
    var pool = ThreadPool(3)
    var drops = Atomic[Int64](0)
    var runs = Atomic[Int64](0)
    var drops_ptr = Pointer(to=drops).unsafe_origin_cast[MutUntrackedOrigin]()
    var runs_ptr = Pointer(to=runs).unsafe_origin_cast[MutUntrackedOrigin]()

    def body(scope: TaskScope) raises {imm drops_ptr, imm runs_ptr}:
        assert_false(scope.cancelled())
        scope.cancel()
        assert_true(scope.cancelled())
        _spawn_tallies(scope, 100, drops_ptr, runs_ptr)

    pool.scope(body)
    assert_equal(runs.load(), 0)
    assert_equal(drops.load(), 100)


def test_threads_scope_running_task_sees_the_cancellation() raises:
    var pool = ThreadPool(2)
    var started = Atomic[Int64](0)
    var stopped = Atomic[Int64](0)
    var sp = Pointer(to=started)
    var xp = Pointer(to=stopped)

    def body(scope: TaskScope) {imm sp, imm xp}:
        def poller() {imm scope, imm sp, imm xp}:
            def cancelled() {imm scope} -> Bool:
                return scope.cancelled()

            sp[].store(1)
            if _eventually(cancelled):
                xp[].store(1)

        scope.spawn(poller)
        while sp[].load() == 0:
            pass
        scope.cancel()

    pool.scope(body)
    assert_equal(stopped.load(), 1)


def test_threads_scope_body_error_wins_after_running_tasks_finish() raises:
    # The body's own error is raised, not a task's, and only once the task
    # already running has finished.
    var pool = ThreadPool(2)
    var started = Atomic[Int64](0)
    var finished = Atomic[Int64](0)
    var sp = Pointer(to=started)
    var fp = Pointer(to=finished)

    def body(scope: TaskScope) raises {imm sp, imm fp}:
        def slow() raises {imm sp, imm fp}:
            sp[].store(1)
            _pause_us(2_000)
            fp[].store(1)
            raise Error("task")

        scope.spawn(slow)
        while sp[].load() == 0:
            pass
        raise Error("body")

    with assert_raises(contains="body"):
        pool.scope(body)
    assert_equal(finished.load(), 1)


def test_threads_scope_keeps_the_kind_of_a_task_error() raises:
    # A scope's tasks are closures of different types, so their errors meet as
    # a `DynError` — which keeps the kind, typed or bare.
    var pool = ThreadPool(2)

    def typed(scope: TaskScope):
        for i in range(10):

            def task() raises KeyError {var i}:
                if i == 4:
                    raise KeyError(t"task {i}")

            scope.spawn(task)

    var kinds = List[Bool]()
    try:
        pool.scope(typed)
    except e:
        kinds.append(e.isa[KeyError]())
        assert_equal(e.message, "task 4")

    def bare(scope: TaskScope):
        def task() raises:
            raise IndexError("bare")

        scope.spawn(task)

    try:
        pool.scope(bare)
    except e:
        kinds.append(e.isa[IndexError]())
        assert_equal(e.message, "bare")
    # Both raised, and each kept its kind.
    assert_true(kinds == [True, True])


def test_threads_scope_error_crosses_nested_scopes() raises:
    var pool = ThreadPool(2)

    def outer(scope: TaskScope) {imm pool}:
        for i in range(4):

            def task() raises {imm pool, var i}:
                def inner(scope: TaskScope) {var i}:
                    def leaf() raises {var i}:
                        if i == 2:
                            raise Error("leaf ", i)

                    scope.spawn(leaf)

                pool.scope(inner)

            scope.spawn(task)

    with assert_raises(contains="leaf 2"):
        pool.scope(outer)
    # The pool is still usable after a failed scope.
    assert_true(_all_once(_visits(pool, 100, 3)))


# ---------------------------------------------------------------------------
# ThreadPool.fan_out
# ---------------------------------------------------------------------------


def test_threads_fan_out_lanes_are_exclusive() raises:
    # A lane runs on one thread at a time, so per-lane scratch needs no lock:
    # no lane is ever entered while it is running, and a plain counter per
    # lane sees every one of its items.
    var pool = ThreadPool(3)
    var lanes = 4
    var per_lane = List[Int](length=lanes, fill=0)
    var seen = List[Int](length=400, fill=0)
    var busy = unsafe_alloc[Atomic[Int64]](lanes)
    for w in range(lanes):
        busy.unsafe_offset(w).unsafe_write(Atomic[Int64](0))
    var overlaps = Atomic[Int64](0)
    var op = Pointer(to=overlaps)

    def visit(
        wid: Int, i: Int
    ) raises {mut per_lane, mut seen, imm lanes, imm busy, imm op}:
        assert_true(wid >= 0 and wid < lanes)
        if busy.unsafe_offset(wid)[].fetch_add(1) != 0:
            _ = op[].fetch_add(1)
        _pause_us(10)
        per_lane[wid] += 1
        seen[i] += 1
        _ = busy.unsafe_offset(wid)[].fetch_sub(1)

    pool.fan_out(400, visit, lanes)
    busy.unsafe_free()
    assert_equal(overlaps.load(), 0)
    assert_true(_all_once(seen))
    var total = 0
    for w in range(lanes):
        total += per_lane[w]
    assert_equal(total, 400)


def test_threads_fan_out_a_slow_item_holds_up_no_other() raises:
    # Item 0 does not finish until all 39 others have. A lane takes the next
    # item when it finishes one, so the other lanes run them; with a fixed
    # share per lane, the items queued behind item 0 in its own lane could
    # not run, and this would time out.
    var pool = ThreadPool(3)
    var done = Atomic[Int64](0)
    var waited = Atomic[Int64](0)
    var dp = Pointer(to=done)
    var wp = Pointer(to=waited)

    def visit(wid: Int, i: Int) raises {imm dp, imm wp}:
        if i == 0:
            if _await(dp, 39):
                wp[].store(1)
        else:
            _ = dp[].fetch_add(1)

    pool.fan_out(40, visit, 4)
    assert_equal(waited.load(), 1)


def test_threads_fan_out_raises_the_lowest_failure_every_time() raises:
    # Item 250 fails last in time and is still the one raised, with every
    # item below it run.
    var pool = ThreadPool(3)
    for _ in range(30 * STRESS):
        var ran = List[Int](length=400, fill=0)

        def visit(wid: Int, i: Int) raises {mut ran}:
            ran[i] = 1
            if i == 250:
                _pause_us(300)
                raise Error("item ", i)
            if i == 251 or i == 397:
                raise Error("item ", i)

        var msg = String()
        try:
            pool.fan_out(400, visit, 4)
        except e:
            msg = String(e)
        assert_equal(msg, "item 250")
        for i in range(250):
            assert_equal(ran[i], 1)


def test_threads_fan_out_keeps_a_typed_error_typed() raises:
    # The body raises `IndexError`, and so does `fan_out`: `e.message()` only
    # exists on a typed error, so this would not compile had the error been
    # flattened to `Error` on its way across threads.
    var pool = ThreadPool(3)

    def visit(wid: Int, i: Int) raises IndexError:
        if i == 7 or i == 31:
            raise IndexError(t"item {i}")

    var message = String()
    try:
        pool.fan_out(40, visit, 4)
    except e:
        message = e.message()
    assert_equal(message, "item 7")


def test_threads_fan_out_skips_items_after_a_failure() raises:
    # Item 0 fails at once; without skipping, the other three lanes would run
    # their 3,000 items to the end.
    var pool = ThreadPool(3)
    var ran = Atomic[Int64](0)
    var rp = Pointer(to=ran)

    def visit(wid: Int, i: Int) raises {imm rp}:
        _ = rp[].fetch_add(1)
        if i == 0:
            raise Error("item 0")
        _pause_us(100)

    with assert_raises(contains="item 0"):
        pool.fan_out(4_000, visit, 4)
    assert_true(ran.load() < 1_000)


comptime _STOP_ITEMS = 100_000
"""Far more than lanes can get through in the time a stop takes to land, so
that every one of them running means the stop never reached the others."""


def _stopping_fan_out(
    pool: ThreadPool, ctx: Optional[ExecContext], lanes: Int
) raises -> Tuple[List[Int], List[Int]]:
    """Fan out over `_STOP_ITEMS` items where only item 100 asks to stop, so
    the other lanes learn of it through the shared bound, not their own
    bodies. Per item, how many times it ran, and the thread that ran it."""
    var ran = List[Int](length=_STOP_ITEMS, fill=0)
    var ids = List[Int](length=_STOP_ITEMS, fill=0)

    def visit(wid: Int, i: Int) raises {mut ran, mut ids} -> Bool:
        ran[i] += 1
        ids[i] = Thread.current_id()
        _pause_us(1)
        return i != 100

    if ctx:
        ctx.value().fan_out(_STOP_ITEMS, visit, lanes)
    else:
        pool.fan_out(_STOP_ITEMS, visit, lanes)
    return (ran^, ids^)


def _assert_stopped(ran: List[Int]) raises:
    """Every item up to the stop ran exactly once — all were handed out before
    it — and the stop reached the other lanes, which a run of every item would
    show it did not.

    How many items above the stop run is not asserted. The stop lands once
    item 100's body has returned, and a lane preempted between the two lets
    the others go on starting items for as long as it is off the core."""
    for i in range(101):
        assert_equal(ran[i], 1)
    var total = 0
    for i in range(len(ran)):
        total += ran[i]
    assert_true(total < len(ran))


def test_threads_fan_out_stops_handing_out_items() raises:
    var pool = ThreadPool(3)
    for lanes in [1, 4]:
        for _ in range(10 * STRESS):
            _assert_stopped(_stopping_fan_out(pool, None, lanes)[0])


def test_threads_fan_out_a_stop_drops_failures_above_it() raises:
    """Items in flight when item 50 stops may still fail, but a serial loop
    would have stopped at 50 and never reached them: nothing is raised."""
    var pool = ThreadPool(3)

    def visit(wid: Int, i: Int) raises -> Bool:
        if i > 50:
            raise Error("item ", i)
        return i != 50

    for lanes in [1, 4]:
        for _ in range(30 * STRESS):
            pool.fan_out(1_000, visit, lanes)


def test_threads_fan_out_a_failure_below_a_stop_is_raised() raises:
    var pool = ThreadPool(3)

    def visit(wid: Int, i: Int) raises -> Bool:
        if i == 30:
            raise Error("item ", i)
        return i != 50

    for lanes in [1, 4]:
        for _ in range(30 * STRESS):
            with assert_raises(contains="item 30"):
                pool.fan_out(1_000, visit, lanes)


def test_threads_fan_out_blocking_runs_on_the_io_pool() raises:
    var ids = List[Int](length=16, fill=0)

    def visit(wid: Int, i: Int) raises {mut ids}:
        ids[i] = Thread.current_id()

    ExecContext.auto().fan_out_blocking(16, visit)
    _assert_ran_on(ThreadPool.shared_io()[], ids)
    # Serial means one at a time, on the caller.
    ExecContext.serial().fan_out_blocking(16, visit)
    assert_equal(_distinct(ids), 1)
    assert_equal(ids[0], Thread.current_id())


# ---------------------------------------------------------------------------
# ThreadPool construction and lifetime
# ---------------------------------------------------------------------------


def test_threads_pool_with_no_workers_runs_on_the_caller() raises:
    var pool = ThreadPool(0)
    assert_equal(pool.size(), 0)
    assert_equal(pool.concurrency(), 1)
    assert_true(_all_once(_visits(pool, 100, 8)))
    assert_equal(_spawned(pool, 10), 10)


def test_threads_pool_rejects_negative_workers() raises:
    with assert_raises(contains="InvalidError: ThreadPool: -1 workers"):
        _ = ThreadPool(-1)


def test_threads_pool_construction_failure_raises() raises:
    with assert_raises(contains="InvalidError: Thread: stack size"):
        _ = ThreadPool(3, stack_size=-1)


def test_threads_pool_destruction_joins_its_workers() raises:
    # Repeatedly build, use and destroy pools: each destruction has to wake
    # and join workers that may be spinning, sleeping or mid-task.
    for rep in range(20 * STRESS):
        var pool = ThreadPool(1 + rep % 4)
        assert_true(_all_once(_visits(pool, 64, pool.concurrency())))


def test_threads_pool_stack_size_is_honoured() raises:
    # A frame that macOS's 512 KiB default for a secondary thread would not
    # hold.
    var pool = ThreadPool(2, stack_size=16 << 20)
    var ok = Atomic[Int64](0)
    var okp = Pointer(to=ok)

    def deep(i: Int) {imm okp}:
        var big = Array[UInt8, 2 << 20](fill=1)
        _ = okp[].fetch_add(Int64(big[i]))

    pool.run(4, deep, 3)
    assert_equal(ok.load(), 4)


def test_threads_pool_pin() raises:
    var pool = ThreadPool(2)
    var idle = ThreadPool(0)
    comptime if CompilationTarget.is_linux():
        pool.pin()
        idle.pin()
        assert_true(_all_once(_visits(pool, 100, 3)))
    else:
        # A pool with no workers to pin says so too.
        with assert_raises(contains="NotImplementedError: ThreadPool.pin"):
            pool.pin()
        with assert_raises(contains="NotImplementedError: ThreadPool.pin"):
            idle.pin()


def test_threads_pool_rejects_a_negative_spin() raises:
    with assert_raises(contains="InvalidError: ThreadPool: spin of -1 ns"):
        _ = ThreadPool(2, spin_ns=-1)


def test_threads_idle_workers_spin_then_sleep() raises:
    # An idle worker polls for work for `spin_ns`, then sleeps. 10 ms after a
    # job, a pool spinning 100 ms still has workers polling — its deadline is
    # wall-clock, so only a worker idle or descheduled for 90 ms could have
    # slept, and not all three; with no spin, every worker falls asleep.
    var spinning = ThreadPool(3, spin_ns=100_000_000)
    var sleeping = ThreadPool(3, spin_ns=0)
    _ = _thread_of_each(spinning, 64, spinning.concurrency(), 100)
    sleep(0.01)
    assert_true(spinning.sleeping() < spinning.size())
    _ = _thread_of_each(sleeping, 64, sleeping.concurrency(), 100)
    assert_true(_all_asleep(sleeping))


def test_threads_workers_carry_their_pools_name() raises:
    """Worker `k` of a pool built with `name` is called `{name}-{k}` — what a
    debugger or a profiler shows — and the caller keeps its own name."""
    var caller = Thread.current_name()
    var pool = ThreadPool(3, name="probe")
    var seen = _names_on_workers(pool)
    var named = 0
    for k in range(len(seen)):
        if seen[k]:
            assert_equal(seen[k], String("probe-", k))
            named += 1
    assert_true(named > 0)
    assert_equal(Thread.current_name(), caller)


def test_threads_a_long_pool_name_is_cut_before_a_split_character() raises:
    # Linux takes 15 bytes of a name. Nine two-byte characters and "-0" put
    # byte 15 inside the eighth, so the name keeps seven.
    var pool = ThreadPool(1, name="ééééééééé")
    var seen = _names_on_workers(pool)
    assert_equal(seen[0], "ééééééé")


def test_threads_shared_pools() raises:
    # One process-wide compute pool, one for blocking work, each a singleton,
    # sized by its default and named after what it is for.
    var a = ThreadPool.shared()
    var b = ThreadPool.shared()
    assert_true(a[].thread_ids() == b[].thread_ids())
    assert_equal(a[].concurrency(), _SharedPool[False].default_concurrency())
    var io = ThreadPool.shared_io()
    assert_true(io[].thread_ids() == ThreadPool.shared_io()[].thread_ids())
    assert_equal(io[].concurrency(), _SharedPool[True].default_concurrency())
    assert_true(io[].size() >= 1)
    # The compute pool polls between kernel calls; the I/O pool, whose next
    # task waits on the network anyway, does not.
    assert_true(a[].spin_ns() > 0)
    assert_equal(io[].spin_ns(), 0)
    for id in io[].thread_ids():
        assert_false(id in a[].thread_ids())
    for name in _names_on_workers(a[]):
        assert_true(not name or name.startswith("marrow-cpu-"))
    for name in _names_on_workers(io[]):
        assert_true(not name or name.startswith("marrow-io-"))


# ---------------------------------------------------------------------------
# Shared pools: size, resize, fork
# ---------------------------------------------------------------------------


def test_threads_a_shared_pool_starts_with_the_threads_it_can_get() raises:
    """When the OS refuses threads, a shared pool starts with half as many
    workers, and half again, rather than aborting the process — down to none,
    which needs no thread."""
    var asked = List[Int]()

    def start(
        workers: Int,
    ) raises DynError {mut asked} -> ArcPointer[ThreadPool]:
        asked.append(workers)
        if workers > 2:
            raise IOError(t"no thread for {workers} workers")
        return ArcPointer(ThreadPool(workers))

    var pool = _start_with_fewer(11, start)
    assert_equal(pool[].size(), 2)
    assert_true(asked == [11, 5, 2])
    assert_true(_all_once(_visits(pool[], 100, 3)))

    def refuse(
        workers: Int,
    ) raises DynError {mut asked} -> ArcPointer[ThreadPool]:
        asked.append(workers)
        if workers > 0:
            raise IOError(t"no thread at all")
        return ArcPointer(ThreadPool(workers))

    asked.clear()
    var serial = _start_with_fewer(5, refuse)
    assert_equal(serial[].size(), 0)
    assert_true(asked == [5, 2, 1, 0])


def test_threads_resize_shared_replaces_the_pool() raises:
    """Every `shared()` after `resize_shared` returns a pool of the new size.
    Whoever still holds the old one can keep using it, and its workers are
    joined when the last holder lets go."""
    var before = ThreadPool.shared()
    var original = before[].concurrency()
    ThreadPool.resize_shared(3)
    var after = ThreadPool.shared()
    assert_equal(after[].concurrency(), 3)
    for id in after[].thread_ids():
        assert_false(id in before[].thread_ids())
    assert_true(_all_once(_visits(before[], 100, 4)))
    assert_true(_all_once(_visits(after[], 100, 4)))
    ThreadPool.resize_shared(original)
    assert_equal(ThreadPool.shared()[].concurrency(), original)


def test_threads_resize_shared_io_and_what_resizing_refuses() raises:
    var original = ThreadPool.shared_io()[].concurrency()
    ThreadPool.resize_shared_io(2)
    assert_equal(ThreadPool.shared_io()[].concurrency(), 2)
    ThreadPool.resize_shared_io(original)
    assert_equal(ThreadPool.shared_io()[].concurrency(), original)
    for bad in [0, -1]:
        with assert_raises(contains="InvalidError: ThreadPool: a concurrency"):
            ThreadPool.resize_shared(bad)
        with assert_raises(contains="InvalidError: ThreadPool: a concurrency"):
            ThreadPool.resize_shared_io(bad)


def test_threads_shared_pool_size_comes_from_the_environment() raises:
    """`MARROW_NUM_THREADS` and `MARROW_IO_THREADS` size a shared pool when
    they hold a positive integer, spaces aside; anything else is ignored. Each
    case starts a pool in a forked child, so this process's stay as they
    are."""
    comptime if not TSAN:
        assert_equal(_concurrency_in_child[False]("3"), 3)
        assert_equal(_concurrency_in_child[False](" 5 "), 5)
        assert_equal(_concurrency_in_child[True]("2"), 2)
        var default = _concurrency_in_child[False]("")
        assert_true(default >= 1)
        for bad in ["0", "-2", "many", "1.5"]:
            assert_equal(_concurrency_in_child[False](bad), default)


def test_threads_a_forked_child_starts_its_own_shared_pool() raises:
    """A child forked from a process whose shared pool is running has none of
    its workers. Its first `shared()` lets go of the parent's copy without
    joining them — which would hang — and starts a pool of its own, which
    spreads work over threads again. Its workers have only just started, so
    under load a first job can finish on the caller before one is scheduled:
    the child retries until one spreads, as a pool with no live workers never
    does."""
    comptime if not TSAN:
        _ = ThreadPool.shared()[].concurrency()
        var pid = _fork()
        if pid == 0:
            var pool = ThreadPool.shared()

            def spreads() {imm pool} -> Bool:
                var ids = _thread_of_each(pool[], 64, pool[].concurrency(), 200)
                return _distinct(ids) > 1

            _exit_child(0 if pool[].size() == 0 or _eventually(spreads) else 1)
        assert_equal(_child_exit_code(pid), 0)


def test_threads_a_pool_dropped_in_a_forked_child_is_not_joined() raises:
    """A pool built before a fork has no workers in the child. Destroying it
    there must neither join them nor take a lock one of them may have held at
    the fork."""
    comptime if not TSAN:
        var pool = ThreadPool(3)
        assert_true(_all_once(_visits(pool, 64, 4)))
        var pid = _fork()
        if pid == 0:
            _ = pool^
            _exit_child(0)
        assert_equal(_child_exit_code(pid), 0)


def test_threads_cgroup_v2_quota_takes_the_lowest_cap_up_the_tree() raises:
    """A cgroup v2 quota is counted in whole CPUs, rounded up, and every group
    from the process's own up to the root may cap it: the lowest cap
    applies."""
    with TemporaryDirectory() as root:
        var membership = root + "/cgroup"
        makedirs(root + "/a/b")
        _write(membership, "0::/a/b\n")
        _write(root + "/a/b/cpu.max", "max 100000\n")
        _write(root + "/a/cpu.max", "150000 100000\n")
        _write(root + "/cpu.max", "800000 100000\n")
        assert_equal(_cgroup_cpu_limit(membership, root).value(), 2)
        _write(root + "/a/cpu.max", "max 100000\n")
        assert_equal(_cgroup_cpu_limit(membership, root).value(), 8)
        _write(root + "/cpu.max", "max 100000\n")
        assert_false(Bool(_cgroup_cpu_limit(membership, root)))


def test_threads_cgroup_v1_quota_is_read_at_the_cpu_controller() raises:
    with TemporaryDirectory() as root:
        var membership = root + "/cgroup"
        makedirs(root + "/cpu")
        _write(membership, "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n")
        _write(root + "/cpu/cpu.cfs_quota_us", "250000\n")
        _write(root + "/cpu/cpu.cfs_period_us", "100000\n")
        assert_equal(_cgroup_cpu_limit(membership, root).value(), 3)
        # A hybrid host: v2 is mounted too, but its group sets no cap.
        _write(membership, "12:cpu,cpuacct:/docker/abc\n0::/docker/abc\n")
        assert_equal(_cgroup_cpu_limit(membership, root).value(), 3)
        _write(root + "/cpu/cpu.cfs_quota_us", "-1\n")
        assert_false(Bool(_cgroup_cpu_limit(membership, root)))


def test_threads_cgroup_files_that_are_missing_or_garbled_cap_nothing() raises:
    with TemporaryDirectory() as root:
        assert_false(Bool(_cgroup_cpu_limit(root + "/missing", root)))
        var membership = root + "/cgroup"
        _write(membership, "0::/\n")
        assert_false(Bool(_cgroup_cpu_limit(membership, root)))
        for garbled in ["", "max", "lots 100000", "100000 0"]:
            _write(root + "/cpu.max", garbled)
            assert_false(Bool(_cgroup_cpu_limit(membership, root)))


# ---------------------------------------------------------------------------
# ExecContext integration
# ---------------------------------------------------------------------------


def test_threads_auto_stripes_one_per_pool_thread() raises:
    var shared = ThreadPool.shared()
    assert_equal(
        ExecContext.auto().resolved_num_threads(), shared[].concurrency()
    )
    var own = ArcPointer(ThreadPool(3))
    assert_equal(
        ExecContext.auto().with_pool(own.copy()).resolved_num_threads(), 4
    )


def test_threads_on_cpu_keeps_threads_and_pool() raises:
    var own = ArcPointer(ThreadPool(2))
    var ctx = ExecContext.parallel(3).with_pool(own.copy()).on_cpu()
    assert_false(ctx.is_gpu())
    assert_equal(ctx.resolved_num_threads(), 3)
    assert_true(ctx.thread_pool()[].thread_ids() == own[].thread_ids())


def test_threads_exec_context_routes_stripe_through_its_pool() raises:
    var pool = ArcPointer(ThreadPool(3))
    var ctx = ExecContext.parallel(4).with_pool(pool.copy())
    var workers = ctx.stripe_workers(4_000)
    var ids = List[Int](length=workers, fill=0)

    @always_inline
    def body(wid: Int, start: Int, end: Int) {mut ids}:
        ids[wid] = Thread.current_id()

    ctx.stripe(4_000, body)
    _assert_ran_on(pool[], ids)


def test_threads_exec_context_routes_fan_out_through_its_pool() raises:
    var pool = ArcPointer(ThreadPool(3))
    var ctx = ExecContext.parallel(4).with_pool(pool.copy())
    var ids = List[Int](length=32, fill=0)

    def visit(wid: Int, i: Int) raises {mut ids}:
        ids[i] = Thread.current_id()

    ctx.fan_out(32, visit, 4)
    _assert_ran_on(pool[], ids)


def test_threads_exec_context_fan_out_stays_within_num_threads() raises:
    """`parallel(2)` fans out on at most two threads, however many lanes are
    asked for and however many the pool holds — the limit a partitioned join
    is held to — and `serial()` on the caller alone."""
    var pool = ArcPointer(ThreadPool(7))
    var ids = List[Int](length=64, fill=0)

    def visit(wid: Int, i: Int) raises {mut ids}:
        ids[i] = Thread.current_id()
        _pause_us(200)

    ExecContext.parallel(2).with_pool(pool.copy()).fan_out(64, visit, 8)
    assert_true(_distinct(ids) <= 2)
    _assert_ran_on(pool[], ids)
    ExecContext.serial().with_pool(pool.copy()).fan_out(64, visit, 8)
    assert_equal(_distinct(ids), 1)
    assert_equal(ids[0], Thread.current_id())


def test_threads_exec_context_fan_out_stops_within_num_threads() raises:
    """The stopping `fan_out` through a context stops, and keeps to the
    context's budget: `parallel(2)` on a pool of eight runs on two threads."""
    var pool = ArcPointer(ThreadPool(7))
    var ctx = ExecContext.parallel(2).with_pool(pool.copy())
    for _ in range(10 * STRESS):
        var result = _stopping_fan_out(pool[], ctx.copy(), 8)
        _assert_stopped(result[0])
        var seen = List[Int]()
        for i in range(len(result[1])):
            if result[1][i] != 0:
                seen.append(result[1][i])
        assert_true(_distinct(seen) <= 2)


def test_threads_exec_context_run_stays_within_num_threads() raises:
    """`ExecContext.run` — a group-by's finishing pass — holds to the same
    limit."""
    var pool = ArcPointer(ThreadPool(7))
    var ids = List[Int](length=64, fill=0)

    def body(i: Int) {mut ids}:
        ids[i] = Thread.current_id()
        _pause_us(200)

    ExecContext.parallel(2).with_pool(pool.copy()).run(64, body)
    assert_true(_distinct(ids) <= 2)
    _assert_ran_on(pool[], ids)
    ExecContext.serial().with_pool(pool.copy()).run(64, body)
    assert_equal(_distinct(ids), 1)
    assert_equal(ids[0], Thread.current_id())


def test_threads_exec_context_keeps_its_pool_alive() raises:
    # The context's copy of the `ArcPointer` is enough: the caller's handle
    # can go before the context does.
    var pool = ArcPointer(ThreadPool(2))
    var ctx = ExecContext.parallel(3).with_pool(pool^)
    var copy = ctx.copy()
    _ = ctx^
    var seen = List[Int](length=3, fill=0)

    @always_inline
    def body(wid: Int, start: Int, end: Int) {mut seen}:
        seen[wid] = Thread.current_id()

    copy.stripe(3_000, body)
    _assert_ran_on(copy.thread_pool()[], seen)
