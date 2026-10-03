# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""CPU threads: the POSIX primitives, and the work-queue pool built on them.

Every CPU thread marrow starts is started here, through libc's pthreads, linked
at build time. The alternative was `max.algorithm.functional.sync_parallelize`,
which is not a thread pool at all but `DeviceContext(api="cpu")
.enqueue_cpu_range` — MAX's device runtime from the `max-core` package. Owning
the pool takes CPU threading off MAX (the GPU paths still import it) and lets
marrow decide how many threads a job gets, where they run, and what happens
when a job starts another one.

Four layers, each built only from the one below:

- `Mutex`, `Condition` and `Thread` wrap one pthread object each. They are the
  only code here that calls `external_call`. `Mutex.locked()` gives a `with`
  block its hold on a mutex, and `Condition.wait` takes that hold, so waiting
  without the lock does not type-check.
- Task erasure turns a closure of any type into an `AnyCoroutine` handle a
  thread can run: `async def` frames over a heap box, through the stdlib's
  private `std.builtin._coroutine` — the reason a closure can sit in a queue
  at all. It and `std.ffi._get_global`, under the shared pools, are the
  private APIs this module stands on.
- `ThreadPool` is a set of worker threads and one lock-free queue of erased
  tasks. `run` is fork-join over an index range, `fan_out` its raising
  counterpart, and `scope` hands out a `TaskScope` whose `spawn`ed tasks all
  finish before `scope` returns — the only way to spawn, so a task can never
  outlive what it borrows or be waited on from the wrong pool.
- `ThreadPool.shared()` is the process-wide compute pool `ExecContext` uses
  unless it is given another, and `ThreadPool.shared_io()` a second one for
  work that blocks, such as an object-store fetch. Their size comes from the
  environment or from the CPUs the process may use, `resize_shared` replaces
  them, and a `fork`ed child starts its own. A pool built by hand does not
  survive a `fork`: in the child, work goes to the shared pools.

**Scheduling.** One lock-free queue shared by every worker — Vyukov's
bounded multi-producer, multi-consumer ring, the family DuckDB's scheduler
uses — and `run` hands out indices from an atomic counter, so a fast thread
takes more of them than a slow one. That is DuckDB's structure: a global queue,
and work claimed from shared state. ClickHouse instead keeps a queue per thread
and steals from the others when its own runs dry, which buys locality once a
task pushes its own successors. Nothing in marrow does that yet, so there is
one queue; per-worker queues can replace it behind the same `run` and `scope`.
The only lock is the one threads sleep under, which a failure — rare — also
takes to record itself.

**Waiting helps.** A thread waiting in `run` or `scope` runs queued tasks
until what it waits on is done and parks only when the queue is empty. That is
what makes a `run` inside a `run` body, or two threads submitting at once,
safe: every task a waiter depends on is either running somewhere or sitting in
the queue where the waiter itself will pick it up. It is also the one way a
task may wait on another: a task blocked on a lock or a condition another task
releases helps nobody, and can deadlock the pool. And `run` waits for its
indices, not for the helpers it asked for, so a helper that is slow to arrive
costs nothing.

A waiter runs *whatever* is queued, not only its own job's tasks, so a call
can return later than its own work finishes — after the unrelated task it
picked up — and nested jobs stack on one thread (hence 8 MiB stacks). That is
the coupling a second pool removes for blocking work: a compute caller never
picks up a fetch. Scoping a waiter to its own job, as TBB's `isolate` does,
needs per-job queues.

**Failures.** `fan_out` and `scope` take bodies that raise, and raise the
failure a serial loop would have stopped at; after a failure they skip the work
that comes after it rather than run it. `fan_out` raises its body's error as
that type; a scope's tasks fail as a `DynError`, which keeps the kind. `run`'s
body cannot raise. Work can also end early without failing: a `fan_out` body
returns False where a serial loop would `break`, and a scope is `cancel`led.
"""

from std.atomic import Atomic
from std.builtin._coroutine import (
    AnyCoroutine,
    Coroutine,
    _coro_destroy_fn,
    _coro_resume_fn,
)
from std.ffi import c_int, c_ssize_t, external_call, _get_global
from std.math import ceildiv
from std.memory import ArcPointer
from std.memory.alloc import unsafe_alloc
from std.os import abort, getenv
from std.sys import CompilationTarget, get_defined_int, stderr
from std.sys.info import num_logical_cores, num_performance_cores
from std.time import perf_counter_ns

from ..errors import (
    DynError,
    IndexError,
    InvalidError,
    IOError,
    NotImplementedError,
)


comptime _Opaque = OpaquePointer[MutUntrackedOrigin]
"""A `void *`: what `pthread_create` hands a new thread."""

comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]

comptime _PTHREAD_OBJECT_BYTES = 128
"""Storage for one `pthread_mutex_t`, `pthread_cond_t` or `pthread_attr_t`.

Their sizes differ by platform — a mutex is 64 bytes on macOS and 40 on
glibc, a condition variable 48 on both, an attribute set 64 and 56 — and
nothing here reads their fields, so one opaque size that covers all of them
replaces a per-platform struct for each.
"""

comptime _CPU_SET_BYTES = 128
"""glibc's `cpu_set_t`: a 1024-bit mask."""

comptime _CAN_PIN = CompilationTarget.is_linux()
"""Whether threads can be bound to a core: `pthread_setaffinity_np` is
Linux's. macOS has no binding affinity — its `THREAD_AFFINITY_POLICY` is a
grouping hint that Apple Silicon ignores."""

comptime _SPIN_NS = get_defined_int["MARROW_THREADS_SPIN_NS", 50_000]()
"""How long an idle thread of a compute pool polls for work before it sleeps
on a condition — `ThreadPool`'s default `spin_ns`.

Kernel calls come tens of microseconds apart, and a wake-up from sleep costs
about as much: polling across that gap keeps the workers on hand for the next
call. Past it, sleeping costs nothing measurable."""

comptime _DEFAULT_STACK_BYTES = 8 << 20
"""8 MiB, the main thread's default. macOS gives a secondary thread 512 KiB,
and marrow's kernels were written against MAX's pool, whose workers do not
run that small."""


# ---------------------------------------------------------------------------
# POSIX primitives
# ---------------------------------------------------------------------------


@no_inline
def _check(what: StaticString, rc: c_int):
    """Abort on a pthread call that fails in a way the caller cannot recover
    from — a mutex that cannot be created, a join that cannot complete.
    Continuing past one would turn a detectable fault into a hang or silent
    memory corruption. Every caller is cold, so this stays out of line."""
    if rc != 0:
        abort(String("threads: ", what, " failed with error ", rc))


def _alloc_pthread_object() -> _Bytes:
    return unsafe_alloc[UInt8](_PTHREAD_OBJECT_BYTES, alignment=16)


def _pid() -> Int64:
    """The calling process's id (`getpid`, which `std.os` does not expose)."""
    return Int64(external_call["getpid", c_int]())


struct Mutex(Movable):
    """A `pthread_mutex_t`.

    The mutex lives on the heap, not in the struct: POSIX makes moving an
    initialised mutex undefined, and a Mojo value moves whenever it is
    returned or stored. Moving a `Mutex` moves only the pointer.

    Prefer `with mutex.locked():` to a `lock`/`unlock` pair; see `locked`.

    Creating one is checked; `lock` and `unlock` — and `Condition`'s `wait`,
    `signal` and `broadcast` — are not. On a default mutex they fail only on a
    corrupted object, and the abort path, inlined into the pool's queue, cost
    ~17 KB on each size gate.
    """

    var _ptr: _Bytes

    def __init__(out self):
        self._ptr = _alloc_pthread_object()
        _check(
            "pthread_mutex_init",
            external_call["pthread_mutex_init", c_int](self._ptr, UInt(0)),
        )

    def __deinit__(deinit self):
        _ = external_call["pthread_mutex_destroy", c_int](self._ptr)
        self._ptr.unsafe_free()

    def lock(self):
        _ = external_call["pthread_mutex_lock", c_int](self._ptr)

    def try_lock(self) -> Bool:
        """Take the mutex if it is free, without waiting; whether it was
        taken."""
        return external_call["pthread_mutex_trylock", c_int](self._ptr) == 0

    def unlock(self):
        _ = external_call["pthread_mutex_unlock", c_int](self._ptr)

    def locked(self) -> MutexLock[origin_of(self)]:
        """A `with` block's hold on this mutex, taken on entering the block and
        released on leaving it — by falling off the end, a `return`, or an
        error, which propagates. `with mutex.locked() as guard:` also names
        the hold, as the `MutexGuard` `Condition.wait` takes.

        A `Mutex` cannot be its own context manager: `with` takes its operand
        by copy, and a mutex does not copy.
        """
        return MutexLock(self)

    def unsafe_handle(self) -> OpaquePointer[MutUntrackedOrigin]:
        """The `pthread_mutex_t *` — C++'s `native_handle()` — for a C call
        that takes one. It carries no proof that the mutex is held."""
        return self._ptr.unsafe_bitcast[NoneType]()


struct MutexLock[origin: ImmOrigin](Movable):
    """What `Mutex.locked()` returns; see there. It borrows the mutex, which
    therefore outlives the block."""

    var _mutex: Pointer[Mutex, Self.origin]

    @doc_hidden
    def __init__(out self, ref[Self.origin] mutex: Mutex):
        self._mutex = Pointer(to=mutex)

    def __enter__(self) -> MutexGuard[Self.origin]:
        self._mutex[].lock()
        return MutexGuard(self._mutex)

    def __exit__(self):
        """Release the mutex — on an error exit too, which then propagates."""
        self._mutex[].unlock()


struct MutexGuard[origin: ImmOrigin](ImplicitlyCopyable, Movable):
    """What `with mutex.locked() as guard:` names, and `Condition.wait`
    takes. It is copyable and tied to the mutex's lifetime, not the block's: a
    guard kept past its block still type-checks, so it proves a block took the
    lock, not that one holds it now."""

    var _mutex: Pointer[Mutex, Self.origin]

    @doc_hidden
    def __init__(out self, mutex: Pointer[Mutex, Self.origin]):
        self._mutex = mutex

    def mutex(self) -> ref[Self.origin] Mutex:
        """The mutex the block holds."""
        return self._mutex[]


struct Condition(Movable):
    """A `pthread_cond_t`, heap-allocated for the same reason as `Mutex`."""

    var _ptr: _Bytes

    def __init__(out self):
        self._ptr = _alloc_pthread_object()
        _check(
            "pthread_cond_init",
            external_call["pthread_cond_init", c_int](self._ptr, UInt(0)),
        )

    def __deinit__(deinit self):
        _ = external_call["pthread_cond_destroy", c_int](self._ptr)
        self._ptr.unsafe_free()

    def wait[origin: ImmOrigin, //](self, guard: MutexGuard[origin]):
        """Release the mutex `guard` holds, sleep until signalled, and take it
        back before returning.

        Taking the guard rather than the mutex is what makes holding the lock
        a precondition the compiler checks. A wake-up can be spurious, so wait
        in a loop that re-checks its condition.
        """
        _ = external_call["pthread_cond_wait", c_int](
            self._ptr, guard.mutex().unsafe_handle()
        )

    def signal(self):
        """Wake at least one waiting thread."""
        _ = external_call["pthread_cond_signal", c_int](self._ptr)

    def broadcast(self):
        """Wake every waiting thread."""
        _ = external_call["pthread_cond_broadcast", c_int](self._ptr)


# ---------------------------------------------------------------------------
# Task erasure
# ---------------------------------------------------------------------------


# An erased task is a coroutine frame. A function instantiated over a closure
# type is itself "capturing", and Mojo will not take a capturing function as a
# runtime value, so a queue of `def(void*) thin` trampolines cannot be built.
# Calling an `async def` builds a frame, `_take_handle` erases it to
# `AnyCoroutine`, and the non-generic `_coro_resume_fn` runs it to completion
# from any thread — what MAX's `enqueue_cpu_range` does, from `std` alone.
#
# Every coroutine here is a top-level function over raw pointers: a frame holds
# a borrowed argument by reference into the caller's stack, so only
# register-passable arguments survive until a worker resumes it (CLAUDE.md,
# "Threads").


async def _call_boxed[F: def() -> None](box: Pointer[F, MutUntrackedOrigin]):
    """Run a heap-boxed closure once, then destroy and free it."""
    box[]()
    _destroy(box)


def _box[T: Movable](var value: T) -> Pointer[T, MutUntrackedOrigin]:
    """`value`, moved into a heap cell of its own: what a raw pointer shared
    with other threads, or a coroutine that outlives its caller, points at."""
    var p = unsafe_alloc[T](1)
    p.unsafe_write(value^)
    return p


def _destroy[T: Deinitable](box: Pointer[T, MutUntrackedOrigin]):
    """Destroy the value `_box` moved onto the heap, and free its cell."""
    box.unsafe_deinit_pointee()
    box.unsafe_free()


def _erase[
    F: def() -> None
](box: Pointer[F, MutUntrackedOrigin]) -> AnyCoroutine:
    """A not-yet-started coroutine that runs `box[]` once and frees it.

    Destroying the handle unrun frees only the frame; the box is the caller's
    to free then.
    """
    return _handle(_call_boxed[F](box))


def _handle[
    T: Deinitable, origins: OriginSet, //
](var coro: Coroutine[T, origins]) -> AnyCoroutine:
    """`coro`, not yet started, as the handle a queue holds."""
    coro._set_noop_callback()
    return coro^._take_handle()


def _run_erased(handle: AnyCoroutine):
    """Run an erased task to completion on this thread and free its frame."""
    _coro_resume_fn(handle)
    _coro_destroy_fn(handle)


def _thread_main(
    start: _Opaque,
) abi("C") -> OptionalPointer[NoneType, MutUntrackedOrigin]:
    """The `start_routine` every `Thread` is created with.

    `start` is a heap cell holding the thread's erased body, allocated by the
    creating thread; this thread owns it from here on.
    """
    var cell = start.unsafe_bitcast[AnyCoroutine]()
    _run_erased(cell[])
    cell.unsafe_free()
    return None


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


struct Thread(Movable):
    """One OS thread running one closure.

    `join` waits for it. A `Thread` destroyed without `join` is detached: it
    runs to completion on its own and its resources are reclaimed when it
    exits.

    **What the closure borrows, the caller must keep alive until `join`.** Mojo
    ends a value's life at its last *visible* use, and a closure boxed into a
    thread is no longer visible: a `Mutex` captured `{imm mutex}` and not
    touched after `join` is destroyed while the threads still lock it. Capture
    by value (`{var x}`), or use each borrowed value after the join — `keep(x)`
    from `std.benchmark` does nothing else. For work on a pool, `run` and
    `ThreadPool.scope` make this automatic; `Thread` is the low-level primitive
    under them.
    """

    var _handle: UInt
    """The `pthread_t`: a pointer on macOS, an `unsigned long` on glibc —
    eight bytes either way."""

    def __init__[
        F: def() -> None
    ](
        out self, var body: F, stack_size: Int = _DEFAULT_STACK_BYTES
    ) raises DynError:
        """Start a thread that runs `body()` once.

        Raises `InvalidError` for a stack size the platform rejects and
        `IOError` if the OS refuses the thread; `body` is destroyed unrun then.
        """
        var box = _box(body^)
        var task = _erase(box)
        try:
            self._handle = Self._spawn(task, stack_size)
        except e:
            _coro_destroy_fn(task)
            _destroy(box)
            raise e^

    @staticmethod
    def _spawn(task: AnyCoroutine, stack_size: Int) raises DynError -> UInt:
        # Kept apart from `__init__` so a failure raises before `self` exists:
        # a half-built `Thread` would run `__deinit__` on a handle of 0, and
        # `pthread_detach(0)` crashes on glibc.
        var start = _box(task)
        var attr = _alloc_pthread_object()
        _ = external_call["pthread_attr_init", c_int](attr)
        var bad_stack = (
            stack_size <= 0
            or external_call["pthread_attr_setstacksize", c_int](
                attr, UInt(stack_size)
            )
            != 0
        )
        var handle = UInt(0)
        var rc = c_int(0)
        if not bad_stack:
            rc = external_call["pthread_create", c_int](
                Pointer(to=handle), attr, _thread_main, start
            )
        _ = external_call["pthread_attr_destroy", c_int](attr)
        attr.unsafe_free()
        if bad_stack:
            start.unsafe_free()
            raise InvalidError(t"Thread: stack size {stack_size} rejected")
        if rc != 0:
            start.unsafe_free()
            raise IOError(t"Thread: pthread_create failed with error {rc}")
        return handle

    def __deinit__(deinit self):
        _ = external_call["pthread_detach", c_int](self._handle)

    def join(deinit self):
        """Wait for the thread to finish. Joining the calling thread itself
        aborts — it would wait forever."""
        _check(
            "pthread_join",
            external_call["pthread_join", c_int](self._handle, UInt(0)),
        )

    def id(self) -> Int:
        """This thread's id, comparable with `Thread.current_id()`."""
        return Int(self._handle)

    @staticmethod
    def current_id() -> Int:
        """The calling thread's id (`pthread_self`)."""
        return Int(external_call["pthread_self", UInt]())

    @staticmethod
    def current_name() -> String:
        """The calling thread's name (`pthread_getname_np`): `"{name}-{k}"` on
        worker `k` of a pool built with `name`, empty on a thread nobody
        named."""
        var buf = Array[UInt8, 64](fill=0)
        _ = external_call["pthread_getname_np", c_int](
            external_call["pthread_self", UInt](), buf.unsafe_ptr(), UInt(64)
        )
        var n = 0
        while n < 64 and buf[n] != 0:
            n += 1
        return String(from_utf8_lossy=Span(buf)[:n])

    @staticmethod
    def yield_now():
        """Give up the rest of the calling thread's time slice
        (`sched_yield`)."""
        _ = external_call["sched_yield", c_int]()

    def pin(self, core: Int) raises DynError:
        """Restrict this thread to logical core `core`.

        Raises `NotImplementedError` where threads cannot be pinned (see
        `_CAN_PIN`), `IndexError` for a core outside the 1,024 a `cpu_set_t`
        holds, and `IOError` if the OS refuses.
        """
        comptime if _CAN_PIN:
            if core < 0 or core >= _CPU_SET_BYTES * 8:
                raise IndexError(t"Thread.pin: core {core} out of range")
            var mask = Array[UInt8, _CPU_SET_BYTES](fill=0)
            mask[core // 8] = UInt8(1) << UInt8(core % 8)
            var rc = external_call["pthread_setaffinity_np", c_int](
                self._handle, UInt(_CPU_SET_BYTES), mask.unsafe_ptr()
            )
            if rc != 0:
                raise IOError(
                    t"Thread.pin: pthread_setaffinity_np failed with {rc}"
                )
        else:
            raise NotImplementedError(
                "Thread.pin: this platform has no binding thread affinity"
            )


comptime _NAME_BYTES = 16
"""The longest thread name Linux takes, its terminator included; macOS takes
64."""


def _name_thread(name: String, k: Int):
    """Name the calling thread `{name}-{k}`, for debuggers and profilers, cut
    to the 15 bytes Linux takes at a character boundary. A name is a label,
    not state, so a failure is ignored.

    The label is built in place rather than formatted as a `String`: this runs
    in every binary that starts a pool, and the formatting machinery cost the
    size-gated ones several kilobytes.
    """
    # One byte past the limit is kept, to tell whether the cut splits a
    # character.
    var label = Array[UInt8, _NAME_BYTES + 1](fill=0)
    var n = 0
    for byte in name.as_bytes():
        if n <= _NAME_BYTES:
            label[n] = byte
            n += 1
    var scale = 1
    while scale * 10 <= k:
        scale *= 10
    if n <= _NAME_BYTES:
        label[n] = UInt8(ord("-"))
        n += 1
    while scale > 0:
        if n <= _NAME_BYTES:
            label[n] = UInt8(ord("0") + (k // scale) % 10)
            n += 1
        scale //= 10
    var end = min(n, _NAME_BYTES - 1)
    while end < n and (label[end] & 0xC0) == 0x80:
        end -= 1
    label[end] = 0
    comptime if CompilationTarget.is_linux():
        _ = external_call["pthread_setname_np", c_int](
            external_call["pthread_self", UInt](), label.unsafe_ptr()
        )
    else:
        # macOS names only the calling thread, and takes no handle for it.
        _ = external_call["pthread_setname_np", c_int](label.unsafe_ptr())


# ---------------------------------------------------------------------------
# State other threads change
# ---------------------------------------------------------------------------


def _raw[
    T: AnyType, origin: Origin, //
](ref[origin] value: T) -> Pointer[T, MutUntrackedOrigin]:
    """`value` as the raw pointer through which state other threads change is
    read and written — the only way this module reaches such state.

    A `mut` reference claims exclusive access, so through one the compiler may
    keep a field's value across another thread's write (CLAUDE.md,
    "Threads"). A raw pointer claims nothing, so every read is a load.
    """
    return (
        Pointer(to=value)
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


comptime _Counter = Pointer[Atomic[Int64], MutUntrackedOrigin]
"""A count other threads change, reached through a raw pointer; see `_raw`."""


# ---------------------------------------------------------------------------
# Failures and stops
# ---------------------------------------------------------------------------


struct _Outcome[E: Movable & Deinitable](Movable):
    """How work numbered in order — a scope's tasks, `fan_out`'s items — ends,
    decided as a serial loop over it would decide it: by the lowest-numbered
    item that failed or asked to stop.

    The threads doing the work change it only through a `_Cutoff`: `stop` and
    `stopped` atomically, `failed` and `error` under the pool's mutex, since
    failures are rare. The thread that waited for the work reads it once the
    work is done.
    """

    var end: Int64
    """Where the numbering ends."""
    var stop: Atomic[Int64]
    """Items numbered at or above this are skipped: `end`, lowered to each
    failure, to just past each stop, and to 0 by a cancel."""
    var stopped: Atomic[Int64]
    """The lowest item that asked to stop."""
    var failed: Int64
    """The item `error` came from."""
    var error: Optional[Self.E]

    def __init__(out self, end: Int64):
        self.end = end
        self.stop = Atomic[Int64](end)
        self.stopped = Atomic[Int64](Int64.MAX)
        self.failed = Int64.MAX
        self.error = None

    def ends_in_failure(self) -> Bool:
        """Whether the work, once done, ends in the failure recorded: there
        is one, and no item below it asked to stop — the loop would have
        stopped there first."""
        return Bool(self.error) and self.failed < self.stopped.load()

    def take_error(mut self) -> Self.E:
        """The failure the work ends in; see `ends_in_failure`."""
        return self.error.take()


struct _Cutoff[E: Movable & Deinitable](ImplicitlyCopyable, Movable):
    """A raw pointer to an `_Outcome`, and what the threads doing the work do
    through it: ask whether an item is still wanted, record a failure, stop
    after an item, cancel what has not started."""

    var outcome: Pointer[_Outcome[Self.E], MutUntrackedOrigin]

    def __init__[
        origin: Origin, //
    ](out self, ref[origin] outcome: _Outcome[Self.E]):
        self.outcome = _raw(outcome)

    def wants(self, seq: Int64) -> Bool:
        """Whether item `seq`, if it has not started, is still to run: it is
        below the end and every failure, no later than every stop, and the work
        is not cancelled."""
        return seq < self.outcome[].stop.load()

    def fail(self, shared: _Shared, seq: Int64, var error: Self.E):
        """Item `seq` raised `error`: keep it unless an item below it failed,
        and skip every item from `seq` on."""
        with shared.state[].mutex.locked():
            if seq < self.outcome[].failed:
                self.outcome[].failed = seq
                self.outcome[].error = error^
        _ = self.outcome[].stop.min(seq)

    def stop_after(self, seq: Int64):
        """Item `seq` asked to stop: skip every item after it."""
        _ = self.outcome[].stopped.min(seq)
        _ = self.outcome[].stop.min(seq + 1)

    def cancel(self):
        """Skip every item that has not started."""
        _ = self.outcome[].stop.min(0)

    def cancelled(self) -> Bool:
        """Whether the work is cut short: an item failed or asked to stop, or
        it was cancelled."""
        return self.outcome[].stop.load() < self.outcome[].end


def _dyn[E: Writable](error: E) -> DynError:
    """`error` as a `DynError` that keeps its kind. Every kind writes its name
    first, which is what `DynError` recovers it from, so a typed error, a
    `DynError` and a bare `Error` all keep theirs. The `Error` below is only
    that text; nothing raises it."""
    var text: Error = error
    return DynError(text)


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


struct _ScopeState(Movable):
    """A scope's bookkeeping, on the heap and reached only through a
    pointer."""

    var pending: Atomic[Int64]
    """Tasks spawned and not yet finished."""
    var spawned: Atomic[Int64]
    """Tasks spawned so far; a task's number is the count before it."""
    var outcome: _Outcome[DynError]

    def __init__(out self):
        self.pending = Atomic[Int64](0)
        self.spawned = Atomic[Int64](0)
        self.outcome = _Outcome[DynError](Int64.MAX)


comptime _ScopePtr = Pointer[_ScopeState, MutUntrackedOrigin]


struct TaskScope:
    """Tasks spawned on a pool, all finished before the `ThreadPool.scope`
    that made this returns — structured concurrency, as Rust's
    `thread::scope`.

    Only `ThreadPool.scope` creates one, and it joins every task before it
    returns. That is what makes spawning sound: a task may borrow anything the
    scope's body can see, because the body is in use — and so is everything it
    borrows — until the scope has joined every task; and a scope is bound to
    the pool it spawns on, so there is no waiting on the wrong one.

    **A task may raise.** The first failure cancels the scope (`cancelled()`
    turns True) and skips every task spawned *after* the failing one that has
    not started; the ones spawned before it still run. `ThreadPool.scope` then
    raises the error of the earliest-spawned task that failed — the one a
    serial loop over the tasks would have stopped at, whichever thread failed
    first — as long as no task stops early because it saw `cancelled()`.
    "Earliest" is the order of the `spawn` calls, which is a race of its own
    when tasks spawn from several threads.

    Tasks are closures of different types, so their errors meet as a
    `DynError`, which keeps each one's kind: `e.isa[KeyError]()` answers for a
    task that raised a `KeyError`, typed or bare.
    """

    var _shared: _Shared
    var _state: _ScopePtr

    @doc_hidden
    def __init__(out self, shared: _Shared):
        # Public only because Mojo has no module-private members:
        # `ThreadPool.scope` is the one caller, and `_Shared` the pool's
        # internal queue.
        self._shared = shared
        self._state = _box(_ScopeState())

    def __deinit__(deinit self):
        """Wait for every spawned task and drop any error they raised."""
        _ = self._wait()

    @doc_hidden
    def join(deinit self) raises DynError:
        """Wait for every spawned task, helping to run queued ones meanwhile,
        and raise the earliest-spawned failure. `ThreadPool.scope` calls it;
        a body only borrows its scope, so it cannot."""
        var error = self._wait()
        if error:
            raise error.take()

    def _wait(self) -> Optional[DynError]:
        self._shared.help_until(_raw(self._state[].pending))
        var error: Optional[DynError] = None
        if self._state[].outcome.ends_in_failure():
            error = self._state[].outcome.take_error()
        _destroy(self._state)
        return error^

    def spawn[F: def() -> None](self, var task: F):
        """Queue `task` to run once on some thread of the pool.

        `task` moves into a heap box and is destroyed once it has run, or
        once it has been skipped. It may itself spawn onto this scope, or open
        a scope of its own. When the queue is full it runs on the calling
        thread instead.
        """
        # Not an adapter over the raising `spawn`: routing a task that cannot
        # fail through the error path hangs the compiler, under
        # `-D ASSERT=all`, once tasks spawn recursively.
        var seq = self._number()
        if seq >= 0:
            var state = self._state
            var shared = self._shared

            def counted() {var task^, var state, var shared, var seq}:
                if _Cutoff(state[].outcome).wants(seq):
                    task()
                shared.count_down(_raw(state[].pending), 1)

            self._queue(counted^)

    def spawn[
        E: Writable & Movable & Deinitable, F: def() raises E -> None
    ](self, var task: F):
        """`spawn` for a task that raises: its error, kind and all, is what
        `ThreadPool.scope` raises if no task spawned earlier failed."""
        var seq = self._number()
        if seq >= 0:
            var state = self._state
            var shared = self._shared

            def counted() {var task^, var state, var shared, var seq}:
                var cutoff = _Cutoff(state[].outcome)
                if cutoff.wants(seq):
                    try:
                        task()
                    except e:
                        cutoff.fail(shared, seq, _dyn(e))
                shared.count_down(_raw(state[].pending), 1)

            self._queue(counted^)

    def _cutoff(self) -> _Cutoff[DynError]:
        return _Cutoff(self._state[].outcome)

    def _number(self) -> Int64:
        """Number the next task and count it pending — or -1, when the scope
        is already stopping and the task is to be dropped unrun."""
        var seq = self._state[].spawned.fetch_add(1)
        if not self._cutoff().wants(seq):
            return -1
        _ = self._state[].pending.fetch_add(1)
        return seq

    def _queue[G: def() -> None](self, var counted: G):
        """Queue a numbered task, or run it here when the queue is full."""
        var handle = _erase(_box(counted^))
        if not self._shared.push(handle):
            _run_erased(handle)

    def cancel(self):
        """Skip every task of this scope that has not started. Running tasks
        finish, and may poll `cancelled()` to stop early. A scope that is
        cancelled, rather than failed, returns normally."""
        self._cutoff().cancel()

    def cancelled(self) -> Bool:
        """Whether `cancel` was called or a task has failed."""
        return self._cutoff().cancelled()


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


comptime _LINE_BYTES = 128
"""A cache line, rounded up to Apple Silicon's 128 bytes; x86 has 64."""

comptime _LINE_WORDS = _LINE_BYTES // 8

comptime _QUEUE_SLOTS = 4096
"""Capacity of a pool's task ring; a power of two. A full ring is not an error:
`run` offers fewer helpers and `spawn` runs the task on the caller."""


@fieldwise_init
struct _Task(ImplicitlyCopyable, Movable):
    """A queued coroutine handle, wrapped so it can sit in an `Optional`."""

    var handle: AnyCoroutine


struct _Ring(Movable):
    """A pool's queue of tasks: Dmitry Vyukov's bounded multi-producer,
    multi-consumer ring, written from his published algorithm (1024cores.net,
    "Bounded MPMC queue"; BSD-2-Clause, see NOTICE.txt).

    A push or a pop is one compare-and-swap on the tail or the head plus a
    store to the slot, so no lock is taken. A free slot's `seq` equals the
    position the next push there takes; a push publishes by storing
    `pos + 1`, and a pop frees the slot for the next lap by storing
    `pos + capacity`. The head and the tail sit on cache lines of their own,
    so a push does not invalidate the line every spinning worker polls.

    Every field is a pointer set once at construction, so the methods take
    `self` and reach what other threads change only through those pointers.
    """

    var ends: _Counter
    """The head at `ends[0]` and the tail at `ends[_LINE_WORDS]`: one
    allocation, two cache lines."""
    var seqs: _Counter
    """Per slot, whose turn it is: `pos` when free for the push at `pos`,
    `pos + 1` once that push has written it."""
    var tasks: Pointer[_Task, MutUntrackedOrigin]
    """Per slot, the task — raw storage, written only by the push that owns
    the slot."""

    def __init__(out self):
        self.seqs = unsafe_alloc[Atomic[Int64]](_QUEUE_SLOTS)
        for i in range(_QUEUE_SLOTS):
            self.seqs.unsafe_offset(i).unsafe_write(Atomic[Int64](Int64(i)))
        self.tasks = unsafe_alloc[_Task](_QUEUE_SLOTS)
        self.ends = unsafe_alloc[Atomic[Int64]](
            2 * _LINE_WORDS, alignment=_LINE_BYTES
        )
        for i in range(2 * _LINE_WORDS):
            self.ends.unsafe_offset(i).unsafe_write(Atomic[Int64](0))

    def __deinit__(deinit self):
        self.ends.unsafe_free()
        self.seqs.unsafe_free()
        self.tasks.unsafe_free()

    def head(self) -> _Counter:
        return self.ends

    def tail(self) -> _Counter:
        return self.ends.unsafe_offset(_LINE_WORDS)

    def enqueue(self, task: AnyCoroutine) -> Bool:
        """Append `task`; False if the ring is full."""
        comptime mask = _QUEUE_SLOTS - 1
        var pos = self.tail()[].load()
        while True:
            var i = Int(pos) & mask
            var dif = self.seqs.unsafe_offset(i)[].load() - pos
            if dif == 0:
                if self.tail()[].compare_exchange(pos, pos + 1):
                    self.tasks.unsafe_offset(i).unsafe_write(_Task(task))
                    self.seqs.unsafe_offset(i)[].store(pos + 1)
                    return True
            elif dif < 0:
                return False
            else:
                pos = self.tail()[].load()

    def dequeue(self) -> Optional[_Task]:
        """Take the oldest task, or None if the ring is empty."""
        comptime mask = _QUEUE_SLOTS - 1
        var pos = self.head()[].load()
        while True:
            var i = Int(pos) & mask
            var dif = self.seqs.unsafe_offset(i)[].load() - (pos + 1)
            if dif == 0:
                if self.head()[].compare_exchange(pos, pos + 1):
                    var task = self.tasks.unsafe_offset(i)[]
                    self.seqs.unsafe_offset(i)[].store(
                        pos + Int64(_QUEUE_SLOTS)
                    )
                    return task
            elif dif < 0:
                return None
            else:
                pos = self.head()[].load()

    def empty(self) -> Bool:
        return self.head()[].load() >= self.tail()[].load()


# ---------------------------------------------------------------------------
# The pool's shared state
# ---------------------------------------------------------------------------


struct _PoolState(Movable):
    """What the workers and the pool share: the queue of tasks, and who is
    asleep. The queue takes no lock; the mutex is for sleeping, and for the
    rare failure a job records. Reached only through `_Shared`'s raw pointer
    (see `_raw`)."""

    var ring: _Ring
    var mutex: Mutex
    var work_cond: Condition
    """Idle workers sleep here."""
    var done_cond: Condition
    """Threads waiting in `run` or `scope` sleep here. They are woken when a
    count they wait on reaches zero, and for new work when no worker is idle —
    a waiter nested inside a task may be the only thread left to run it."""
    var idle_workers: Atomic[Int64]
    var idle_waiters: Atomic[Int64]
    """Threads asleep, or about to be, on each condition. Raised under the
    mutex, read without it by pushers and finishers — see `_Shared.push` and
    `_Shared.count_down` for why that cannot lose a wake-up."""
    var shutdown: Bool
    var spin_ns: Int
    """How long an idle thread polls for work before it sleeps."""

    def __init__(out self, spin_ns: Int):
        self.ring = _Ring()
        self.mutex = Mutex()
        self.work_cond = Condition()
        self.done_cond = Condition()
        self.idle_workers = Atomic[Int64](0)
        self.idle_waiters = Atomic[Int64](0)
        self.shutdown = False
        self.spin_ns = spin_ns


comptime _StatePtr = Pointer[_PoolState, MutUntrackedOrigin]


@fieldwise_init
struct _Shared(ImplicitlyCopyable, Movable):
    """A raw pointer to a pool's `_PoolState`, and the protocol its threads
    follow on it: queue a task and wake a sleeper, run what is queued, poll,
    and sleep until there is work or a count reaches zero.

    A plain pointer carries no exclusivity claim, so every field read here is a
    real load. The pool joins its workers before it drops the state, so the
    pointer never outlives what it points at.
    """

    var state: _StatePtr

    @staticmethod
    def of(arc: ArcPointer[_PoolState]) -> Self:
        return Self(_raw(arc[]))

    def empty(self) -> Bool:
        return self.state[].ring.empty()

    # Kept as one `push` per task rather than a batched publish: split into an
    # inlined enqueue loop and a wake-up, it was copied into every `run[Body]`
    # and cost the size-gated binaries ~14 KB each.

    def push(self, task: AnyCoroutine) -> Bool:
        """Enqueue `task` and wake a sleeping thread to take it; False if the
        ring is full.

        No wake-up is lost: this side publishes the task and then reads the
        idle counts, and a sleeper raises its count and then re-checks the
        ring, holding the mutex until it sleeps. Both are sequentially
        consistent, so one sees the other's write — and the signal, taken
        under the mutex, cannot land between the sleeper's check and its wait.
        """
        if not self.state[].ring.enqueue(task):
            return False
        if self.state[].idle_workers.load() > 0:
            with self.state[].mutex.locked():
                self.state[].work_cond.signal()
        elif self.state[].idle_waiters.load() > 0:
            with self.state[].mutex.locked():
                self.state[].done_cond.broadcast()
        return True

    def try_run_one(self) -> Bool:
        var task = self.state[].ring.dequeue()
        if task:
            _run_erased(task.value().handle)
            return True
        return False

    def count_down(self, counter: _Counter, n: Int):
        """Subtract `n` finished items from `counter`, waking the waiters if
        that was the last of them.

        The mutex is taken only when a waiter is asleep, by the same argument
        as `push`: this side decrements and then reads `idle_waiters`; a
        waiter raises `idle_waiters` and then reads the counter, under the
        mutex.
        """
        if (
            counter[].fetch_sub(Int64(n)) == Int64(n)
            and self.state[].idle_waiters.load() > 0
        ):
            with self.state[].mutex.locked():
                self.state[].done_cond.broadcast()

    def spin(self, counter: Optional[_Counter]):
        """Poll for work — or `counter` reaching zero — for up to the pool's
        `spin_ns`, reading the clock every 64 polls."""
        var deadline = perf_counter_ns() + self.state[].spin_ns
        var polls = 0
        while self.empty():
            if counter and counter.value()[].load() == 0:
                break
            polls += 1
            if polls & 63 == 0 and perf_counter_ns() >= deadline:
                break

    def work(self):
        """A worker's whole life: run tasks until the pool shuts down and the
        ring is empty, spinning briefly before each sleep."""
        while True:
            if self.try_run_one():
                continue
            self.spin(None)
            if not self.empty():
                continue
            var done: Bool
            with self.state[].mutex.locked() as guard:
                _ = self.state[].idle_workers.fetch_add(1)
                if self.empty() and not self.state[].shutdown:
                    self.state[].work_cond.wait(guard)
                _ = self.state[].idle_workers.fetch_sub(1)
                done = self.state[].shutdown and self.empty()
            if done:
                break

    def help_until(self, counter: _Counter):
        """Run queued tasks until `counter` reaches zero; spin, then sleep,
        when there are none."""
        while counter[].load() != 0:
            if self.try_run_one():
                continue
            self.spin(counter)
            if counter[].load() == 0 or not self.empty():
                continue
            with self.state[].mutex.locked() as guard:
                _ = self.state[].idle_waiters.fetch_add(1)
                if counter[].load() != 0 and self.empty():
                    self.state[].done_cond.wait(guard)
                _ = self.state[].idle_waiters.fetch_sub(1)

    def shut_down(self):
        with self.state[].mutex.locked():
            self.state[].shutdown = True
            self.state[].work_cond.broadcast()
            self.state[].done_cond.broadcast()


# ---------------------------------------------------------------------------
# Fork-join: `run`
# ---------------------------------------------------------------------------


struct _ForJob[Body: def(Int) -> None](Movable):
    """`run`'s shared state, on the heap: the body, the range, the next index to
    hand out, the indices not yet finished, and who still holds it.

    `run` waits for the indices, not for the helpers it offered: an offer that
    starts late finds no index left and never touches `body`. That is why the
    job lives on the heap and is freed by whoever lets go of it last.

    `body` points at the *caller's* closure — `run` takes it by `ref`. A
    pointer to a by-value argument would point at a callee's own copy.
    """

    var body: Pointer[Self.Body, ImmUntrackedOrigin]
    var n: Int
    var next: Atomic[Int64]
    var pending: Atomic[Int64]
    var holders: Atomic[Int64]

    def __init__(
        out self,
        body: Pointer[Self.Body, ImmUntrackedOrigin],
        n: Int,
        holders: Int,
    ):
        self.body = body
        self.n = n
        self.next = Atomic[Int64](0)
        self.pending = Atomic[Int64](Int64(n))
        self.holders = Atomic[Int64](Int64(holders))


comptime _JobPtr[Body: def(Int) -> None] = Pointer[
    _ForJob[Body], MutUntrackedOrigin
]


@no_inline
def _claim[Body: def(Int) -> None](job: _JobPtr[Body], shared: _Shared):
    """Run indices from `job` until none are left, then count them done.

    Counted once per claimer, not per index: the job cannot finish while any
    claimer still holds an index, and a claimer holds one until it has run it
    and found the counter exhausted.

    Out of line, and the one place `body` is called: the caller of `run`, its
    offers of help and its serial case all claim through it, so each body is
    compiled once.
    """
    var done = 0
    while True:
        var i = Int(job[].next.fetch_add(1))
        if i >= job[].n:
            break
        job[].body[](i)
        done += 1
    if done > 0:
        shared.count_down(_raw(job[].pending), done)


def _release[Body: def(Int) -> None](job: _JobPtr[Body]):
    if job[].holders.fetch_sub(1) == 1:
        _destroy(job)


async def _offer[Body: def(Int) -> None](job: _JobPtr[Body], state: _StatePtr):
    """A queued offer of help with `job`."""
    _claim[Body](job, _Shared(state))
    _release[Body](job)


# ---------------------------------------------------------------------------
# How many threads
# ---------------------------------------------------------------------------


def _read_text(path: String) -> Optional[String]:
    """A small text file's contents — a `/proc` or `/sys` file, which reads
    to its end whatever size it reports — or None if it cannot be read.

    Read through libc, not `open()`: on Linux, where this runs on every
    shared pool's start, the stdlib's read path here deadlocked the compiler
    on the golden suite's unit. The calls are declared exactly as the
    stdlib's own, which share their symbols.
    """
    var c_path = path.copy()
    var fd = external_call["open", c_int, num_fixed_args=2](
        c_path.as_c_string_span(), c_int(0), c_int(0o666)
    )
    if fd < 0:
        return None
    var bytes = List[UInt8](length=4096, fill=0)
    var size = 0
    var got = c_ssize_t(1)
    while got > 0:
        if size == len(bytes):
            bytes.resize(2 * size, 0)
        got = external_call["read", c_ssize_t](
            Int(fd), bytes.unsafe_ptr().unsafe_offset(size), len(bytes) - size
        )
        if got > 0:
            size += Int(got)
    _ = external_call["close", c_int](fd)
    if got < 0:
        return None
    return String(from_utf8_lossy=Span(bytes)[:size])


def _positive_int(text: StringSlice) -> Int:
    """`text` as a positive decimal integer, spaces around it aside, or 0 when
    it is not one.

    Not `atol`: its error path links into every binary that starts a shared
    pool, and the size-gated ones paid for it.
    """
    var value = 0
    for byte in text.strip().as_bytes():
        var digit = Int(byte) - ord("0")
        if digit >= 0 and digit <= 9 and value < (1 << 40):
            value = value * 10 + digit
        else:
            value = 0
            break
    return value


def _warn(var message: String):
    """`message` and a newline on stderr, in one `write(2)`. Not `print`,
    which the size-gated binaries do not otherwise link."""
    message += "\n"
    var err = stderr
    err.write_bytes(message.as_bytes())


def _quota_cpus(quota: StringSlice, period: StringSlice) -> Optional[Int]:
    """`quota` microseconds of CPU time in every `period` as whole CPUs,
    rounded up; None for no quota — `max`, -1 — or text that is not one."""
    var q = _positive_int(quota)
    var p = _positive_int(period)
    if q > 0 and p > 0:
        return ceildiv(q, p)
    return None


def _cpu_max_cap(path: String) -> Optional[Int]:
    """The cap a cgroup v2 `cpu.max` file (`<quota> <period>`, or `max
    <period>`) sets, in whole CPUs rounded up; None for no cap or no file."""
    var text = _read_text(path)
    if text:
        var fields = text.value().split()
        if len(fields) == 2:
            return _quota_cpus(fields[0], fields[1])
    return None


def _cgroup_cpu_limit(proc_cgroup: String, root: String) -> Optional[Int]:
    """The CPU quota this process's cgroup sets, in whole CPUs rounded up, or
    None if it sets none — what `docker --cpus` and a Kubernetes CPU limit
    set. `proc_cgroup` is `/proc/self/cgroup` and `root` the cgroup mount,
    `/sys/fs/cgroup`; a test passes others.

    Under cgroup v2 the process's group is its `0::` line, and each group from
    there up to the root may cap it in `cpu.max` (see `_cpu_max_cap`); the
    lowest cap applies. When none does, a v1 `cpu` controller is read at its
    root, which is a container's own group: `cpu.cfs_quota_us` (-1 for none)
    over `cpu.cfs_period_us`.
    """
    var limit: Optional[Int] = None
    var membership = _read_text(proc_cgroup)
    if membership:
        for line in membership.value().splitlines():
            if line.startswith("0::"):
                var group = String(line.removeprefix("0::").strip())
                if group == "/":
                    group = ""
                while True:
                    var cap = _cpu_max_cap(root + group + "/cpu.max")
                    if cap and (not limit or cap.value() < limit.value()):
                        limit = cap
                    if group == "":
                        break
                    var parent = String(group[byte = 0 : group.rfind("/")])
                    group = parent^
        # A v1 `cpu` controller, on a host with only v1 or with v1 and v2
        # mounted side by side, the `0::` line naming a group without one.
        if not limit:
            var quota = _read_text(root + "/cpu/cpu.cfs_quota_us")
            var period = _read_text(root + "/cpu/cpu.cfs_period_us")
            if quota and period:
                limit = _quota_cpus(quota.value(), period.value())
    return limit


def _available_cpus() -> Int:
    """The logical CPUs this process may run on.

    The runtime's core counts already follow the affinity mask — under
    `taskset -c 0,1` or `docker --cpuset-cpus=0,1` they answer 2 — but not a
    cgroup CPU quota: under `docker --cpus=2` on an 8-core machine they still
    answer 8. So on Linux the quota is read here, and caps them.
    """
    var cpus = num_logical_cores()
    comptime if CompilationTarget.is_linux():
        var quota = _cgroup_cpu_limit("/proc/self/cgroup", "/sys/fs/cgroup")
        if quota:
            cpus = min(cpus, quota.value())
    return max(cpus, 1)


# ---------------------------------------------------------------------------
# The shared pools
# ---------------------------------------------------------------------------


def _start_with_fewer[
    Start: def(Int) raises DynError -> ArcPointer[ThreadPool]
](workers: Int, start: Start) -> ArcPointer[ThreadPool]:
    """The pool `start(workers)` gives or, when the OS refuses that many
    threads, the one `start` gives with half as many workers, and so on down
    to none, which needs no thread at all.

    A process can run out of threads — a container's pids limit, a cloud
    function's cap on threads — and a shared pool is started implicitly, by
    the first kernel that stripes: aborting there would take the process down
    for want of parallelism. A smaller pool is reported on stderr. A pool
    asked for by size, `ThreadPool(n)` or `resize_shared`, raises instead.
    """
    var want = workers
    while True:
        try:
            var pool = start(want)
            if want < workers:
                _warn(
                    String("marrow: a shared pool started with ")
                    + String(want)
                    + " of "
                    + String(workers)
                    + " workers, as the OS would not start more threads"
                )
            return pool^
        except e:
            if want == 0:
                abort(String("ThreadPool: cannot start a shared pool: ", e))
            want //= 2


struct _PoolCell(Movable):
    """Where a shared pool lives, in the runtime's registry of globals: a
    lock, the pool, and the process that started it.

    The registry runs the creating function *outside* any lock and keeps
    whichever result lands first, so it cannot hold the pool itself: threads
    racing on first use would each start one. A cell is cheap to create and to
    lose, and its lock lets exactly one thread start the pool — or replace it,
    for `resize_shared` and in a forked child.

    The cell is registered with `destroy` as its destructor. The stdlib's
    `_Global` wrapper would free it without running one, so the pool would
    never shut its workers down and LeakSanitizer would report its blocks at
    exit. Through `destroy`, the runtime's teardown of its globals, after
    `main` returns, drops each pool, and its destructor joins the workers, as
    Arrow C++ does with its own global pool.

    Reached only through a raw pointer, like every state other threads change.
    """

    var lock: Atomic[Int64]
    """0 when free, else the id of the process whose thread holds it. A
    `fork` can copy the cell while another of the parent's threads holds the
    lock; that thread does not exist in the child, which therefore takes the
    lock over from any other process."""
    var pool: Optional[ArcPointer[ThreadPool]]
    var pid: Int64
    """The process `pool` was started in. A forked child holds a copy of the
    parent's pool but none of its threads, and starts its own."""

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.pool = None
        self.pid = 0

    @staticmethod
    def create() -> _GlobalPtr:
        """A fresh cell on the heap: the registry's creating function."""
        return _box(_PoolCell()).unsafe_bitcast[NoneType]()

    @staticmethod
    def destroy(cell: _GlobalPtr):
        """The registry's destructor: at exit, or once a cell has lost the race
        to be registered. Dropping the last reference to its pool shuts the
        workers down and joins them."""
        if cell:
            _destroy(cell.unsafe_value().unsafe_bitcast[_PoolCell]())

    @staticmethod
    def lock_for(cell: _CellPtr, pid: Int64):
        """Take `cell`'s lock for process `pid`, waiting while another of its
        threads holds it, and taking it over from another process's."""
        while True:
            var held = Int64(0)
            if cell[].lock.compare_exchange(held, pid):
                break
            if held != pid and cell[].lock.compare_exchange(held, pid):
                break
            Thread.yield_now()

    @staticmethod
    def unlock(cell: _CellPtr):
        cell[].lock.store(0)


comptime _GlobalPtr = OptionalPointer[NoneType, MutUntrackedOrigin]
"""What the runtime's registry of globals holds: a `void *`."""

comptime _CellPtr = Pointer[_PoolCell, MutUntrackedOrigin]


struct _SharedPool[io: Bool]:
    """One of the two process-wide pools — the compute pool, or with `io` the
    one for blocking work: what tells them apart, and how one is found,
    started, replaced and restarted after a fork. `ThreadPool.shared`,
    `shared_io` and the two `resize_shared` methods are its public face."""

    comptime key = "marrow.threads.io" if Self.io else "marrow.threads.shared"
    """The pool's entry in the runtime's registry of globals."""
    comptime variable = (
        "MARROW_IO_THREADS" if Self.io else "MARROW_NUM_THREADS"
    )
    """The environment variable that sizes it."""
    comptime name = "marrow-io" if Self.io else "marrow-cpu"
    """What its workers are named after."""
    comptime spin_ns = 0 if Self.io else _SPIN_NS
    """How long its idle threads poll; see `ThreadPool.shared_io`."""

    @staticmethod
    def default_concurrency() -> Int:
        """How many threads the pool starts with; see `ThreadPool.shared` and
        `ThreadPool.shared_io`. A value of `variable` that is not a positive
        integer is reported on stderr and ignored."""
        var text = getenv(Self.variable)
        var chosen = _positive_int(text)
        if text and chosen == 0:
            _warn(
                String("marrow: ignoring ")
                + Self.variable
                + "='"
                + text
                + "', which is not a positive integer"
            )
        if chosen == 0:
            var cpus = _available_cpus()
            comptime if Self.io:
                chosen = max(cpus, 2)
            else:
                chosen = max(min(num_performance_cores(), cpus), 1)
        return chosen

    @staticmethod
    def start(workers: Int) raises DynError -> ArcPointer[ThreadPool]:
        """A pool of this kind with `workers` workers."""
        return ArcPointer(
            ThreadPool(workers, spin_ns=Self.spin_ns, name=Self.name)
        )

    @staticmethod
    def cell() -> _CellPtr:
        var registered = _get_global[
            Self.key, _PoolCell.create, _PoolCell.destroy
        ]()
        if not registered:
            abort("ThreadPool: cannot reach a shared pool")
        return registered.unsafe_value().unsafe_bitcast[_PoolCell]()

    @no_inline
    @staticmethod
    def get() -> ArcPointer[ThreadPool]:
        """The pool, started on the first call in this process. Out of line so
        its start-up path is not copied into every kernel that reaches
        `ExecContext.thread_pool()` (~5 KB on the size gates)."""
        var cell = Self.cell()
        var pid = _pid()
        _PoolCell.lock_for(cell, pid)
        var stale: Optional[ArcPointer[ThreadPool]] = None
        if not cell[].pool or cell[].pid != pid:
            stale = cell[].pool.copy()
            cell[].pool = _start_with_fewer(
                Self.default_concurrency() - 1, Self.start
            )
            cell[].pid = pid
        var pool = cell[].pool.value().copy()
        _PoolCell.unlock(cell)
        # Let go outside the lock. In a forked child the parent's copy is
        # abandoned rather than joined; see `ThreadPool.__deinit__`.
        _ = stale^
        return pool^

    @staticmethod
    def resize(concurrency: Int) raises DynError:
        """Replace the pool with one of `concurrency` threads; see
        `ThreadPool.resize_shared`."""
        if concurrency < 1:
            raise InvalidError(t"ThreadPool: a concurrency of {concurrency}")
        var fresh = Self.start(concurrency - 1)
        var cell = Self.cell()
        var pid = _pid()
        _PoolCell.lock_for(cell, pid)
        var stale = cell[].pool.copy()
        cell[].pool = fresh^
        cell[].pid = pid
        _PoolCell.unlock(cell)
        # The old pool's workers are joined once its last user lets go —
        # here, unless work is still running on it; never under the lock,
        # which a task on that pool may be waiting for.
        _ = stale^


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def _abandon[T: Movable](var value: T):
    """Make sure `value` is never destroyed, by moving it into a heap cell
    nothing points at. For what a forked child must not touch; see
    `ThreadPool.__deinit__`."""
    _ = _box(value^)


struct ThreadPool(Movable):
    """Worker threads and one queue of tasks.

    A pool built with `ThreadPool(n)` owns `n` threads and joins them when it
    is destroyed, after they have drained the queue. `shared()` and
    `shared_io()` are the process-wide pools, for compute and for blocking
    work.

    Workers run at the default QoS class, the one placement control macOS
    honours, while a command-line caller runs at `USER_INTERACTIVE`. Raising
    them to `USER_INITIATED` or `USER_INTERACTIVE` was measured on 1M-element
    stripes over 2-12 threads at load 62-70 — the contention it would have to
    win against — and changed nothing at the median or the 90th percentile.
    """

    var _state: ArcPointer[_PoolState]
    var _threads: List[Thread]
    var _pid: Int64
    """The process the workers were started in; see `__deinit__`."""

    def __init__(
        out self,
        workers: Int,
        stack_size: Int = _DEFAULT_STACK_BYTES,
        spin_ns: Int = _SPIN_NS,
        name: String = "marrow",
    ) raises DynError:
        """Start `workers` threads now, each with a `stack_size`-byte stack.

        An idle worker, and a thread waiting in `run` or `scope`, polls for
        work for `spin_ns` nanoseconds before it sleeps; see `_SPIN_NS` for
        the default and `shared_io` for a pool that does not poll. Worker `k`
        is named `"{name}-{k}"`, cut to the 15 bytes Linux takes, which is
        what a debugger or profiler shows for it.

        Raises `InvalidError` for a negative worker count or spin. If a thread
        cannot be started, the ones already running are shut down and joined
        — by `__deinit__`, which Mojo runs when `__init__` raises after every
        field is set — before the error propagates.
        """
        if workers < 0:
            raise InvalidError(t"ThreadPool: {workers} workers")
        if spin_ns < 0:
            raise InvalidError(t"ThreadPool: spin of {spin_ns} ns")
        self._state = ArcPointer(_PoolState(spin_ns))
        self._threads = List[Thread](capacity=workers)
        self._pid = _pid()
        var shared = _Shared.of(self._state)
        for k in range(workers):
            var label = name.copy()

            def body() {var shared, var k, var label^}:
                _name_thread(label, k)
                shared.work()

            self._threads.append(Thread(body^, stack_size))

    def __deinit__(deinit self):
        """Shut the workers down and join them, after they drain the queue.

        Aborts when called from one of the pool's own workers — a task that
        drops the last handle to its pool — since that worker would join
        itself and then run on in freed state.

        In a process `fork`ed from the one that started them, the workers do
        not exist, and the state may hold a lock one of them held at the fork:
        a pool destroyed there is abandoned instead — nothing is locked,
        joined or freed.
        """
        if self._pid != _pid():
            _abandon(self._state^)
            _abandon(self._threads^)
        else:
            var me = Thread.current_id()
            for k in range(len(self._threads)):
                if self._threads[k].id() == me:
                    abort("ThreadPool destroyed from one of its own workers")
            _Shared.of(self._state).shut_down()
            while self._threads:
                self._threads.pop().join()
            # The workers read the state until they are joined, and a field of
            # a `deinit self` dies at its last use — which, without this, would
            # be the `shut_down` call above.
            _ = self._state^

    @staticmethod
    def shared() -> ArcPointer[Self]:
        """The process-wide compute pool, started on first use — by one thread,
        however many make that first use at once.

        It runs `MARROW_NUM_THREADS` threads, counting the caller of `run`,
        when that is set to a positive integer; otherwise one per performance
        core, since an efficiency core would set the pace of any stripe it
        drew, within the CPUs the process may use — on Linux its affinity
        mask and cgroup CPU quota can allow fewer. `resize_shared` replaces
        it, and a `fork`ed child starts its own on first use there.

        A Mojo program shuts it down — its workers joined — when the runtime
        tears down its globals after `main` returns; a Python process never
        runs that teardown, so there the pool lasts until the process ends.
        """
        return _SharedPool[False].get()

    @staticmethod
    def shared_io() -> ArcPointer[Self]:
        """The process-wide pool for work that blocks, started on first use.

        Separate from `shared()` on two grounds: blocking work should not
        occupy the compute workers, and a thread waiting on the compute pool
        runs whatever is queued there, so a kernel's caller would otherwise
        pick up a fetch. Its threads do not poll when idle (`spin_ns=0`): the
        next task waits on the network anyway, so polling would only burn a
        core.

        It runs `MARROW_IO_THREADS` threads when that is set to a positive
        integer, and otherwise one per logical core the process may use, at
        least two; `resize_shared_io` replaces it. Otherwise it behaves as
        `shared()` does.
        """
        return _SharedPool[True].get()

    @staticmethod
    def resize_shared(concurrency: Int) raises DynError:
        """Replace the process-wide compute pool with one on which a `run`
        occupies up to `concurrency` threads: `concurrency - 1` workers and
        the caller.

        Every later `shared()` call returns the new pool. Work already running
        on the old one finishes there, and its workers are joined once the
        last of that work lets go of it. Raises `InvalidError` for a
        concurrency below 1, and `IOError` if a thread cannot be started — the
        old pool stays then.
        """
        _SharedPool[False].resize(concurrency)

    @staticmethod
    def resize_shared_io(concurrency: Int) raises DynError:
        """`resize_shared` for the I/O pool, `shared_io()`."""
        _SharedPool[True].resize(concurrency)

    def size(self) -> Int:
        """Worker threads, not counting a caller of `run` or `scope`."""
        return len(self._threads)

    def concurrency(self) -> Int:
        """How many threads one `run` can occupy at once: the workers and the
        caller."""
        return len(self._threads) + 1

    def spin_ns(self) -> Int:
        """How long an idle thread of this pool polls for work before it
        sleeps."""
        return _Shared.of(self._state).state[].spin_ns

    def sleeping(self) -> Int:
        """How many workers are asleep waiting for work, or about to be — a
        snapshot for tests and diagnostics, stale as soon as it is read."""
        return Int(_Shared.of(self._state).state[].idle_workers.load())

    def thread_ids(self) -> List[Int]:
        """Every worker's `Thread.id()` — which threads a job may run on."""
        var ids = List[Int](capacity=len(self._threads))
        for k in range(len(self._threads)):
            ids.append(self._threads[k].id())
        return ids^

    def pin(self) raises DynError:
        """Pin worker `k` to logical core `k + 1`, leaving core 0 to the
        caller. Raises `NotImplementedError` where threads cannot be pinned
        (see `_CAN_PIN`) — a pool with no workers included."""
        comptime if not _CAN_PIN:
            raise NotImplementedError(
                "ThreadPool.pin: this platform has no binding thread affinity"
            )
        var cores = num_logical_cores()
        for k in range(len(self._threads)):
            self._threads[k].pin((k + 1) % cores)

    def fan_out[
        E: Movable & Deinitable, Body: def(Int, Int) raises E -> None
    ](self, n: Int, body: Body, lanes: Int) raises E:
        """Run `body(wid, i)` for every `i` in `[0, n)` on up to `lanes`
        lanes, and raise what the lowest failing `i` raised.

        `run`'s counterpart for a body that **raises** — a fetch, a decode, an
        operator over a partition — where an error has to reach the caller.

        A lane takes the next item as soon as it has finished one, so an
        expensive item holds up only its own lane while the others take what
        is left. Items are handed out in increasing order, each to exactly one
        lane. A lane runs on one thread at a time and at most `concurrency()`
        lanes run at once, so `wid` — in `[0, min(lanes, n))` — can index
        per-lane scratch the caller allocated. Which items a lane draws is not
        fixed: a body writes its result to a slot indexed by `i`. The price is
        one atomic increment per item on a counter every lane shares — about
        15 ns an item with twelve lanes on empty items, nothing next to an item
        worth a thread. Claiming items in batches would amortise it, but then a
        slow item holds up the rest of its batch.

        It raises what `body` raises, as that type: a typed error stays typed
        across threads, and a bare `raises` stays `Error`.

        **The error raised is deterministic.** A lane stops at its first
        error, and a failure stops every lane at the first item above the
        lowest failure seen so far. Items are handed out in order, so every
        item below the eventual lowest failure has already started and runs
        to its end, and that failure is the error raised — the one a serial
        loop would have stopped at, whatever the core count and whichever
        thread failed first.

        With one effective lane the loop runs on the calling thread and the
        error propagates directly, with no dispatch and no parking.

        A body that returns a `Bool` can also stop early; see the overload
        below. It is a second copy of this loop rather than this one adapting
        `body`: a closure that raises a generic `E` cannot be declared inside
        a function, and a bare `raises` would flatten typed errors to `Error`.
        """
        var nt = max(1, min(lanes, n, self.concurrency()))
        if nt == 1:
            for i in range(n):
                body(0, i)
        else:
            var shared = _Shared.of(self._state)
            # `cursor` is the next item to hand out. The closure captures it
            # and `outcome` themselves, which keeps them alive until `run`
            # returns — a pointer taken out here would not.
            var cursor = Atomic[Int64](0)
            var outcome = _Outcome[E](Int64(n))

            def task(wid: Int) {imm body, imm shared, imm cursor, imm outcome}:
                var claim = _raw(cursor)
                var cutoff = _Cutoff(outcome)
                var i = claim[].fetch_add(1)
                try:
                    while cutoff.wants(i):
                        body(wid, Int(i))
                        i = claim[].fetch_add(1)
                except e:
                    cutoff.fail(shared, i, e^)

            self.run(nt, task, nt)
            if outcome.ends_in_failure():
                raise outcome.take_error()

    def fan_out[
        E: Movable & Deinitable, Body: def(Int, Int) raises E -> Bool
    ](self, n: Int, body: Body, lanes: Int) raises E:
        """`fan_out` for a body that can stop early: `body(wid, i)` returns
        whether the items after `i` are still wanted — True to go on, False to
        stop, as a serial loop would `break`. A source that only has to
        produce enough rows for a `LIMIT` stops this way.

        When `body(wid, s)` returns False, the stop takes effect as it returns,
        not when item `s` was handed out. The other lanes have gone on claiming
        items while `s` ran, and those finish; from then on no lane starts
        another item, though each may start the one it had already claimed.
        The call then returns normally. Every item below `s` has started,
        since items are handed out in order, and runs to its end.

        A stop and a failure resolve as they would in that serial loop: the
        lower one decides. A failure below the stop is raised, and one above it
        is dropped, since the loop would never have reached it.
        """
        var nt = max(1, min(lanes, n, self.concurrency()))
        if nt == 1:
            for i in range(n):
                if not body(0, i):
                    break
        else:
            var shared = _Shared.of(self._state)
            # As above: the closure captures `cursor` and `outcome` itself.
            var cursor = Atomic[Int64](0)
            var outcome = _Outcome[E](Int64(n))

            def task(wid: Int) {imm body, imm shared, imm cursor, imm outcome}:
                var claim = _raw(cursor)
                var cutoff = _Cutoff(outcome)
                var i = claim[].fetch_add(1)
                try:
                    while cutoff.wants(i):
                        if not body(wid, Int(i)):
                            cutoff.stop_after(i)
                            break
                        i = claim[].fetch_add(1)
                except e:
                    cutoff.fail(shared, i, e^)

            self.run(nt, task, nt)
            if outcome.ends_in_failure():
                raise outcome.take_error()

    def scope[
        Body: def(TaskScope) -> None
    ](self, ref body: Body) raises DynError:
        """Call `body(scope)` and return once every task it spawned on `scope`
        has finished; see `TaskScope`. While it waits, the calling thread runs
        queued tasks rather than sleeping.

        Raises the earliest-spawned task's error once every task has finished,
        as a `DynError` that keeps its kind.

        **A task must not block on another task** — on a `Mutex` held across
        a wait, a `Condition` another task signals, a value another task
        produces. Waiting *through the pool* is safe, since a waiter runs
        queued work; a thread blocked any other way does not, and once every
        thread is blocked on a task still in the queue, nothing runs it.
        """

        def raising(scope: TaskScope) raises {imm body}:
            body(scope)

        self.scope(raising)

    def scope[
        E: Writable & Movable & Deinitable,
        Body: def(TaskScope) raises E -> None,
    ](self, ref body: Body) raises DynError:
        """`scope` for a body that raises. If it does, the scope is cancelled,
        the tasks already running are waited for, and `body`'s error is
        raised instead of any task's."""
        var scope = TaskScope(_Shared.of(self._state))
        try:
            body(scope)
        except e:
            scope.cancel()
            _ = scope^
            raise _dyn(e)
        scope^.join()

    def run[
        Body: def(Int) -> None
    ](self, n: Int, ref body: Body, max_threads: Int):
        """Call `body(i)` for every `i` in `[0, n)`, on at most `max_threads`
        threads counting the caller, and return once all calls have returned.

        Indices are handed out one at a time from a shared counter, so which
        thread runs which index is not fixed; a `body` that keeps per-thread
        scratch must index it by `i`. `n <= 0` runs nothing, and
        `max_threads <= 1` runs every index on the caller. `body` cannot raise;
        `fan_out` is the raising counterpart.

        A call that uses other threads allocates the job and one coroutine
        frame per offer of help, and returns as soon as every index has run,
        whether or not the offers were taken up; see `_ForJob`.

        `body(i)` must not block on another index, or on a task of this pool,
        other than through the pool; see `scope`.
        """
        var k = min(max_threads, n, self.concurrency())
        var body_ptr = (
            Pointer(to=body)
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        if k <= 1:
            # Through `_claim` rather than a loop of its own: `body` is inlined
            # where it is called, so a loop here would be a second copy of
            # every kernel striped through the pool.
            var serial = _ForJob[Body](body_ptr, n, holders=1)
            _claim[Body](_raw(serial), _Shared.of(self._state))
            _ = serial^
        else:
            var job = _box(_ForJob[Body](body_ptr, n, holders=k))
            var shared = _Shared.of(self._state)
            for _ in range(k - 1):
                var offer = _handle(_offer[Body](job, shared.state))
                if not shared.push(offer):
                    _coro_destroy_fn(offer)
                    _release[Body](job)
            _claim[Body](job, shared)
            shared.help_until(_raw(job[].pending))
            _release[Body](job)
