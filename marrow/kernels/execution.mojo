"""Execution context for kernel dispatch.

Bundles the two axes of parallelism that kernels need to know about:

- **Device** — an optional GPU ``DeviceContext``. When set, kernels run
  on the GPU; when ``None``, they run on the CPU. Mirrors today's
  ``Optional[DeviceContext]`` parameter that appears on every apply /
  kernel.
- **Threads** — CPU worker count for striped parallelism on the non-GPU
  path. ``1`` is serial (current pre-parallel behavior), ``>1`` uses
  ``sync_parallelize``, ``0`` means "auto" → ``num_physical_cores()``.
  Below ``min_parallel_size`` (per-kernel threshold) the dispatch
  collapses to serial so stripe overhead never exceeds the work.

Kernels take one of these instead of a bare ``Optional[DeviceContext]``
so the CPU multi-thread path can be enabled uniformly — rather than each
kernel implementing its own ``sync_parallelize`` stripe loop.

Implicit conversions from ``Optional[DeviceContext]`` keep all existing
call sites working without source changes.
"""

from std.algorithm.functional import sync_parallelize
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.sys.info import num_physical_cores


struct ExecutionContext(
    ConvertibleFromPython, ConvertibleToPython, Copyable, Movable, Writable
):
    """How a kernel should dispatch its work.

    See the module docstring for the full contract. Construct via one of
    the factory methods (``.serial()``, ``.parallel()``, ``.gpu()``) or
    via the default constructor (= serial, no GPU), or pass an
    ``Optional[DeviceContext]`` directly — it implicitly converts to a
    CPU-serial context with the given device.
    """

    var num_threads: Int
    """CPU dispatch strategy, encoded as an Int sentinel.

    - ``1``           — **Serial (forced)**: always run on the calling thread,
      irrespective of problem size.
    - ``N >= 2``      — **Multi(N) (forced)**: always stripe across exactly
      ``N`` workers via ``sync_parallelize``, irrespective of problem size.
    - ``0`` (default) — **Auto**: dispatch picks serial vs all-cores-multi
      based on the per-kernel ``min_parallel_size`` threshold consulted by
      ``wants_parallel()``.
    - ``< 0``         — reserved for future strategies
      (``adaptive``, ``work_stealing``…)."""

    var device: Optional[DeviceContext]
    """GPU ``DeviceContext``, or ``None`` for CPU execution."""

    def __init__(
        out self,
        num_threads: Int = 1,
        device: Optional[DeviceContext] = None,
    ):
        self.num_threads = num_threads
        self.device = device.copy() if device else None

    @implicit
    def __init__(out self, device: DeviceContext):
        """Implicit conversion from ``DeviceContext``."""
        self.num_threads = 1
        self.device = Optional[DeviceContext](device)

    @implicit
    def __init__(out self, device: Optional[DeviceContext]):
        """Implicit conversion from ``Optional[DeviceContext]``.

        Enables existing call sites that still pass a bare
        ``Optional[DeviceContext]`` (``None`` or ``Some(ctx)``) to keep
        working without source changes. Resulting context is
        ``num_threads=1`` — callers that want CPU parallelism build an
        ``ExecutionContext`` explicitly.
        """
        self.num_threads = 1
        self.device = device.copy() if device else None

    def __init__(out self, *, copy: Self):
        self.num_threads = copy.num_threads
        self.device = copy.device.copy() if copy.device else None

    # --- factories ----------------------------------------------------

    @staticmethod
    def serial() -> Self:
        """Single-threaded CPU execution. Forced — bypasses the size threshold.
        """
        return Self(num_threads=1, device=None)

    @staticmethod
    def parallel(num_threads: Int = 0) -> Self:
        """CPU execution with ``num_threads`` workers.

        - ``num_threads == 0`` (default) — **auto**: the dispatch picks serial
          vs all-cores-multi based on the per-kernel size threshold consulted
          by ``wants_parallel()``. Equivalent to ``auto()``.
        - ``num_threads >= 2`` — **forced multi**: always stripes across
          exactly ``num_threads`` workers, bypassing the size threshold.
        """
        return Self(num_threads=num_threads, device=None)

    @staticmethod
    def auto() -> Self:
        """CPU execution that picks serial vs all-cores-multi based on the
        per-kernel ``min_parallel_size`` threshold consulted by
        ``wants_parallel()``. Equivalent to ``parallel()`` with default
        ``num_threads=0``."""
        return Self(num_threads=0, device=None)

    @staticmethod
    def gpu(device: DeviceContext) -> Self:
        """GPU execution on the given device."""
        return Self(num_threads=1, device=Optional[DeviceContext](device))

    # --- queries ------------------------------------------------------

    def is_gpu(self) -> Bool:
        """True when work should be dispatched to the GPU."""
        return Bool(self.device)

    def resolved_num_threads(self) -> Int:
        """Normalize ``num_threads`` into a concrete worker count.

        - ``num_threads >= 1`` → returned as-is.
        - ``num_threads == 0`` (auto) → ``num_physical_cores()``.
        - ``num_threads < 0`` → treated as auto for now (reserved range).
        """
        if self.num_threads >= 1:
            return self.num_threads
        return num_physical_cores()

    def wants_parallel(self, n: Int, min_parallel_size: Int = 32768) -> Bool:
        """Decide whether a CPU kernel of size ``n`` should stripe work.

        Strategy contract (see ``num_threads`` doc):

        - ``num_threads == 1`` → **serial (forced)**: always ``False``.
        - ``num_threads >= 2`` → **multi(N) (forced)**: always ``True``,
          regardless of ``n``.
        - ``num_threads == 0`` → **auto**: ``True`` iff
          ``n >= min_parallel_size``. Below that threshold,
          ``sync_parallelize`` dispatch overhead dominates the actual compute.
        - GPU path always returns ``False`` (GPU handles its own parallelism).
        """
        if self.is_gpu():
            return False
        if self.num_threads == 1:
            return False
        if self.num_threads >= 2:
            return True
        return n >= min_parallel_size

    # --- striped execution ------------------------------------------------

    def stripe_workers(
        self, length: Int, min_parallel_size: Int = 32768
    ) -> Int:
        """How many stripes ``stripe`` will run for this length — 1 when serial.

        Callers whose body owns per-worker scratch (a histogram, a partials
        slot, a write-offset block) must allocate it *before* the call, sized by
        this. Without it every such caller re-derives `wants_parallel` and
        `resolved_num_threads` for itself, which is the reach-in this type
        exists to remove — and the two must agree with what `stripe` then does,
        or the body indexes past its scratch.
        """
        if self.wants_parallel(length, min_parallel_size):
            return self.resolved_num_threads()
        else:
            return 1

    @always_inline
    def stripe[
        body: def(Int, Int, Int) capturing[_] -> None
    ](self, length: Int, min_parallel_size: Int = 32768, align: Int = 1):
        """Run ``body(wid, start, end)`` over ``[0, length)``, striped or serial.

        ``wid`` is the stripe index, always in ``[0, stripe_workers(length))``.
        A body with no per-worker state simply ignores it; one that owns scratch
        indexes into it with ``wid``. That is why there is a single primitive
        rather than a plain and an indexed variant — the second would be a second
        way to do the same thing, and the unused parameter costs nothing.

        This is the CPU dispatch decision, in one place. Callers write their
        range loop **once**; whether it runs on the calling thread or across
        ``resolved_num_threads()`` workers is decided here.

        The duplication this removes is not the `wants_parallel` / `ceildiv` /
        clamp / empty-stripe preamble — that was only the visible half. Each
        kernel that striped by hand also had to write its inner loop *twice*,
        once inside the worker and once for the serial arm, so the two could
        (and did) drift independently.

        The serial arm is a single ``body(0, length)`` call rather than a
        one-worker stripe, so the no-parallelism path keeps exactly the shape it
        had before — no closure per stripe, no chunk arithmetic.

        **Mark ``body`` ``@always_inline``.** Measured on `Take.apply`'s gather
        over 1M elements: without it the conversion was ~10 % *slower* than the
        hand-rolled loop it replaced (median 3.1 ms vs 2.8 ms); with it, ~18 %
        faster (2.3 ms), and the striped path went 373 µs → 309 µs. A hand-rolled
        worker got inlined for free because `sync_parallelize` consumed it
        directly; routed through here it will not unless you say so.

        ``align`` rounds each stripe **up** to a multiple of itself. Pass the
        SIMD width when ``body`` is a vectorized loop with a scalar tail: it
        keeps every stripe boundary on a vector boundary, so the tail runs once
        at the end of the last stripe rather than once per stripe. Leaving it at
        1 for such a body is a silent throughput loss, not a correctness bug,
        which is exactly why it is a parameter rather than an assumption.

        **``body`` may not raise, and widening it is not a small change.**
        Tried and reverted 2026-07-28. `sync_parallelize` accepts a raising
        worker only in its *parameter* form (`sync_parallelize[w](n)`), which
        needs an implicitly-capturing `@parameter` closure; the *value* form
        used here takes an explicit capture list and rejects `raises`. Switching
        to the parameter form compiles — with new "assignment was never used"
        warnings on the very buffers the body writes — and then **crashes at
        run time**: the captures are not made and the body reads garbage. The
        warnings are the tell. `test_partition.mojo`'s coverage assertions catch
        it immediately, which is what they are for.

        Consequence: a kernel whose stripe body raises keeps its hand-rolled
        loop. `GroupBy._thread_local_columns` is the one such caller
        (`groupby.mojo`) — its worker hashes keys inside the stripe.

        GPU kernels do not use this: ``wants_parallel`` is always False on the
        GPU path, since the device handles its own parallelism, so a caller with
        a device branch takes it before reaching here.
        """
        if self.wants_parallel(length, min_parallel_size):
            var workers = self.resolved_num_threads()
            var chunk = ceildiv(ceildiv(length, workers), align) * align

            @always_inline
            def task(
                wid: Int,
            ) {imm chunk, imm length,}:
                var start = wid * chunk
                var end = min(start + chunk, length)
                # The last stripes are empty when `length < workers`.
                if start < end:
                    body(wid, start, end)

            sync_parallelize(task, workers)
        else:
            # Stripe 0 of 1 — matches `stripe_workers` returning 1 here, so a
            # body with per-worker scratch reads slot 0 and the caller only
            # allocated one.
            body(0, 0, length)

    # --- Writable ---------------------------------------------------------

    def write_to[W: Writer](self, mut writer: W):
        if self.is_gpu():
            writer.write("ExecutionContext(gpu)")
        else:
            writer.write("ExecutionContext(cpu)")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    # --- ConvertibleFromPython / ConvertibleToPython --------------------------

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[ExecutionContext]()[].copy()

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)
