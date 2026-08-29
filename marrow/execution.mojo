"""How work is dispatched: thread count, optional device, striping.

This is a *core* module, not a kernel one. It imports nothing from marrow — it
is a pure policy object — and it has three sets of consumers, only one of which
is `kernels/`: `views.apply` takes one to pick the CPU/GPU path, `tabular` and
`expr` thread one through execution, and the Python bindings expose it. Filing
it under `kernels/` was the tree's only `core -> kernels` import edge.

Bundles the two axes of parallelism a dispatch site needs to know about:

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

from max.algorithm.functional import sync_parallelize
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.sys.info import num_physical_cores
from std.sys import has_accelerator, CompilationTarget, get_defined_bool


# The single switch for GPU code generation across marrow.  **Off by default**:
# GPU work is opt-in, so build with `-D MARROW_GPU=true` to get it.  With it
# off, every device path is eliminated at elaboration time — device allocations
# in the kernels, the accelerator arms of `_apply_dispatch`, and
# `has_accelerator_support`, which answers False and so makes a GPU
# `ExecContext` raise at the dispatch site rather than misbehave.  Applies
# to `mojo build` / `mojo run`; `mojo precompile` rejects `-D` outright.
#
# This is marrow's largest single compile-time lever.  Cold builds (fresh
# `MODULAR_CACHE_DIR` — the transform cache makes a repeated identical compile
# useless as a measurement):
#
#                        GPU off (default)   GPU on
#   cast, numeric x numeric      14.6s        40.1s
#   cast + sort_indices          43.7s        85.0s
#
# **Both halves have to be gated to get any of it.** Device paths only vanish
# when the allocations are wrapped in `comptime if GPU_ENABLED` *and*
# `has_accelerator_support` answers False.  Gating either one alone measures as
# no change at all (45.2s and 84.4s respectively against 42.5s / 84.1s
# baselines), which is why this looked like a dead end for a long time.  So:
# anything that touches device code needs a `comptime if GPU_ENABLED` around
# it; a runtime `if ctx.is_gpu()` cannot be eliminated at elaboration time and
# silently keeps the whole device path alive.
#
# It does not shed the MAX runtime, though — a binary built with the flag off
# still links `libmax` / AsyncRT exactly as one built with it on.
comptime GPU_ENABLED = get_defined_bool["MARROW_GPU", False]()


struct ExecContext(
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
        ``ExecContext`` explicitly.
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

    # --- derivation ---------------------------------------------------

    def with_threads(self, num_threads: Int) -> Self:
        """This context with a different worker count and the **same device**.

        Use this — never ``parallel(n)`` or ``serial()`` — whenever an internal
        site needs to re-derive a context it was *given*. The factories build a
        fresh CPU context, so calling one to change a thread count silently sets
        ``device=None``: the work then runs on the CPU while the caller believes
        it asked for the GPU, and no CPU test can observe it.

        This has already happened three times. `HashJoin` destructured to a bare
        `_num_threads` and rebuilt at five internal sites; the eager group-by
        driver did the same and rebuilt at two; `Aggregation.whole` took
        `num_threads: Int` across its API boundary so the device was gone before
        it was ever called.
        """
        return Self(num_threads=num_threads, device=self.device.copy())

    # --- queries ------------------------------------------------------

    @staticmethod
    def has_accelerator_support[*dtypes: DType]() -> Bool:
        """Check if there is accelerator support for all given dtypes.

        For example Metal doesn't support float64 as of April 2026.

        Must use `comptime if`, not runtime `if`: these guards have to eliminate
        the accelerator branches at elaboration time, not at runtime.

        Note `has_accelerator()` is itself `is_gpu() or _accelerator_arch() != ""`,
        so on a machine reporting an accelerator this enables GPU codegen — which
        since 1.0.0b3.dev2026072406 requires a MAX runtime (`lib/libmax.dylib`),
        hence the `max` dependency pinned alongside `mojo` in `pixi.toml`.

        A previous `_accelerator_arch()` check here validated the GPU architecture
        string, working around a toolchain regression that reported a malformed
        target (e.g. 'metal:2-metal4' on an M2 with the Metal 4 API). It has been
        removed; reinstate it if that regression reappears.

        GPU codegen is **opt-in**: this answers False unless the build passes
        `-D MARROW_GPU=true` (see `GPU_ENABLED`). Without it a GPU
        `ExecContext` raises at the dispatch site rather than misbehaving.
        Applies to `mojo build`/`mojo run` only — `mojo precompile` rejects `-D`.

        Answering False here is one half of eliminating GPU codegen; the other half
        is the `comptime if GPU_ENABLED` guards around the kernels' device
        allocations. Either half alone measures as no improvement whatsoever — this
        call returning False while the allocations stay behind a runtime
        `if ctx.is_gpu()` was 45.2s against a 42.5s baseline. Together they take the
        same build to 14.6s. Do not "simplify" one side away.
        """
        comptime if not GPU_ENABLED:
            return False
        comptime if not has_accelerator():
            return False
        comptime if not CompilationTarget.is_apple_silicon():
            return True
        comptime for dtype in dtypes:
            if dtype == DType.float64:
                return False
        return True

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

    def worth_parallel(self, n: Int, min_parallel_size: Int) -> Bool:
        """Is a problem of size ``n`` big enough that a *parallel algorithm*
        pays for itself?

        The companion to ``wants_parallel``, and they differ on exactly one
        input: ``parallel(N)`` below the threshold. That is the difference
        between the two costs being weighed:

        - ``wants_parallel`` guards ``stripe``, where going parallel costs one
          dispatch. A forced count is an **instruction** — the caller asked for
          N workers, so a 1,000-row loop is split N ways. Tests rely on this to
          exercise the striped path on small inputs.
        - ``worth_parallel`` guards a *choice of algorithm*, where going
          parallel costs radix partitioning and N hash tables. A forced count is
          a **budget**, not a demand to use it: ``parallel(4)`` on 1,000 rows
          must still take the serial path, or the setup dwarfs the query.

        ``min_parallel_size`` is deliberately required rather than defaulted.
        There is no meaningful default — each of the three callers has measured
        its own crossover (60k rows for group-by, 100k for join, 200k for
        distinct) and they are not the same number, nor `stripe`'s 32768.

        Like ``wants_parallel`` this answers False on a GPU context, which the
        three hand-rolled copies of this test did not: they asked
        ``resolved_num_threads()``, which knows nothing about the device.
        """
        if self.is_gpu():
            return False
        if self.resolved_num_threads() <= 1:
            return False
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
        Body: def(Int, Int, Int) -> None
    ](
        self,
        length: Int,
        body: Body,
        min_parallel_size: Int = 32768,
        align: Int = 1,
    ):
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

        **Mark ``body`` ``@always_inline``.** Measured on `TakeKernel.apply`'s gather
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

        **``body`` may not raise.** `sync_parallelize`'s value form — the one
        used here — takes a non-raising worker. A caller whose body genuinely
        raises should park the first error and re-raise after the join, which is
        what `RadixPartitioner.map_partitions` and the Parquet row-group reader
        do. Do **not** reach for
        `sync_parallelize`'s parameter form instead: it accepts a raising worker
        but needs an implicitly-capturing closure whose captures are silently
        not made, and the body then reads garbage at run time. The tell is an
        "assignment was never used" warning on a buffer the body writes.

        ``body`` is a **unified closure passed by value**, so it carries an
        explicit capture list (``{imm}``, ``{mut scratch, imm}``, …) rather than
        capturing implicitly. Keep ``@always_inline`` on it — see above.

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
            ) {imm chunk, imm length, imm body,}:
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
            writer.write("ExecContext(gpu)")
        else:
            writer.write("ExecContext(cpu)")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    # --- ConvertibleFromPython / ConvertibleToPython --------------------------

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[ExecContext]()[].copy()

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)
