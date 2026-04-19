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

from std.gpu.host import DeviceContext
from std.sys.info import num_physical_cores


struct ExecutionContext(Copyable, Movable):
    """How a kernel should dispatch its work.

    See the module docstring for the full contract. Construct via one of
    the factory methods (``.serial()``, ``.parallel()``, ``.gpu()``) or
    via the default constructor (= serial, no GPU), or pass an
    ``Optional[DeviceContext]`` directly — it implicitly converts to a
    CPU-serial context with the given device.
    """

    var num_threads: Int
    """Worker count for CPU striped parallelism. ``1`` = serial;
    ``>1`` = ``sync_parallelize`` with that many workers; ``0`` is
    treated as "auto" and resolved to ``num_physical_cores()`` in
    ``resolved_num_threads()``."""

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
        """Single-threaded CPU execution."""
        return Self(num_threads=1, device=None)

    @staticmethod
    def parallel(num_threads: Int = 0) -> Self:
        """CPU execution with ``num_threads`` workers (0 = auto)."""
        return Self(num_threads=num_threads, device=None)

    @staticmethod
    def gpu(device: DeviceContext) -> Self:
        """GPU execution on the given device."""
        return Self(num_threads=1, device=Optional[DeviceContext](device))

    # --- queries ------------------------------------------------------

    def is_gpu(self) -> Bool:
        """True when work should be dispatched to the GPU."""
        return Bool(self.device)

    def resolved_num_threads(self) -> Int:
        """Normalize ``num_threads``: ``0`` → ``num_physical_cores()``,
        else the value itself (lower-bounded at 1)."""
        if self.num_threads <= 0:
            return num_physical_cores()
        return self.num_threads

    def wants_parallel(self, n: Int, min_parallel_size: Int = 32768) -> Bool:
        """Decide whether a CPU kernel of size ``n`` should stripe work.

        Returns ``False`` on the GPU path (GPU handles its own
        parallelism) and when ``n`` is below the kernel's grain-size
        threshold — below that, ``sync_parallelize`` dispatch overhead
        dominates the actual compute.
        """
        if self.is_gpu():
            return False
        return self.resolved_num_threads() > 1 and n >= min_parallel_size
