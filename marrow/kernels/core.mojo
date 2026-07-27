"""The root of the kernel trait hierarchy.

Every kernel family in `marrow.kernels` — element-wise arithmetic, comparison,
boolean, string, temporal, aggregate — descends from `Kernel`, so a kernel is
nameable without knowing which family it belongs to. Family traits add the
call shape (`core`/`apply`/`dispatch`, `combine`/`finalize`, ...).
"""


trait Kernel:
    """Base trait for all compute kernels."""

    comptime name: String
    """This kernel's identity — for display and debugging, never dispatch."""
