"""Nested (list/struct) compute kernels.

Markers NOT IMPLEMENTED yet (compute `core`/`apply` are TODO) — they exist so the
typed expression layer (`marrow.expr.ibis`) can name list operations; execution
wires up later.
"""

from .helpers import Kernel


struct ArrayLengthKernel(Kernel):
    comptime name = "array_length"


struct ArrayContainsKernel(Kernel):
    comptime name = "array_contains"
