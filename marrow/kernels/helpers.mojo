"""Shared helpers for compute kernels.

Provides:
  - `bitmap_and` — null bitmap propagation (bitwise AND of two validity bitmaps).
  - `Kernel` — base trait for all SIMD compute kernels.
"""

from marrow.buffers import Bitmap


# ---------------------------------------------------------------------------
# Null bitmap kernel
# ---------------------------------------------------------------------------


def bitmap_and(
    a: Optional[Bitmap[mut=False]], b: Optional[Bitmap[mut=False]]
) raises -> Optional[Bitmap[mut=False]]:
    """Compute the output validity bitmap as the bitwise AND of two input bitmaps.

    Output bit i is True iff both a[i] and b[i] are True (valid).
    None represents an all-valid bitmap.

    Args:
        a: First input bitmap (None = all valid).
        b: Second input bitmap (None = all valid).

    Returns:
        None if both are all-valid; otherwise the AND of the two bitmaps.
    """
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a
    return (a.value().view() & b.value().view()).to_immutable()


# ---------------------------------------------------------------------------
# Base kernel trait
# ---------------------------------------------------------------------------


trait Kernel:
    """Base trait for all SIMD compute kernels."""

    comptime name: String
