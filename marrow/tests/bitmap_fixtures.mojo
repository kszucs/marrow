"""The bitmaps the bitmap benchmarks measure over.

Shared rather than repeated because those benchmarks are spread across four
files, and a file is one `-O3` compilation unit -- see `bench_bitmap.mojo` for
why they are spread.  Every one of them builds the same two patterns.

Not named `bench_*`, so neither the harness nor the benchmark workflow's
`find` picks it up as a selection of its own.
"""

from ..buffers import Bitmap


def make_alternating(size: Int) -> Bitmap[mut=False]:
    """Bitmap with alternating 0/1 bits (worst-case for popcount branching)."""
    var b = Bitmap.alloc_zeroed(size)
    var i = 0
    while i < size:
        b.set(i)
        i += 2
    return b.to_immutable()


def make_half_set(size: Int) -> Bitmap[mut=False]:
    """Bitmap with the first half of bits set."""
    var b = Bitmap.alloc_zeroed(size)
    b.set_range(0, size // 2, True)
    return b.to_immutable()
