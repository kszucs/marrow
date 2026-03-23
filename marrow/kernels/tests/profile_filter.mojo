"""Profiling workload for the filter kernel.

Run via:  pixi run profile marrow/kernels/tests/profile_filter.mojo
"""

from std.benchmark import keep

from marrow.arrays import PrimitiveArray
from marrow.builders import arange, PrimitiveBuilder
from marrow.dtypes import int64, bool_
from marrow.kernels.filter import filter_


def _make_mask(size: Int, selectivity_pct: Int) raises -> PrimitiveArray[bool_]:
    var b = PrimitiveBuilder[bool_](size)
    for i in range(size):
        b.append(Bool((i * 100) // size < selectivity_pct))
    return b.finish_typed()


def main() raises:
    var n = 100_000_000
    var iterations = 1000

    var arr = arange[int64](0, n)
    var mask = _make_mask(n, 50)

    for _ in range(iterations):
        var r = filter_[int64](arr, mask)
        keep(len(r))
