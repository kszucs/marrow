"""Single-shot profiling driver for the partition-parallel hash join.

Builds a 10M × 10M INNER join and runs ``hash_join()`` in a loop so
Instruments can record a dense sample of where time is spent.  Meant
for use via:

    pixi run profile marrow/kernels/tests/profile_join.mojo

The file deliberately avoids the benchmark harness — no warmup, no
calibration, no pytest overhead — so the resulting trace only contains
join work and the AsyncRT task-pool it dispatches through. Override the
per-side row count with ``MARROW_PROFILE_N`` (default 10_000_000) and
iteration count with ``MARROW_PROFILE_ITERS`` (default 8).
"""

from std.benchmark import keep
from std.os.env import getenv

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.builders import PrimitiveBuilder, Int64Builder
from marrow.dtypes import int64, Int64Type, struct_, Field
from marrow.expr.relations import JOIN_INNER, JOIN_ALL
from marrow.kernels.join import hash_join


def _make_struct(n: Int) raises -> StructArray:
    """Build a StructArray with columns (k: int64, v: int64)."""
    var kb = Int64Builder(capacity=n)
    var vb = Int64Builder(capacity=n)
    for i in range(n):
        kb.append(Scalar[int64.native](i))
        vb.append(Scalar[int64.native](i * 10))
    var cols = List[AnyArray]()
    cols.append(kb.finish().to_any())
    cols.append(vb.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", int64), Field("v", int64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        children=cols^,
    )


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def main() raises:
    var n = _parse_int("MARROW_PROFILE_N", 10_000_000)
    var iters = _parse_int("MARROW_PROFILE_ITERS", 8)
    print("profile_join: n =", n, " iters =", iters)

    # Build the data once outside the timed region.
    var left = _make_struct(n)
    var right = _make_struct(n)
    var left_on = List[Int]()
    left_on.append(0)
    var right_on = List[Int]()
    right_on.append(0)

    # Repeatedly run the join so Instruments has enough samples to
    # attribute time. ``num_threads=0`` → auto (num_physical_cores()).
    for _ in range(iters):
        var out = hash_join(
            left,
            right,
            left_on,
            right_on,
            JOIN_INNER,
            JOIN_ALL,
            num_threads=0,
        )
        keep(out.length)

    keep(left)
    keep(right)
    print("profile_join: done")
