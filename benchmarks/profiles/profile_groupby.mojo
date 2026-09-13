"""Single-shot profiling driver for the radix-partitioned parallel group-by.

Groups a 10M-row int32 column over 5M distinct keys in a loop, so a sampling
profiler can attribute time across the six phases `_consume_keys_radix` runs:
key hashing, the radix histogram, the scatter, the 64 per-partition
`SwissHashTable` inserts, the id write-back, and the closing `take` + builder
extend that materialises the new group keys.

    pixi run profile benchmarks/profiles/profile_groupby.mojo --sample --no-open

Like `profile_join.mojo` this avoids the benchmark harness entirely — no
warmup, no calibration, no pytest — so the trace holds only grouping work and
the AsyncRT pool it dispatches through.

A fresh `HashGrouping` per iteration is deliberate: `assign` accumulates, and
constructing the 64 per-partition tables is real per-query cost that the
benchmark also pays.

Overrides: `MARROW_PROFILE_N` (default 10_000_000), `MARROW_PROFILE_CARD`
(default 5_000_000), `MARROW_PROFILE_ITERS` (default 20) and
`MARROW_PROFILE_THREADS` (default 0 = auto). Set `MARROW_PROFILE_CARD` low —
say 1000 — to profile the serial path instead: the cardinality gate then keeps
radix off, which is the comparison that says which phases radix added.
"""

from std.benchmark import keep
from std.os.env import getenv

from marrow.arrays import DynArray
from marrow.builders import Int32Builder
from marrow.execution import ExecContext
from marrow.kernels.groupby import HashGrouping


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def _int_keys(n: Int, card: Int) raises -> List[DynArray]:
    """`n` int32 keys over `card` distinct values, interleaved rather than
    blocked — the same generator `bench_groupby.mojo` uses, so the profile and
    the benchmark describe the same work."""
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32((i * 7919) % card))
    var cols = List[DynArray]()
    cols.append(b.finish())
    return cols^


def main() raises:
    var n = _parse_int("MARROW_PROFILE_N", 10_000_000)
    var card = _parse_int("MARROW_PROFILE_CARD", 5_000_000)
    var iters = _parse_int("MARROW_PROFILE_ITERS", 20)
    var threads = _parse_int("MARROW_PROFILE_THREADS", 0)
    print(
        "profile_groupby: n =",
        n,
        " card =",
        card,
        " iters =",
        iters,
        " threads =",
        threads,
    )

    # Build the keys once, outside the sampled region.
    var cols = _int_keys(n, card)

    for _ in range(iters):
        var ctx = ExecContext(num_threads=threads)
        var g = HashGrouping(ctx^)
        var groups = g.assign(cols.copy(), n)
        keep(groups.num_groups)

    keep(cols)
    print("profile_groupby: done")
