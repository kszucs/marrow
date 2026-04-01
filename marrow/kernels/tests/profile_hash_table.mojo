"""Profiling workload for SwissHashTable.

Runs a single large workload (no iteration averaging) so that
Instruments can capture a clean flame graph.

Prints wall-clock times for each phase: insert, build (insert + CSR),
and probe.

Profile with Instruments (builds with debug info, records, opens trace)::

    pixi run profile marrow/kernels/tests/profile_hash_table.mojo

Run without profiler (prints timings only)::

    pixi run mojo run -I . marrow/kernels/tests/profile_hash_table.mojo
    pixi run mojo run -I . marrow/kernels/tests/profile_hash_table.mojo 10000000
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.buffers import Bitmap
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import uint64, struct_, Field
from marrow.kernels.hashtable import SwissHashTable
from marrow.kernels.hashing import rapidhash
from std.sys import argv


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------


def _make_keys(n: Int) raises -> StructArray:
    """Generate a single-column StructArray with n distinct uint64 keys."""
    var b = PrimitiveBuilder[UInt64Type](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i * 0x9E3779B97F4A7C15 + 1))
    var children = List[AnyArray]()
    children.append(b.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", uint64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=Optional[Bitmap[]](None),
        children=children^,
    )


def _fmt_us(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


def _fmt_ms(ns: UInt) -> String:
    return String(Int(ns // 1_000_000)) + " ms"


def _fmt(ns: UInt) -> String:
    if ns >= 1_000_000:
        return _fmt_ms(ns)
    return _fmt_us(ns)


def _mops(n: Int, ns: UInt) -> String:
    return String(Int(UInt(n) * 1_000 // (ns + 1))) + " Mops/s"


# ---------------------------------------------------------------------------
# Profiling phases
# ---------------------------------------------------------------------------


def profile(n: Int) raises:
    print("SwissHashTable profile: n =", n)
    print("=" * 50)

    # --- Data generation ---
    var t0 = perf_counter_ns()
    var keys = _make_keys(n)
    var t_gen = perf_counter_ns() - t0
    print("  data gen:   ", _fmt(t_gen))

    # --- Phase 1: insert only (no CSR) ---
    t0 = perf_counter_ns()
    var table_insert = SwissHashTable[rapidhash]()
    var bids = table_insert.insert(keys)
    var t_insert = perf_counter_ns() - t0
    keep(table_insert.num_keys())
    print(
        "  insert:     ",
        _fmt(t_insert),
        " (",
        _mops(n, t_insert),
        ")",
    )

    # --- Phase 2: build (insert + CSR construction) ---
    t0 = perf_counter_ns()
    var table = SwissHashTable[rapidhash]()
    table.build(keys)
    var t_build = perf_counter_ns() - t0
    keep(table.num_keys())
    print(
        "  build:      ",
        _fmt(t_build),
        " (",
        _mops(n, t_build),
        ")",
    )

    # --- Phase 3: probe (all keys match, 1:1) ---
    t0 = perf_counter_ns()
    var pairs = table.probe(keys, keys, n)
    var t_probe = perf_counter_ns() - t0
    keep(len(pairs[0]))
    print(
        "  probe:      ",
        _fmt(t_probe),
        " (",
        _mops(n, t_probe),
        ")",
    )

    # --- Phase 4: probe with single_match (semi-join) ---
    t0 = perf_counter_ns()
    var pairs_single = table.probe(keys, keys, n, single_match=True)
    var t_probe_single = perf_counter_ns() - t0
    keep(len(pairs_single[0]))
    print(
        "  probe semi: ",
        _fmt(t_probe_single),
        " (",
        _mops(n, t_probe_single),
        ")",
    )

    # --- Summary ---
    var t_total = t_build + t_probe
    print()
    print("  total (build+probe): ", _fmt(t_total))
    print("  num_keys:           ", table.num_keys())
    print("  matches:            ", len(pairs[0]))
    print("  matches (semi):     ", len(pairs_single[0]))


def main() raises:
    var n = 1_000_000
    var iters = 1000

    # Usage: profile_hash_table [n] [iters]
    if len(argv()) > 1:
        n = Int(argv()[1])
    if len(argv()) > 2:
        iters = Int(argv()[2])

    for i in range(iters):
        if iters > 1:
            print("\n--- iteration", i + 1, "of", iters, "---")
        profile(n)
