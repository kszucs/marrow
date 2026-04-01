"""Profiling workload for the hash join kernel.

Prints wall-clock times for build, probe, and total join phases.

Run via:
    pixi run profile marrow/kernels/tests/profile_join.mojo

Or build and run directly:
    mojo build -I . -O2 marrow/kernels/tests/profile_join.mojo -o /tmp/profile_join
    /tmp/profile_join
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int64, int32, struct_, Field
from marrow.kernels.join import HashJoin, hash_join
from marrow.kernels.hashing import rapidhash
from marrow.expr.relations import JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ALL


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------


def _make_struct(n: Int, key_offset: Int = 0) raises -> StructArray:
    """Build a StructArray with columns (k: int64, v: int64)."""
    var kb = PrimitiveBuilder[Int64Type](capacity=n)
    var vb = PrimitiveBuilder[Int64Type](capacity=n)
    for i in range(n):
        kb.append(Scalar[int64.native](i + key_offset))
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


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _fmt_us(ns: UInt) -> String:
    return String(Int(ns // 1_000)) + " µs"


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_build(
    left: StructArray, key_indices: List[Int], warmup: Int, iters: Int
) raises:
    """Time the build phase only."""
    # Warmup
    for _ in range(warmup):
        var j = HashJoin()
        j.build(left, key_indices)
        keep(j.num_left_rows())

    var total_ns = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var j = HashJoin()
        j.build(left, key_indices)
        total_ns += perf_counter_ns() - t0
        keep(j.num_left_rows())

    var avg = total_ns // UInt(iters)
    print("  build:  ", _fmt_us(avg), " avg over", iters, "iters")


def bench_probe(
    left: StructArray,
    right: StructArray,
    left_keys: List[Int],
    right_keys: List[Int],
    warmup: Int,
    iters: Int,
) raises:
    """Time build + probe separately, with phase breakdown."""
    from marrow.kernels.join import IndexPairs  # type alias
    from marrow.kernels.filter import take

    var j = HashJoin()
    j.build(left, left_keys)

    # Warmup
    for _ in range(warmup):
        var r = j.probe(right, right_keys, JOIN_INNER, JOIN_ALL)
        keep(len(r))

    # Total probe (includes assemble)
    var total_ns = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var r = j.probe(right, right_keys, JOIN_INNER, JOIN_ALL)
        total_ns += perf_counter_ns() - t0
        keep(len(r))
    var avg = total_ns // UInt(iters)
    print("  probe:  ", _fmt_us(avg), " avg over", iters, "iters")

    # --- Phase breakdown (single run) ---

    # Phase A+B: probe (hash + lookup + equality verify)
    var lk = left.select(left_keys)
    var rk = right.select(right_keys)
    var n = len(rk)
    var t0 = perf_counter_ns()
    var pairs = j._table.probe(lk, rk, len(left))
    var t_pairs = perf_counter_ns() - t0
    print("    probe:    ", _fmt_us(t_pairs))

    # Phase C: assemble (take)
    t0 = perf_counter_ns()
    var result = j._assemble(right, pairs, JOIN_INNER)
    var t_asm = perf_counter_ns() - t0
    print("    assemble: ", _fmt_us(t_asm))


def bench_full(
    left: StructArray,
    right: StructArray,
    left_keys: List[Int],
    right_keys: List[Int],
    warmup: Int,
    iters: Int,
) raises:
    """Time full hash_join (build + probe + assemble)."""
    # Warmup
    for _ in range(warmup):
        var r = hash_join(
            left, right, left_keys, right_keys, JOIN_INNER, JOIN_ALL
        )
        keep(len(r))

    var total_ns = UInt(0)
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var r = hash_join(
            left, right, left_keys, right_keys, JOIN_INNER, JOIN_ALL
        )
        total_ns += perf_counter_ns() - t0
        keep(len(r))

    var avg = total_ns // UInt(iters)
    print("  total:  ", _fmt_us(avg), " avg over", iters, "iters")


def run_size(n: Int, warmup: Int, iters: Int) raises:
    print("\n=== INNER JOIN", n, "x", n, "rows (1:1 unique keys) ===")
    var left = _make_struct(n)
    var right = _make_struct(n)
    var keys = List[Int]()
    keys.append(0)

    bench_build(left, keys, warmup, iters)
    bench_probe(left, right, keys, keys, warmup, iters)
    bench_full(left, right, keys, keys, warmup, iters)


def main() raises:
    print("Marrow hash join profiling")
    print("==========================")

    run_size(10_000, warmup=10, iters=100)
    run_size(100_000, warmup=5, iters=50)
    run_size(1_000_000, warmup=3, iters=20)
