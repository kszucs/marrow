"""Profiling driver for the sort kernel.

Calls the production sort_indices() and take() kernels directly so that the
macOS `sample` profiler (or Instruments Time Profiler) can attribute hot
samples to the real call tree:

    main → _bench_sort_indices → sort_indices → _sort_indices_primitive → _radix_sort_indices
    main → _bench_sort    → sort_indices + take → _radix_sort_indices + _gather

Run:
    pixi run profile --sample marrow/kernels/tests/profile_sort.mojo

Override defaults via env vars:
  MARROW_PROFILE_N      (default 10_000_000)
  MARROW_PROFILE_ITERS  (default 50)
  MARROW_PROFILE_TYPE   int64 | float64  (default int64)
"""

from std.benchmark import keep
from std.os.env import getenv
from std.time import perf_counter_ns

from marrow.arrays import AnyArray, Int64Array, Float64Array, PrimitiveArray
from marrow.builders import Int64Builder, Float64Builder
from marrow.dtypes import int64, float64, Int64Type, Float64Type, PrimitiveType
from marrow.kernels.sort import sort_indices
from marrow.kernels.filter import take


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _random_int64(n: Int) raises -> Int64Array:
    var b = Int64Builder(capacity=n)
    var s: UInt64 = 0xFEDCBA9876543210
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        b.append(s.cast[int64.native]())
    return b.finish()


def _random_float64(n: Int) raises -> Float64Array:
    var b = Float64Builder(capacity=n)
    var s: UInt64 = 0xABCDEF0123456789
    for _ in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        var f = (s >> 11).cast[float64.native]() * (1.0 / Float64(1 << 53))
        b.append(f)
    return b.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def _parse_str(name: String, default: String) -> String:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    return s


def _ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


def _throughput(n: Int, avg_ns: Int) -> Float64:
    """Rows per second in millions."""
    return Float64(n) / (Float64(avg_ns) / 1_000_000_000.0) / 1_000_000.0


# ---------------------------------------------------------------------------
# Benchmark functions — one per kernel path so sample shows clean frames
# ---------------------------------------------------------------------------


def _bench_sort_indices[
    T: PrimitiveType
](data: PrimitiveArray[T], iters: Int) raises:
    var n = len(data)
    var arr: AnyArray = data.copy()

    # warmup (not counted)
    keep(sort_indices(arr))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var idx = sort_indices(arr)
        keep(idx)
    var elapsed = perf_counter_ns() - t0

    var avg = elapsed // iters
    print(
        "  sort_indices   ",
        _ns_to_ms(avg),
        "ms avg",
        " |",
        _throughput(n, avg),
        "M rows/s",
        " | total:",
        _ns_to_ms(elapsed),
        "ms",
    )
    keep(arr)


def _bench_sort[T: PrimitiveType](data: PrimitiveArray[T], iters: Int) raises:
    """sort_indices + take — the full sort pipeline."""
    var n = len(data)
    var arr: AnyArray = data.copy()

    # warmup
    var wi = sort_indices(arr)
    keep(take(arr, wi))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var idx = sort_indices(arr)
        var sorted = take(arr, idx)
        keep(idx)
        keep(sorted)
    var elapsed = perf_counter_ns() - t0

    var avg = elapsed // iters
    print(
        "  sort      ",
        _ns_to_ms(avg),
        "ms avg",
        " |",
        _throughput(n, avg),
        "M rows/s",
        " | total:",
        _ns_to_ms(elapsed),
        "ms",
    )
    keep(arr)


def _bench_sort_indices_desc[
    T: PrimitiveType
](data: PrimitiveArray[T], iters: Int) raises:
    var n = len(data)
    var arr: AnyArray = data.copy()

    keep(sort_indices(arr, ascending=False))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var idx = sort_indices(arr, ascending=False)
        keep(idx)
    var elapsed = perf_counter_ns() - t0

    var avg = elapsed // iters
    print(
        "  sort_indices↓  ",
        _ns_to_ms(avg),
        "ms avg",
        " |",
        _throughput(n, avg),
        "M rows/s",
    )
    keep(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var n = _parse_int("MARROW_PROFILE_N", 10_000_000)
    var iters = _parse_int("MARROW_PROFILE_ITERS", 50)
    var dtype = _parse_str("MARROW_PROFILE_TYPE", "int64")

    print("profile_sort: n =", n, " iters =", iters, " type =", dtype)
    print("-----------------------------------------------------------")

    if dtype == "float64":
        var data = _random_float64(n)
        _bench_sort_indices[Float64Type](data, iters)
        _bench_sort_indices_desc[Float64Type](data, iters)
        _bench_sort[Float64Type](data, iters)
        keep(data)
    else:
        var data = _random_int64(n)
        _bench_sort_indices[Int64Type](data, iters)
        _bench_sort_indices_desc[Int64Type](data, iters)
        _bench_sort[Int64Type](data, iters)
        keep(data)

    print("-----------------------------------------------------------")
    print("done")
