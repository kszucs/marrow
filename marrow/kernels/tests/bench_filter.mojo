"""Benchmarks for filter kernel.

Covers different selectivities (10%, 50%, 90%) with and without nulls,
across sizes 10k–1M for int64 arrays.

Uses manual perf_counter_ns timing rather than the Bench framework, because
the Bench framework's tight loop rapidly allocates and frees the filter
result buffer each iteration, crashing libKGENCompilerRTShared (same issue
as GPU benchmarks in bench_arithmetic.mojo).

Run with: pixi run bench_mojo -k bench_filter
"""

from std.benchmark import keep
from std.time import perf_counter_ns

from marrow.arrays import PrimitiveArray, arange
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int64, bool_
from marrow.kernels.filter import filter_


def _make_mask(size: Int, selectivity_pct: Int) raises -> PrimitiveArray[bool_]:
    var b = PrimitiveBuilder[bool_](size)
    for i in range(size):
        b.append(Bool((i * 100) // size < selectivity_pct))
    return b.finish_typed()


def _make_array_with_nulls(size: Int) raises -> PrimitiveArray[int64]:
    var b = PrimitiveBuilder[int64](size)
    for i in range(size):
        if i % 10 == 0:
            b.append_null()
        else:
            b.append(Scalar[int64.native](i))
    return b.finish_typed()


def _bench_filter(
    name: String, size: Int, selectivity_pct: Int, with_nulls: Bool, iters: Int
) raises -> Float64:
    var arr: PrimitiveArray[int64]
    if with_nulls:
        arr = _make_array_with_nulls(size)
    else:
        arr = arange[int64](0, size)
    var mask = _make_mask(size, selectivity_pct)

    # Warmup
    for _ in range(3):
        var r = filter_[int64](arr, mask)
        keep(len(r))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = filter_[int64](arr, mask)
        keep(len(r))
    return Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0


def main() raises:
    comptime sizes = (10_000, 100_000, 1_000_000)
    comptime size_labels = ("10k", "100k", "1M")
    comptime iters_list = (500, 100, 20)

    print("bench_id                              us/call")
    print("--------                              -------")

    # --- 50% selectivity, no nulls ---
    comptime for si in range(3):
        var us = _bench_filter(
            "filter_50pct[int64]", sizes[si], 50, False, iters_list[si]
        )
        print(t"filter_50pct[int64]/{size_labels[si]}          {us} us")

    # --- 10% selectivity, no nulls ---
    comptime for si in range(3):
        var us = _bench_filter(
            "filter_10pct[int64]", sizes[si], 10, False, iters_list[si]
        )
        print(t"filter_10pct[int64]/{size_labels[si]}          {us} us")

    # --- 90% selectivity, no nulls ---
    comptime for si in range(3):
        var us = _bench_filter(
            "filter_90pct[int64]", sizes[si], 90, False, iters_list[si]
        )
        print(t"filter_90pct[int64]/{size_labels[si]}          {us} us")

    # --- 50% selectivity, with nulls ---
    comptime for si in range(3):
        var us = _bench_filter(
            "filter_50pct_nulls[int64]", sizes[si], 50, True, iters_list[si]
        )
        print(t"filter_50pct_nulls[int64]/{size_labels[si]}    {us} us")
