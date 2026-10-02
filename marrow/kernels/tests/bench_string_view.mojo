# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`string` against `string_view`, one operation at a time.

Each operation runs over the same values in both layouts, tagged with
`extra_info("lib", ...)`, so `--competition` prints them side by side:

    pixi run -e dev pytest marrow/kernels/tests/bench_string_view.mojo \\
        --benchmark --competition

`urls` are 24-40 bytes, so every view points into a data buffer; `codes` are
at most 12 bytes, so every view holds its value inline. The two casts and the
adoption of a foreign layout are measured on their own: they have no
counterpart in the other layout.
"""

from std.benchmark import BenchMetric, keep

from ...arrays import DynArray, Int32Array, StringArray, StringViewArray
from ...builders import Int32Builder, BoolBuilder
from ...dtypes import string, string_view
from ...kernels.cast import cast
from ...kernels.filter import filter, take
from ...kernels.hashing import RapidHashKernel
from ...kernels.sort import SortIndices
from ...kernels.string import LikeKernel, StringEqKernel
from ...utils.testing import Benchmark
from .string_fixtures import broadcast, codes, urls

comptime N = 1_000_000


def _layout(var data: StringArray, view: Bool) raises -> DynArray:
    """`data` in the layout under test."""
    if view:
        return cast(data^.to_dyn(), string_view.to_dyn())
    return data^.to_dyn()


def _lib(view: Bool) -> String:
    return "string_view" if view else "string"


def _every_other(n: Int) raises -> DynArray:
    var b = BoolBuilder(n)
    for i in range(n):
        b.append(i % 2 == 0)
    return b.finish().to_dyn()


def _scattered(n: Int) raises -> Int32Array:
    var b = Int32Builder(n)
    for i in range(n):
        b.append(Int32((i * 7919) % n))
    return b.finish()


# --- filter / take: the view layout gathers 16 bytes, never the value ------


def _bench_filter(mut b: Benchmark, var data: StringArray, view: Bool) raises:
    var arr = _layout(data^, view)
    var mask = _every_other(N)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(filter(arr, mask).length())

    b.iter(call)
    keep(arr)
    keep(mask)


def bench_string_filter_urls(mut b: Benchmark) raises:
    _bench_filter(b, urls(N), False)


def bench_string_view_filter_urls(mut b: Benchmark) raises:
    _bench_filter(b, urls(N), True)


def bench_string_filter_codes(mut b: Benchmark) raises:
    _bench_filter(b, codes(N), False)


def bench_string_view_filter_codes(mut b: Benchmark) raises:
    _bench_filter(b, codes(N), True)


def _bench_take(mut b: Benchmark, view: Bool) raises:
    var arr = _layout(urls(N), view)
    var idx = _scattered(N)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(take(arr, idx).length())

    b.iter(call)
    keep(arr)
    keep(idx)


def bench_string_take(mut b: Benchmark) raises:
    _bench_take(b, False)


def bench_string_view_take(mut b: Benchmark) raises:
    _bench_take(b, True)


# --- element readers: one `BytesArray` loop, two ways to find the bytes ----


def _bench_hash(mut b: Benchmark, view: Bool) raises:
    var arr = _layout(urls(N), view)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(len(RapidHashKernel.dispatch(arr)))

    b.iter(call)
    keep(arr)


def bench_string_hash(mut b: Benchmark) raises:
    _bench_hash(b, False)


def bench_string_view_hash(mut b: Benchmark) raises:
    _bench_hash(b, True)


def _bench_sort(mut b: Benchmark, var data: StringArray, view: Bool) raises:
    var n = len(data)
    var arr = _layout(data^, view)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(len(SortIndices.dispatch(arr)))

    b.iter(call)
    keep(arr)


def bench_string_sort_urls(mut b: Benchmark) raises:
    """Every URL shares its first eight bytes, so the sort key decides
    nothing here -- the worst case for it."""
    _bench_sort(b, urls(100_000), False)


def bench_string_view_sort_urls(mut b: Benchmark) raises:
    _bench_sort(b, urls(100_000), True)


def bench_string_sort_codes(mut b: Benchmark) raises:
    _bench_sort(b, codes(100_000), False)


def bench_string_view_sort_codes(mut b: Benchmark) raises:
    _bench_sort(b, codes(100_000), True)


def _bench_eq(mut b: Benchmark, var data: StringArray, view: Bool) raises:
    """Equality against a broadcast literal -- the runtime lane's shape for
    `col == 'x'`, with the literal already on the column's layout."""
    var arr = _layout(data^, view)
    var lit = _layout(broadcast("c4242", N), view)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(StringEqKernel.dispatch(arr, lit).length())

    b.iter(call)
    keep(arr)
    keep(lit)


def bench_string_eq_codes(mut b: Benchmark) raises:
    _bench_eq(b, codes(N), False)


def bench_string_view_eq_codes(mut b: Benchmark) raises:
    _bench_eq(b, codes(N), True)


def _bench_like(mut b: Benchmark, view: Bool) raises:
    var arr = _layout(urls(N), view)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", _lib(view))

    @always_inline
    def call() raises {imm}:
        keep(LikeKernel.dispatch(arr, "%google%").length())

    b.iter(call)
    keep(arr)


def bench_string_like(mut b: Benchmark) raises:
    _bench_like(b, False)


def bench_string_view_like(mut b: Benchmark) raises:
    _bench_like(b, True)


# --- conversions and adoption, which have no counterpart -------------------


def bench_cast_string_to_view(mut b: Benchmark) raises:
    """Zero-copy: one view per row into the existing values buffer."""
    var arr = urls(N).to_dyn()
    b.throughput(BenchMetric.elements, N)

    @always_inline
    def call() raises {imm}:
        keep(cast(arr, string_view.to_dyn()).length())

    b.iter(call)
    keep(arr)


def bench_cast_view_to_string(mut b: Benchmark) raises:
    """A copy: the bytes have to become contiguous."""
    var arr = _layout(urls(N), True)
    b.throughput(BenchMetric.elements, N)

    @always_inline
    def call() raises {imm}:
        keep(cast(arr, string.to_dyn()).length())

    b.iter(call)
    keep(arr)


def bench_adopt_view_layout(mut b: Benchmark) raises:
    """`DynArray.from_data` over a view layout -- what every C Data and IPC
    import pays: `validate` and the null-view scan, no copy for a
    well-formed producer."""
    var data = _layout(urls(N), True).to_data()
    b.throughput(BenchMetric.elements, N)

    @always_inline
    def call() raises {imm}:
        keep(DynArray.from_data(data).length())

    b.iter(call)
    keep(data)
