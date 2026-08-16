"""Benchmarks for the hash primitives themselves.

`kernels/tests/bench_hashing.mojo` measures hashing an *array* — dtype dispatch,
validity, buffer allocation and the SIMD driver all included. These measure the
primitives on their own, so a change to the mixing steps shows up here without
the array machinery masking it.

`XxHash64.hash` is reported in bytes/second: it walks a span, and its cost is
per byte with a 32-byte-block fast path. `RapidHash64` is reported in
elements/second: it hashes one fixed-width value (or one SIMD lane) at a time
and never sees a length.

Run with: pixi run -e dev pytest marrow/utils/tests/bench_hashing.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ..hashing import RapidHash64, XxHash64
from ..testing import Benchmark


def _bytes(n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(UInt8((i * 31 + 7) & 0xFF))
    return out^


# ---------------------------------------------------------------------------
# XxHash64.hash — byte-span throughput
# ---------------------------------------------------------------------------


def _bench_xxhash64(mut b: Benchmark, size: Int) raises:
    var data = _bytes(size)
    b.throughput(BenchMetric.bytes, size)

    @always_inline
    def call() {imm}:
        keep(XxHash64.hash(Span(data)))

    b.iter(call)
    keep(len(data))


def bench_xxhash64_8b(mut b: Benchmark) raises:
    """Below the 32-byte block: seed + P5, one 8-byte round, no block path."""
    _bench_xxhash64(b, 8)


def bench_xxhash64_64b(mut b: Benchmark) raises:
    """Two 32-byte blocks — the four-accumulator loop with no tail."""
    _bench_xxhash64(b, 64)


def bench_xxhash64_1k(mut b: Benchmark) raises:
    _bench_xxhash64(b, 1_024)


def bench_xxhash64_64k(mut b: Benchmark) raises:
    """Past L1, so the block loop is memory-bound rather than issue-bound."""
    _bench_xxhash64(b, 65_536)


# ---------------------------------------------------------------------------
# RapidHash64 — scalar mixing
# ---------------------------------------------------------------------------


def _bench_rapidhash64_fixed[byte_width: Int](mut b: Benchmark, n: Int) raises:
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() {imm}:
        var acc = UInt64(0)
        for i in range(n):
            acc ^= RapidHash64.hash_fixed[byte_width](UInt64(i))
        keep(acc)

    b.iter(call)


def bench_rapidhash64_fixed4_10k(mut b: Benchmark) raises:
    """The int32-shaped value hash — `byte_width` is folded into the seed."""
    _bench_rapidhash64_fixed[4](b, 10_000)


def bench_rapidhash64_fixed8_10k(mut b: Benchmark) raises:
    _bench_rapidhash64_fixed[8](b, 10_000)


def bench_rapidhash64_mix_10k(mut b: Benchmark) raises:
    """`mix` alone: one 128-bit multiply plus an XOR of the halves."""
    var n = 10_000
    b.throughput(BenchMetric.elements, n)

    @always_inline
    def call() {imm}:
        var acc = UInt64(0)
        for i in range(n):
            acc = RapidHash64.mix(acc ^ UInt64(i), RapidHash64.SECRET1)
        keep(acc)

    b.iter(call)


# ---------------------------------------------------------------------------
# RapidHash64 — per-lane mixing
#
# `mix_wide` is the GPU-compatible path: no `uint128`, so it builds the 128-bit
# product from four 32-bit sub-products. It is what the array kernel actually
# runs, and the width sweep shows whether that reconstruction vectorizes.
# ---------------------------------------------------------------------------


def _bench_rapidhash64_mix_wide[W: Int](mut b: Benchmark, n: Int) raises:
    b.throughput(BenchMetric.elements, n * W)
    var secret = SIMD[DType.uint64, W](RapidHash64.SECRET1)

    @always_inline
    def call() {imm}:
        var acc = SIMD[DType.uint64, W](0)
        for i in range(n):
            acc = RapidHash64.mix_wide[W](
                acc ^ SIMD[DType.uint64, W](UInt64(i)), secret
            )
        keep(acc)

    b.iter(call)
    keep(secret)


def bench_rapidhash64_mix_wide_w1_10k(mut b: Benchmark) raises:
    _bench_rapidhash64_mix_wide[1](b, 10_000)


def bench_rapidhash64_mix_wide_w4_10k(mut b: Benchmark) raises:
    _bench_rapidhash64_mix_wide[4](b, 10_000)


def bench_rapidhash64_mix_wide_w8_10k(mut b: Benchmark) raises:
    _bench_rapidhash64_mix_wide[8](b, 10_000)
