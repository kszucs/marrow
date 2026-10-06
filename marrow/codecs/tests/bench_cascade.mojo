# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The runtime `DynCascade` against the fused chain, on `Delta -> Zigzag ->
BitPack` over 1M int64 microsecond timestamps.

- `runtime`: `DynCascade.encode` / `DynCascade.decode`, each codec dispatched
  through `DynCodec` and run over the whole column in turn.
- `fused`: `Cascade[Delta, Zigzag, BitPack]`, the elementwise codecs fused into one loop; the
  bytes are checked identical to `runtime`'s.
- `plain`: a chain of no codecs, an untouched baseline.

Run with:
    pixi run -e dev pytest marrow/codecs/tests/bench_cascade.mojo --benchmark --competition
"""

from std.benchmark import BenchMetric, keep

from ...codecs import BitPack, DynCascade, Delta, DynCodec, Cascade, Zigzag
from ...utils.testing import Benchmark

comptime N = 1_000_000


def _timestamps() -> List[Int64]:
    var out = List[Int64](capacity=N)
    var t = Int64(1_700_000_000_000_000)
    for i in range(N):
        t += Int64(1_000 + (i * 7919) % 500)
        out.append(t)
    return out^


def _bench[lib: StaticString, decode: Bool](mut b: Benchmark) raises:
    comptime F = Cascade[Delta, Zigzag, BitPack]
    var values = _timestamps()
    var chain = DynCascade(Delta(), Zigzag(), BitPack())
    comptime if lib == "plain":
        chain = DynCascade(List[DynCodec]())
    var data = chain.encode[DType.int64](values)
    comptime if lib == "fused":
        if F.encode[DType.int64](values) != data:
            raise Error("bench: fused bytes differ from the runtime cascade's")
    if chain.decode[DType.int64](data) != values:
        raise Error("bench: the stream does not round-trip")
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", String(lib))

    @always_inline
    def call() raises {imm}:
        comptime if decode:
            comptime if lib == "fused":
                keep(len(F.decode[DType.int64](data)))
            else:
                keep(len(chain.decode[DType.int64](data)))
        else:
            comptime if lib == "fused":
                keep(len(F.encode[DType.int64](values)))
            else:
                keep(len(chain.encode[DType.int64](values)))

    b.iter(call)
    keep(values)
    keep(data)
    keep(chain)


def bench_runtime_encode(mut b: Benchmark) raises:
    _bench["runtime", False](b)


def bench_runtime_decode(mut b: Benchmark) raises:
    _bench["runtime", True](b)


def bench_fused_encode(mut b: Benchmark) raises:
    _bench["fused", False](b)


def bench_fused_decode(mut b: Benchmark) raises:
    _bench["fused", True](b)


def bench_plain_encode(mut b: Benchmark) raises:
    _bench["plain", False](b)


def bench_plain_decode(mut b: Benchmark) raises:
    _bench["plain", True](b)
