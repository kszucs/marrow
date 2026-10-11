# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Each codec alone, then the cascades, over 64K values: encode, and decode
what was encoded. The inputs are shaped for each codec.

Run with:
    pixi run -e dev pytest marrow/codecs/tests/bench_codecs.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...codecs import (
    Alp,
    BitPack,
    Buckets,
    ByteStreamSplit,
    Cascade,
    Codec,
    Compressed,
    Delta,
    Dictionary,
    Frequency,
    Gcd,
    Gorilla,
    Lengths,
    Ordered,
    Patch,
    Plain,
    Prefixes,
    FrameOfReference,
    Runs,
    Xor,
    Zigzag,
)
from ...compress import Huffman, Lz4, Snappy, Zstd
from ...utils.testing import Benchmark
from ...views import BufferView

comptime N = 1 << 16


def _data[T: DType, shape: StaticString]() -> List[Scalar[T]]:
    var out = List[Scalar[T]](capacity=N)
    var t = Int64(1_700_000_000)
    for i in range(N):
        comptime if shape == "timestamps":
            t += Int64(1_000 + (i * 7919) % 500)
            out.append(Scalar[T](t))
        elif shape == "small":
            out.append(Scalar[T]((i * 7919) % 100))
        elif shape == "runs":
            out.append(Scalar[T]((i // 50) % 7))
        elif shape == "frequent":
            out.append(Scalar[T](3 if i % 97 != 0 else i % 11))
        elif shape == "cardinality":
            out.append(Scalar[T]((i * 7919) % 1000) * 1_003)
        elif shape == "constant":
            out.append(Scalar[T](42))
        elif shape == "bytes":
            out.append(Scalar[T]((i // 3) % 17))
        elif shape == "interval":
            # Timestamps every second, a jitter of a few milliseconds.
            t += Int64(1_000 + (i * 7919) % 7 - 3)
            out.append(Scalar[T](t))
        elif shape == "seconds":
            # Milliseconds, in whole seconds.
            t += Int64(1_000 * (1 + i % 3))
            out.append(Scalar[T](t))
        elif shape == "outliers":
            out.append(Scalar[T]((i * 7919) % 100 if i % 100 != 0 else 1 << 20))
        elif shape == "signed":
            out.append(Scalar[T]((i * 7919) % 201 - 100))
        elif shape == "sparse":
            out.append(Scalar[T](0 if i % 1000 < 990 else (i * 7919) % 100_000))
        elif shape == "gauge":
            # A reading held for ten samples at a time.
            out.append(
                Scalar[T](
                    Float64(20 + (i // 100) % 10)
                    + Float64((i // 10) % 7) * 0.125
                )
            )
        else:
            out.append(Scalar[T](Float64(i % 1000) * 0.25 + 20.0))
    return out^


def _bench[
    T: DType, shape: StaticString, decode: Bool, *Ts: Codec
](mut b: Benchmark, var cascade: Cascade[*Ts]) raises:
    var values = _data[T, shape]()
    var encoded = cascade.encode_view(BufferView(Span(values)))
    var out = List[Scalar[T]](length=N, fill=0)
    cascade.decode_into[T](encoded, BufferView(Span(out)))
    for i in range(N):
        if out[i] != values[i]:
            raise Error("bench: the cascade does not round-trip")
    b.throughput(BenchMetric.elements, N)

    @always_inline
    def call() raises {mut cascade, mut out, imm}:
        comptime if decode:
            cascade.decode_into[T](encoded, BufferView(Span(out)))
            keep(out[N - 1])
        else:
            keep(cascade.encode_view(BufferView(Span(values))).count)

    b.iter(call)
    keep(values)
    keep(out)
    keep(encoded.count)


def _strings(mut offsets: List[Int32], mut data: List[UInt8]):
    offsets.append(0)
    for i in range(N):
        data.extend(String("row-", i // 10, "-", i * 7919 % 1000).as_bytes())
        offsets.append(Int32(len(data)))


def _bench_bytes[
    decode: Bool, *Ts: Codec
](mut b: Benchmark, var cascade: Cascade[*Ts]) raises:
    var offsets = List[Int32](capacity=N + 1)
    var data = List[UInt8]()
    _strings(offsets, data)
    var encoded = cascade.encode(Span(offsets), Span(data))
    b.throughput(BenchMetric.elements, N)

    @always_inline
    def call() raises {mut cascade, imm}:
        comptime if decode:
            var ends: List[Int32] = [0]
            var bytes = List[UInt8]()
            cascade.decode(encoded, ends, bytes)
            keep(len(bytes))
        else:
            keep(cascade.encode(Span(offsets), Span(data)).count)

    b.iter(call)
    keep(offsets)
    keep(data)
    keep(encoded.count)


# --- each codec alone ---------------------------------------------------------


def bench_plain_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, Cascade[Plain]())


def bench_plain_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, Cascade[Plain]())


def bench_bss_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, Cascade[ByteStreamSplit]())


def bench_bss_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, Cascade[ByteStreamSplit]())


def bench_bitpack_encode(mut b: Benchmark) raises:
    _bench[DType.uint32, "small", False](b, Cascade[BitPack[]]())


def bench_bitpack_decode(mut b: Benchmark) raises:
    _bench[DType.uint32, "small", True](b, Cascade[BitPack[]]())


def bench_reference_encode(mut b: Benchmark) raises:
    _bench[DType.uint64, "cardinality", False](
        b, Cascade[FrameOfReference[], Plain]()
    )


def bench_reference_decode(mut b: Benchmark) raises:
    _bench[DType.uint64, "cardinality", True](
        b, Cascade[FrameOfReference[], Plain]()
    )


def bench_runs_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", False](b, Cascade[Runs[]]())


def bench_runs_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", True](b, Cascade[Runs[]]())


def bench_dictionary_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](b, Cascade[Dictionary[], Plain]())


def bench_dictionary_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](b, Cascade[Dictionary[], Plain]())


def bench_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, Cascade[Delta, Plain]())


def bench_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, Cascade[Delta, Plain]())


def bench_zigzag_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, Cascade[Zigzag, Plain]())


def bench_zigzag_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, Cascade[Zigzag, Plain]())


def bench_ordered_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, Cascade[Ordered, Plain]())


def bench_ordered_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, Cascade[Ordered, Plain]())


def bench_lengths_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Lengths[], Plain]())


def bench_lengths_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Lengths[], Plain]())


def bench_prefixes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Prefixes[], Plain]())


def bench_prefixes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Prefixes[], Plain]())


def bench_dictionary_bytes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Dictionary[], Plain]())


def bench_dictionary_bytes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Dictionary[], Plain]())


# --- cascades, after ClickHouse's codec chains ---------------------------------


def _delta_t64() -> Cascade[Delta, Ordered, FrameOfReference[], BitPack[]]:
    return Cascade((Delta(), Ordered(), FrameOfReference(128), BitPack(32)))


def _double_delta_t64() -> (
    Cascade[Delta, Delta, Ordered, FrameOfReference[], BitPack[]]
):
    return Cascade(
        (Delta(), Delta(), Ordered(), FrameOfReference(128), BitPack(32))
    )


def _t64_blocks() -> Cascade[Ordered, FrameOfReference[], BitPack[]]:
    return Cascade((Ordered(), FrameOfReference(64), BitPack(64)))


def bench_cascade_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](
        b, Cascade[Delta, Zigzag, BitPack[]]()
    )


def bench_cascade_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](
        b, Cascade[Delta, Zigzag, BitPack[]]()
    )


def bench_cascade_double_delta_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", False](
        b, Cascade[Delta, Delta, Zigzag, BitPack[]]()
    )


def bench_cascade_double_delta_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", True](
        b, Cascade[Delta, Delta, Zigzag, BitPack[]]()
    )


def bench_cascade_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](
        b, Cascade[Ordered, FrameOfReference[], BitPack[]]()
    )


def bench_cascade_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](
        b, Cascade[Ordered, FrameOfReference[], BitPack[]]()
    )


def bench_cascade_t64_blocks_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](b, _t64_blocks())


def bench_cascade_t64_blocks_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](b, _t64_blocks())


def bench_cascade_delta_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", False](b, _delta_t64())


def bench_cascade_delta_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "timestamps", True](b, _delta_t64())


def bench_cascade_double_delta_t64_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", False](b, _double_delta_t64())


def bench_cascade_double_delta_t64_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", True](b, _double_delta_t64())


def bench_cascade_zigzag_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "signed", False](b, Cascade[Zigzag, BitPack[]]())


def bench_cascade_zigzag_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "signed", True](b, Cascade[Zigzag, BitPack[]]())


def bench_cascade_low_cardinality_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", False](
        b, Cascade[Dictionary[], BitPack[]]()
    )


def bench_cascade_low_cardinality_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "cardinality", True](
        b, Cascade[Dictionary[], BitPack[]]()
    )


def bench_cascade_low_cardinality_runs_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", False](
        b, Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]()
    )


def bench_cascade_low_cardinality_runs_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "runs", True](
        b, Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]()
    )


def bench_cascade_sparse_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", False](
        b,
        Cascade[
            Runs[
                Cascade[Ordered, FrameOfReference[], BitPack[]],
                Cascade[BitPack[]],
            ]
        ](),
    )


def bench_cascade_sparse_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "sparse", True](
        b,
        Cascade[
            Runs[
                Cascade[Ordered, FrameOfReference[], BitPack[]],
                Cascade[BitPack[]],
            ]
        ](),
    )


def bench_cascade_float_planes_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", False](
        b,
        Cascade[Runs[ByteStreamSplit, BitPack[]]](),
    )


def bench_cascade_float_planes_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", True](
        b,
        Cascade[Runs[ByteStreamSplit, BitPack[]]](),
    )


def bench_cascade_lengths_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Lengths[], Delta, Zigzag, BitPack[]]())


def bench_cascade_lengths_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Lengths[], Delta, Zigzag, BitPack[]]())


def bench_cascade_prefixes_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Prefixes[], Delta, Zigzag, BitPack[]]())


def bench_cascade_prefixes_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Prefixes[], Delta, Zigzag, BitPack[]]())


def bench_cascade_strings_dictionary_encode(mut b: Benchmark) raises:
    _bench_bytes[False](b, Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]())


def bench_cascade_strings_dictionary_decode(mut b: Benchmark) raises:
    _bench_bytes[True](b, Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]())


def bench_xor_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, Cascade[Xor, Plain]())


def bench_xor_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, Cascade[Xor, Plain]())


def bench_gcd_encode(mut b: Benchmark) raises:
    _bench[DType.uint64, "seconds", False](b, Cascade[Gcd[], Plain]())


def bench_gcd_decode(mut b: Benchmark) raises:
    _bench[DType.uint64, "seconds", True](b, Cascade[Gcd[], Plain]())


def bench_patch_encode(mut b: Benchmark) raises:
    _bench[DType.uint32, "outliers", False](b, Cascade[Patch[], Plain]())


def bench_patch_decode(mut b: Benchmark) raises:
    _bench[DType.uint32, "outliers", True](b, Cascade[Patch[], Plain]())


def bench_alp_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](b, Cascade[Alp[], Plain]())


def bench_alp_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](b, Cascade[Alp[], Plain]())


def bench_buckets_encode(mut b: Benchmark) raises:
    _bench[DType.uint64, "small", False](b, Cascade[Buckets]())


def bench_buckets_decode(mut b: Benchmark) raises:
    _bench[DType.uint64, "small", True](b, Cascade[Buckets]())


def bench_gorilla_encode(mut b: Benchmark) raises:
    _bench[DType.uint64, "small", False](b, Cascade[Gorilla]())


def bench_gorilla_decode(mut b: Benchmark) raises:
    _bench[DType.uint64, "small", True](b, Cascade[Gorilla]())


def bench_cascade_pfor_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "outliers", False](
        b, Cascade[Ordered, FrameOfReference[], Patch[], BitPack[]]()
    )


def bench_cascade_pfor_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "outliers", True](
        b, Cascade[Ordered, FrameOfReference[], Patch[], BitPack[]]()
    )


def bench_cascade_alp_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", False](
        b, Cascade[Alp[], Ordered, FrameOfReference[], BitPack[]]()
    )


def bench_cascade_alp_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "floats", True](
        b, Cascade[Alp[], Ordered, FrameOfReference[], BitPack[]]()
    )


def bench_cascade_gcd_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "seconds", False](
        b, Cascade[Ordered, FrameOfReference[], Gcd[], BitPack[]]()
    )


def bench_cascade_gcd_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "seconds", True](
        b, Cascade[Ordered, FrameOfReference[], Gcd[], BitPack[]]()
    )


def bench_cascade_double_delta_buckets_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", False](
        b, Cascade[Delta, Delta, Zigzag, Buckets]()
    )


def bench_cascade_double_delta_buckets_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "interval", True](
        b, Cascade[Delta, Delta, Zigzag, Buckets]()
    )


def bench_cascade_gorilla_bits_encode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", False](b, Cascade[Xor, Gorilla]())


def bench_cascade_gorilla_bits_decode(mut b: Benchmark) raises:
    _bench[DType.float64, "gauge", True](b, Cascade[Xor, Gorilla]())


def bench_huffman_encode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", False](b, Cascade[Compressed[Huffman]]())


def bench_huffman_decode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", True](b, Cascade[Compressed[Huffman]]())


def bench_cascade_lengths_huffman_encode(mut b: Benchmark) raises:
    _bench_bytes[False](
        b,
        Cascade[
            Lengths[Cascade[Compressed[Huffman]]], Delta, Zigzag, BitPack[]
        ](),
    )


def bench_cascade_lengths_huffman_decode(mut b: Benchmark) raises:
    _bench_bytes[True](
        b,
        Cascade[
            Lengths[Cascade[Compressed[Huffman]]], Delta, Zigzag, BitPack[]
        ](),
    )


def bench_frequency_encode(mut b: Benchmark) raises:
    _bench[DType.int64, "frequent", False](b, Cascade[Frequency[]]())


def bench_frequency_decode(mut b: Benchmark) raises:
    _bench[DType.int64, "frequent", True](b, Cascade[Frequency[]]())


def bench_compressed_zstd_encode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", False](b, Cascade[Compressed[Zstd]]())


def bench_compressed_zstd_decode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", True](b, Cascade[Compressed[Zstd]]())


def bench_compressed_lz4_encode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", False](b, Cascade[Compressed[Lz4]]())


def bench_compressed_lz4_decode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", True](b, Cascade[Compressed[Lz4]]())


def bench_compressed_snappy_encode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", False](b, Cascade[Compressed[Snappy]]())


def bench_compressed_snappy_decode(mut b: Benchmark) raises:
    _bench[DType.uint8, "bytes", True](b, Cascade[Compressed[Snappy]]())
