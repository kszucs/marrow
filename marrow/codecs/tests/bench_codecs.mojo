# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The codecs' kernels and every codec, over 1M values.

- `bitpack_{pack,unpack}_w<W>`: `BitPack`'s kernels at the widths that take
  different paths -- 1, the SIMD widths up to 32, and the scalar ones above.
- `codecs_<codec>_{encode,decode}`: each codec alone in a chain on an input
  shaped for it, so a change to a shared driver shows in every row.
- `{views,bitpack}_{pack,unpack}_bits`: width-1 packing, `BitmapView`'s
  `store[64]` / `load[64]` against `BitPack`; run with `--competition` to read
  them side by side.

Run with:
    pixi run -e dev pytest marrow/codecs/tests/bench_codecs.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...buffers import Bitmap
from ...codecs import (
    BitPack,
    ByteStreamSplit,
    DynCascade,
    Codec,
    Constant,
    Delta,
    Dictionary,
    Frequency,
    Rle,
    Varint,
    Xor,
    Zigzag,
)
from ...utils.testing import Benchmark, Rng
from ...views import BufferView

comptime N = 1_000_000


# ---------------------------------------------------------------------------
# BitPack's kernels
# ---------------------------------------------------------------------------


def _values(width: Int, n: Int) -> List[UInt64]:
    """`n` random values of `width` bits."""
    var mask = UInt64(0) if width == 0 else UInt64.MAX >> UInt64(64 - width)
    var rng = Rng(11)
    var out = List[UInt64](capacity=n)
    for _ in range(n):
        out.append(rng.next() & mask)
    return out^


def _bench_pack(mut b: Benchmark, width: Int, n: Int) raises:
    var values = _values(width, n)
    var out = List[UInt8](capacity=(n * width + 7) // 8)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", "bitpack")

    @always_inline
    def call() {mut out, imm}:
        out.clear()
        BitPack.pack(Span(values), width, out)
        keep(len(out))

    b.iter(call)
    keep(values)
    keep(out)


def _bench_unpack(mut b: Benchmark, width: Int, n: Int) raises:
    var values = _values(width, n)
    var data = List[UInt8]()
    BitPack.pack(Span(values), width, data)
    var out = List[UInt64](capacity=n)
    b.throughput(BenchMetric.elements, n)
    b.extra_info("lib", "bitpack")

    @always_inline
    def call() {mut out, imm}:
        out.clear()
        BitPack.unpack(Span(data), 0, width, n, out)
        keep(len(out))

    b.iter(call)
    if out != values:
        raise Error("bench: unpack does not round-trip")
    keep(data)
    keep(out)


def bench_bitpack_pack_w1(mut b: Benchmark) raises:
    _bench_pack(b, 1, N)


def bench_bitpack_pack_w3(mut b: Benchmark) raises:
    _bench_pack(b, 3, N)


def bench_bitpack_pack_w8(mut b: Benchmark) raises:
    _bench_pack(b, 8, N)


def bench_bitpack_pack_w17(mut b: Benchmark) raises:
    _bench_pack(b, 17, N)


def bench_bitpack_pack_w32(mut b: Benchmark) raises:
    _bench_pack(b, 32, N)


def bench_bitpack_pack_w33(mut b: Benchmark) raises:
    _bench_pack(b, 33, N)


def bench_bitpack_pack_w64(mut b: Benchmark) raises:
    _bench_pack(b, 64, N)


def bench_bitpack_pack_w8_1k(mut b: Benchmark) raises:
    _bench_pack(b, 8, 1_000)


def bench_bitpack_pack_w8_100k(mut b: Benchmark) raises:
    _bench_pack(b, 8, 100_000)


def bench_bitpack_unpack_w1(mut b: Benchmark) raises:
    _bench_unpack(b, 1, N)


def bench_bitpack_unpack_w3(mut b: Benchmark) raises:
    _bench_unpack(b, 3, N)


def bench_bitpack_unpack_w8(mut b: Benchmark) raises:
    _bench_unpack(b, 8, N)


def bench_bitpack_unpack_w17(mut b: Benchmark) raises:
    _bench_unpack(b, 17, N)


def bench_bitpack_unpack_w32(mut b: Benchmark) raises:
    _bench_unpack(b, 32, N)


def bench_bitpack_unpack_w33(mut b: Benchmark) raises:
    _bench_unpack(b, 33, N)


def bench_bitpack_unpack_w64(mut b: Benchmark) raises:
    _bench_unpack(b, 64, N)


def bench_bitpack_unpack_w8_1k(mut b: Benchmark) raises:
    _bench_unpack(b, 8, 1_000)


def bench_bitpack_unpack_w8_100k(mut b: Benchmark) raises:
    _bench_unpack(b, 8, 100_000)


def bench_bitpack_get_w17(mut b: Benchmark) raises:
    """Random access: every value read on its own."""
    var data = List[UInt8]()
    BitPack.pack(Span(_values(17, N)), 17, data)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", "bitpack")

    @always_inline
    def call() {imm}:
        var acc = UInt64(0)
        for i in range(N):
            acc ^= BitPack.get(Span(data), i * 17, 17)
        keep(acc)

    b.iter(call)
    keep(data)


# ---------------------------------------------------------------------------
# Width 1: BitmapView against BitPack
# ---------------------------------------------------------------------------


def _bits() -> List[UInt8]:
    """`N` zeros and ones, at random."""
    var rng = Rng(5)
    var out = List[UInt8](capacity=N)
    for _ in range(N):
        out.append(UInt8(rng.next() & 1))
    return out^


def bench_views_pack_bits(mut b: Benchmark) raises:
    var bits = _bits()
    var bm = Bitmap.alloc_zeroed(N)
    var dst = bm.view()
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", "views")

    @always_inline
    def call() {imm}:
        var src = BufferView(Span(bits))
        for i in range(0, N - 63, 64):
            dst.store[64](i, src.load[64](i).cast[DType.bool]())
        keep(dst.load_bytes[DType.uint8](0))

    b.iter(call)
    keep(bits)


def bench_bitpack_pack_bits(mut b: Benchmark) raises:
    var bits = _bits()
    var out = List[UInt8](capacity=N // 8 + 8)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", "bitpack")

    @always_inline
    def call() {mut out, imm}:
        out.clear()
        BitPack.pack(Span(bits), 1, out)
        keep(len(out))

    b.iter(call)
    keep(bits)
    keep(out)


def bench_views_unpack_bits(mut b: Benchmark) raises:
    var bits = _bits()
    var packed = List[UInt8]()
    BitPack.pack(Span(bits), 1, packed)
    var bm = Bitmap.alloc_zeroed(N)
    var view = bm.view()
    for i in range(0, N - 63, 64):
        view.store[64](i, BufferView(Span(bits)).load[64](i).cast[DType.bool]())
    var out = List[UInt8](length=N, fill=0)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", "views")

    @always_inline
    def call() {mut out, imm}:
        var dst = BufferView(Span(out))
        for i in range(0, N - 63, 64):
            dst.store[64](i, view.load[64](i).cast[DType.uint8]())
        keep(out[0])

    b.iter(call)
    keep(bits)
    keep(packed)


def bench_bitpack_unpack_bits(mut b: Benchmark) raises:
    var bits = _bits()
    var packed = List[UInt8]()
    BitPack.pack(Span(bits), 1, packed)
    var out = List[UInt8](capacity=N)
    b.throughput(BenchMetric.elements, N)
    b.extra_info("lib", "bitpack")

    @always_inline
    def call() {mut out, imm}:
        out.clear()
        BitPack.unpack(Span(packed), 0, 1, N, out)
        keep(len(out))

    b.iter(call)
    keep(bits)
    keep(packed)


# ---------------------------------------------------------------------------
# Every codec, through a one-node cascade
# ---------------------------------------------------------------------------


def _cascade[C: Codec]() raises -> DynCascade:
    """`C` alone, so the row times `C` and as little else as a chain
    allows."""
    return DynCascade(C())


def _bench_codec[
    C: Codec, T: DType, decode: Bool
](mut b: Benchmark, values: List[Scalar[T]]) raises:
    var cascade = _cascade[C]()
    var data = cascade.encode[T](values)
    if cascade.decode[T](data) != values:
        raise Error(String("bench: ", C.name(), " does not round-trip"))
    b.throughput(BenchMetric.elements, len(values))
    b.extra_info("lib", "codecs")
    comptime if decode:

        @always_inline
        def call_decode() raises {imm}:
            keep(len(cascade.decode[T](data)))

        b.iter(call_decode)
    else:

        @always_inline
        def call_encode() raises {imm}:
            keep(len(cascade.encode[T](values)))

        b.iter(call_encode)
    keep(values)
    keep(data)
    keep(cascade)


def _timestamps() -> List[Int64]:
    var out = List[Int64](capacity=N)
    var t = Int64(1_700_000_000_000_000)
    for i in range(N):
        t += Int64(1_000 + (i * 7919) % 500)
        out.append(t)
    return out^


def _small() -> List[UInt32]:
    """Unsigned values under 2^17, at random."""
    var rng = Rng(3)
    var out = List[UInt32](capacity=N)
    for _ in range(N):
        out.append(UInt32(rng.next() & 0x1FFFF))
    return out^


def _signed() -> List[Int32]:
    """Signed values of small magnitude, at random."""
    var rng = Rng(4)
    var out = List[Int32](capacity=N)
    for _ in range(N):
        out.append(Int32(rng.below(2001)) - 1000)
    return out^


def _floats() -> List[Float64]:
    """A slowly moving series, as sensor readings are."""
    var out = List[Float64](capacity=N)
    for i in range(N):
        out.append(20.0 + Float64(i % 1000) * 0.01)
    return out^


def _runs() -> List[Int32]:
    """Runs of 1 to 64 equal values."""
    var rng = Rng(6)
    var out = List[Int32](capacity=N)
    while len(out) < N:
        var v = Int32(rng.below(100))
        for _ in range(min(1 + rng.below(64), N - len(out))):
            out.append(v)
    return out^


def _categories() -> List[Int64]:
    """1,000 distinct values, at random."""
    var rng = Rng(7)
    var out = List[Int64](capacity=N)
    for _ in range(N):
        out.append(Int64(rng.below(1000)) * 1_000_003)
    return out^


def _dominated() -> List[Int16]:
    """One value, and 1% exceptions."""
    var rng = Rng(8)
    var out = List[Int16](capacity=N)
    for _ in range(N):
        out.append(Int16(rng.below(1000)) if rng.below(100) == 0 else 42)
    return out^


def bench_codecs_bitpack_encode(mut b: Benchmark) raises:
    _bench_codec[BitPack, DType.uint32, False](b, _small())


def bench_codecs_bitpack_decode(mut b: Benchmark) raises:
    _bench_codec[BitPack, DType.uint32, True](b, _small())


def bench_codecs_varint_encode(mut b: Benchmark) raises:
    _bench_codec[Varint, DType.uint32, False](b, _small())


def bench_codecs_varint_decode(mut b: Benchmark) raises:
    _bench_codec[Varint, DType.uint32, True](b, _small())


def bench_codecs_zigzag_encode(mut b: Benchmark) raises:
    _bench_codec[Zigzag, DType.int32, False](b, _signed())


def bench_codecs_zigzag_decode(mut b: Benchmark) raises:
    _bench_codec[Zigzag, DType.int32, True](b, _signed())


def bench_codecs_delta_encode(mut b: Benchmark) raises:
    _bench_codec[Delta, DType.int64, False](b, _timestamps())


def bench_codecs_delta_decode(mut b: Benchmark) raises:
    _bench_codec[Delta, DType.int64, True](b, _timestamps())


def bench_codecs_xor_encode(mut b: Benchmark) raises:
    _bench_codec[Xor, DType.float64, False](b, _floats())


def bench_codecs_xor_decode(mut b: Benchmark) raises:
    _bench_codec[Xor, DType.float64, True](b, _floats())


def bench_codecs_byte_stream_split_encode(mut b: Benchmark) raises:
    _bench_codec[ByteStreamSplit, DType.float64, False](b, _floats())


def bench_codecs_byte_stream_split_decode(mut b: Benchmark) raises:
    _bench_codec[ByteStreamSplit, DType.float64, True](b, _floats())


def bench_codecs_constant_encode(mut b: Benchmark) raises:
    _bench_codec[Constant, DType.int32, False](b, List[Int32](length=N, fill=7))


def bench_codecs_constant_decode(mut b: Benchmark) raises:
    _bench_codec[Constant, DType.int32, True](b, List[Int32](length=N, fill=7))


def bench_codecs_rle_encode(mut b: Benchmark) raises:
    _bench_codec[Rle, DType.int32, False](b, _runs())


def bench_codecs_rle_decode(mut b: Benchmark) raises:
    _bench_codec[Rle, DType.int32, True](b, _runs())


def bench_codecs_dictionary_encode(mut b: Benchmark) raises:
    _bench_codec[Dictionary, DType.int64, False](b, _categories())


def bench_codecs_dictionary_decode(mut b: Benchmark) raises:
    _bench_codec[Dictionary, DType.int64, True](b, _categories())


def bench_codecs_frequency_encode(mut b: Benchmark) raises:
    _bench_codec[Frequency, DType.int16, False](b, _dominated())


def bench_codecs_frequency_decode(mut b: Benchmark) raises:
    _bench_codec[Frequency, DType.int16, True](b, _dominated())
