# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every codec through one harness: edge cases and random columns round-trip
bit for bit, every truncation of a stream is refused, and a type the codec
does not take is refused."""

from std.memory import bitcast
from std.testing import assert_raises, assert_true

from ...codecs import (
    BinaryCodec,
    BitPack,
    Bits,
    ByteStreamSplit,
    DynCascade,
    Codec,
    Delta,
    DeltaBinaryPacked,
    DeltaByteArray,
    DeltaLengthByteArray,
    Dictionary,
    Frequency,
    Hybrid,
    PlainBinary,
    Rle,
    Varint,
    Xor,
    Zigzag,
)
from ...utils.testing import Rng


def _cascade[C: Codec]() raises -> DynCascade:
    """`C` alone: an elementwise codec's values then stored plain."""
    return DynCascade(C())


def _same[T: DType](a: List[Scalar[T]], b: List[Scalar[T]]) -> Bool:
    """Equal bit for bit -- so a NaN equals itself and -0.0 is not 0.0."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[Bits.unsigned[T]](a[i]) != bitcast[Bits.unsigned[T]](b[i]):
            return False
    return True


def _cases[T: DType]() -> List[List[Scalar[T]]]:
    """The empty column, single values, the extremes, runs and a sorted
    column -- and for floats the values whose bits matter."""
    var cases = List[List[Scalar[T]]]()
    cases.append(List[Scalar[T]]())
    cases.append([Scalar[T](0)])
    cases.append([Scalar[T].MIN_FINITE])
    cases.append([Scalar[T].MAX_FINITE])
    cases.append(
        [Scalar[T](0), Scalar[T].MAX_FINITE, Scalar[T].MIN_FINITE, Scalar[T](1)]
    )
    var runs = List[Scalar[T]]()
    for i in range(40):
        runs.append(Scalar[T]((i // 7) % 3))
    cases.append(runs^)
    var sorted = List[Scalar[T]]()
    for i in range(100):
        sorted.append(Scalar[T](i * 3))
    cases.append(sorted^)
    comptime if T.is_floating_point():
        cases.append(
            [
                Scalar[T](-0.0),
                Scalar[T](0.0),
                Scalar[T].MAX,
                Scalar[T].MIN,
                Scalar[T](0.0) / Scalar[T](0.0),
            ]
        )
    return cases^


def _random[T: DType](n: Int, seed: UInt64) -> List[Scalar[T]]:
    """`n` values of random bits -- NaN payloads included, for floats."""
    var rng = Rng(seed)
    var out = List[Scalar[T]](capacity=n)
    for _ in range(n):
        out.append(bitcast[T](rng.next().cast[Bits.unsigned[T]]()))
    return out^


def _roundtrip[
    C: Codec, T: DType
](values: List[Scalar[T]], every_prefix: Bool) raises:
    var chain = _cascade[C]()
    var data = chain.encode[T](values)
    var back = chain.decode[T](data)
    assert_true(
        _same(back, values),
        String(
            C.name(), "[", T, "] did not round-trip ", len(values), " values"
        ),
    )
    if every_prefix:
        for n in range(len(data)):
            with assert_raises(contains="CorruptError"):
                _ = chain.decode[T](Span(data)[:n])


def _check[C: Codec, T: DType]() raises:
    for ref values in _cases[T]():
        _roundtrip[C, T](values, every_prefix=True)
    _roundtrip[C, T](_random[T](10_000, 42), every_prefix=False)


def _refuses[C: Codec, T: DType]() raises:
    var one: List[Scalar[T]] = [Scalar[T](1)]
    with assert_raises(contains="InvalidError"):
        _ = _cascade[C]().encode[T](one)


def test_codecs_bitpack() raises:
    _check[BitPack, DType.uint8]()
    _check[BitPack, DType.uint16]()
    _check[BitPack, DType.uint32]()
    _check[BitPack, DType.uint64]()
    _check[BitPack, DType.int8]()
    _check[BitPack, DType.int64]()
    _check[BitPack, DType.float64]()


def test_codecs_bitpack_widths() raises:
    """Every width from 0 to 64, for both the SIMD and the scalar unpack."""
    for w in range(65):
        var values = List[UInt64]()
        for i in range(37):
            var top = UInt64(0) if w == 0 else (
                UInt64.MAX if w == 64 else (UInt64(1) << UInt64(w)) - 1
            )
            values.append(top - UInt64(i % 3) if w > 1 else top)
        _roundtrip[BitPack, DType.uint64](values, every_prefix=False)


def test_codecs_bitpack_get() raises:
    """Bits read least significant first, across a byte boundary, and past
    the end as zero."""
    var b: List[UInt8] = [0xB2]
    assert_true(BitPack.get(Span(b), 0, 1) == 0)
    assert_true(BitPack.get(Span(b), 1, 1) == 1)
    assert_true(BitPack.get(Span(b), 0, 4) == 0x2)
    assert_true(BitPack.get(Span(b), 4, 4) == 0xB)
    assert_true(BitPack.get(Span(b), 0, 8) == 0xB2)
    assert_true(BitPack.get(Span(b), 4, 8) == 0xB)
    var c: List[UInt8] = [0xFF, 0x01]
    assert_true(BitPack.get(Span(c), 4, 8) == 0x1F)
    assert_true(BitPack.get(Span(c), 0, 0) == 0)


def test_codecs_bitpack_pack_unpack() raises:
    """`pack` keeps each value's low `width` bits and `unpack` reads them back
    from any byte, for every width; `width` gives the narrowest."""
    assert_true(Bits.width(0) == 0)
    assert_true(Bits.width(1) == 1)
    assert_true(Bits.width(2) == 2)
    assert_true(Bits.width(7) == 3)
    assert_true(Bits.width(8) == 4)
    assert_true(Bits.width(255) == 8)
    assert_true(Bits.width(UInt64.MAX) == 64)
    var rng = Rng(3)
    for w in range(65):
        var mask = UInt64(0) if w == 0 else UInt64.MAX >> UInt64(64 - w)
        var values = List[UInt64]()
        for _ in range(29):
            values.append(rng.next())
        var data: List[UInt8] = [0xAB]
        BitPack.pack(Span(values), w, data)
        assert_true(len(data) == 1 + (29 * w + 7) // 8)
        var back = List[UInt64]()
        BitPack.unpack(Span(data), 1, w, 29, back)
        for i in range(29):
            assert_true(back[i] == values[i] & mask, String("width ", w))


def test_codecs_bitpack_registers() raises:
    """`pack_bools` and `unpack_lanes` invert each other, at every lane
    count a bitmap uses and every width a word holds."""
    var m = SIMD[DType.bool, 64](fill=False)
    for j in range(0, 64, 3):
        m[j] = True
    var word = bitcast[DType.uint64, 1](Bits.pack_bools[64](m))
    assert_true(word == 0x9249249249249249)
    var back = Bits.unpack_lanes[DType.uint64, 64](word, 1)
    for j in range(64):
        assert_true((back[j] == 1) == m[j])
    assert_true(Bits.pack_bools[8](SIMD[DType.bool, 8](fill=True)) == 0xFF)
    var le: List[UInt8] = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]
    for w in range(1, 9):
        var v = Bits.unpack_lanes[DType.uint64, 8](0x0123456789ABCDEF, w)
        for j in range(8):
            assert_true(v[j] == BitPack.get(Span(le), j * w, w))


def test_codecs_varint() raises:
    _check[Varint, DType.uint8]()
    _check[Varint, DType.uint32]()
    _check[Varint, DType.uint64]()
    _check[Varint, DType.int64]()
    _refuses[Varint, DType.float32]()


def test_codecs_zigzag() raises:
    _check[Zigzag, DType.int8]()
    _check[Zigzag, DType.int16]()
    _check[Zigzag, DType.int32]()
    _check[Zigzag, DType.int64]()
    _refuses[Zigzag, DType.uint32]()
    _refuses[Zigzag, DType.float32]()


def test_codecs_delta() raises:
    _check[Delta, DType.int32]()
    _check[Delta, DType.int64]()
    _check[Delta, DType.uint8]()
    _check[Delta, DType.uint64]()
    _refuses[Delta, DType.float64]()


def test_codecs_rle() raises:
    _check[Rle, DType.uint8]()
    _check[Rle, DType.int32]()
    _check[Rle, DType.float64]()


def test_codecs_byte_stream_split() raises:
    _check[ByteStreamSplit, DType.uint16]()
    _check[ByteStreamSplit, DType.int32]()
    _check[ByteStreamSplit, DType.float32]()
    _check[ByteStreamSplit, DType.float64]()


def test_codecs_dictionary() raises:
    _check[Dictionary, DType.uint8]()
    _check[Dictionary, DType.int64]()
    _check[Dictionary, DType.float32]()


def test_codecs_frequency() raises:
    _check[Frequency, DType.int16]()
    _check[Frequency, DType.uint64]()
    _check[Frequency, DType.float64]()


def test_codecs_xor() raises:
    _check[Xor, DType.int32]()
    _check[Xor, DType.float32]()
    _check[Xor, DType.float64]()


def test_codecs_hybrid() raises:
    _check[Hybrid, DType.uint8]()
    _check[Hybrid, DType.uint32]()
    _check[Hybrid, DType.uint64]()
    _refuses[Hybrid, DType.int32]()
    _refuses[Hybrid, DType.float32]()


def test_codecs_hybrid_runs() raises:
    """Both run kinds read back at every width a level or index takes, and
    the two encoders write Parquet's bytes."""
    for w in range(1, 33):
        var top = UInt32.MAX >> UInt32(32 - w)
        var values = List[UInt32]()
        for i in range(83):
            values.append(top if (i // 9) % 2 == 0 else UInt32(i) & top)
        for packed in [False, True]:
            var data = List[UInt8]()
            if packed:
                Hybrid.encode_packed(Span(values), w, data)
            else:
                Hybrid.encode_runs(Span(values), w, data)
            var back = List[UInt32]()
            var used = Hybrid.decode_runs(Span(data), w, len(values), back)
            assert_true(used == len(data), String("width ", w))
            assert_true(back == values, String("width ", w))
            var ones = Hybrid.count_matches(
                Span(data), w, len(values), UInt64(top)
            )
            var want = 0
            for v in values:
                if v == top:
                    want += 1
            assert_true(ones == want, String("width ", w))
    # One bit-packed run of 0..7 at width 3, and its zero padding.
    var eight: List[UInt8] = [0, 1, 2, 3, 4, 5, 6, 7]
    var packed = List[UInt8]()
    Hybrid.encode_packed(Span(eight), 3, packed)
    var expected: List[UInt8] = [3, 0x88, 0xC6, 0xFA]
    assert_true(packed == expected)
    eight.append(5)
    packed.clear()
    Hybrid.encode_packed(Span(eight), 3, packed)
    assert_true(len(packed) == 1 + 2 * 3)


def test_codecs_hybrid_refuses_a_truncated_run() raises:
    var values: List[UInt8] = [1, 2, 3, 4, 5, 6, 7, 1, 2]
    var data = List[UInt8]()
    Hybrid.encode_packed(Span(values), 3, data)
    var out = List[UInt8]()
    with assert_raises(contains="CorruptError"):
        _ = Hybrid.decode_runs(Span(data)[: len(data) - 1], 3, 9, out)


def test_codecs_delta_binary_packed() raises:
    _check[DeltaBinaryPacked, DType.int8]()
    _check[DeltaBinaryPacked, DType.int16]()
    _check[DeltaBinaryPacked, DType.int32]()
    _check[DeltaBinaryPacked, DType.int64]()
    _refuses[DeltaBinaryPacked, DType.uint32]()
    _refuses[DeltaBinaryPacked, DType.float64]()


def test_codecs_delta_binary_packed_blocks() raises:
    """Differences as wide as 64 bits, a last block that is not full, and
    the stream's end where what follows it begins."""
    var vals: List[Int64] = [Int64.MIN, Int64.MAX, 0, -1]
    var x = Int64(1)
    for i in range(300):
        x = x * 6364136223846793005 + 1442695040888963407
        vals.append(x >> Int64(i % 60))
    var data = List[UInt8]()
    DeltaBinaryPacked.encode_blocks(Span(vals), data)
    data.append(0xAB)
    var back = List[Int64]()
    var end = DeltaBinaryPacked.decode_blocks(Span(data), 0, len(vals), back)
    assert_true(back == vals)
    assert_true(end == len(data) - 1)
    var none = List[Int64]()
    with assert_raises(contains="CorruptError"):
        _ = DeltaBinaryPacked.decode_blocks(Span(data), 0, len(vals) + 1, none)


def _column(
    values: List[String], mut offsets: List[Int32], mut data: List[UInt8]
):
    offsets.append(Int32(len(data)))
    for v in values:
        data.extend(v.as_bytes())
        offsets.append(Int32(len(data)))


def _check_binary_column[
    C: BinaryCodec
](offsets: List[Int32], data: List[UInt8], every_prefix: Bool) raises:
    var count = len(offsets) - 1
    var out = List[UInt8]()
    C.encode(Span(offsets), Span(data), out)
    var back_offsets: List[Int32] = [0]
    var back = List[UInt8]()
    var end = C.decode(Span(out), 0, count, back_offsets, back)
    assert_true(end == len(out), String(C.name()))
    assert_true(back_offsets == offsets, String(C.name()))
    assert_true(back == data, String(C.name()))
    if every_prefix:
        for n in range(len(out)):
            var cut_offsets: List[Int32] = [0]
            var cut = List[UInt8]()
            with assert_raises(contains="CorruptError"):
                _ = C.decode(Span(out)[:n], 0, count, cut_offsets, cut)


def _check_binary[C: BinaryCodec]() raises:
    var cases = List[List[String]]()
    cases.append(List[String]())
    cases.append([String("")])
    cases.append(
        [
            String(""),
            String("abc"),
            String(""),
            String("abd"),
            String("abd"),
            String("abdx"),
            String("b"),
        ]
    )
    for ref values in cases:
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        _check_binary_column[C](offsets, data, every_prefix=True)
    # Bytes that are not UTF-8.
    var odd_offsets: List[Int32] = [0, 4, 7]
    var odd: List[UInt8] = [0xFF, 0x00, 0xC3, 0x28, 0x00, 0xC3, 0x28]
    _check_binary_column[C](odd_offsets, odd, every_prefix=True)
    # A thousand values sharing prefixes.
    var many = List[String]()
    for i in range(1000):
        many.append(String("row-", i // 10, "-", i * 7919 % 1000))
    var offsets = List[Int32]()
    var data = List[UInt8]()
    _column(many, offsets, data)
    _check_binary_column[C](offsets, data, every_prefix=False)


def test_codecs_plain_binary() raises:
    _check_binary[PlainBinary]()


def test_codecs_delta_length_byte_array() raises:
    _check_binary[DeltaLengthByteArray]()


def test_codecs_delta_byte_array() raises:
    _check_binary[DeltaByteArray]()


def test_codecs_dictionary_build() raises:
    """Values are told apart by their bits."""
    var nan = Float64(0.0) / Float64(0.0)
    var values: List[Float64] = [-0.0, 0.0, nan, nan, 1.0, 0.0]
    var distinct = List[Float64]()
    var codes = List[Int32]()
    Dictionary.build(Span(values), distinct, codes)
    assert_true(len(distinct) == 4)
    var want: List[Int32] = [0, 1, 2, 2, 3, 1]
    assert_true(codes == want)
