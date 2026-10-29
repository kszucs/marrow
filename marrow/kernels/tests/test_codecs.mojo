# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every codec round-trips every column bit for bit: stream stages composed,
packers alone and nested, the codecs built from both, and the byte-string
packers; and a packed part altered is refused."""

from std.builtin.rebind import rebind_var
from std.memory import bitcast
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import Int64Array
from ...buffers import Buffer
from ...builders import array
from ...dtypes import Int64Type, int64
from ...utils.integers import Unsigned
from ...utils.testing import Rng
from ...views import BufferView
from ...kernels.bitwise import (
    ByteSwap,
    OrderedDecode,
    OrderedEncode,
    ZigzagDecode,
    ZigzagEncode,
)
from ...kernels.cast import NumericCast
from ...kernels.compose import Codec, Compose, Decoder, Encoder, With
from ...kernels.core import PackKernel, UnpackKernel
from ...kernels.dictionary import (
    DictionaryBytesPack,
    DictionaryBytesPacked,
    DictionaryBytesUnpack,
    DictionaryPack,
    DictionaryPacked,
    DictionaryUnpack,
)
from ...kernels.numeric import AddKernel, SubKernel
from ...kernels.packing import (
    BLOCK,
    BitPack,
    BitPacked,
    BitUnpack,
    FittedPacked,
    LengthsPack,
    LengthsPacked,
    LengthsUnpack,
    PlainPack,
    PlainPacked,
    PlainUnpack,
    PrefixesPack,
    PrefixesPacked,
    PrefixesUnpack,
    ReferencePack,
    ReferenceUnpack,
    RlePack,
    RlePacked,
    RleUnpack,
    SplitPack,
    SplitPacked,
    SplitUnpack,
)
from ...kernels.runs import RunsPack, RunsPacked, RunsUnpack
from ...kernels.scan import DiffScan, SumScan
from ...kernels.segments import PerSegment


# --- columns ------------------------------------------------------------------


def _columns[T: DType]() -> List[List[Scalar[T]]]:
    """The empty column, single values, the extremes, runs, a sorted column,
    random bits -- and for floats the values whose bits matter."""
    var cols = List[List[Scalar[T]]]()
    cols.append(List[Scalar[T]]())
    cols.append([Scalar[T](0)])
    cols.append([Scalar[T].MIN_FINITE])
    cols.append([Scalar[T].MAX_FINITE])
    cols.append(
        [Scalar[T](0), Scalar[T].MAX_FINITE, Scalar[T].MIN_FINITE, Scalar[T](1)]
    )
    var runs = [Scalar[T]((i // 37) % 3) for i in range(1000)]
    cols.append(runs^)
    var sorted = [Scalar[T](i * 3) for i in range(301)]
    cols.append(sorted^)
    comptime if T.is_floating_point():
        cols.append(
            [
                Scalar[T](-0.0),
                Scalar[T](0.0),
                Scalar[T].MAX,
                Scalar[T].MIN,
                Scalar[T](0.0) / Scalar[T](0.0),
            ]
        )
    var rng = Rng(7)
    var noise = List[Scalar[T]]()
    for _ in range(10_003):
        noise.append(bitcast[T](rng.next().cast[Unsigned[T]]()))
    cols.append(noise^)
    return cols^


def _same[T: DType](a: List[Scalar[T]], b: List[Scalar[T]]) -> Bool:
    """Equal bit for bit -- so a NaN equals itself and -0.0 is not 0.0."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[Unsigned[T]](a[i]) != bitcast[Unsigned[T]](b[i]):
            return False
    return True


def _decode[
    T: DType, E: PackKernel, D: UnpackKernel
](codec: Codec[E, D], var packed: E.Packed, count: Int) raises -> List[
    Scalar[T]
]:
    var out = List[Scalar[T]](length=count, fill=0)
    codec.decode[T](packed^, BufferView(Span(out)))
    return out^


def check[T: DType, E: PackKernel, D: UnpackKernel](codec: Codec[E, D]) raises:
    """Every column round-trips bit for bit."""
    for ref values in _columns[T]():
        var n = len(values)
        var packed = codec.encode(BufferView(Span(values)))
        assert_true(
            _same(_decode[T](codec, packed^, n), values),
            String(T, " did not round-trip ", n),
        )


def _strings() -> List[List[String]]:
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
    var many = List[String]()
    for i in range(1000):
        many.append(String("row-", i // 10, "-", i * 7919 % 1000))
    cases.append(many^)
    return cases^


def _column(
    values: List[String], mut offsets: List[Int32], mut data: List[UInt8]
):
    offsets.append(Int32(len(data)))
    for v in values:
        data.extend(v.as_bytes())
        offsets.append(Int32(len(data)))


# `check`, for byte strings: every column of `_strings` round-trips, offsets
# and bytes alike. One per pair, as the pairs share no trait.


def check_bytes[
    L: PackKernel, M: UnpackKernel
](packer: LengthsPack[L], unpacker: LengthsUnpack[M]) raises:
    comptime assert L.Packed == M.Packed, "the unpacker takes what is packed"
    for ref values in _strings():
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        var packed = packer.pack(Span(offsets), Span(data))
        var ends: List[Int32] = [0]
        var bytes = List[UInt8]()
        unpacker.unpack(
            rebind_var[LengthsPacked[M.Packed]](packed^),
            len(values),
            ends,
            bytes,
        )
        assert_true(ends == offsets and bytes == data)


def check_bytes[
    P: PackKernel, L: PackKernel, Q: UnpackKernel, M: UnpackKernel
](packer: PrefixesPack[P, L], unpacker: PrefixesUnpack[Q, M]) raises:
    comptime assert (
        P.Packed == Q.Packed and L.Packed == M.Packed
    ), "the unpacker takes what is packed"
    for ref values in _strings():
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        var packed = packer.pack(Span(offsets), Span(data))
        var ends: List[Int32] = [0]
        var bytes = List[UInt8]()
        unpacker.unpack(
            rebind_var[PrefixesPacked[Q.Packed, M.Packed]](packed^),
            len(values),
            ends,
            bytes,
        )
        assert_true(ends == offsets and bytes == data)


def check_bytes[
    C: PackKernel, D: UnpackKernel
](packer: DictionaryBytesPack[C], unpacker: DictionaryBytesUnpack[D]) raises:
    comptime assert C.Packed == D.Packed, "the unpacker takes what is packed"
    for ref values in _strings():
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        var packed = packer.pack(Span(offsets), Span(data))
        var ends: List[Int32] = [0]
        var bytes = List[UInt8]()
        unpacker.unpack(
            rebind_var[DictionaryBytesPacked[D.Packed]](packed^),
            len(values),
            ends,
            bytes,
        )
        assert_true(ends == offsets and bytes == data)


# --- codecs --------------------------------------------------------------------


def test_codecs_delta() raises:
    check[DType.int64](
        Codec(
            Encoder(Compose((DiffScan(),)), PlainPack()),
            Decoder[PlainUnpack, Compose[SumScan], DType.int64](
                PlainUnpack(), Compose((SumScan(),))
            ),
        )
    )
    check[DType.int8](
        Codec(
            Encoder(Compose((DiffScan(),)), PlainPack()),
            Decoder[PlainUnpack, Compose[SumScan], DType.int8](
                PlainUnpack(), Compose((SumScan(),))
            ),
        )
    )
    # The value before the first is 7.
    check[DType.uint16](
        Codec(
            Encoder(
                Compose((With[SubKernel, DType.uint16](7), DiffScan())),
                PlainPack(),
            ),
            Decoder[
                PlainUnpack,
                Compose[SumScan, With[AddKernel, DType.uint16]],
                DType.uint16,
            ](
                PlainUnpack(),
                Compose((SumScan(), With[AddKernel, DType.uint16](7))),
            ),
        )
    )


# --- codec stages --------------------------------------------------------------
# Each codec's encoder, and its decoder for a column of `T`; `In`, what the
# decoder's unpacker yields, is the dtype the encoder packed.


def _zigzag_pack() -> Encoder[Compose[ZigzagEncode], PlainPack]:
    return Encoder(Compose((ZigzagEncode(),)), PlainPack())


def _zigzag_unpack[
    T: DType
]() -> Decoder[PlainUnpack, Compose[ZigzagDecode], Unsigned[T]]:
    return Decoder[PlainUnpack, Compose[ZigzagDecode], Unsigned[T]](
        PlainUnpack(), Compose((ZigzagDecode(),))
    )


def test_codecs_zigzag() raises:
    check[DType.int32](Codec(_zigzag_pack(), _zigzag_unpack[DType.int32]()))
    check[DType.int64](Codec(_zigzag_pack(), _zigzag_unpack[DType.int64]()))


def _ordered_unpack[
    T: DType
]() -> Decoder[PlainUnpack, Compose[OrderedDecode[T]], Unsigned[T]]:
    return Decoder[PlainUnpack, Compose[OrderedDecode[T]], Unsigned[T]](
        PlainUnpack(), Compose((OrderedDecode[T](),))
    )


def test_codecs_ordered() raises:
    check[DType.float32](
        Codec(
            Encoder(Compose((OrderedEncode(),)), PlainPack()),
            _ordered_unpack[DType.float32](),
        )
    )
    check[DType.int16](
        Codec(
            Encoder(Compose((OrderedEncode(),)), PlainPack()),
            _ordered_unpack[DType.int16](),
        )
    )


def test_codecs_cast() raises:
    check[DType.int8](
        Codec(
            Encoder(Compose((NumericCast[DType.int32](),)), PlainPack()),
            Decoder[PlainUnpack, Compose[NumericCast[DType.int8]], DType.int32](
                PlainUnpack(), Compose((NumericCast[DType.int8](),))
            ),
        )
    )
    check[DType.uint16](
        Codec(
            Encoder(Compose((NumericCast[DType.uint32](),)), PlainPack()),
            Decoder[
                PlainUnpack, Compose[NumericCast[DType.uint16]], DType.uint32
            ](PlainUnpack(), Compose((NumericCast[DType.uint16](),))),
        )
    )
    check[DType.int32](
        Codec(
            Encoder(
                Compose(
                    (NumericCast[DType.int64](), DiffScan(), ZigzagEncode())
                ),
                BitPack(),
            ),
            Decoder[
                BitUnpack,
                Compose[ZigzagDecode, SumScan, NumericCast[DType.int32]],
                DType.uint64,
            ](
                BitUnpack(),
                Compose(
                    (ZigzagDecode(), SumScan(), NumericCast[DType.int32]())
                ),
            ),
        )
    )


def _byte_swap[
    T: DType
]() -> Codec[
    Encoder[Compose[ByteSwap], PlainPack],
    Decoder[PlainUnpack, Compose[ByteSwap], T],
]:
    return Codec(
        Encoder(Compose((ByteSwap(),)), PlainPack()),
        Decoder[PlainUnpack, Compose[ByteSwap], T](
            PlainUnpack(), Compose((ByteSwap(),))
        ),
    )


def test_codecs_byte_swap() raises:
    check[DType.int64](_byte_swap[DType.int64]())
    # Values wider than a block of scratch holds as `uint64`s.
    var values = [Int128(i) << 100 | Int128(i) for i in range(3000)]
    var codec = _byte_swap[DType.int128]()
    var packed = codec.encode(BufferView(Span(values)))
    # The second value, big-endian: bit 100 in its fourth byte, 1 in its
    # last.
    var bytes = packed.bytes.view[DType.uint8]()
    assert_true(bytes[16 + 3] == 0x10 and bytes[16 + 15] == 1)
    assert_true(_decode[DType.int128](codec, packed^, len(values)) == values)


# --- packers -------------------------------------------------------------------


def test_codecs_plain() raises:
    check[DType.float64](Codec(PlainPack(), PlainUnpack()))
    check[DType.int8](Codec(PlainPack(), PlainUnpack()))
    check[DType.uint64](Codec(PlainPack(), PlainUnpack()))


def test_codecs_byte_stream_split() raises:
    check[DType.float32](Codec(SplitPack(), SplitUnpack()))
    check[DType.float64](Codec(SplitPack(), SplitUnpack()))
    check[DType.int64](Codec(SplitPack(), SplitUnpack()))


def test_codecs_bitpack() raises:
    check[DType.uint8](Codec(BitPack(), BitUnpack()))
    check[DType.uint32](Codec(BitPack(), BitUnpack()))
    check[DType.uint64](Codec(BitPack(32), BitUnpack(32)))
    check[DType.uint8](Codec(BitPack(8), BitUnpack(8)))
    check[DType.uint16](Codec(BitPack(64), BitUnpack(64)))


def test_codecs_reference() raises:
    check[DType.uint32](
        Codec(
            ReferencePack(BLOCK, PlainPack()),
            ReferenceUnpack(BLOCK, PlainUnpack()),
        )
    )
    check[DType.uint16](
        Codec(
            ReferencePack(128, PlainPack()), ReferenceUnpack(128, PlainUnpack())
        )
    )
    check[DType.uint64](
        Codec(
            ReferencePack(256, BitPack(32)), ReferenceUnpack(256, BitUnpack(32))
        )
    )
    check[DType.uint8](
        Codec(ReferencePack(64, BitPack()), ReferenceUnpack(64, BitUnpack()))
    )


def test_codecs_frames_whole() raises:
    """A fit's segments are whole registers, a bit packer's whole groups, and
    encoding takes the segments that divide a block."""
    var one: List[UInt8] = [1]
    for size in [32, 100, 384, 2048]:
        var reference = Codec(
            ReferencePack(size, PlainPack()),
            ReferenceUnpack(size, PlainUnpack()),
        )
        with assert_raises(contains="InvalidError"):
            _ = reference.encode(BufferView(Span(one)))
    var bitpack = Codec(BitPack(12), BitUnpack(12))
    with assert_raises(contains="InvalidError"):
        _ = bitpack.encode(BufferView(Span(one)))


def test_codecs_runs() raises:
    check[DType.int64](
        Codec(
            RunsPack(PlainPack(), PlainPack()),
            RunsUnpack(PlainUnpack(), PlainUnpack()),
        )
    )
    check[DType.float32](
        Codec(
            RunsPack(PlainPack(), PlainPack()),
            RunsUnpack(PlainUnpack(), PlainUnpack()),
        )
    )
    check[DType.uint32](
        Codec(
            RunsPack(BitPack(), BitPack()), RunsUnpack(BitUnpack(), BitUnpack())
        )
    )


def test_cascade_float_planes() raises:
    """Gauges: readings that repeat, as runs, their values' byte planes
    apart."""
    check[DType.float64](
        Codec(
            RunsPack(SplitPack(), BitPack()),
            RunsUnpack(SplitUnpack(), BitUnpack()),
        )
    )
    check[DType.float32](
        Codec(
            RunsPack(SplitPack(), BitPack()),
            RunsUnpack(SplitUnpack(), BitUnpack()),
        )
    )


def test_codecs_arrays_plain_decodes_a_prefix() raises:
    """`PlainPacked` holds the values as they are: decoding three of a
    buffer's four values gives those three."""
    var four: List[Int64] = [5, 6, 7, 8]
    var arr = array[Int64Type](four, int64)
    var back = _decode[DType.int64](
        Codec(PlainPack(), PlainUnpack()), PlainPacked(arr.buffer, 3 * 8), 3
    )
    assert_equal(len(back), 3)
    assert_true(back[0] == 5 and back[2] == 7)


# --- dictionaries ----------------------------------------------------------------


def test_codecs_dictionary() raises:
    check[DType.float64](
        Codec(DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack()))
    )
    check[DType.int32](
        Codec(DictionaryPack(BitPack()), DictionaryUnpack(BitUnpack()))
    )
    check_bytes(
        DictionaryBytesPack(PlainPack()), DictionaryBytesUnpack(PlainUnpack())
    )
    check_bytes(
        DictionaryBytesPack(BitPack()), DictionaryBytesUnpack(BitUnpack())
    )


# --- byte strings ----------------------------------------------------------------


def _delta_zigzag_pack() -> Encoder[Compose[DiffScan, ZigzagEncode], BitPack]:
    return Encoder(Compose((DiffScan(), ZigzagEncode())), BitPack())


def _delta_zigzag_unpack[
    T: DType
]() -> Decoder[BitUnpack, Compose[ZigzagDecode, SumScan], Unsigned[T]]:
    return Decoder[BitUnpack, Compose[ZigzagDecode, SumScan], Unsigned[T]](
        BitUnpack(), Compose((ZigzagDecode(), SumScan()))
    )


def test_codecs_lengths() raises:
    check_bytes(LengthsPack(PlainPack()), LengthsUnpack(PlainUnpack()))
    check_bytes(
        LengthsPack(_delta_zigzag_pack()),
        LengthsUnpack(_delta_zigzag_unpack[DType.int32]()),
    )


def test_codecs_prefixes() raises:
    check_bytes(
        PrefixesPack(PlainPack(), PlainPack()),
        PrefixesUnpack(PlainUnpack(), PlainUnpack()),
    )
    check_bytes(
        PrefixesPack(
            Encoder(Compose((ZigzagEncode(),)), BitPack()), PlainPack()
        ),
        PrefixesUnpack(
            Decoder[BitUnpack, Compose[ZigzagDecode], DType.uint32](
                BitUnpack(), Compose((ZigzagDecode(),))
            ),
            PlainUnpack(),
        ),
    )


# --- RlePack, Parquet's RLE / bit-packed hybrid ---------------------------------


def test_codecs_rle() raises:
    check[DType.uint8](Codec(RlePack(8), RleUnpack(8)))
    check[DType.uint32](Codec(RlePack(32), RleUnpack(32)))
    check[DType.uint64](Codec(RlePack(64), RleUnpack(64)))


def test_codecs_rle_width_wider_than_the_type() raises:
    var data: List[UInt8] = [UInt8(0x02), 0x01]
    var out = List[UInt32]()
    with assert_raises(contains="of 33 bits"):
        _ = RleUnpack.read_runs[DType.uint32](Span(data), 33, 1, out)


def test_codecs_rle_packed_run_past_the_data() raises:
    var data: List[UInt8] = [UInt8(0x05), 0xFF]
    var out = List[UInt32]()
    with assert_raises(contains="truncated"):
        _ = RleUnpack.read_runs[DType.uint32](Span(data), 3, 16, out)


def test_codecs_rle_header_past_the_data() raises:
    var data: List[UInt8] = [UInt8(0x80)]
    var out = List[UInt32]()
    with assert_raises(contains="truncated"):
        _ = RleUnpack.read_runs[DType.uint32](Span(data), 1, 1, out)


def test_codecs_rle_codes_with_no_width() raises:
    var data = List[UInt8]()
    var codes = List[Int32]()
    with assert_raises(contains="no codes"):
        RleUnpack.read_codes[DType.int32](Span(data), 3, codes)


def test_codecs_rle_no_codes_with_no_width() raises:
    var data = List[UInt8]()
    var codes = List[Int32]()
    RleUnpack.read_codes[DType.int32](Span(data), 0, codes)
    assert_true(len(codes) == 0)


# --- codecs, after ClickHouse's codec chains -------------------------------------


def test_cascade_delta() raises:
    """CODEC(Delta): monotonic counters and timestamps."""
    check[DType.int64](
        Codec(_delta_zigzag_pack(), _delta_zigzag_unpack[DType.int64]())
    )
    check[DType.int32](
        Codec(_delta_zigzag_pack(), _delta_zigzag_unpack[DType.int32]())
    )


def _double_delta_unpack[
    T: DType
]() -> Decoder[BitUnpack, Compose[ZigzagDecode, SumScan, SumScan], Unsigned[T]]:
    return Decoder[
        BitUnpack, Compose[ZigzagDecode, SumScan, SumScan], Unsigned[T]
    ](BitUnpack(), Compose((ZigzagDecode(), SumScan(), SumScan())))


def test_cascade_double_delta() raises:
    """CODEC(DoubleDelta): timestamps at a near-constant interval."""
    check[DType.int64](
        Codec(
            Encoder(
                Compose((DiffScan(), DiffScan(), ZigzagEncode())), BitPack()
            ),
            _double_delta_unpack[DType.int64](),
        )
    )
    check[DType.int16](
        Codec(
            Encoder(
                Compose((DiffScan(), DiffScan(), ZigzagEncode())), BitPack()
            ),
            _double_delta_unpack[DType.int16](),
        )
    )


def _t64[
    T: DType
](size: Int, width: Int) -> Codec[
    Encoder[Compose[OrderedEncode], ReferencePack[BitPack]],
    Decoder[ReferenceUnpack[BitUnpack], Compose[OrderedDecode[T]], Unsigned[T]],
]:
    """Each value's ordered bits, less the least of its segment of `size`,
    packed at the bits each segment of `width` needs."""
    return Codec(
        Encoder(
            Compose((OrderedEncode(),)), ReferencePack(size, BitPack(width))
        ),
        Decoder[
            ReferenceUnpack[BitUnpack], Compose[OrderedDecode[T]], Unsigned[T]
        ](
            ReferenceUnpack(size, BitUnpack(width)),
            Compose((OrderedDecode[T](),)),
        ),
    )


def test_cascade_t64() raises:
    """CODEC(T64): frame of reference, then the bits the range needs."""
    check[DType.int64](_t64[DType.int64](BLOCK, BLOCK))
    check[DType.float32](_t64[DType.float32](BLOCK, BLOCK))


def test_cascade_t64_blocks() raises:
    """T64 a frame of 64 values at a time."""
    check[DType.int32](_t64[DType.int32](64, 64))
    check[DType.float64](_t64[DType.float64](64, 64))


def _delta_t64_pack() -> (
    Encoder[Compose[DiffScan, OrderedEncode], ReferencePack[BitPack]]
):
    return Encoder(
        Compose((DiffScan(), OrderedEncode())), ReferencePack(128, BitPack(32))
    )


def _delta_t64_unpack[
    T: DType
]() -> Decoder[
    ReferenceUnpack[BitUnpack], Compose[OrderedDecode[T], SumScan], Unsigned[T]
]:
    return Decoder[
        ReferenceUnpack[BitUnpack],
        Compose[OrderedDecode[T], SumScan],
        Unsigned[T],
    ](
        ReferenceUnpack(128, BitUnpack(32)),
        Compose((OrderedDecode[T](), SumScan())),
    )


def test_cascade_delta_t64() raises:
    """CODEC(Delta, T64) -- Parquet's DELTA_BINARY_PACKED after its first
    value."""
    check[DType.int64](
        Codec(_delta_t64_pack(), _delta_t64_unpack[DType.int64]())
    )
    check[DType.int32](
        Codec(_delta_t64_pack(), _delta_t64_unpack[DType.int32]())
    )


def test_cascade_double_delta_t64() raises:
    check[DType.int64](
        Codec(
            Encoder(
                Compose((DiffScan(), DiffScan(), OrderedEncode())),
                ReferencePack(128, BitPack(32)),
            ),
            Decoder[
                ReferenceUnpack[BitUnpack],
                Compose[OrderedDecode[DType.int64], SumScan, SumScan],
                DType.uint64,
            ](
                ReferenceUnpack(128, BitUnpack(32)),
                Compose((OrderedDecode[DType.int64](), SumScan(), SumScan())),
            ),
        )
    )


def _zigzag_bitpack_unpack[
    T: DType
]() -> Decoder[BitUnpack, Compose[ZigzagDecode], Unsigned[T]]:
    return Decoder[BitUnpack, Compose[ZigzagDecode], Unsigned[T]](
        BitUnpack(), Compose((ZigzagDecode(),))
    )


def test_cascade_zigzag() raises:
    """Small signed integers, packed."""
    check[DType.int8](
        Codec(
            Encoder(Compose((ZigzagEncode(),)), BitPack()),
            _zigzag_bitpack_unpack[DType.int8](),
        )
    )
    check[DType.int64](
        Codec(
            Encoder(Compose((ZigzagEncode(),)), BitPack()),
            _zigzag_bitpack_unpack[DType.int64](),
        )
    )


def test_cascade_low_cardinality_runs() raises:
    """LowCardinality whose codes repeat: Parquet's RLE_DICTIONARY."""
    check[DType.int64](
        Codec(
            DictionaryPack(RunsPack(BitPack(), BitPack())),
            DictionaryUnpack(RunsUnpack(BitUnpack(), BitUnpack())),
        )
    )
    check_bytes(
        DictionaryBytesPack(RunsPack(BitPack(), BitPack())),
        DictionaryBytesUnpack(RunsUnpack(BitUnpack(), BitUnpack())),
    )


def test_cascade_sparse() raises:
    """Long runs of one value, the rest framed and packed."""
    check[DType.int64](
        Codec(
            RunsPack(
                Encoder(
                    Compose((OrderedEncode(),)), ReferencePack(BLOCK, BitPack())
                ),
                BitPack(),
            ),
            RunsUnpack(
                Decoder[
                    ReferenceUnpack[BitUnpack],
                    Compose[OrderedDecode[DType.int64]],
                    DType.uint64,
                ](
                    ReferenceUnpack(BLOCK, BitUnpack()),
                    Compose((OrderedDecode[DType.int64](),)),
                ),
                BitUnpack(),
            ),
        )
    )


def test_cascade_strings() raises:
    """Byte strings: their lengths, or the prefixes they share, delta
    packed."""
    check_bytes(
        LengthsPack(_delta_t64_pack()),
        LengthsUnpack(_delta_t64_unpack[DType.int32]()),
    )
    check_bytes(
        PrefixesPack(_delta_zigzag_pack(), PlainPack()),
        PrefixesUnpack(_delta_zigzag_unpack[DType.int32](), PlainUnpack()),
    )


def test_cascade_delta_byte_array() raises:
    """Parquet's DELTA_BYTE_ARRAY: the prefixes delta-packed, the suffixes
    as DELTA_LENGTH_BYTE_ARRAY."""
    check_bytes(
        PrefixesPack(_delta_t64_pack(), _delta_t64_pack()),
        PrefixesUnpack(
            _delta_t64_unpack[DType.int32](), _delta_t64_unpack[DType.int32]()
        ),
    )


# --- what the docstrings show -------------------------------------------------


def _bytes(bytes: Buffer[mut=False], size: Int) -> List[UInt8]:
    return List(bytes.view[DType.uint8](0, size).as_span())


def test_codecs_docstring_examples() raises:
    var small: List[UInt8] = [0, 1, 3]
    var bitpack = Codec(BitPack(), BitUnpack()).encode(BufferView(Span(small)))
    assert_true(_bytes(bitpack.bytes, bitpack.size) == [UInt8(0x34), 0])
    assert_true(
        len(bitpack.widths) == 1 and bitpack.widths.get[DType.uint8](0) == 2
    )

    var signed: List[Int8] = [0, -1, 1, -2]
    var zigzag = Codec(_zigzag_pack(), _zigzag_unpack[DType.int8]()).encode(
        BufferView(Span(signed))
    )
    assert_true(_bytes(zigzag.bytes, zigzag.size) == [UInt8(0), 1, 2, 3])

    var clustered: List[UInt8] = [100, 101, 103]
    var reference = Codec(
        ReferencePack(BLOCK, PlainPack()), ReferenceUnpack(BLOCK, PlainUnpack())
    ).encode(BufferView(Span(clustered)))
    assert_true(
        _bytes(reference.inner.bytes, reference.inner.size) == [UInt8(0), 1, 3]
    )
    assert_true(
        len(reference.fits) == 1 and reference.fits.get[DType.uint8](0) == 100
    )

    var pairs: List[UInt16] = [0x0102, 0x0304]
    var bss = Codec(SplitPack(), SplitUnpack()).encode(BufferView(Span(pairs)))
    assert_true(_bytes(bss.bytes, bss.size) == [UInt8(2), 4, 1, 3])

    var repeated: List[UInt8] = [5, 5, 9, 5]
    var dictionary = Codec(
        DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack())
    ).encode(BufferView(Span(repeated)))
    assert_true(
        _bytes(dictionary.values.bytes, dictionary.values.size) == [UInt8(5), 9]
    )
    assert_true(dictionary.codes.bytes.view[DType.uint32]()[2] == 1)

    var held: List[UInt8] = [5, 5, 5, 1, 2, 2]
    var runs = Codec(
        RunsPack(PlainPack(), PlainPack()),
        RunsUnpack(PlainUnpack(), PlainUnpack()),
    ).encode(BufferView(Span(held)))
    assert_true(_bytes(runs.values.bytes, runs.values.size) == [UInt8(5), 1, 2])
    var ends = runs.ends.bytes.view[DType.uint64]()
    assert_true(ends[0] == 3 and ends[1] == 4 and ends[2] == 6)

    # "ab", "abc", "ab"
    var offsets: List[Int32] = [0, 2, 5, 7]
    var data = List(String("ababcab").as_bytes())
    var abc = Span(offsets)[:3]
    var lengths = LengthsPack(PlainPack()).pack(abc, Span(data))
    assert_true(
        _bytes(lengths.bytes, lengths.size) == List(String("ababc").as_bytes())
    )
    var sizes = lengths.lengths.bytes.view[DType.int32]()
    assert_true(sizes[0] == 2 and sizes[1] == 3)
    var prefixes = PrefixesPack(PlainPack(), PlainPack()).pack(abc, Span(data))
    var shared = prefixes.prefixes.bytes.view[DType.int32]()
    assert_true(shared[0] == 0 and shared[1] == 2)
    assert_true(
        _bytes(prefixes.suffixes.bytes, prefixes.suffixes.size)
        == List(String("abc").as_bytes())
    )
    var codes = DictionaryBytesPack(PlainPack()).pack(Span(offsets), Span(data))
    assert_true(
        _bytes(codes.strings.bytes, codes.strings.size)
        == List(String("ababc").as_bytes())
    )
    var named = codes.codes.bytes.view[DType.uint32]()
    assert_true(named[0] == 0 and named[1] == 1 and named[2] == 0)


def test_codecs_parquet_delta_binary_packed() raises:
    """Parquet's own example, 1 to 5 as DELTA_BINARY_PACKED: after the
    first value, the deltas through `With[SubKernel](first)`, `DiffScan`,
    `OrderedEncode` and `ReferencePack(128, BitPack(32))`."""
    var dbp = Codec(
        Encoder(
            Compose(
                (With[SubKernel, DType.int32](1), DiffScan(), OrderedEncode())
            ),
            ReferencePack(128, BitPack(32)),
        ),
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[
                OrderedDecode[DType.int32],
                SumScan,
                With[AddKernel, DType.int32],
            ],
            DType.uint32,
        ](
            ReferenceUnpack(128, BitUnpack(32)),
            Compose(
                (
                    OrderedDecode[DType.int32](),
                    SumScan(),
                    With[AddKernel, DType.int32](1),
                )
            ),
        ),
    )
    var deltas: List[Int32] = [2, 3, 4, 5]
    var packed = dbp.encode(BufferView(Span(deltas)))
    # The smallest delta's key, and one width of 0 -- so no bytes.
    assert_true(packed.fits.get[DType.uint32](0) == 0x80000001)
    assert_true(
        len(packed.inner.widths) == 1
        and packed.inner.widths.get[DType.uint8](0) == 0
    )
    assert_equal(packed.inner.size, 0)
    assert_true(_decode[DType.int32](dbp, packed^, 4) == [Int32(2), 3, 4, 5])


# --- typed arrays at the edges ------------------------------------------------


def test_codecs_arrays() raises:
    """A column goes in as an array's values, and comes out into any view:
    a new array's buffer, or a buffer at an offset."""
    var values = [Int64(i * 7 - (i % 13) * 1000) for i in range(3000)]
    var n = len(values)
    var arr = array[Int64Type](values.copy(), int64)
    var codec = Codec(_delta_zigzag_pack(), _delta_zigzag_unpack[DType.int64]())
    var packed = codec.encode(arr.values())
    var buffer = Buffer.alloc_uninit[DType.int64](n)
    codec.decode[DType.int64](packed.copy(), buffer.view[DType.int64](0, n))
    var back = Int64Array(length=n, buffer=buffer^.to_immutable())
    assert_true(back == arr)
    var wide = List[Int64](length=n + 5, fill=0)
    codec.decode[DType.int64](packed^, BufferView(Span(wide))[5:])
    assert_true(wide[0] == 0 and wide[5 + 42] == values[42])
    # A sliced array is encoded where it lies.
    var tail = codec.encode(arr.slice(1000, 500).values())
    assert_true(_decode[DType.int64](codec, tail^, 500)[0] == values[1000])


# --- a packed part altered -----------------------------------------------------


def _refused[
    T: DType, E: PackKernel, D: UnpackKernel
](codec: Codec[E, D], var packed: E.Packed, count: Int) raises:
    """`packed` is refused: a `CorruptError`, from `start` or `check`."""
    with assert_raises(contains="CorruptError"):
        _ = _decode[T](codec, packed^, count)


def _with[
    T: DType
](bytes: Buffer[mut=False], length: Int, at: Int, value: Scalar[T]) -> Buffer[
    mut=False
]:
    """A copy of the first `length` values of `T` in `bytes`, value `at` set
    to `value`."""
    var out = Buffer.alloc_uninit[T](length)
    out.extend(bytes.view[T](0, length), 0, length)
    out.view[T]().unsafe_set(at, value)
    return out^.to_immutable()


def test_codecs_refuse_cut_packed_parts() raises:
    comptime U = DType.uint64
    var sorted = [UInt64(i * 3) for i in range(301)]
    var n = len(sorted)
    var values = BufferView(Span(sorted))

    var plain = Codec(PlainPack(), PlainUnpack())
    var p = plain.encode(values)
    _refused[U](plain, PlainPacked(p.bytes, p.size - 1), n)

    var split = Codec(SplitPack(), SplitUnpack())
    var s = split.encode(values)
    _refused[U](split, SplitPacked(s.bytes, s.size - 1), n)

    var bits = Codec(BitPack(64), BitUnpack(64))
    var b = bits.encode(values)
    var segments = len(b.widths)
    _refused[U](
        bits,
        BitPacked(b.bytes, b.size, PerSegment(b.widths.values, segments - 1)),
        n,
    )
    _refused[U](
        bits,
        BitPacked(
            b.bytes,
            b.size,
            PerSegment(
                _with[DType.uint8](b.widths.values, segments, 0, 65), segments
            ),
        ),
        n,
    )
    _refused[U](bits, BitPacked(b.bytes, b.size - 1, b.widths.copy()), n)

    var fitted = Codec(
        ReferencePack(64, PlainPack()), ReferenceUnpack(64, PlainUnpack())
    )
    var f = fitted.encode(values)
    _refused[U](
        fitted,
        FittedPacked(
            PerSegment(f.fits.values, len(f.fits) - 1), f.inner.copy()
        ),
        n,
    )

    var rle = Codec(RlePack(16), RleUnpack(16))
    var r = rle.encode(values)
    _refused[U](rle, RlePacked(r.bytes, 3), n)

    var runs = Codec(
        RunsPack(PlainPack(), PlainPack()),
        RunsUnpack(PlainUnpack(), PlainUnpack()),
    )
    var u = runs.encode(values)
    # The last run ends one short of the column.
    var ends = u.ends.bytes.view[U](0, u.runs)
    var short = Buffer.alloc_uninit[U](u.runs)
    short.extend(ends, 0, u.runs)
    short.view[U]().unsafe_set(u.runs - 1, UInt64(n - 1))
    _refused[U](
        runs,
        RunsPacked(
            u.runs,
            u.values.copy(),
            PlainPacked(short^.to_immutable(), u.ends.size),
        ),
        n,
    )
    _refused[U](runs, RunsPacked(u.runs - 1, u.values.copy(), u.ends.copy()), n)


def test_codecs_refuse_a_size_past_its_buffer() raises:
    """The runs' values claim the bytes of every run, in a buffer of 64."""
    comptime U = DType.uint64
    var sorted = [UInt64(i * 3) for i in range(301)]
    var runs = Codec(
        RunsPack(PlainPack(), PlainPack()),
        RunsUnpack(PlainUnpack(), PlainUnpack()),
    )
    var u = runs.encode(BufferView(Span(sorted)))
    var small = Buffer.alloc_zeroed[U](1).to_immutable()
    _refused[U](
        runs,
        RunsPacked(u.runs, PlainPacked(small, u.values.size), u.ends.copy()),
        len(sorted),
    )


def test_codecs_refuse_a_count_past_its_buffer() raises:
    """100 segments' widths claimed from a buffer of 64."""
    comptime U = DType.uint8
    var values = [UInt8(i % 7) for i in range(800)]
    var bits = Codec(BitPack(8), BitUnpack(8))
    var b = bits.encode(BufferView(Span(values)))
    assert_equal(len(b.widths), 100)
    var small = Buffer.alloc_zeroed[U](1).to_immutable()
    # The count check, not the byte-size check behind it.
    with assert_raises(contains="widths in a buffer of 64 bytes"):
        _ = _decode[U](
            bits,
            BitPacked(b.bytes, b.size, PerSegment(small, 100)),
            len(values),
        )


def test_packers_refuse_blocks_of_another_size() raises:
    """`store` takes `BLOCK` values a block, and fewer in the last only: a
    larger block is refused, and so is one after a short block."""
    comptime U = DType.uint64
    var values = [UInt64(i) for i in range(2 * BLOCK)]
    var block = BufferView(Span(values))
    var fitted = ReferencePack(64, BitPack(64))
    var state = fitted.start[U](len(values))
    with assert_raises(contains="InvalidError"):
        fitted.store[U](state, block)
    var bits = BitPack(64)
    var packing = bits.start[U](len(values))
    bits.store[U](packing, block[:100])
    with assert_raises(contains="InvalidError"):
        bits.store[U](packing, block[100:200])


def test_codecs_refuse_cut_packed_strings() raises:
    comptime U = DType.uint64
    var sorted = [UInt64(i * 3) for i in range(301)]
    var dictionary = Codec(
        DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack())
    )
    var d = dictionary.encode(BufferView(Span(sorted)))
    # The last value is gone; the last code still names it.
    _refused[U](
        dictionary,
        DictionaryPacked(
            PlainPacked(d.values.bytes, d.values.size - 8), d.codes.copy()
        ),
        len(sorted),
    )

    # "", "abc", "", "abd", "abd", "abdx", "b"
    var strings = _strings()[2].copy()
    var n = len(strings)
    var offsets = List[Int32]()
    var data = List[UInt8]()
    _column(strings, offsets, data)
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()

    var lengths = LengthsUnpack(PlainUnpack())
    var l = LengthsPack(PlainPack()).pack(Span(offsets), Span(data))
    with assert_raises(contains="CorruptError"):
        lengths.unpack(
            LengthsPacked(l.bytes, l.size - 1, l.lengths.copy()),
            n,
            ends,
            bytes,
        )
    with assert_raises(contains="CorruptError"):
        lengths.unpack(
            LengthsPacked(
                l.bytes,
                l.size,
                PlainPacked(
                    _with[DType.int32](l.lengths.bytes, n, 0, -1),
                    l.lengths.size,
                ),
            ),
            n,
            ends,
            bytes,
        )

    var prefixes = PrefixesUnpack(PlainUnpack(), PlainUnpack())
    var p = PrefixesPack(PlainPack(), PlainPack()).pack(
        Span(offsets), Span(data)
    )
    # The first string is empty, so the second shares no prefix with it.
    with assert_raises(contains="CorruptError"):
        prefixes.unpack(
            PrefixesPacked(
                PlainPacked(
                    _with[DType.int32](p.prefixes.bytes, n, 1, 1),
                    p.prefixes.size,
                ),
                p.suffixes.copy(),
            ),
            n,
            ends,
            bytes,
        )
    with assert_raises(contains="CorruptError"):
        prefixes.unpack(
            PrefixesPacked(
                p.prefixes.copy(),
                LengthsPacked(
                    p.suffixes.bytes,
                    p.suffixes.size,
                    PlainPacked(
                        p.suffixes.lengths.bytes, p.suffixes.lengths.size - 4
                    ),
                ),
            ),
            n,
            ends,
            bytes,
        )

    var c = DictionaryBytesPack(PlainPack()).pack(Span(offsets), Span(data))
    # Five distinct strings, so code 5 names none.
    with assert_raises(contains="CorruptError"):
        DictionaryBytesUnpack(PlainUnpack()).unpack(
            DictionaryBytesPacked(
                c.strings.copy(),
                PlainPacked(
                    _with[DType.uint32](c.codes.bytes, n, 0, 5), c.codes.size
                ),
            ),
            n,
            ends,
            bytes,
        )
