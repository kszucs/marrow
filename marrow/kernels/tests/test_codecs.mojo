# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every codec round-trips every column bit for bit: stream stages composed,
packers alone and nested, the codecs built from both, and the byte-string
packers; and a packed part altered is refused."""

from std.builtin.rebind import rebind_var
from std.memory import bitcast
from std.sys import bit_width_of, size_of
from std.testing import assert_equal, assert_raises, assert_true
from std.utils.numerics import inf, neg_inf

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
from ...kernels.compose import (
    Auto,
    ChoicePack,
    ChoiceUnpack,
    Codec,
    Compose,
    Decoder,
    Encoder,
    With,
)
from ...kernels.core import PackKernel, UnpackKernel
from ...kernels.dictionary import (
    DictionaryBytesPack,
    DictionaryBytesPacked,
    DictionaryBytesUnpack,
    DictionaryPack,
    DictionaryPacked,
    DictionaryUnpack,
)
from ...kernels.float import (
    Alp,
    AlpPack,
    AlpRdPack,
    AlpRdUnpack,
    AlpUnpack,
)
from ...kernels.numeric import AddKernel, SubKernel
from ...kernels.packing import (
    BLOCK,
    BitPack,
    BitPacked,
    BitUnpack,
    ConstantPack,
    ConstantPacked,
    ConstantUnpack,
    FittedPacked,
    FrequencyPack,
    FrequencyUnpack,
    FsstPack,
    FsstPacked,
    FsstUnpack,
    LengthsPack,
    LengthsPacked,
    LengthsUnpack,
    PlainPack,
    PlainPacked,
    PlainUnpack,
    PforPack,
    PforUnpack,
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


def _round_trips[
    T: DType, E: PackKernel, D: UnpackKernel
](codec: Codec[E, D], values: List[Scalar[T]]) raises -> Bool:
    var packed = codec.encode(BufferView(Span(values)))
    return _same(_decode[T](codec, packed^, len(values)), values)


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


def check_bytes[
    L: PackKernel, M: UnpackKernel
](packer: FsstPack[L], unpacker: FsstUnpack[M]) raises:
    for ref values in _strings():
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        _fsst_round_trips(packer, unpacker, offsets, data)


def _fsst_round_trips[
    L: PackKernel, M: UnpackKernel
](
    packer: FsstPack[L],
    unpacker: FsstUnpack[M],
    offsets: List[Int32],
    data: List[UInt8],
) raises:
    comptime assert L.Packed == M.Packed, "the unpacker takes what is packed"
    var packed = packer.pack(Span(offsets), Span(data))
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    unpacker.unpack(
        rebind_var[FsstPacked[M.Packed]](packed^),
        len(offsets) - 1,
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


def _fsst_pack() -> FsstPack[Encoder[Compose[DiffScan, ZigzagEncode], BitPack]]:
    """FSST with the code strings' lengths delta-packed."""
    return FsstPack(_delta_zigzag_pack())


def _fsst_unpack() -> (
    FsstUnpack[Decoder[BitUnpack, Compose[ZigzagDecode, SumScan], DType.uint32]]
):
    return FsstUnpack(_delta_zigzag_unpack[DType.int32]())


def _binary(mut offsets: List[Int32], mut data: List[UInt8]):
    """Every byte value, a string each, and one string of all 256."""
    offsets.append(Int32(len(data)))
    for b in range(256):
        data.append(UInt8(b))
        offsets.append(Int32(len(data)))
    for b in range(256):
        data.append(UInt8(255 - b))
    offsets.append(Int32(len(data)))


def _text(mut offsets: List[Int32], mut data: List[UInt8], strings: Int):
    """Strings of a kilobyte of words picked at random, 40 KB of them past
    the sample's 16 KB at `strings = 40`."""
    var words = [
        "the ",
        "quick ",
        "brown ",
        "fox ",
        "jumps ",
        "over ",
        "lazy ",
        "dog ",
        "and ",
        "keeps ",
        "running ",
        "through ",
        "fields ",
        "of ",
        "golden ",
        "wheat ",
        "1984 ",
        "2026 ",
        "Mojo ",
        "Arrow ",
    ]
    var rng = Rng(11)
    offsets.append(Int32(len(data)))
    for _ in range(strings):
        var start = len(data)
        while len(data) - start < 1000:
            data.extend(words[Int(rng.next() % 20)].as_bytes())
        offsets.append(Int32(len(data)))


def _noise(mut offsets: List[Int32], mut data: List[UInt8]):
    """Strings of up to eleven random letters and digits."""
    var letters = String("abcdefghijklmnopqrstuvwxyz0123456789").as_bytes()
    var rng = Rng(5)
    offsets.append(Int32(len(data)))
    for _ in range(2000):
        for _ in range(Int(rng.next() % 12)):
            data.append(letters[Int(rng.next() % 36)])
        offsets.append(Int32(len(data)))


def _fsst_columns() -> List[Tuple[List[Int32], List[UInt8]]]:
    """The `row-...` strings, every byte value, text and noise, each as
    offsets and bytes."""
    var columns = List[Tuple[List[Int32], List[UInt8]]]()
    for column in range(4):
        var offsets = List[Int32]()
        var data = List[UInt8]()
        if column == 0:
            _column(_strings()[3], offsets, data)
        elif column == 1:
            _binary(offsets, data)
        elif column == 2:
            _text(offsets, data, 40)
        else:
            _noise(offsets, data)
        columns.append((offsets^, data^))
    return columns^


def test_codecs_fsst() raises:
    check_bytes(
        FsstPack(PlainPack()),
        FsstUnpack(PlainUnpack()),
    )
    var packer = _fsst_pack()
    var unpacker = _fsst_unpack()
    check_bytes(packer, unpacker)
    var columns = _fsst_columns()
    assert_true(len(columns[2][1]) >= 40_000)
    for ref column in columns:
        _fsst_round_trips(packer, unpacker, column[0], column[1])
    var empty = List[Int32](length=1001, fill=0)
    _fsst_round_trips(packer, unpacker, empty, List[UInt8]())


def test_codecs_fsst_symbols() raises:
    var packer = _fsst_pack()
    var columns = _fsst_columns()
    var counts = List[Int]()
    for i in range(len(columns)):
        var offsets = columns[i][0].copy()
        var data = columns[i][1].copy()
        var p = packer.pack(Span(offsets), Span(data))
        assert_true(0 < p.count and p.count <= 255)
        assert_true(len(p.symbols) >= 8 * p.count)
        for c in range(p.count):
            var length = p.lengths.unsafe_get(c)
            assert_true(length >= 1 and length <= 8)
        counts.append(p.count)
        # The `row-...` strings, to under half their bytes as offsets and
        # data.
        if i == 0:
            var raw = len(data) + 4 * len(offsets)
            assert_true(
                2 * packer.nbytes(p) < raw,
                String(packer.nbytes(p), " of ", raw),
            )
    # Every byte value, each the most common byte, fills the table.
    assert_true(counts[1] == 255, String(counts[1]))


def test_codecs_fsst_is_deterministic() raises:
    var packer = _fsst_pack()
    var offsets = List[Int32]()
    var data = List[UInt8]()
    _text(offsets, data, 40)
    var a = packer.pack(Span(offsets), Span(data))
    var b = packer.pack(Span(offsets), Span(data))
    assert_true(a.count == b.count)
    assert_true(a.symbols == b.symbols and a.lengths == b.lengths)
    assert_true(a.codes.size == b.codes.size and a.codes.bytes == b.codes.bytes)
    assert_true(a.codes.lengths.widths.values == b.codes.lengths.widths.values)
    # The bit-packed bytes past their size are scratch.
    var size = a.codes.lengths.size
    assert_true(size == b.codes.lengths.size)
    assert_true(
        a.codes.lengths.bytes.view(0, size).as_span()
        == b.codes.lengths.bytes.view(0, size).as_span()
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


# --- one value, and the values that are not -----------------------------------


def _constant[T: DType](value: Scalar[T]) raises:
    """3,000 copies of `value` round-trip bit for bit, and are packed as the
    one value; the empty column is packed as nothing."""
    var codec = Codec(ConstantPack(), ConstantUnpack())
    var values = List[Scalar[T]](length=3_000, fill=value)
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(codec.encoder.nbytes[T](packed), size_of[Scalar[T]]())
    assert_true(_same(_decode[T](codec, packed^, len(values)), values))
    var empty = List[Scalar[T]]()
    var none = codec.encode(BufferView(Span(empty)))
    assert_equal(codec.encoder.nbytes[T](none), 0)
    assert_true(_same(_decode[T](codec, none^, 0), empty))


def test_codecs_constant() raises:
    comptime for T in [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    ]:
        _constant[T](0)
        _constant[T](Scalar[T].MAX_FINITE)
        comptime if T.is_floating_point():
            # A NaN with a payload, and -0.0: each kept by its bits.
            var nan = bitcast[Unsigned[T]](Scalar[T](0.0) / Scalar[T](0.0))
            _constant[T](bitcast[T](nan | 0x123))
            _constant[T](-0.0)


def test_codecs_constant_refuses_a_second_value() raises:
    var codec = Codec(ConstantPack(), ConstantUnpack())
    var values: List[Int32] = [4, 4, 5]
    with assert_raises(contains="InvalidError"):
        _ = codec.encode(BufferView(Span(values)))
    # Equal as values, not as bits.
    var zeros: List[Float64] = [0.0, -0.0]
    with assert_raises(contains="InvalidError"):
        _ = codec.encode(BufferView(Span(zeros)))


def test_codecs_frequency() raises:
    var codec = Codec(
        FrequencyPack(64, ConstantPack()),
        FrequencyUnpack(64, ConstantUnpack()),
    )
    comptime for T in [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    ]:
        check[T](codec)
    check[DType.int64](
        Codec(
            FrequencyPack(BLOCK, ConstantPack()),
            FrequencyUnpack(BLOCK, ConstantUnpack()),
        )
    )


def test_codecs_frequency_keeps_the_most_common_value() raises:
    var codec = Codec(
        FrequencyPack(64, ConstantPack()),
        FrequencyUnpack(64, ConstantUnpack()),
    )
    var values: List[Int64] = [7, 7, 3, 7, 9]
    var packed = codec.encode(BufferView(Span(values)))
    assert_true(len(packed.fits) == 1 and packed.fits.get[DType.uint64](0) == 7)
    assert_true(len(packed.ends) == 1 and packed.ends.get[DType.uint32](0) == 2)
    var positions = packed.positions.bytes.view[DType.uint16]()
    assert_true(
        packed.positions.size == 4 and positions[0] == 2 and positions[1] == 4
    )
    var kept = packed.values.bytes.view[DType.int64]()
    assert_true(packed.values.size == 16 and kept[0] == 3 and kept[1] == 9)
    # The inner packer sees five zeros.
    assert_true(packed.inner.value == 0 and packed.inner.count == 5)
    assert_equal(codec.encoder.nbytes[DType.int64](packed), 8 + 4 + 4 + 16 + 8)
    assert_true(_decode[DType.int64](codec, packed^, 5) == values)
    # A tie goes to the smaller bits.
    var tied: List[Int64] = [5, 3, 5, 3]
    var t = codec.encode(BufferView(Span(tied)))
    assert_true(t.fits.get[DType.uint64](0) == 3)


def _pfor[
    T: DType
]() -> Codec[
    Encoder[Compose[OrderedEncode], ReferencePack[PforPack]],
    Decoder[
        ReferenceUnpack[PforUnpack],
        Compose[OrderedDecode[T]],
        Unsigned[T],
    ],
]:
    """Patched frame of reference: each value's ordered bits less the least
    of its block, at the width that packs each segment of 128 smallest, the
    values wider than it patched."""
    return Codec(
        Encoder(
            Compose((OrderedEncode(),)),
            ReferencePack(BLOCK, PforPack(128, BitPack(128))),
        ),
        Decoder[
            ReferenceUnpack[PforUnpack],
            Compose[OrderedDecode[T]],
            Unsigned[T],
        ](
            ReferenceUnpack(BLOCK, PforUnpack(128, BitUnpack(128))),
            Compose((OrderedDecode[T](),)),
        ),
    )


def test_codecs_pfor() raises:
    comptime for T in [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]:
        check[T](_pfor[T]())


def _outliers() -> List[UInt64]:
    """1,024 values below 16, and five of `2**40`: one in the first segment
    of 128, two in the second, one in the sixth and one in the last."""
    var values = [UInt64(i % 16) for i in range(1024)]
    for p in [3, 130, 131, 700, 1023]:
        values[p] = UInt64(1) << 40
    return values^


def _pfor_alone() -> Codec[PforPack, PforUnpack]:
    return Codec(
        PforPack(128, BitPack(128)),
        PforUnpack(128, BitUnpack(128)),
    )


def test_codecs_pfor_patches_outliers() raises:
    var values = _outliers()
    var codec = _pfor_alone()
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(len(packed.fits), 8)
    for s in range(8):
        assert_true(packed.fits.get[DType.uint64](s) == 4)
    for s in range(len(packed.inner.widths)):
        assert_true(packed.inner.widths.get[DType.uint8](s) <= 4)
    var ends: List[UInt32] = [1, 3, 3, 3, 3, 4, 4, 5]
    for s in range(8):
        assert_true(packed.ends.get[DType.uint32](s) == ends[s])
    var positions = packed.positions.bytes.view[DType.uint16]()
    var within: List[UInt16] = [3, 2, 3, 60, 127]
    assert_equal(packed.positions.size, 10)
    assert_equal(packed.values.size, 40)
    for k in range(5):
        assert_true(positions[k] == within[k])
        assert_true(
            packed.values.bytes.view[DType.uint64]()[k] == UInt64(1) << 40
        )
    assert_true(_decode[DType.uint64](codec, packed^, len(values)) == values)


# --- floats as their decimal digits -------------------------------------------


def _alp[T: DType]() -> Codec[AlpPack, AlpUnpack[T]]:
    """Floats as integers in segments of a block, the integers framed and
    packed."""
    comptime I = Alp.Encoded[T]
    return Codec(
        AlpPack(
            BLOCK,
            Encoder(
                Compose((OrderedEncode(),)), ReferencePack(BLOCK, BitPack())
            ),
        ),
        AlpUnpack[T](
            BLOCK,
            Decoder[
                ReferenceUnpack[BitUnpack],
                Compose[OrderedDecode[I]],
                Unsigned[I],
            ](
                ReferenceUnpack(BLOCK, BitUnpack()),
                Compose((OrderedDecode[I](),)),
            ),
        ),
    )


def test_codecs_alp() raises:
    check[DType.float64](_alp[DType.float64]())
    check[DType.float32](_alp[DType.float32]())


def _edge_floats[T: DType](big: Scalar[T]) -> List[Scalar[T]]:
    """`0.0` and `-0.0`, two NaN payloads, the infinities, the subnormals,
    the smallest normal, the extremes, and values with more digits than an
    integer holds."""
    comptime U = Unsigned[T]
    comptime M = 52 if T == DType.float64 else 23
    comptime sign = Scalar[U](1) << Scalar[U](bit_width_of[T]() - 1)
    var quiet = bitcast[U](inf[T]()) | (Scalar[U](1) << Scalar[U](M - 1))
    return [
        0.0,
        -0.0,
        bitcast[T](quiet),
        bitcast[T](quiet | sign | 0x123),
        inf[T](),
        neg_inf[T](),
        bitcast[T](Scalar[U](1)),
        bitcast[T]((Scalar[U](1) << Scalar[U](M)) - 1),
        bitcast[T](Scalar[U](1) << Scalar[U](M)),
        Scalar[T].MAX_FINITE,
        big,
        3.141592653589793,
        Scalar[T](1) / 3,
        0.1,
        9007199254740994.0,
        -12.5,
    ]


def _alp_edge_cases[T: DType](big: Scalar[T]) raises:
    """The values no decimal integer gives back -- `-0.0`, NaN, the
    infinities, the subnormals, the extremes and the values with more digits
    than the integer holds -- are the exceptions, and no other value is."""
    comptime U = Unsigned[T]
    var values = _edge_floats[T](big)
    # All but 0.0 and 0.1. The smallest normal rounds to 0 and `MAX_FINITE`
    # past the limit under every pair; and the pairs that keep -12.5 as well
    # give the integers a range six bits wider, which on 16 values costs
    # more than its exception.
    var exceptions: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
    var codec = _alp[T]()
    var packed = codec.encode(BufferView(Span(values)))
    assert_true(len(packed.ends) == 1)
    assert_equal(Int(packed.ends.get[DType.uint32](0)), len(exceptions))
    var positions = packed.positions.bytes.view[DType.uint16]()
    var kept = packed.values.bytes.view[T]()
    for k in range(len(exceptions)):
        assert_equal(Int(positions[k]), exceptions[k])
        assert_true(bitcast[U](kept[k]) == bitcast[U](values[exceptions[k]]))
    assert_true(_same(_decode[T](codec, packed^, len(values)), values))


def test_codecs_alp_edge_cases() raises:
    _alp_edge_cases[DType.float64](1e300)
    _alp_edge_cases[DType.float32](3e38)


def _prices() -> List[Float64]:
    """3,000 prices in quarters."""
    return [Float64(i % 1000) * 0.25 + 20.0 for i in range(3000)]


def test_codecs_alp_decimals_need_no_exceptions() raises:
    var codec = _alp[DType.float64]()
    var example: List[Float64] = [1.5, 2.25, 0.5]
    var packed = codec.encode(BufferView(Span(example)))
    assert_true(packed.ends.get[DType.uint32](0) == 0)
    var v = SIMD[DType.float64, 4](1.5, 2.25, 0.5, 0.0)
    var ints = SIMD[DType.int64, 4](150, 225, 50, 0)
    assert_true(Alp.encode[DType.float64, 4](v, 2 << 8) == ints)
    assert_true(
        Alp.encode[DType.float64, 4](v, packed.fits.get[DType.uint64](0))
        == ints
    )
    assert_true(_same(_decode[DType.float64](codec, packed^, 3), example))
    # Every segment's pair answers what (2, 0) does.
    var prices = _prices()
    var p = codec.encode(BufferView(Span(prices)))
    assert_equal(len(p.ends), 3)
    for s in range(3):
        assert_true(p.ends.get[DType.uint32](s) == 0)
    for i in range(len(prices)):
        var fit = p.fits.get[DType.uint64](i // BLOCK)
        var x = SIMD[DType.float64, 1](prices[i])
        assert_true(
            Alp.encode[DType.float64, 1](x, fit)
            == Alp.encode[DType.float64, 1](x, 2 << 8)
        )
    assert_true(_same(_decode[DType.float64](codec, p^, 3000), prices))


def test_codecs_alp_refits_every_hundred_segments() raises:
    """The pairs are searched for again on the 101st segment, so its values,
    in eighths where those before were in quarters, need no exception."""
    var n = 101 * BLOCK
    var values = List[Float64](capacity=n)
    for i in range(n):
        if i < 100 * BLOCK:
            values.append(Float64(i % 1000) * 0.25 + 20.0)
        else:
            values.append(Float64(i % 1000) * 0.125 + 20.0)
    var codec = _alp[DType.float64]()
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(len(packed.ends), 101)
    assert_true(packed.ends.get[DType.uint32](100) == 0)
    assert_true(_same(_decode[DType.float64](codec, packed^, n), values))


def _alp_rd() -> (
    Codec[
        AlpRdPack[DictionaryPack[BitPack], BitPack],
        AlpRdUnpack[DictionaryUnpack[BitUnpack], BitUnpack],
    ]
):
    """Floats cut in segments of a block, the left parts through a dictionary
    and the right bit-packed."""
    return Codec(
        AlpRdPack(BLOCK, DictionaryPack(BitPack()), BitPack()),
        AlpRdUnpack(BLOCK, DictionaryUnpack(BitUnpack()), BitUnpack()),
    )


def test_codecs_alp_rd() raises:
    check[DType.float64](_alp_rd())
    check[DType.float32](_alp_rd())


def test_codecs_alp_rd_keeps_every_bit() raises:
    var codec = _alp_rd()
    assert_true(_round_trips(codec, _edge_floats[DType.float64](1e300)))
    assert_true(_round_trips(codec, _edge_floats[DType.float32](3e38)))
    # The docstring's example, part by part: one cut, two left parts coded
    # 0 and 1 in a bit each, and right parts of no bits.
    var example: List[Float64] = [1.0, 1.5]
    var packed = codec.encode(BufferView(Span(example)))
    assert_true(len(packed.cuts) == 1 and packed.cuts.get[DType.uint8](0) == 48)
    var left = packed.left.values.bytes.view[DType.uint16]()
    assert_equal(packed.left.values.size, 4)
    assert_true(left[0] == 0x3FF0 and left[1] == 0x3FF8)
    assert_true(
        _bytes(packed.left.codes.bytes, packed.left.codes.size) == [UInt8(2)]
    )
    assert_true(packed.left.codes.widths.get[DType.uint8](0) == 1)
    assert_equal(packed.right.size, 0)
    assert_true(packed.right.widths.get[DType.uint8](0) == 0)
    # The cut, two left parts of two bytes, a byte of codes, and the width
    # of each bit packer's one segment.
    assert_equal(codec.encoder.nbytes[DType.float64](packed), 8)
    assert_true(_same(_decode[DType.float64](codec, packed^, 2), example))


# --- packers chosen per segment ---------------------------------------------


def _choice[T: DType, U: DType]() -> Codec[Auto.Int1, Auto.IntUnpack1[T, U]]:
    """Integers of `T`, unsigned `U`, chosen for in segments of 4,096."""
    return Codec(Auto.int1(4096), Auto.int_unpack1[T, U]())


def _float_choice[
    I: DType, U: DType
]() -> Codec[Auto.Float1, Auto.FloatUnpack1[I, U]]:
    """Floats as the integers `I` and unsigned `U` of their digits, chosen
    for in segments of 4,096."""
    return Codec(Auto.float1(4096), Auto.float_unpack1[I, U]())


def test_codecs_choice() raises:
    check[DType.int8](_choice[DType.int8, DType.uint8]())
    check[DType.int32](_choice[DType.int32, DType.uint32]())
    check[DType.int64](_choice[DType.int64, DType.uint64]())
    check[DType.uint64](_choice[DType.uint64, DType.uint64]())
    check[DType.float32](_float_choice[DType.int32, DType.uint32]())
    check[DType.float64](_float_choice[DType.int64, DType.uint64]())


# `Auto.Int1`'s candidates, in the order they are listed.
comptime _FREQUENCY = 0
comptime _PLAIN = 1
comptime _FOR = 2
comptime _PFOR = 3
comptime _DELTA = 5
comptime _RUNS = 6


def test_codecs_choice_picks_the_smallest() raises:
    """Each segment goes to the candidate that packs its sample smallest, or
    to the first listed within 2% of it."""
    var values = List[Int64]()
    # One value, and every 128th far past it: Frequency, PFOR a byte more.
    for i in range(4096):
        values.append(Int64(1) << 40 | Int64(i) if i % 128 == 5 else 7)
    # One value alone: FOR, 9 bytes on the sample -- its least and a width
    # of 0 -- the dictionary 9 too and listed after it, the runs 16 and
    # Frequency 20.
    for _ in range(4096):
        values.append(7)
    # A sorted run: its differences, framed.
    for i in range(4096):
        values.append(Int64(i * 3))
    # Random bits: plain, FOR a fit and a width more.
    var rng = Rng(11)
    for _ in range(4096):
        values.append(bitcast[DType.int64](rng.next()))
    var codec = _choice[DType.int64, DType.uint64]()
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(len(packed.chosen), 4)
    assert_equal(packed.chose(0), _FREQUENCY)
    assert_equal(packed.chose(1), _FOR)
    assert_equal(packed.chose(2), _DELTA)
    assert_equal(packed.chose(3), _PLAIN)
    assert_true(
        _same(_decode[DType.int64](codec, packed^, len(values)), values)
    )


def _keeps_to_its_sample[
    T: DType, D: UnpackKernel, *Ps: PackKernel
](codec: Codec[ChoicePack[*Ps], D]) raises:
    """Over every column of at least 640 values, no segment packs more than
    5% past what its sample predicted."""
    for ref values in _columns[T]():
        if len(values) >= 640:
            var packed = codec.encode(BufferView(Span(values)))
            for s in range(len(packed.chosen)):
                var nbytes = codec.encoder.segment_nbytes[T](packed, s)
                assert_true(
                    nbytes * 100 <= packed.predicted[s] * 105,
                    String(
                        T,
                        " segment ",
                        s,
                        ": ",
                        nbytes,
                        " bytes, ",
                        packed.predicted[s],
                        " predicted",
                    ),
                )


def test_codecs_choice_keeps_to_its_sample() raises:
    _keeps_to_its_sample[DType.int64](_choice[DType.int64, DType.uint64]())
    _keeps_to_its_sample[DType.uint64](_choice[DType.uint64, DType.uint64]())
    _keeps_to_its_sample[DType.float64](
        _float_choice[DType.int64, DType.uint64]()
    )
    _keeps_to_its_sample[DType.int64](
        _choice_three_levels[DType.int64, DType.uint64]()
    )
    _keeps_to_its_sample[DType.float64](
        _float_choice_three_levels[DType.int64, DType.uint64]()
    )


def _choice_three_levels[
    T: DType, U: DType
]() -> Codec[Auto.Int2, Auto.IntUnpack2[T, U]]:
    """`_choice`, a level deeper."""
    return Codec(Auto.int2(4096), Auto.int_unpack2[T, U]())


def _float_choice_three_levels[
    I: DType, U: DType
]() -> Codec[Auto.Float2, Auto.FloatUnpack2[I, U]]:
    """`_float_choice`, a level deeper."""
    return Codec(Auto.float2(4096), Auto.float_unpack2[I, U]())


def test_codecs_choice_repicks_past_its_prediction() raises:
    """Small values with every hundredth far past them: three levels deep,
    the sample's best is the differences through a dictionary, which packs
    half again past its prediction, as each outlier adds two differences
    the sample never saw. The segment goes to the next best, patched frame
    of reference, within its own prediction."""
    var values = List[Int64]()
    for i in range(65_536):
        if i % 100 == 0:
            values.append((Int64(1) << 40) + Int64(i))
        else:
            values.append(Int64((i * 7919) % 100))
    var codec = Codec(
        Auto.int2(), Auto.int_unpack2[DType.int64, DType.uint64]()
    )
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(len(packed.chosen), 1)
    assert_equal(packed.chose(0), _PFOR)
    var nbytes = codec.encoder.segment_nbytes[DType.int64](packed, 0)
    assert_true(nbytes * 100 <= packed.predicted[0] * 105)
    assert_true(
        _same(_decode[DType.int64](codec, packed^, len(values)), values)
    )


comptime _ForInt64 = Encoder[Compose[OrderedEncode], ReferencePack[BitPack]]
comptime _ForInt64Unpack = Decoder[
    ReferenceUnpack[BitUnpack],
    Compose[OrderedDecode[DType.int64]],
    DType.uint64,
]


def test_codecs_choice_keeps_the_smaller_of_two_picks() raises:
    """Values of four bits with one of 21 in each block, where no sampled run
    lies: frame of reference, predicted at four bits a value, packs at 21,
    and the next best, the values as they are, packs larger still, so frame
    of reference is kept."""
    var values = List[Int64]()
    for i in range(4096):
        if i % BLOCK == 1000:
            values.append(Int64(1) << 20)
        else:
            values.append(Int64((i * 7919) % 16))
    var codec = _plain_or_for()
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(packed.chose(0), 1)
    var nbytes = codec.encoder.segment_nbytes[DType.int64](packed, 0)
    # Past its prediction, so the values as they are were packed too.
    assert_true(nbytes * 100 > packed.predicted[0] * 105)
    assert_true(nbytes < 4096 * 8)
    assert_true(
        _same(_decode[DType.int64](codec, packed^, len(values)), values)
    )


def test_codecs_choice_three_levels() raises:
    check[DType.int64](_choice_three_levels[DType.int64, DType.uint64]())
    check[DType.float64](
        _float_choice_three_levels[DType.int64, DType.uint64]()
    )


def _three_shapes() -> List[Int64]:
    """Three segments of 4,096 -- random bits, sorted, long runs -- and a
    tail of 100 random values."""
    var values = List[Int64]()
    var rng = Rng(5)
    for _ in range(4096):
        values.append(bitcast[DType.int64](rng.next()))
    for i in range(4096):
        values.append(Int64(i * 3))
    for i in range(4096):
        values.append(Int64(i // 512) * 1_000_003)
    for _ in range(100):
        values.append(bitcast[DType.int64](rng.next()))
    return values^


def test_codecs_choice_is_deterministic() raises:
    var codec = _choice[DType.int64, DType.uint64]()
    for ref values in _columns[DType.int64]():
        var a = codec.encode(BufferView(Span(values)))
        var b = codec.encode(BufferView(Span(values)))
        assert_equal(len(a.chosen), len(b.chosen))
        for s in range(len(a.chosen)):
            assert_equal(a.chose(s), b.chose(s))
        assert_true(a.predicted == b.predicted)
        assert_equal(
            codec.encoder.nbytes[DType.int64](a),
            codec.encoder.nbytes[DType.int64](b),
        )
    var values = _three_shapes()
    var a = codec.encode(BufferView(Span(values)))
    var b = codec.encode(BufferView(Span(values)))
    assert_true(a.predicted == b.predicted)
    assert_equal(
        codec.encoder.nbytes[DType.int64](a),
        codec.encoder.nbytes[DType.int64](b),
    )
    # Byte for byte, over candidates whose parts are buffers: the values as
    # they are, and frame of reference.
    var two = _plain_or_for()
    var x = two.encode(BufferView(Span(values)))
    var y = two.encode(BufferView(Span(values)))
    comptime Plain = type_of(x).Slot[0]
    comptime For = type_of(x).Slot[1]
    assert_equal(len(x.chosen), 4)
    assert_true(x.chose(0) == 0 and x.chose(1) == 1 and x.chose(2) == 1)
    assert_true(len(y.chosen) == 4 and x.predicted == y.predicted)
    for s in range(len(x.chosen)):
        assert_equal(x.chose(s), y.chose(s))
        if x.chosen[s].isa[Plain]():
            ref l = rebind[PlainPacked](x.chosen[s][Plain].packed)
            ref r = rebind[PlainPacked](y.chosen[s][Plain].packed)
            assert_true(_same_bytes(l.bytes, r.bytes, l.size, r.size))
        else:
            ref l = rebind[FittedPacked[BitPacked]](x.chosen[s][For].packed)
            ref r = rebind[FittedPacked[BitPacked]](y.chosen[s][For].packed)
            assert_true(
                _same_bytes(
                    l.fits.values,
                    r.fits.values,
                    8 * len(l.fits),
                    8 * len(r.fits),
                )
            )
            assert_true(
                _same_bytes(
                    l.inner.bytes, r.inner.bytes, l.inner.size, r.inner.size
                )
            )
            ref lw = l.inner.widths
            ref rw = r.inner.widths
            assert_true(_same_bytes(lw.values, rw.values, len(lw), len(rw)))


def _same_bytes(
    a: Buffer[mut=False], b: Buffer[mut=False], m: Int, n: Int
) -> Bool:
    """The first `m` bytes of `a` are the first `n` of `b`."""
    return (
        m == n
        and m <= len(a)
        and n <= len(b)
        and a.view(0, m).as_span() == b.view(0, n).as_span()
    )


def _plain_or_for() -> (
    Codec[
        ChoicePack[PlainPack, _ForInt64],
        ChoiceUnpack[PlainUnpack, _ForInt64Unpack],
    ]
):
    """`int64`s as they are or by frame of reference, in segments of
    4,096."""
    return Codec(
        ChoicePack[PlainPack, _ForInt64](
            4096,
            (
                PlainPack(),
                _ForInt64(
                    Compose((OrderedEncode(),)),
                    ReferencePack(BLOCK, BitPack()),
                ),
            ),
        ),
        ChoiceUnpack[PlainUnpack, _ForInt64Unpack](
            (
                PlainUnpack(),
                _ForInt64Unpack(
                    ReferenceUnpack(BLOCK, BitUnpack()),
                    Compose((OrderedDecode[DType.int64](),)),
                ),
            )
        ),
    )


def test_codecs_choice_reads_segments_in_any_order() raises:
    """Each segment read whole and in order, the segments not: a register
    comes from the segment that holds it."""
    var values = _three_shapes()
    var n = len(values)
    var codec = _choice[DType.int64, DType.uint64]()
    var packed = codec.encode(BufferView(Span(values)))
    var state = codec.decoder.start[DType.int64](
        rebind_var[Auto.IntUnpack1[DType.int64, DType.uint64].Packed](packed^),
        n,
    )
    var out = List[Int64](length=n, fill=0)
    var dst = BufferView(Span(out))
    for s in [2, 0, 3, 1]:
        var i = s * 4096
        var hi = min(i + 4096, n)
        while i + 8 <= hi:
            dst.store[8](i, codec.decoder.load[DType.int64, 8](state, i))
            i += 8
        while i < hi:
            dst.store[1](i, codec.decoder.load[DType.int64, 1](state, i))
            i += 1
    codec.decoder.check[DType.int64](state)
    assert_true(_same(out, values))


def test_codecs_choice_segments() raises:
    var values = _three_shapes()
    var codec = _choice[DType.int64, DType.uint64]()
    var packed = codec.encode(BufferView(Span(values)))
    assert_equal(len(packed.chosen), 4)
    assert_equal(packed.chose(0), _PLAIN)
    assert_equal(packed.chose(1), _DELTA)
    assert_equal(packed.chose(2), _RUNS)
    assert_equal(packed.chose(3), _PLAIN)
    # The tail is its own sample, so it packs as predicted.
    assert_equal(
        codec.encoder.segment_nbytes[DType.int64](packed, 3),
        packed.predicted[3],
    )
    # Each segment adds a byte naming its candidate.
    var parts = 0
    for s in range(len(packed.chosen)):
        parts += codec.encoder.segment_nbytes[DType.int64](packed, s)
    assert_equal(
        codec.encoder.nbytes[DType.int64](packed),
        len(packed.chosen) + parts,
    )
    assert_true(
        _same(_decode[DType.int64](codec, packed^, len(values)), values)
    )


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


def _nbytes[
    T: DType, E: PackKernel, D: UnpackKernel
](codec: Codec[E, D], values: List[Scalar[T]]) raises -> Int:
    """What the encoder says it packed `values` as."""
    return codec.encoder.nbytes[T](codec.encode(BufferView(Span(values))))


def test_codecs_nbytes() raises:
    """Each packer counts every part at its logical size, the parts it nests
    included, and no count or size field."""
    var sorted = [UInt64(i * 3) for i in range(301)]
    assert_equal(_nbytes(Codec(PlainPack(), PlainUnpack()), sorted), 2_408)
    assert_equal(_nbytes(Codec(SplitPack(), SplitUnpack()), sorted), 2_408)
    # 800 values of 3 bits in 300 bytes, and 100 widths.
    var small = [UInt8(i % 7) for i in range(800)]
    assert_equal(_nbytes(Codec(BitPack(8), BitUnpack(8)), small), 400)

    # The docstrings' examples.
    var bits: List[UInt8] = [0, 1, 3]
    assert_equal(_nbytes(Codec(BitPack(), BitUnpack()), bits), 3)
    assert_equal(_nbytes(Codec(RlePack(2), RleUnpack(2)), bits), 3)
    var clustered: List[UInt8] = [100, 101, 103]
    assert_equal(
        _nbytes(
            Codec(
                ReferencePack(BLOCK, PlainPack()),
                ReferenceUnpack(BLOCK, PlainUnpack()),
            ),
            clustered,
        ),
        4,
    )
    var held: List[UInt8] = [5, 5, 5, 1, 2, 2]
    assert_equal(
        _nbytes(
            Codec(
                RunsPack(PlainPack(), PlainPack()),
                RunsUnpack(PlainUnpack(), PlainUnpack()),
            ),
            held,
        ),
        3 + 3 * 8,
    )
    var repeated: List[UInt8] = [5, 5, 9, 5]
    assert_equal(
        _nbytes(
            Codec(DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack())),
            repeated,
        ),
        2 + 4 * 4,
    )
    # An encoder answers its packer's, at the dtype its stages yield.
    var signed: List[Int8] = [0, -1, 1, -2]
    assert_equal(
        _nbytes(Codec(_zigzag_pack(), _zigzag_unpack[DType.int8]()), signed), 4
    )
    assert_equal(
        _nbytes(
            Codec(
                Encoder(Compose((NumericCast[DType.int32](),)), PlainPack()),
                Decoder[
                    PlainUnpack, Compose[NumericCast[DType.int8]], DType.int32
                ](PlainUnpack(), Compose((NumericCast[DType.int8](),))),
            ),
            signed,
        ),
        16,
    )

    # "ab", "abc"
    var offsets: List[Int32] = [0, 2, 5]
    var data = List(String("ababc").as_bytes())
    var lengths = LengthsPack(PlainPack())
    assert_equal(
        lengths.nbytes(lengths.pack(Span(offsets), Span(data))), 5 + 2 * 4
    )
    var prefixes = PrefixesPack(PlainPack(), PlainPack())
    # Prefixes [0, 2], suffixes "ab" and "c".
    assert_equal(
        prefixes.nbytes(prefixes.pack(Span(offsets), Span(data))),
        2 * 4 + 3 + 2 * 4,
    )
    var dictionary = DictionaryBytesPack(PlainPack())
    assert_equal(
        dictionary.nbytes(dictionary.pack(Span(offsets), Span(data))),
        5 + 2 * 4 + 2 * 4,
    )


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


def _fsst_refused(
    unpacker: FsstUnpack[PlainUnpack],
    n: Int,
    var packed: FsstPacked[PlainPacked],
) raises:
    var ends: List[Int32] = [0]
    var bytes = List[UInt8]()
    with assert_raises(contains="CorruptError"):
        unpacker.unpack(packed^, n, ends, bytes)


def test_codecs_refuse_cut_fsst_parts() raises:
    var packer = FsstPack(PlainPack())
    var unpacker = FsstUnpack(PlainUnpack())
    var offsets = List[Int32]()
    var data = List[UInt8]()
    _column(_strings()[3], offsets, data)
    var n = len(offsets) - 1
    var p = packer.pack(Span(offsets), Span(data))
    var k = p.count
    assert_true(k < 255)
    var q = p.copy()
    q.count = 256
    _fsst_refused(unpacker, n, q^)
    q = p.copy()
    q.lengths = _with[DType.uint8](p.lengths, k, 0, 0)
    _fsst_refused(unpacker, n, q^)
    q = p.copy()
    q.lengths = _with[DType.uint8](p.lengths, k, 0, 9)
    _fsst_refused(unpacker, n, q^)
    # One length short: past a 64-byte boundary the buffer is too small,
    # else the padding reads as a length of 0.
    var short = Buffer.alloc_uninit(k - 1)
    short.extend(p.lengths.view(0, k - 1), 0, k - 1)
    q = p.copy()
    q.lengths = short^.to_immutable()
    _fsst_refused(unpacker, n, q^)
    # Eight symbols short, so the buffer falls short of the count even
    # padded.
    var kept = max(k - 8, 0)
    var cut = Buffer.alloc_uninit[DType.uint64](kept)
    cut.extend(p.symbols.view[DType.uint64](0, kept), 0, kept)
    q = p.copy()
    q.symbols = cut^.to_immutable()
    assert_true(len(q.symbols) < 8 * k)
    _fsst_refused(unpacker, n, q^)
    # The first string's first code names no symbol.
    ref codes = p.codes
    q = p.copy()
    q.codes.bytes = _with[DType.uint8](codes.bytes, codes.size, 0, UInt8(k))
    _fsst_refused(unpacker, n, q^)
    # The first string's last code an escape; it holds none to begin with.
    var first = Int(codes.lengths.bytes.view[DType.int32]()[0])
    for i in range(first):
        assert_true(codes.bytes.unsafe_get(i) != 255)
    q = p.copy()
    q.codes.bytes = _with[DType.uint8](codes.bytes, codes.size, first - 1, 255)
    _fsst_refused(unpacker, n, q^)
    q = p.copy()
    q.codes.size -= 1
    _fsst_refused(unpacker, n, q^)


def test_codecs_refuse_cut_patch_parts() raises:
    comptime U = DType.uint64
    var values = _outliers()
    var n = len(values)
    var codec = _pfor_alone()
    var p = codec.encode(BufferView(Span(values)))
    # 8 segments; ends [1, 3, 3, 3, 3, 4, 4, 5]; positions [3, 2, 3, 60, 127].
    var q = p.copy()
    q.fits = PerSegment(p.fits.values, 7)
    _refused[U](codec, q^, n)
    # A width wider than the values.
    q = p.copy()
    q.fits = PerSegment(_with[U](p.fits.values, 8, 0, 65), 8)
    _refused[U](codec, q^, n)
    q = p.copy()
    q.ends = PerSegment(p.ends.values, 7)
    _refused[U](codec, q^, n)
    # The last end past the five positions.
    q = p.copy()
    q.ends = PerSegment(_with[DType.uint32](p.ends.values, 8, 7, 6), 8)
    _refused[U](codec, q^, n)
    # An end below the one before it.
    q = p.copy()
    q.ends = PerSegment(_with[DType.uint32](p.ends.values, 8, 1, 0), 8)
    _refused[U](codec, q^, n)
    q = p.copy()
    q.positions.size -= 2
    _refused[U](codec, q^, n)
    # A position past the segment of 128 it is in.
    q = p.copy()
    q.positions.bytes = _with[DType.uint16](p.positions.bytes, 5, 0, 128)
    _refused[U](codec, q^, n)
    # The second segment's two positions swapped.
    var swapped = _with[DType.uint16](p.positions.bytes, 5, 1, 3)
    q = p.copy()
    q.positions.bytes = _with[DType.uint16](swapped, 5, 2, 2)
    _refused[U](codec, q^, n)
    q = p.copy()
    q.values.size -= 8
    _refused[U](codec, q^, n)

    var constant = Codec(ConstantPack(), ConstantUnpack())
    var same = List[UInt8](length=100, fill=42)
    var c = constant.encode(BufferView(Span(same)))
    _refused[DType.uint8](constant, ConstantPacked(c.value, c.count - 1), 100)
    # Bits past a `uint8`.
    _refused[DType.uint8](constant, ConstantPacked(0x1FF, c.count), 100)

    var frequency = Codec(
        FrequencyPack(64, ConstantPack()),
        FrequencyUnpack(64, ConstantUnpack()),
    )
    var common: List[Int8] = [7, 7, 3, 7, 9]
    var f = frequency.encode(BufferView(Span(common)))
    # A most common value with bits past an `int8`.
    f.fits = PerSegment(_with[DType.uint64](f.fits.values, 1, 0, 0x1FF), 1)
    _refused[DType.int8](frequency, f^, len(common))


def test_codecs_refuse_cut_alp_parts() raises:
    comptime U = DType.uint64
    var values = _prices()
    var n = len(values)
    var codec = _alp[DType.float64]()
    var p = codec.encode(BufferView(Span(values)))
    # e = 19: past the powers of ten a float64 holds exactly.
    var q = p.copy()
    q.fits = PerSegment(_with[U](p.fits.values, 3, 0, 19 << 8), 3)
    _refused[DType.float64](codec, q^, n)
    # f above e.
    q = p.copy()
    q.fits = PerSegment(_with[U](p.fits.values, 3, 1, 2 << 8 | 3), 3)
    _refused[DType.float64](codec, q^, n)
    # e = 11: past the powers of ten a float32 holds exactly.
    var narrow = [Float32(i % 1000) * 0.25 + 20.0 for i in range(3000)]
    var codec32 = _alp[DType.float32]()
    var r = codec32.encode(BufferView(Span(narrow)))
    r.fits = PerSegment(_with[U](r.fits.values, 3, 2, 11 << 8), 3)
    _refused[DType.float32](codec32, r^, n)


def test_codecs_refuse_cut_alp_rd_parts() raises:
    comptime T = DType.float64
    comptime U8 = DType.uint8
    var values = _prices()
    var n = len(values)
    var codec = _alp_rd()
    var p = codec.encode(BufferView(Span(values)))
    assert_equal(len(p.cuts), 3)
    # The cuts one short.
    var q = p.copy()
    q.cuts = PerSegment(p.cuts.values, 2)
    _refused[T](codec, q^, n)
    # A cut of 64: no left part.
    q = p.copy()
    q.cuts = PerSegment(_with[U8](p.cuts.values, 3, 0, 64), 3)
    _refused[T](codec, q^, n)
    # A cut of 40: a left part of 24 bits, past the 16 one holds.
    q = p.copy()
    q.cuts = PerSegment(_with[U8](p.cuts.values, 3, 1, 40), 3)
    _refused[T](codec, q^, n)
    # The left parts' codes, and the right parts, a segment short.
    q = p.copy()
    q.left.codes.widths = PerSegment(p.left.codes.widths.values, 2)
    _refused[T](codec, q^, n)
    q = p.copy()
    q.right.widths = PerSegment(p.right.widths.values, 2)
    _refused[T](codec, q^, n)


def test_codecs_refuse_cut_choice_parts() raises:
    comptime T = DType.int64
    var values = _three_shapes()
    var n = len(values)
    var codec = _choice[T, DType.uint64]()
    var p = codec.encode(BufferView(Span(values)))
    comptime Packed = type_of(p)
    var short = p.chosen.copy()
    _ = short.pop()
    _refused[T](codec, Packed(p.size, short^, p.predicted.copy()), n)
    _refused[T](codec, Packed(0, p.chosen.copy(), p.predicted.copy()), n)
    # As many segments, but not of whole registers.
    _refused[T](codec, Packed(4000, p.chosen.copy(), p.predicted.copy()), n)
    # Segment 1's differences, their frame of reference one fit short: the
    # nested candidate refuses it once the segment is read, and both
    # segments are named.
    comptime Slot = Packed.Slot[_DELTA]
    comptime Inner = Auto.Int0.Packed
    var slot = p.chosen[1][Slot].copy()
    var deltas = rebind[Inner](slot.packed).copy()
    assert_equal(deltas.chose(0), 1)
    var framed = deltas.chosen[0][Inner.Slot[1]].copy()
    ref f = rebind[FittedPacked[BitPacked]](framed.packed)
    var fits = PerSegment(f.fits.values, len(f.fits) - 1)
    framed.packed = rebind_var[type_of(framed.packed)](
        FittedPacked(fits^, f.inner.copy())
    )
    var inner = deltas.chosen.copy()
    inner[0] = Inner.Chosen(framed^)
    slot.packed = rebind_var[type_of(slot.packed)](
        Inner(deltas.size, inner^, deltas.predicted.copy())
    )
    var cut = p.chosen.copy()
    cut[1] = Packed.Chosen(slot^)
    with assert_raises(
        contains=(
            "choice_unpack: segment 1: CorruptError: choice_unpack: segment"
            " 0: CorruptError"
        )
    ):
        _ = _decode[T](codec, Packed(p.size, cut^, p.predicted.copy()), n)
