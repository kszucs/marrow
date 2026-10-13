# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every codec round-trips bit for bit, alone and in cascades, its slots
stored by cascades of their own; and a tree with a byte or a value cut away
is refused or decodes to other values."""

from std.memory import bitcast
from std.testing import assert_equal, assert_raises, assert_true

from ...codecs import (
    BitPack,
    ByteStreamSplit,
    ByteSwap,
    Cascade,
    Cast,
    Codec,
    Delta,
    Dictionary,
    Encoded,
    FrameOfReference,
    Lengths,
    Ordered,
    Plain,
    Prefixes,
    Rle,
    Runs,
    Zigzag,
)
from ...builders import array
from ...dtypes import Int64Type, int64
from ...utils.integers import Unsigned
from ...utils.testing import Rng
from ...views import BufferView


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
    var runs = List[Scalar[T]]()
    for i in range(1000):
        runs.append(Scalar[T]((i // 37) % 3))
    cols.append(runs^)
    var sorted = List[Scalar[T]]()
    for i in range(301):
        sorted.append(Scalar[T](i * 3))
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


def _cut(node: Encoded) -> List[Encoded]:
    """`node` with one thing taken away -- a node's last byte, or a
    cascade's last value -- in every way there is."""
    var out = List[Encoded]()
    if node.size > 0:
        var cut = node.copy()
        cut.size -= 1
        out.append(cut^)
    if node.count > 0:
        var cut = node.copy()
        cut.count -= 1
        out.append(cut^)
    if len(node.children) == 0:
        return out^
    for i in range(len(node.children)):
        for ref child in _cut(node.children[i]):
            var cut = node.copy()
            cut.children[i] = child.copy()
            out.append(cut^)
    return out^


def _same[T: DType](a: List[Scalar[T]], b: List[Scalar[T]]) -> Bool:
    """Equal bit for bit -- so a NaN equals itself and -0.0 is not 0.0."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[Unsigned[T]](a[i]) != bitcast[Unsigned[T]](b[i]):
            return False
    return True


def _decode[
    T: DType, C: Codec
](mut cascade: C, node: Encoded) raises -> List[Scalar[T]]:
    var out = List[Scalar[T]](length=node.count, fill=0)
    cascade.decode_into[T](node, BufferView(Span(out)))
    return out^


def check[T: DType, C: Codec](var cascade: C) raises:
    """Every column round-trips, and a tree cut short is refused or decodes
    to other values -- never read past."""
    for ref values in _columns[T]():
        var n = len(values)
        var encoded = cascade.encode_view(BufferView(Span(values)))
        assert_true(
            _same(_decode[T](cascade, encoded), values),
            String(T, " did not round-trip ", n),
        )
        if n > 0 and n < 400:
            for ref cut in _cut(encoded):
                try:
                    var wrong = _decode[T](cascade, cut)
                    assert_true(not _same(wrong, values), "a cut went unseen")
                except e:
                    assert_true("CorruptError" in String(e), String(e))


def _column(
    values: List[String], mut offsets: List[Int32], mut data: List[UInt8]
):
    offsets.append(Int32(len(data)))
    for v in values:
        data.extend(v.as_bytes())
        offsets.append(Int32(len(data)))


def _decode_bytes[
    C: Codec
](mut cascade: C, node: Encoded) raises -> Tuple[List[Int32], List[UInt8]]:
    var offsets: List[Int32] = [0]
    var data = List[UInt8]()
    cascade.decode(node, offsets, data)
    return (offsets^, data^)


def check_bytes[C: Codec](var cascade: C) raises:
    """`check`, for byte strings."""
    for ref values in _strings():
        var offsets = List[Int32]()
        var data = List[UInt8]()
        _column(values, offsets, data)
        var encoded = cascade.encode(Span(offsets), Span(data))
        var back = _decode_bytes(cascade, encoded)
        assert_true(back[0] == offsets and back[1] == data)
        if len(values) > 0 and len(values) < 400:
            for ref cut in _cut(encoded):
                try:
                    var wrong = _decode_bytes(cascade, cut)
                    assert_true(
                        not (wrong[0] == offsets and wrong[1] == data),
                        "a cut went unseen",
                    )
                except e:
                    assert_true("CorruptError" in String(e), String(e))


# --- each codec ----------------------------------------------------------------


def test_codecs_plain() raises:
    check[DType.float64](Plain())
    check[DType.int8](Plain())
    check[DType.uint64](Cascade[Plain]())


def test_codecs_byte_stream_split() raises:
    check[DType.float32](Cascade[ByteStreamSplit]())
    check[DType.float64](Cascade[ByteStreamSplit]())
    check[DType.int64](Cascade[ByteStreamSplit]())


def test_codecs_bitpack() raises:
    check[DType.uint8](BitPack[]())
    check[DType.uint32](Cascade[BitPack[]]())
    check[DType.uint64](Cascade((BitPack(32),)))
    check[DType.uint8](Cascade((BitPack(8),)))
    check[DType.uint16](Cascade((BitPack(64),)))


def test_codecs_reference() raises:
    check[DType.uint32](Cascade[FrameOfReference[], Plain]())
    check[DType.uint16](Cascade((FrameOfReference(128), Plain())))
    check[DType.uint64](Cascade((FrameOfReference(256), BitPack(32))))
    check[DType.uint8](Cascade((FrameOfReference(64), BitPack())))


def test_codecs_frames_whole() raises:
    """A lane's frames are whole registers, a store's whole groups, and
    encoding takes the frames that divide a block."""
    var one: List[UInt8] = [1]
    for size in [32, 100, 384, 2048]:
        var reference = Cascade((FrameOfReference(size), Plain()))
        with assert_raises(contains="InvalidError"):
            _ = reference.encode_view(BufferView(Span(one)))
    var bitpack = Cascade((BitPack(12),))
    with assert_raises(contains="InvalidError"):
        _ = bitpack.encode_view(BufferView(Span(one)))


def test_codecs_runs() raises:
    check[DType.int64](Cascade[Runs[]]())
    check[DType.float32](Cascade[Runs[]]())
    check[DType.uint32](Cascade[Runs[BitPack[], BitPack[]]]())


def test_codecs_dictionary() raises:
    check[DType.float64](Cascade[Dictionary[], Plain]())
    check[DType.int32](Cascade[Dictionary[], BitPack[]]())
    check_bytes(Cascade[Dictionary[], Plain]())
    check_bytes(Cascade[Dictionary[], BitPack[]]())


def test_codecs_delta() raises:
    check[DType.int64](Cascade[Delta, Plain]())
    check[DType.int8](Cascade[Delta, Plain]())
    check[DType.uint16](Cascade((Delta(UInt16(7)), Plain())))


def test_codecs_zigzag() raises:
    check[DType.int32](Cascade[Zigzag, Plain]())
    check[DType.int64](Cascade[Zigzag, Plain]())


def test_codecs_ordered() raises:
    check[DType.float32](Cascade[Ordered, Plain]())
    check[DType.int16](Cascade[Ordered, Plain]())


def test_codecs_cast() raises:
    check[DType.int8](Cascade[Cast[DType.int32], Plain]())
    check[DType.uint16](Cascade[Cast[DType.uint32], Plain]())
    check[DType.int32](Cascade[Cast[DType.int64], Delta, Zigzag, BitPack[]]())


def test_codecs_byte_swap() raises:
    check[DType.int64](Cascade[ByteSwap, Plain]())
    # Values wider than a block of scratch holds as `uint64`s.
    var values = List[Int128]()
    for i in range(3000):
        values.append(Int128(i) << 100 | Int128(i))
    var cascade = Cascade[ByteSwap, Plain]()
    var e = cascade.encode_view(BufferView(Span(values)))
    # The second value, big-endian: bit 100 in its fourth byte, 1 in its
    # last.
    var bytes = e[1].bytes()
    assert_true(bytes[16 + 3] == 0x10 and bytes[16 + 15] == 1)
    var back = List[Int128](length=len(values), fill=0)
    cascade.decode_into[DType.int128](e, BufferView(Span(back)))
    assert_true(back == values)


def test_codecs_lengths() raises:
    check_bytes(Cascade[Lengths[], Plain]())
    check_bytes(Cascade[Lengths[], Delta, Zigzag, BitPack[]]())


def test_codecs_prefixes() raises:
    check_bytes(Cascade[Prefixes[], Plain]())
    check_bytes(Cascade[Prefixes[], Zigzag, BitPack[]]())


# --- cascades, after ClickHouse's codec chains ---------------------------------


def test_cascade_delta() raises:
    """CODEC(Delta): monotonic counters and timestamps."""
    check[DType.int64](Cascade[Delta, Zigzag, BitPack[]]())
    check[DType.int32](Cascade[Delta, Zigzag, BitPack[]]())


def test_cascade_double_delta() raises:
    """CODEC(DoubleDelta): timestamps at a near-constant interval."""
    check[DType.int64](Cascade[Delta, Delta, Zigzag, BitPack[]]())
    check[DType.int16](Cascade[Delta, Delta, Zigzag, BitPack[]]())


def test_cascade_t64() raises:
    """CODEC(T64): frame of reference, then the bits the range needs."""
    check[DType.int64](Cascade[Ordered, FrameOfReference[], BitPack[]]())
    check[DType.float32](Cascade[Ordered, FrameOfReference[], BitPack[]]())


def test_cascade_t64_blocks() raises:
    """T64 a frame of 64 values at a time."""
    check[DType.int32](Cascade((Ordered(), FrameOfReference(64), BitPack(64))))
    check[DType.float64](
        Cascade((Ordered(), FrameOfReference(64), BitPack(64)))
    )


def test_cascade_delta_t64() raises:
    """CODEC(Delta, T64) -- Parquet's DELTA_BINARY_PACKED after its first
    value."""
    check[DType.int64](
        Cascade((Delta(), Ordered(), FrameOfReference(128), BitPack(32)))
    )
    check[DType.int32](
        Cascade((Delta(), Ordered(), FrameOfReference(128), BitPack(32)))
    )


def test_cascade_double_delta_t64() raises:
    check[DType.int64](
        Cascade(
            (Delta(), Delta(), Ordered(), FrameOfReference(128), BitPack(32))
        )
    )


def test_cascade_zigzag() raises:
    """Small signed integers, packed."""
    check[DType.int8](Cascade[Zigzag, BitPack[]]())
    check[DType.int64](Cascade[Zigzag, BitPack[]]())


def test_cascade_low_cardinality() raises:
    """LowCardinality: a dictionary, its codes packed."""
    check[DType.float64](Cascade[Dictionary[], BitPack[]]())
    check_bytes(Cascade[Dictionary[], BitPack[]]())


def test_cascade_low_cardinality_runs() raises:
    """LowCardinality whose codes repeat: Parquet's RLE_DICTIONARY."""
    check[DType.int64](Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]())
    check_bytes(Cascade[Dictionary[], Runs[BitPack[], BitPack[]]]())


def test_cascade_sparse() raises:
    """Long runs of one value, the rest framed and packed."""
    check[DType.int64](
        Cascade[
            Runs[
                Cascade[Ordered, FrameOfReference[], BitPack[]],
                Cascade[BitPack[]],
            ]
        ]()
    )


def test_cascade_float_planes() raises:
    """Gauges: readings that repeat, as runs, their values' byte planes
    apart."""
    check[DType.float64](Cascade[Runs[ByteStreamSplit, BitPack[]]]())
    check[DType.float32](Cascade[Runs[ByteStreamSplit, BitPack[]]]())


def test_cascade_strings() raises:
    """Byte strings: their lengths, or the prefixes they share, delta
    packed."""
    check_bytes(Cascade[Lengths[], Delta, Zigzag, BitPack[]]())
    check_bytes(
        Cascade(
            (Lengths(), Delta(), Ordered(), FrameOfReference(128), BitPack(32))
        )
    )
    check_bytes(Cascade[Prefixes[], Delta, Zigzag, BitPack[]]())


def test_cascade_slots() raises:
    """Slots stored by cascades of their own: the smallest values of each
    frame delta-packed, the widths run-encoded."""
    check[DType.int64](
        Cascade(
            (
                Ordered(),
                FrameOfReference(128, Cascade[Delta, BitPack[]]()),
                BitPack(32, Runs()),
            )
        )
    )


def test_cascade_delta_byte_array() raises:
    """Parquet's DELTA_BYTE_ARRAY: the prefixes delta-packed, the suffixes
    as DELTA_LENGTH_BYTE_ARRAY."""
    var dlba = Cascade(
        (Lengths(), Delta(), Ordered(), FrameOfReference(128), BitPack(32))
    )
    check_bytes(
        Cascade(
            (
                Prefixes(dlba^),
                Delta(),
                Ordered(),
                FrameOfReference(128),
                BitPack(32),
            )
        )
    )


# --- what the docstrings show -------------------------------------------------


def _bytes(node: Encoded) -> List[UInt8]:
    var out = List[UInt8]()
    for v in node.bytes():
        out.append(v)
    return out^


def _encode[
    T: DType, C: Codec
](var cascade: C, values: List[Scalar[T]]) raises -> Encoded:
    return cascade.encode_view(BufferView(Span(values)))


def test_codecs_docstring_examples() raises:
    var bitpack = _encode(BitPack[](), [UInt8(0), 1, 3])
    assert_true(_bytes(bitpack) == [UInt8(0x34), 0])
    var widths = BitPack.widths(bitpack)
    assert_true(len(widths) == 1 and widths[0] == 2)

    var zigzag = _encode(Cascade[Zigzag, Plain](), [Int8(0), -1, 1, -2])
    assert_true(_bytes(zigzag[1]) == [UInt8(0), 1, 2, 3])

    var delta = _encode(Cascade[Delta, Plain](), [Int8(10), 12, 15, 15])
    assert_true(_bytes(delta[1]) == [UInt8(10), 2, 3, 0])

    var reference = _encode(
        Cascade[FrameOfReference[], Plain](), [UInt8(100), 101, 103]
    )
    assert_true(_bytes(reference[1]) == [UInt8(0), 1, 3])
    var lows = FrameOfReference.lows[DType.uint8](reference[0])
    assert_true(len(lows) == 1 and lows[0] == 100)

    var bss = _encode(ByteStreamSplit(), [UInt16(0x0102), 0x0304])
    assert_true(_bytes(bss) == [UInt8(2), 4, 1, 3])

    var dictionary = _encode(
        Cascade[Dictionary[], Plain](), [UInt8(5), 5, 9, 5]
    )
    assert_true(_bytes(Dictionary.distinct(dictionary[0])) == [UInt8(5), 9])
    assert_true(dictionary[1].values[DType.uint32]()[2] == 1)

    var runs = _encode(Runs[](), [UInt8(5), 5, 5, 1, 2, 2])
    assert_true(_bytes(runs[Runs[].VALUES]) == [UInt8(5), 1, 2])
    var lengths = runs[Runs[].LENGTHS].values[DType.uint64]()
    assert_true(lengths[0] == 3 and lengths[1] == 1 and lengths[2] == 2)


def test_codecs_parquet_delta_binary_packed() raises:
    """Parquet's own example, 1 to 5 as DELTA_BINARY_PACKED: after the
    first value, the deltas through `Delta(first), Ordered,
    FrameOfReference(128), BitPack(32)`."""
    var dbp = Cascade(
        (Delta(Int32(1)), Ordered(), FrameOfReference(128), BitPack(32))
    )
    var deltas: List[Int32] = [2, 3, 4, 5]
    var e = dbp.encode_view(BufferView(Span(deltas)))
    # The smallest delta's key, and one width of 0 -- so no bytes.
    assert_true(FrameOfReference.lows[DType.uint32](e[2])[0] == 0x80000001)
    var widths = BitPack.widths(e[3])
    assert_true(len(widths) == 1 and widths[0] == 0)
    assert_equal(len(e[3].bytes()), 0)
    var back = _decode[DType.int32](dbp, e)
    assert_true(back == [Int32(2), 3, 4, 5])


# --- typed arrays at the edges ------------------------------------------------


def test_codecs_arrays() raises:
    """A column goes in and comes out as an array; `decode_into` writes
    into any view, an array's buffer at an offset among them."""
    var values = List[Int64]()
    for i in range(3000):
        values.append(Int64(i * 7 - (i % 13) * 1000))
    var arr = array[Int64Type](values.copy(), int64)
    var cascade = Cascade[Delta, Zigzag, BitPack[]]()
    var node = cascade.encode_view(arr.values())
    var back = cascade.decode(node, int64)
    assert_equal(len(back), len(values))
    for i in range(len(values)):
        assert_true(back.values()[i] == values[i])
    var wide = List[Int64](length=len(values) + 5, fill=0)
    cascade.decode_into[DType.int64](node, BufferView(Span(wide))[5:])
    assert_true(wide[0] == 0 and wide[5 + 42] == values[42])
    # A sliced array is encoded where it lies.
    var tail = cascade.encode_view(arr.slice(1000, 500).values())
    var sliced = cascade.decode(tail, int64)
    assert_true(sliced.values()[0] == values[1000])


def test_codecs_arrays_plain_decodes_a_prefix() raises:
    """`Plain` keeps the values as they are: decoding a node holding three
    of a buffer's four values gives those three."""
    var four: List[Int64] = [5, 6, 7, 8]
    var arr = array[Int64Type](four^, int64)
    var node = Encoded(arr.buffer, 3 * 8)
    node.count = 3
    var plain = Plain()
    var back = plain.decode(node, int64)
    assert_equal(len(back), 3)
    assert_true(back.values()[0] == 5 and back.values()[2] == 7)


# --- Rle, Parquet's RLE / bit-packed hybrid -----------------------------------


def test_codecs_rle() raises:
    check[DType.uint8](Rle(8))
    check[DType.uint32](Rle(32))
    check[DType.uint64](Cascade((Rle(64),)))


def test_codecs_rle_width_wider_than_the_type() raises:
    var data: List[UInt8] = [UInt8(0x02), 0x01]
    var out = List[UInt32]()
    with assert_raises(contains="of 33 bits"):
        _ = Rle.read_runs[DType.uint32](Span(data), 33, 1, out)


def test_codecs_rle_packed_run_past_the_data() raises:
    var data: List[UInt8] = [UInt8(0x05), 0xFF]
    var out = List[UInt32]()
    with assert_raises(contains="truncated"):
        _ = Rle.read_runs[DType.uint32](Span(data), 3, 16, out)


def test_codecs_rle_header_past_the_data() raises:
    var data: List[UInt8] = [UInt8(0x80)]
    var out = List[UInt32]()
    with assert_raises(contains="truncated"):
        _ = Rle.read_runs[DType.uint32](Span(data), 1, 1, out)


def test_codecs_rle_codes_with_no_width() raises:
    var data = List[UInt8]()
    var codes = List[Int32]()
    with assert_raises(contains="no codes"):
        Rle.read_codes[DType.int32](Span(data), 3, codes)


def test_codecs_rle_no_codes_with_no_width() raises:
    var data = List[UInt8]()
    var codes = List[Int32]()
    Rle.read_codes[DType.int32](Span(data), 0, codes)
    assert_true(len(codes) == 0)
