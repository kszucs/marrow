# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page encodings, as codecs of `marrow.kernels`.

Each encoding is a codec -- stream stages and a packer, paired with the
unpacker and stages that undo them; what is here is Parquet's byte layout
around it -- the block headers, the zigzag varints, the length prefixes --
and how a page's present values come out of an Arrow array and go back into
one. Its levels, dictionary codes and booleans are `RlePack`'s runs.

- `DeltaBinaryPackedEncoding` -- the first value, then each value less the
  one before, `OrderedEncode`d, packed by `ReferencePack(128)` over
  `BitPack(32)`: each block's smallest difference is its fit, the widths
  are `BitPack`'s.
- `DeltaLengthEncoding` -- `LengthsPack`, its lengths framed as
  DELTA_BINARY_PACKED.
- `DeltaByteArrayEncoding` -- `PrefixesPack`, its prefixes and its suffixes'
  lengths framed as DELTA_BINARY_PACKED.
- `PlainValues` · `DictionaryValues` · `ByteStreamSplitValues` -- a page's
  present values to and from an array, through `PlainPack`,
  `DictionaryPack` over `RlePack`, and `SplitPack`.
- `Encoding` -- the enum that dispatches a data page's present values to the
  right one, so the flat and nested reader paths share one decoder.
"""

from std.sys import size_of

from ...arrays import (
    BinaryLikeArray,
    BoolArray,
    BytesArray,
    DynArray,
    FixedSizeBinaryArray,
    PrimitiveArray,
)
from ...buffers import Buffer
from ...kernels.bitwise import ByteSwap, OrderedDecode, OrderedEncode
from ...kernels.cast import NumericCast
from ...kernels.compose import Codec, Compose, Decoder, Encoder, With
from ...kernels.dictionary import (
    DictionaryBytesPack,
    DictionaryPack,
    DictionaryPacked,
    DictionaryUnpack,
)
from ...kernels.numeric import AddKernel, SubKernel
from ...kernels.packing import (
    BitPack,
    BitPacked,
    BitUnpack,
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
    ReferencePacked,
    ReferenceUnpack,
    RlePack,
    RlePacked,
    RleUnpack,
    SplitPack,
    SplitPacked,
    SplitUnpack,
)
from ...kernels.scan import DiffScan, SumScan
from ...kernels.segments import PerSegment
from ...utils.byteorder import Leb128, LittleEndian
from ...utils.integers import Unsigned, ordered, unordered, unzigzag, zigzag
from ... import dtypes as dt
from ...errors import CorruptError, NotImplementedError
from ...views import BitmapView, BufferView


comptime _DbpCodec[T: DType] = Codec[
    Encoder[
        Compose[With[SubKernel, T], DiffScan, OrderedEncode],
        ReferencePack[BitPack],
    ],
    Decoder[
        ReferenceUnpack[BitUnpack],
        Compose[OrderedDecode[T], SumScan, With[AddKernel, T]],
        Unsigned[T],
    ],
]
"""DELTA_BINARY_PACKED after its first value: each value less the one before
-- the first value before the first -- as the unsigned integers that order as
the differences do, each block less its smallest, bit-packed by miniblock."""

comptime _PlainCodec[store: DType, phys: DType] = Codec[
    Encoder[Compose[NumericCast[phys]], PlainPack],
    Decoder[PlainUnpack, Compose[NumericCast[store]], phys],
]
"""PLAIN: `store` values as little-endian `phys`."""

comptime _SplitCodec[store: DType, phys: DType] = Codec[
    Encoder[Compose[NumericCast[phys]], SplitPack],
    Decoder[SplitUnpack, Compose[NumericCast[store]], phys],
]
"""BYTE_STREAM_SPLIT: `store` values as the byte planes of `phys`."""


def _dbp_codec[
    T: DType
](first: Scalar[T], block: Int, per: Int) -> _DbpCodec[T]:
    """`_DbpCodec` after `first`, in blocks of `block` values and miniblocks
    of `per`."""
    return _DbpCodec[T](
        Encoder(
            Compose((With[SubKernel, T](first), DiffScan(), OrderedEncode())),
            ReferencePack(block, BitPack(per)),
        ),
        Decoder[
            ReferenceUnpack[BitUnpack],
            Compose[OrderedDecode[T], SumScan, With[AddKernel, T]],
            Unsigned[T],
        ](
            ReferenceUnpack(block, BitUnpack(per)),
            Compose((OrderedDecode[T](), SumScan(), With[AddKernel, T](first))),
        ),
    )


def _plain_codec[store: DType, phys: DType]() -> _PlainCodec[store, phys]:
    return _PlainCodec[store, phys](
        Encoder(Compose((NumericCast[phys](),)), PlainPack()),
        Decoder[PlainUnpack, Compose[NumericCast[store]], phys](
            PlainUnpack(), Compose((NumericCast[store](),))
        ),
    )


def _split_codec[store: DType, phys: DType]() -> _SplitCodec[store, phys]:
    return _SplitCodec[store, phys](
        Encoder(Compose((NumericCast[phys](),)), SplitPack()),
        Decoder[SplitUnpack, Compose[NumericCast[store]], phys](
            SplitUnpack(), Compose((NumericCast[store](),))
        ),
    )


# Out of line: inlined, the copy is repeated in every decoder instantiated
# per dtype.
@no_inline
def _plain[T: DType](values: Span[Scalar[T], _]) -> PlainPacked:
    """`values`, copied as `PlainPack` packs them."""
    var n = len(values)
    var bytes = Buffer.alloc_uninit[T](n)
    bytes.extend(BufferView(values), 0, n)
    return PlainPacked(bytes^.to_immutable(), n * size_of[Scalar[T]]())


# ---------------------------------------------------------------------------
# DELTA_BINARY_PACKED and the byte-array encodings over it
# ---------------------------------------------------------------------------


struct DeltaBinaryPackedEncoding:
    """Parquet's DELTA_BINARY_PACKED: a header -- block size, miniblocks a
    block, how many values, the first one -- then each block's smallest
    difference, its miniblocks' widths and the miniblocks. After the first
    value it is `_DbpCodec`: each block's smallest is what `ReferencePack`
    fits, the widths are `BitPack`'s."""

    comptime BLOCK = 128
    comptime MINIBLOCKS = 4

    @staticmethod
    def encode[
        T: DType
    ](values: Span[Scalar[T], _], mut out: List[UInt8]) raises:
        """Append `values` in blocks of 128 and miniblocks of 32. A short
        last block's unused miniblocks get width 0 and no bytes."""
        comptime U = Unsigned[T]
        comptime B = Self.BLOCK
        comptime M = Self.MINIBLOCKS
        comptime PER = B // M
        var n = len(values)
        var first = values[0] if n > 0 else Scalar[T](0)
        Leb128.write(out, UInt64(B))
        Leb128.write(out, UInt64(M))
        Leb128.write(out, UInt64(n))
        Leb128.write(out, zigzag(first).cast[DType.uint64]())
        var packed = _dbp_codec(first, B, PER).encode(
            BufferView(values[1:] if n > 0 else values)
        )
        ref widths = packed.inner.widths
        var bits = packed.inner.bytes.view[DType.uint8](0, packed.inner.size)
        var deltas = max(n - 1, 0)
        var at = 0
        for b in range((deltas + B - 1) // B):
            var smallest = unordered[T](packed.fits.get[U](b))
            Leb128.write(out, zigzag(smallest).cast[DType.uint64]())
            var left = min(B, deltas - b * B)
            var used = (left + PER - 1) // PER
            for m in range(M):
                out.append(
                    widths.get[DType.uint8](b * M + m) if m < used else 0
                )
            for m in range(used):
                var width = Int(widths.get[DType.uint8](b * M + m))
                var bytes = (min(PER, left - m * PER) + 7) // 8 * width
                out.extend(bits.as_span()[at : at + bytes])
                at += bytes
                # A miniblock is stored whole: zeros past its last value.
                for _ in range(PER * width // 8 - bytes):
                    out.append(0)

    @staticmethod
    def decode[
        T: DType, exact: Bool = False
    ](data: Span[UInt8, _], count: Int, mut out: List[Scalar[T]]) raises -> Int:
        """Append the first `count` values of the blocks at the start of
        `data`; return where the last miniblock read ends. The blocks' parts
        -- smallest values, widths, miniblocks -- are gathered into the
        `ReferencePacked[BitPacked]` the codec decodes, which writes the
        values in place."""
        comptime U = Unsigned[T]
        var pos = 0
        var block_size: UInt64
        block_size, pos = Leb128.read(data, pos)
        var miniblocks: UInt64
        miniblocks, pos = Leb128.read(data, pos)
        var total: UInt64
        total, pos = Leb128.read(data, pos)
        var first_z: UInt64
        first_z, pos = Leb128.read(data, pos)
        # A byte-array page's bytes follow every length the stream holds,
        # so there the count must be the stream's.
        if UInt64(count) > total or (exact and UInt64(count) != total):
            raise CorruptError(
                t"parquet: {count} values from a stream of {total}"
            )
        if (
            miniblocks == 0
            or miniblocks > block_size
            or block_size > UInt64(1 << 31)
            or block_size % miniblocks != 0
            or (block_size // miniblocks) % UInt64(BitPack.GROUP) != 0
        ):
            raise CorruptError(
                t"parquet: {miniblocks} miniblocks in a block of {block_size}"
            )
        var block = Int(block_size)
        var mbs = Int(miniblocks)
        var per = block // mbs
        var first = unzigzag[T](first_z.cast[U]())
        var deltas = max(count - 1, 0)
        # A fit a block, a width a miniblock, and the miniblocks, which take
        # at most the bytes left.
        var fits = Buffer.alloc_uninit[U]((deltas + block - 1) // block)
        var widths = Buffer.alloc_uninit[DType.uint8]((deltas + per - 1) // per)
        var bits = Buffer.alloc_uninit[DType.uint8](len(data) - pos)
        var blocks = 0
        var used = 0
        var size = 0
        var done = 0
        while done < deltas:
            var smallest_z: UInt64
            smallest_z, pos = Leb128.read(data, pos)
            fits.unsafe_set[U](
                blocks, ordered(unzigzag[T](smallest_z.cast[U]()))
            )
            blocks += 1
            if pos + mbs > len(data):
                raise CorruptError("parquet: truncated miniblock widths")
            var widths_at = pos
            pos += mbs
            var left = min(block, deltas - done)
            for m in range((left + per - 1) // per):
                var width = Int(data[widths_at + m])
                if width > 64:
                    raise CorruptError(t"parquet: miniblock width {width}")
                var bytes = (min(per, left - m * per) + 7) // 8 * width
                if pos + bytes > len(data):
                    raise CorruptError("parquet: truncated miniblock")
                widths.unsafe_set[DType.uint8](used, UInt8(width))
                used += 1
                bits.extend(BufferView(data[pos : pos + bytes]), size, bytes)
                size += bytes
                # A miniblock is stored whole, its padding included.
                pos += min(per * width // 8, len(data) - pos)
            done += left
        var base = len(out)
        out.resize(unsafe_uninit_length=base + count)
        if count > 0:
            out[base] = first
            _dbp_codec(first, block, per).decode(
                ReferencePacked[BitPacked](
                    PerSegment(fits^.to_immutable(), blocks),
                    BitPacked(
                        bits^.to_immutable(),
                        size,
                        PerSegment(widths^.to_immutable(), used),
                    ),
                ),
                BufferView(Span(out)[base + 1 :]),
            )
        return pos


struct DeltaLengthEncoding:
    """Parquet's DELTA_LENGTH_BYTE_ARRAY: the byte strings' lengths as
    DELTA_BINARY_PACKED, then their bytes -- `LengthsPack`, its lengths
    framed as DELTA_BINARY_PACKED."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var packed = LengthsPack(PlainPack()).pack(offsets, data)
        DeltaBinaryPackedEncoding.encode(
            packed.lengths.bytes.view[DType.int32](
                0, packed.lengths.size // 4
            ).as_span(),
            out,
        )
        out.extend(packed.bytes.view[DType.uint8](0, packed.size).as_span())

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        """Append the `count` byte strings at the start of `src` to `data`,
        and where each ends to `offsets`; return the bytes read."""
        var lengths = List[Int32](capacity=count)
        var pos = DeltaBinaryPackedEncoding.decode[DType.int32, exact=True](
            src, count, lengths
        )
        var total = Self.total(lengths, len(src) - pos)
        LengthsUnpack(PlainUnpack()).unpack(
            Self.packed(lengths, src[pos : pos + total]), count, offsets, data
        )
        return pos + total

    @staticmethod
    def total(lengths: List[Int32], room: Int) raises -> Int:
        """The bytes `lengths` take, refused past `room`."""
        var total = 0
        for length in lengths:
            total += Int(length)
        if total < 0 or total > room:
            raise CorruptError("parquet: byte strings run past the page")
        return total

    @staticmethod
    def packed(
        lengths: List[Int32], bytes: Span[UInt8, _]
    ) raises -> LengthsPacked[PlainPacked]:
        """What `LengthsPack[PlainPack]` packs `lengths` and `bytes` as."""
        var held = _plain(bytes)
        return LengthsPacked(held.bytes, held.size, _plain(Span(lengths)))


struct DeltaByteArrayEncoding:
    """Parquet's DELTA_BYTE_ARRAY: the prefix each byte string shares with
    the one before as DELTA_BINARY_PACKED, then the rest as
    DELTA_LENGTH_BYTE_ARRAY -- `PrefixesPack`, its prefixes and its suffixes'
    lengths framed as DELTA_BINARY_PACKED."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var packed = PrefixesPack(PlainPack(), PlainPack()).pack(offsets, data)
        ref suffixes = packed.suffixes
        DeltaBinaryPackedEncoding.encode(
            packed.prefixes.bytes.view[DType.int32](
                0, packed.prefixes.size // 4
            ).as_span(),
            out,
        )
        DeltaBinaryPackedEncoding.encode(
            suffixes.lengths.bytes.view[DType.int32](
                0, suffixes.lengths.size // 4
            ).as_span(),
            out,
        )
        out.extend(suffixes.bytes.view[DType.uint8](0, suffixes.size).as_span())

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        """Append the `count` byte strings at the start of `src` to `data`,
        and where each ends to `offsets`; return the bytes read."""
        var prefixes = List[Int32](capacity=count)
        var pos = DeltaBinaryPackedEncoding.decode[DType.int32, exact=True](
            src, count, prefixes
        )
        var lengths = List[Int32](capacity=count)
        pos += DeltaBinaryPackedEncoding.decode[DType.int32, exact=True](
            src[pos:], count, lengths
        )
        var total = DeltaLengthEncoding.total(lengths, len(src) - pos)
        var packed = PrefixesPacked(
            _plain(Span(prefixes)),
            DeltaLengthEncoding.packed(lengths, src[pos : pos + total]),
        )
        PrefixesUnpack(PlainUnpack(), PlainUnpack()).unpack(
            packed^, count, offsets, data
        )
        return pos + total


# ---------------------------------------------------------------------------
# A page's present values, to and from an array
# ---------------------------------------------------------------------------


struct PlainValues:
    """The PLAIN codec — values laid out in order: fixed-width little-endian for
    primitives, bit-packed for booleans, 4-byte-length-prefixed for byte arrays.
    Encode takes a present-value Arrow array; decode returns the present values.
    """

    @staticmethod
    def encode_primitive[
        store: dt.PrimitiveType, phys: DType, big_endian: Bool = False
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """PLAIN fixed-width encode of the present values, `phys`-wide. `store`
        may be numeric, temporal, decimal, or interval; `big_endian` (for DECIMAL
        FIXED_LEN_BYTE_ARRAY) writes the two's-complement value most-significant
        byte first."""
        comptime T = store.native
        var present = Self.present(arr)
        comptime if big_endian:
            var packed = Codec(
                Encoder(Compose((ByteSwap(),)), PlainPack()),
                Decoder[PlainUnpack, Compose[ByteSwap], T](
                    PlainUnpack(), Compose((ByteSwap(),))
                ),
            ).encode(BufferView(Span(present)))
            out.extend(packed.bytes.view[DType.uint8](0, packed.size).as_span())
        else:
            var packed = _plain_codec[T, phys]().encode(
                BufferView(Span(present))
            )
            out.extend(packed.bytes.view[DType.uint8](0, packed.size).as_span())

    @staticmethod
    def present[
        T: dt.PrimitiveType
    ](arr: PrimitiveArray[T]) -> List[Scalar[T.native]]:
        """The values of `arr` that are not null, in order."""
        var present = List[Scalar[T.native]](
            capacity=arr.length - arr.null_count()
        )
        var values = arr.values()
        for i in range(arr.length):
            if arr.is_valid(i):
                present.append(values[i])
        return present^

    @staticmethod
    def encode_bool(arr: BoolArray, mut out: List[UInt8]) raises:
        """PLAIN booleans: the present values a bit each, least significant
        first."""
        var bits = arr.values()
        var start = len(out)
        var n = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                if n % 8 == 0:
                    out.append(0)
                if bits.test(i):
                    out[start + n // 8] |= UInt8(1) << UInt8(n % 8)
                n += 1

    @staticmethod
    def encode_bytes[A: BytesArray](arr: A, mut out: List[UInt8]) raises:
        """PLAIN byte arrays: each present value as its 4-byte little-endian
        length, then its bytes. Serves string/binary and their large_
        variants alike."""
        for i in range(len(arr)):
            if arr.is_valid(i):
                PlainValues.append_bytes(
                    out, arr.unsafe_get(UInt(i)).as_bytes()
                )

    @staticmethod
    def append_bytes(mut out: List[UInt8], bytes: Span[UInt8, _]):
        """One PLAIN byte array: its 4-byte little-endian length, then its
        bytes."""
        LittleEndian.append[DType.uint32](out, UInt32(len(bytes)))
        out.extend(bytes)

    @staticmethod
    def present_bytes[
        A: BytesArray
    ](arr: A, mut offsets: List[Int32], mut data: List[UInt8]) raises:
        """The present values of a byte-array column, nulls left out, as the
        offsets and data a byte-string packer takes."""
        offsets.append(Int32(len(data)))
        for i in range(len(arr)):
            if arr.is_valid(i):
                data.extend(arr.unsafe_get(UInt(i)).as_bytes())
                offsets.append(Int32(len(data)))

    @staticmethod
    def encode_fixed_size_binary(
        arr: FixedSizeBinaryArray, mut out: List[UInt8]
    ) raises:
        """FIXED_LEN_BYTE_ARRAY: the present values' `byte_width` bytes, no
        length prefix (the width is in the schema)."""
        for i in range(arr.length):
            if arr.is_valid(i):
                out.extend(Span(arr[i].value()))

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], dst: BufferView[mut=True, store, _]) raises:
        """Decode the `len(dst)` values at the start of `values`, each stored
        as a little-endian `phys`, into `dst`."""
        comptime PW = size_of[Scalar[phys]]()
        var np = len(dst)
        if len(values) < np * PW:
            raise CorruptError(t"parquet: {len(values)} bytes for {np} values")
        _plain_codec[store, phys]().decode(_plain(values[: np * PW]), dst)

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _],
        np: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        offsets.reserve(len(offsets) + np)
        var pos = 0
        for _ in range(np):
            var start, n = Self.locate(values, pos)
            data.extend(values[start : start + n])
            offsets.append(Int32(len(data)))

    @staticmethod
    @always_inline
    def locate(
        data: Span[UInt8, _], mut pos: Int
    ) raises CorruptError -> Tuple[Int, Int]:
        """Where the PLAIN byte array at `pos` starts and how many bytes it
        holds; `pos` moves past it."""
        if pos < 0 or pos + 4 > len(data):
            raise CorruptError(
                t"parquet: a byte array's length at byte {pos} runs past"
                t" {len(data)}"
            )
        var n = Int(LittleEndian.fixed[DType.uint32](data, pos))
        var start = pos + 4
        if n > len(data) - start:
            raise CorruptError(
                t"parquet: a byte array of {n} bytes at byte {start} runs past"
                t" {len(data)}"
            )
        pos = start + n
        return (start, n)

    @staticmethod
    def decode_bool(values: Span[UInt8, _], np: Int) raises -> List[Bool]:
        if (np + 7) // 8 > len(values):
            raise CorruptError(
                t"parquet: {len(values)} bytes for {np} booleans"
            )
        var bits = BitmapView(values)
        var out = List[Bool](capacity=np)
        for i in range(np):
            out.append(bits.test(i))
        return out^


struct DictionaryValues:
    """The dictionary codec — a dictionary page of distinct values, then data
    pages of RLE/bit-packed codes into it. `encode` builds both from a column;
    `read_page` and `read_page_bytes` read the dictionary page, once a column
    chunk, and `decode_into` a data page's codes, gathering the values."""

    # --- encode: column -> dictionary-page bytes + per-present-value indices ---

    @staticmethod
    def decode_into[
        T: DType
    ](
        values: Span[UInt8, _],
        count: Int,
        dict: PlainPacked,
        dst: BufferView[mut=True, T, _],
    ) raises:
        """Decode a data page's `count` codes -- their width in a byte, then
        their runs -- into `dst`, each the value of `dict`, the chunk's
        dictionary as `read_page` gave it, it names."""
        if len(values) == 0:
            if count > 0:
                raise CorruptError("parquet: a dictionary page with no codes")
        else:
            var width = Int(values[0])
            var runs = _plain(values[1:])
            Codec(
                DictionaryPack(RlePack(width)),
                DictionaryUnpack(RleUnpack(width)),
            ).decode(
                DictionaryPacked(dict.copy(), RlePacked(runs.bytes, runs.size)),
                dst,
            )

    @staticmethod
    def _encode_prim[
        store: dt.NumericType, phys: DType
    ](
        arr: PrimitiveArray[store],
        mut page: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a primitive column: PLAIN-encode each distinct value
        (widened to `phys`) into `page`, collect the per-value index.
        Values are keyed by their bits, so `0.0` and `-0.0` stay distinct."""
        comptime T = store.native
        var present = PlainValues.present(arr)
        var packed = Codec(
            DictionaryPack(PlainPack()), DictionaryUnpack(PlainUnpack())
        ).encode(BufferView(Span(present)))
        var distinct = packed.values.size // size_of[Scalar[T]]()
        var codes = packed.codes.bytes.view[DType.uint32](0, len(present))
        indices.reserve(len(indices) + len(codes))
        for i in range(len(codes)):
            indices.append(Int32(codes[i]))
        var plain = _plain_codec[T, phys]().encode(
            packed.values.bytes.view[T](0, distinct)
        )
        page.extend(plain.bytes.view[DType.uint8](0, plain.size).as_span())
        return distinct

    @staticmethod
    def _encode_bytes[
        A: BytesArray
    ](arr: A, mut page: List[UInt8], mut indices: List[Int32]) raises -> Int:
        """Dictionary-encode a byte-array column (string/binary, their large_
        and view variants): length-prefixed distinct values in the dictionary
        page, one index per present value."""
        var offsets = List[Int32]()
        var data = List[UInt8]()
        PlainValues.present_bytes(arr, offsets, data)
        var packed = DictionaryBytesPack(PlainPack()).pack(
            Span(offsets), Span(data)
        )
        # The distinct strings, their lengths and their bytes, make the
        # dictionary page: each as PLAIN writes it.
        ref strings = packed.strings
        var lengths = strings.lengths.bytes.view[DType.int32](
            0, strings.lengths.size // 4
        )
        var bytes = strings.bytes.view[DType.uint8](0, strings.size).as_span()
        var at = 0
        for i in range(len(lengths)):
            var length = Int(lengths[i])
            PlainValues.append_bytes(page, bytes[at : at + length])
            at += length
        var codes = packed.codes.bytes.view[DType.uint32](0, len(offsets) - 1)
        indices.reserve(len(indices) + len(codes))
        for i in range(len(codes)):
            indices.append(Int32(codes[i]))
        return len(lengths)

    @staticmethod
    def encode(
        dtype: dt.DynType,
        col: DynArray,
        mut page: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Build the dictionary page bytes + per-value indices for `col`; returns
        the dictionary size. Dispatches on the Arrow type like the PLAIN encoders
        (bool is never dictionary-encoded)."""

        # Integers narrower than 32 bits are stored as INT32; every other
        # numeric keeps its own type (FLOAT16 included).
        def numeric[
            T: dt.NumericType
        ](t: T) raises {imm col, mut page, mut indices} -> Int:
            comptime phys = DType.int32 if T.native.is_integral() and size_of[
                T.native
            ]() < 4 else T.native
            return Self._encode_prim[T, phys](
                col.get[PrimitiveArray[T]](), page, indices
            )

        if dtype.is_numeric():
            return dtype.try_visit[Int, dt.NumericType](
                numeric, "dtype is not numeric"
            )
        elif (
            dtype.is_binary_like()
            or dtype.is_string_view()
            or dtype.is_binary_view()
        ):

            def encode_bytes[
                A: BytesArray
            ](arr: A) raises {mut page, mut indices, imm} -> Int:
                return Self._encode_bytes(arr, page, indices)

            return col.try_visit[Int, BytesArray](
                encode_bytes, "expected a binary or string array"
            )
        else:
            raise NotImplementedError(
                t"parquet: cannot dictionary-encode column type {dtype}"
            )

    # --- decode ---

    @staticmethod
    def read_page[
        store: DType, phys: DType
    ](body: Span[UInt8, _], num_values: Int) raises -> PlainPacked:
        """A primitive dictionary page -- PLAIN `phys` values -- as the
        `store` values it holds: read once a column chunk, shared by every
        data page of it."""
        var values = Buffer.alloc_uninit[store](num_values)
        PlainValues.decode_primitive[store, phys](
            body, values.view[store](0, num_values)
        )
        return PlainPacked(
            values^.to_immutable(), num_values * size_of[Scalar[store]]()
        )

    @staticmethod
    def read_page_bytes[
        T: dt.BinaryLikeType
    ](body: Span[UInt8, _], num_values: Int) raises -> BinaryLikeArray[T]:
        """A byte-array dictionary page -- PLAIN length-prefixed values -- as
        an array: read once a column chunk, and taken from by its data
        pages."""
        comptime O = T.offset
        var offsets = Buffer.alloc_uninit[O](num_values + 1)
        offsets.unsafe_set[O](0, 0)
        # The values take less than the page, which holds a length each too.
        var values = Buffer.alloc_uninit[DType.uint8](len(body))
        var pos = 0
        var size = 0
        for i in range(num_values):
            var start, n = PlainValues.locate(body, pos)
            values.extend(BufferView(body[start : start + n]), size, n)
            size += n
            offsets.unsafe_set[O](i + 1, Scalar[O](size))
        return BinaryLikeArray[T](
            length=num_values,
            offsets=offsets^.to_immutable(),
            values=values^.to_immutable(),
        )


struct ByteStreamSplitValues:
    """The BYTE_STREAM_SPLIT codec — each value's bytes are transposed into
    per-byte planes (byte `k` of value `i` at `values[k*np + i]`)."""

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], dst: BufferView[mut=True, store, _]) raises:
        """Decode the `len(dst)` values at the start of `values`, the byte
        planes of `phys` values, into `dst`."""
        comptime B = size_of[Scalar[phys]]()
        var np = len(dst)
        if len(values) < np * B:
            raise CorruptError(t"parquet: {len(values)} bytes for {np} values")
        var planes = _plain(values[: np * B])
        _split_codec[store, phys]().decode(
            SplitPacked(planes.bytes, planes.size), dst
        )

    @staticmethod
    def encode[
        store: dt.NumericType, phys: DType
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """Transpose the present values into per-byte planes: byte `k` of value
        `i` at `out[base + k*np + i]` (inverse of `decode_primitive`)."""
        comptime T = store.native
        var present = PlainValues.present(arr)
        var packed = _split_codec[T, phys]().encode(BufferView(Span(present)))
        out.extend(packed.bytes.view[DType.uint8](0, packed.size).as_span())


@fieldwise_init
struct Encoding(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `Encoding` enum value. The decode methods dispatch a data page's
    *present* values to the per-encoding codec above; each appends `num_present`
    values from a value byte-span (nulls are placed later by the caller from the
    definition levels), so the same logic serves the flat and nested reader
    paths."""

    var code: Int

    comptime PLAIN = Self(0)
    comptime PLAIN_DICTIONARY = Self(2)
    comptime RLE = Self(3)
    comptime BIT_PACKED = Self(4)
    comptime DELTA_BINARY_PACKED = Self(5)
    comptime DELTA_LENGTH_BYTE_ARRAY = Self(6)
    comptime DELTA_BYTE_ARRAY = Self(7)
    comptime RLE_DICTIONARY = Self(8)
    comptime BYTE_STREAM_SPLIT = Self(9)

    def is_plain(self) -> Bool:
        return self == Self.PLAIN

    def is_dictionary(self) -> Bool:
        return self == Self.RLE_DICTIONARY or self == Self.PLAIN_DICTIONARY

    def decode_primitive[
        store: DType, phys: DType
    ](
        self,
        values: Span[UInt8, _],
        num_present: Int,
        dict: PlainPacked,
        mut out: List[Scalar[store]],
    ) raises:
        """Append the present fixed-width values (widened `phys` -> `store`);
        `dict` is the chunk's dictionary, as `DictionaryValues.read_page`
        gave it."""
        var base = len(out)
        out.resize(unsafe_uninit_length=base + num_present)
        var dst = BufferView(Span(out))[base:]
        if self.is_plain():
            PlainValues.decode_primitive[store, phys](values, dst)
        elif self.is_dictionary():
            DictionaryValues.decode_into[store](values, num_present, dict, dst)
        elif self == Self.DELTA_BINARY_PACKED:
            comptime if phys.is_integral() and phys.is_signed():
                if num_present > 0:
                    var decoded = List[Scalar[phys]](capacity=num_present)
                    _ = DeltaBinaryPackedEncoding.decode[phys](
                        values, num_present, decoded
                    )
                    # Iterated, not indexed: an index is bounds-checked.
                    var i = 0
                    for v in decoded:
                        dst.unsafe_set(i, v.cast[store]())
                        i += 1
            else:
                raise CorruptError(
                    t"parquet: DELTA_BINARY_PACKED over a {phys} column"
                )
        elif self == Self.BYTE_STREAM_SPLIT:
            ByteStreamSplitValues.decode_primitive[store, phys](values, dst)
        else:
            raise NotImplementedError(
                t"parquet: unsupported data page encoding {self.code}"
            )

    def decode_bytes(
        self,
        values: Span[UInt8, _],
        num_present: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """Append the present variable-length byte values of a page that is
        not dictionary-encoded to `data`, and where each ends to
        `offsets`."""
        if self.is_plain():
            PlainValues.decode_bytes(values, num_present, offsets, data)
        elif num_present == 0:
            pass
        elif self == Self.DELTA_LENGTH_BYTE_ARRAY:
            _ = DeltaLengthEncoding.decode(values, num_present, offsets, data)
        elif self == Self.DELTA_BYTE_ARRAY:
            _ = DeltaByteArrayEncoding.decode(
                values, num_present, offsets, data
            )
        else:
            raise NotImplementedError(
                t"parquet: unsupported byte-array encoding {self.code}"
            )

    def decode_flba(
        self, values: Span[UInt8, _], num_present: Int, width: Int
    ) raises -> List[UInt8]:
        """Decode the present FIXED_LEN_BYTE_ARRAY values (each `width` bytes) of
        a non-PLAIN/dict page into a contiguous `num_present * width` buffer — the
        DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT encodings PyArrow emits for decimal
        / fixed_size_binary. PLAIN and dictionary pages are read in place by the
        leaf builders, so they never reach here."""
        if self == Self.DELTA_BYTE_ARRAY:
            var offsets: List[Int32] = [0]
            var data = List[UInt8](capacity=num_present * width)
            if num_present > 0:
                _ = DeltaByteArrayEncoding.decode(
                    values, num_present, offsets, data
                )
            if len(data) != num_present * width:
                raise CorruptError(
                    t"parquet: DELTA_BYTE_ARRAY values are not {width} bytes"
                )
            return data^
        elif self == Self.BYTE_STREAM_SPLIT:
            # The width is the schema's, known only at run time: the planes
            # merged a byte at a time.
            if len(values) < num_present * width:
                raise CorruptError("parquet: BYTE_STREAM_SPLIT runs short")
            var out = List[UInt8](length=num_present * width, fill=0)
            for i in range(num_present):
                for k in range(width):
                    out[i * width + k] = values[k * num_present + i]
            return out^
        else:
            raise NotImplementedError(
                t"parquet: unsupported FIXED_LEN_BYTE_ARRAY encoding "
                t"{self.code}"
            )

    def decode_bool(
        self, values: Span[UInt8, _], num_present: Int
    ) raises -> List[Bool]:
        """Return the present booleans — PLAIN bit-packed, or RLE (the encoding
        arrow/PyArrow use for boolean values in DataPage v2). An RLE boolean
        stream is a 4-byte little-endian length then a width-1 RLE/bit-packed
        hybrid run, exactly like a level stream."""
        if self.is_plain():
            return PlainValues.decode_bool(values, num_present)
        elif self == Self.RLE:
            var length = Int(LittleEndian.checked[DType.uint32](values, 0))
            if length > len(values) - 4:
                raise CorruptError(
                    t"parquet: {length} bytes of RLE booleans in a page of"
                    t" {len(values) - 4}"
                )
            var decoded = List[Int32](capacity=num_present)
            _ = RleUnpack.read_runs[DType.int32](
                values[4 : 4 + length], 1, num_present, decoded
            )
            var out = List[Bool](capacity=num_present)
            for b in decoded:
                out.append(b == 1)
            return out^
        else:
            raise NotImplementedError(
                "parquet: non-plain bool encoding not supported"
            )
