# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page encodings, as cascades of `marrow.codecs`.

Each encoding is a cascade; what is here is Parquet's byte layout around it
-- the block headers, the zigzag varints, the length prefixes -- and how a
page's present values come out of an Arrow array and go back into one. Its
levels, dictionary codes and booleans are `marrow.codecs.Rle`'s runs.

- `DeltaBinaryPackedEncoding` -- the first value, then `Delta(first)`,
  `Ordered`, `FrameOfReference(128)`, `BitPack(32)`.
- `DeltaLengthEncoding` -- `Lengths`, the lengths as DELTA_BINARY_PACKED.
- `DeltaByteArrayEncoding` -- `Prefixes`, its prefixes and its suffixes'
  lengths as DELTA_BINARY_PACKED.
- `PlainValues` · `DictionaryValues` · `ByteStreamSplitValues` -- a page's
  present values to and from an array, through `Plain`, `Dictionary` over
  `Rle`, and `ByteStreamSplit`.
- `Encoding` -- the enum that dispatches a data page's present values to the
  right one, so the flat and nested reader paths share one decoder.
"""

from std.sys import size_of

from ..arrays import (
    BoolArray,
    BytesArray,
    DynArray,
    FixedSizeBinaryArray,
    PrimitiveArray,
)
from ..buffers import Buffer
from ..codecs import (
    BitPack,
    ByteStreamSplit,
    ByteSwap,
    Cascade,
    Cast,
    Delta,
    Dictionary,
    Encoded,
    FrameOfReference,
    Lengths,
    Ordered,
    Plain,
    Prefixes,
    Rle,
)
from ..utils.byteorder import Leb128, LittleEndian
from ..utils.integers import Unsigned, ordered, unordered, unzigzag, zigzag
from .. import dtypes as dt
from ..errors import CorruptError, NotImplementedError
from ..views import BitmapView, BufferView


# ---------------------------------------------------------------------------
# DELTA_BINARY_PACKED and the byte-array encodings over it
# ---------------------------------------------------------------------------


struct DeltaBinaryPackedEncoding:
    """Parquet's DELTA_BINARY_PACKED: a header -- block size, miniblocks a
    block, how many values, the first one -- then each block's smallest
    difference, its miniblocks' widths and the miniblocks. After the first
    value it frames `Delta(first), Ordered, FrameOfReference(block),
    BitPack(miniblock)`: each block's smallest is what `FrameOfReference` sets
    apart, the widths are `BitPack`'s."""

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
        var cascade = Cascade(
            (Delta(first), Ordered(), FrameOfReference(B), BitPack(PER))
        )
        var encoded = cascade.encode_view(
            BufferView(values[1:] if n > 0 else values)
        )
        var low = FrameOfReference.lows[U](encoded[2])
        var w = BitPack.widths(encoded[3])
        var packed = encoded[3].bytes()
        var deltas = max(n - 1, 0)
        var at = 0
        for b in range((deltas + B - 1) // B):
            var smallest = unordered[T](low[b])
            Leb128.write(out, zigzag(smallest).cast[DType.uint64]())
            var left = min(B, deltas - b * B)
            var used = (left + PER - 1) // PER
            for m in range(M):
                out.append(w[b * M + m] if m < used else 0)
            for m in range(used):
                var width = Int(w[b * M + m])
                var bytes = (min(PER, left - m * PER) + 7) // 8 * width
                out.extend(packed[at : at + bytes])
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
        -- smallest values, widths, miniblocks -- are gathered into the tree
        the cascade decodes, which writes the values in place."""
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
        var lows = List[Scalar[U]]()
        var widths = List[UInt8]()
        var packed = List[UInt8]()
        var deltas = max(count - 1, 0)
        var done = 0
        while done < deltas:
            var smallest_z: UInt64
            smallest_z, pos = Leb128.read(data, pos)
            lows.append(ordered(unzigzag[T](smallest_z.cast[U]())))
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
                widths.append(UInt8(width))
                packed.extend(data[pos : pos + bytes])
                # A miniblock is stored whole, its padding included.
                pos += min(per * width // 8, len(data) - pos)
            done += left
        var bits = Encoded(bytes=Span(packed))
        bits.children.append(Encoded(Span(widths)))
        var tree = Encoded(
            deltas,
            [
                Encoded(),
                Encoded(),
                Encoded(0, [Encoded(Span(lows))]),
                bits^,
            ],
        )
        var cascade = Cascade(
            (Delta(first), Ordered(), FrameOfReference(block), BitPack(per))
        )
        var base = len(out)
        out.resize(unsafe_uninit_length=base + count)
        if count > 0:
            out[base] = first
            cascade.decode_into[T](tree, BufferView(Span(out)[base + 1 :]))
        return pos


struct DeltaLengthEncoding:
    """Parquet's DELTA_LENGTH_BYTE_ARRAY: the byte strings' lengths as
    DELTA_BINARY_PACKED, then their bytes -- `Lengths`, its lengths framed
    as DELTA_BINARY_PACKED."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var cascade = Cascade[Lengths[], Plain]()
        var encoded = cascade.encode(offsets, data)
        DeltaBinaryPackedEncoding.encode(
            encoded[1].values[DType.int32]().as_span(), out
        )
        out.extend(Lengths.bytes(encoded[0]))

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
        var cascade = Cascade[Lengths[], Plain]()
        cascade.decode(
            Self.tree(lengths, src[pos : pos + total]), offsets, data
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
    def tree(lengths: List[Int32], bytes: Span[UInt8, _]) -> Encoded:
        """What `Cascade[Lengths[], Plain]` stores `lengths` and `bytes`
        as."""
        return Encoded(
            len(lengths),
            [
                Encoded(0, [Encoded(bytes)]),
                Encoded(Span(lengths)),
            ],
        )


struct DeltaByteArrayEncoding:
    """Parquet's DELTA_BYTE_ARRAY: the prefix each byte string shares with
    the one before as DELTA_BINARY_PACKED, then the rest as
    DELTA_LENGTH_BYTE_ARRAY -- `Prefixes`, its prefixes and its suffixes'
    lengths framed as DELTA_BINARY_PACKED."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var cascade = Cascade[Prefixes[], Plain]()
        var encoded = cascade.encode(offsets, data)
        ref suffixes = Prefixes.suffixes(encoded[0])
        DeltaBinaryPackedEncoding.encode(
            encoded[1].values[DType.int32]().as_span(), out
        )
        DeltaBinaryPackedEncoding.encode(
            suffixes[1].values[DType.int32]().as_span(), out
        )
        out.extend(Lengths.bytes(suffixes[0]))

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
        var suffixes = DeltaLengthEncoding.tree(lengths, src[pos : pos + total])
        var tree = Encoded(
            count, [Encoded(0, [suffixes^]), Encoded(Span(prefixes))]
        )
        var cascade = Cascade[Prefixes[], Plain]()
        cascade.decode(tree, offsets, data)
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
        var present = Self.present(arr)
        comptime if big_endian:
            var cascade = Cascade[ByteSwap, Plain]()
            out.extend(
                cascade.encode_view(BufferView(Span(present)))[1].bytes()
            )
        else:
            var cascade = Cascade[Cast[phys], Plain]()
            out.extend(
                cascade.encode_view(BufferView(Span(present)))[1].bytes()
            )

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
        offsets and data a byte-string codec takes."""
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
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        """Append the `np` values at the start of `values`, each stored as
        a little-endian `phys`."""
        comptime PW = size_of[Scalar[phys]]()
        if len(values) < np * PW:
            raise CorruptError(t"parquet: {len(values)} bytes for {np} values")
        var tree = Encoded(np, [Encoded(), Encoded(bytes=values[: np * PW])])
        var base = len(out)
        out.resize(unsafe_uninit_length=base + np)
        var cascade = Cascade[Cast[phys], Plain]()
        cascade.decode_into[store](tree, BufferView(Span(out))[base:])

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
    pages of RLE/bit-packed indices into it. `encode` builds both from a column;
    `decode_page_*` reads the dictionary page and `decode_*` reads a data page's
    indices and gathers the values."""

    # --- encode: column -> dictionary-page bytes + per-present-value indices ---

    @staticmethod
    def decode_into[
        T: DType
    ](
        values: Span[UInt8, _],
        count: Int,
        ref dict: Encoded,
        dst: BufferView[mut=True, T, _],
    ) raises:
        """Decode a data page's `count` codes -- their width in a byte, then
        their runs -- into `dst`, each the value of `dict`, the chunk's
        dictionary as `read_page` gave it, it names."""
        if len(values) == 0:
            if count > 0:
                raise CorruptError("parquet: a dictionary page with no codes")
            return
        # The `Dictionary` node holds the distinct values as its one slot;
        # the `Rle` node holds the codes.
        var tree = Encoded(
            count, [Encoded(0, [dict.copy()]), Encoded(bytes=values[1:])]
        )
        var cascade = Cascade((Dictionary(), Rle(Int(values[0]))))
        cascade.decode_into[T](tree, dst)

    @staticmethod
    def _encode_prim[
        store: dt.NumericType, phys: DType
    ](
        arr: PrimitiveArray[store],
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a primitive column: PLAIN-encode each distinct value
        (widened to `phys`) into `dict_body`, collect the per-value index.
        Values are keyed by their bits, so `0.0` and `-0.0` stay distinct."""
        var present = PlainValues.present(arr)
        var cascade = Cascade[Dictionary[], Plain]()
        var encoded = cascade.encode_view(BufferView(Span(present)))
        var distinct = Dictionary.distinct(encoded[0]).values[store.native]()
        var codes = encoded[1].values[DType.uint32]()
        indices.reserve(len(indices) + len(codes))
        for i in range(len(codes)):
            indices.append(Int32(codes[i]))
        var page = Cascade[Cast[phys], Plain]()
        dict_body.extend(
            page.encode_view(BufferView(distinct.as_span()))[1].bytes()
        )
        return len(distinct)

    @staticmethod
    def _encode_bytes[
        A: BytesArray
    ](
        arr: A,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a byte-array column (string/binary, their large_
        and view variants): length-prefixed distinct values in the dictionary
        page, one index per present value."""
        var offsets = List[Int32]()
        var data = List[UInt8]()
        PlainValues.present_bytes(arr, offsets, data)
        var cascade = Cascade[Dictionary[], Plain]()
        var encoded = cascade.encode(Span(offsets), Span(data))
        # The distinct strings, stored as `Lengths`, then `Plain`, make the
        # dictionary page: each as PLAIN writes it.
        ref strings = Dictionary.distinct(encoded[0])
        var lengths = strings[1].values[DType.int32]()
        var bytes = Lengths.bytes(strings[0])
        var at = 0
        for i in range(len(lengths)):
            var length = Int(lengths[i])
            PlainValues.append_bytes(dict_body, bytes[at : at + length])
            at += length
        var codes = encoded[1].values[DType.uint32]()
        indices.reserve(len(indices) + len(codes))
        for i in range(len(codes)):
            indices.append(Int32(codes[i]))
        return len(lengths)

    @staticmethod
    def encode(
        dtype: dt.DynType,
        col: DynArray,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Build the dictionary page bytes + per-value indices for `col`; returns
        the dictionary size. Dispatches on the Arrow type like the PLAIN encoders
        (bool is never dictionary-encoded)."""
        if dtype == dt.int32:
            return Self._encode_prim[dt.Int32Type, DType.int32](
                col.as_int32(), dict_body, indices
            )
        elif dtype == dt.int64:
            return Self._encode_prim[dt.Int64Type, DType.int64](
                col.as_int64(), dict_body, indices
            )
        elif dtype == dt.uint32:
            return Self._encode_prim[dt.UInt32Type, DType.uint32](
                col.as_uint32(), dict_body, indices
            )
        elif dtype == dt.uint64:
            return Self._encode_prim[dt.UInt64Type, DType.uint64](
                col.as_uint64(), dict_body, indices
            )
        elif dtype == dt.float32:
            return Self._encode_prim[dt.Float32Type, DType.float32](
                col.as_float32(), dict_body, indices
            )
        elif dtype == dt.float64:
            return Self._encode_prim[dt.Float64Type, DType.float64](
                col.as_float64(), dict_body, indices
            )
        elif dtype == dt.float16:
            return Self._encode_prim[dt.Float16Type, DType.float16](
                col.as_float16(), dict_body, indices
            )
        elif dtype == dt.int8:
            return Self._encode_prim[dt.Int8Type, DType.int32](
                col.as_int8(), dict_body, indices
            )
        elif dtype == dt.int16:
            return Self._encode_prim[dt.Int16Type, DType.int32](
                col.as_int16(), dict_body, indices
            )
        elif dtype == dt.uint8:
            return Self._encode_prim[dt.UInt8Type, DType.int32](
                col.as_uint8(), dict_body, indices
            )
        elif dtype == dt.uint16:
            return Self._encode_prim[dt.UInt16Type, DType.int32](
                col.as_uint16(), dict_body, indices
            )
        elif (
            dtype.is_binary_like()
            or dtype.is_string_view()
            or dtype.is_binary_view()
        ):

            def encode_bytes[
                A: BytesArray
            ](arr: A) raises {mut dict_body, mut indices, imm} -> Int:
                return Self._encode_bytes(arr, dict_body, indices)

            return col.dispatch_bytes(encode_bytes)
        else:
            raise NotImplementedError(
                t"parquet: cannot dictionary-encode column type {dtype}"
            )

    # --- decode ---

    @staticmethod
    def read_page[
        store: DType, phys: DType
    ](body: Span[UInt8, _], num_values: Int) raises -> Encoded:
        """A primitive dictionary page -- PLAIN `phys` values -- as `Plain`'s
        node of `store` values: read once a column chunk, shared by every
        data page of it."""
        var values = List[Scalar[store]](capacity=num_values)
        PlainValues.decode_primitive[store, phys](body, num_values, values)
        return Encoded(Span(values))

    @staticmethod
    def decode_page_bytes(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_body: List[UInt8],
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Read a byte-array dictionary page: its bytes into `dict_body`, and
        where each value sits in them (see `byte_offsets`)."""
        dict_body.clear()
        dict_body.extend(body)
        Self.byte_offsets(body, num_values, dict_off, dict_len)

    @staticmethod
    def byte_offsets(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Where each value of a byte-array dictionary page starts and how
        long it is: the page is length-prefixed values, so each offset points
        past its 4-byte prefix. Replaces what the lists held -- the indices of
        a data page address one chunk's dictionary."""
        dict_off.clear()
        dict_len.clear()
        var pos = 0
        for _ in range(num_values):
            var start, n = PlainValues.locate(body, pos)
            dict_off.append(start)
            dict_len.append(n)

    @staticmethod
    def decode_primitive[
        store: DType
    ](
        values: Span[UInt8, _],
        np: Int,
        ref dict: Encoded,
        mut out: List[Scalar[store]],
    ) raises:
        var base = len(out)
        out.resize(unsafe_uninit_length=base + np)
        Self.decode_into[store](values, np, dict, BufferView(Span(out))[base:])


struct ByteStreamSplitValues:
    """The BYTE_STREAM_SPLIT codec — each value's bytes are transposed into
    per-byte planes (byte `k` of value `i` at `values[k*np + i]`)."""

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        comptime B = size_of[Scalar[phys]]()
        if len(values) < np * B:
            raise CorruptError(t"parquet: {len(values)} bytes for {np} values")
        var tree = Encoded(np, [Encoded(), Encoded(bytes=values[: np * B])])
        var base = len(out)
        out.resize(unsafe_uninit_length=base + np)
        var cascade = Cascade[Cast[phys], ByteStreamSplit[]]()
        cascade.decode_into[store](tree, BufferView(Span(out))[base:])

    @staticmethod
    def encode[
        store: dt.NumericType, phys: DType
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """Transpose the present values into per-byte planes: byte `k` of value
        `i` at `out[base + k*np + i]` (inverse of `decode_primitive`)."""
        var present = PlainValues.present(arr)
        var cascade = Cascade[Cast[phys], ByteStreamSplit[]]()
        out.extend(cascade.encode_view(BufferView(Span(present)))[1].bytes())


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
        ref dict: Encoded,
        mut out: List[Scalar[store]],
    ) raises:
        """Append the present fixed-width values (widened `phys` -> `store`);
        `dict` is the chunk's dictionary, as `DictionaryValues.read_page`
        gave it."""
        if self.is_plain():
            PlainValues.decode_primitive[store, phys](values, num_present, out)
        elif self.is_dictionary():
            DictionaryValues.decode_primitive[store](
                values, num_present, dict, out
            )
        elif self == Self.DELTA_BINARY_PACKED:
            comptime if phys.is_integral() and phys.is_signed():
                if num_present == 0:
                    return
                var decoded = List[Scalar[phys]](capacity=num_present)
                _ = DeltaBinaryPackedEncoding.decode[phys](
                    values, num_present, decoded
                )
                for v in decoded:
                    out.append(v.cast[store]())
            else:
                raise CorruptError(
                    t"parquet: DELTA_BINARY_PACKED over a {phys} column"
                )
        elif self == Self.BYTE_STREAM_SPLIT:
            ByteStreamSplitValues.decode_primitive[store, phys](
                values, num_present, out
            )
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
            _ = Rle.read_runs[DType.int32](
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
