# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page encodings, as cascades of `marrow.codecs`.

Each encoding is a cascade; what is here is Parquet's byte layout around it
-- the run headers, the block headers, the zigzag varints -- and how a page's
present values come out of an Arrow array and go back into one:

- `RleEncoding` -- RLE / bit-packed runs over `Runs`: levels, dictionary
  codes, booleans.
- `DeltaBinaryPackedEncoding` -- the first value, then `Delta(first)`,
  `Ordered`, `FrameOfReference(128)`, `BitPack(32)`.
- `DeltaLengthEncoding` -- `Lengths`, the lengths as DELTA_BINARY_PACKED.
- `DeltaByteArrayEncoding` -- `Prefixes`, its prefixes and its suffixes'
  lengths as DELTA_BINARY_PACKED.
- `PlainValues` · `DictionaryValues` · `ByteStreamSplitValues` -- a page's
  present values to and from an array, through `Plain`, `Dictionary` and
  `ByteStreamSplit`.
- `Encoding` -- the enum that dispatches a data page's present values to the
  right one, so the flat and nested reader paths share one decoder.
"""

from std.memory import bitcast
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
    Codec,
    Delta,
    Dictionary,
    Encoded,
    FrameOfReference,
    Lengths,
    Ordered,
    Plain,
    Prefixes,
    Runs,
)
from ..utils.byteorder import Leb128, LittleEndian
from ..utils.integers import Unsigned, ordered, unordered, unzigzag, zigzag
from ..codecs.core import Unpacked
from .. import dtypes as dt
from ..errors import CorruptError, NotImplementedError
from ..views import BitmapView, BufferView, PackedGroup


# ---------------------------------------------------------------------------
# RLE / bit-packed runs
# ---------------------------------------------------------------------------


struct RleRuns(Copyable, Movable):
    """Where reading Parquet's RLE runs is: the next header, and the run
    being read -- one value repeated, or bit-packed groups of eight. Read a
    stretch of values at a time, each picking up where the last left off."""

    var pos: Int
    """The byte after the run being read."""
    var left: Int
    """Its values not yet read."""
    var packed: Bool
    var value: UInt64
    """A repeated run's value."""
    var at: Int
    """The byte a packed run's groups start at."""
    var read: Int
    """A packed run's values read."""

    def __init__(out self):
        self.pos = 0
        self.left = 0
        self.packed = False
        self.value = 0
        self.at = 0
        self.read = 0

    def fill[
        T: DType
    ](
        mut self,
        data: BufferView[DType.uint8, _],
        group: PackedGroup,
        dst: BufferView[mut=True, T, _],
    ) -> Bool:
        """Read `len(dst)` values into `dst`; False when the runs end, or a
        run runs past `data`, first."""
        comptime G = RleEncoding.GROUP
        var width = group.width
        var produced = 0
        while produced < len(dst):
            if self.left == 0:
                var header, end = Leb128.scan(data.as_span(), self.pos)
                if end < 0:
                    return False
                self.pos = end
                self.packed = (header & 1) == 1
                var n = header >> 1
                if self.packed:
                    if n > UInt64((len(data) - self.pos) // max(width, 1)):
                        return False
                    self.at = self.pos
                    self.read = 0
                    self.left = Int(n) * G
                    self.pos += Int(n) * width
                else:
                    var bytes = (width + 7) // 8
                    if self.pos + bytes > len(data) or n > UInt64(Int.MAX):
                        return False
                    self.value = LittleEndian.partial[DType.uint64](
                        data.as_span()[self.pos : self.pos + bytes], 0
                    )
                    self.pos += bytes
                    self.left = Int(n)
                continue
            var take = min(self.left, len(dst) - produced)
            if not self.packed:
                dst.as_span()[produced : produced + take].fill(
                    self.value.cast[T]()
                )
            else:
                var end = self.read + take
                var to = produced
                while self.read < end:
                    var lane = self.read % G
                    var values = group.unpack(
                        data, self.at + (self.read // G) * width
                    ).cast[T]()
                    if lane == 0 and self.read + G <= end:
                        dst.store[G](to, values)
                        to += G
                        self.read += G
                    else:
                        var stop = min(G, lane + end - self.read)
                        for k in range(lane, stop):
                            dst.unsafe_set(to, values[k])
                            to += 1
                        self.read += stop - lane
            self.left -= take
            produced += take
        return True


struct RleEncoding(Codec):
    """Parquet's RLE encoding: runs of one value -- a header, then the value
    in its whole bytes -- or bit-packed groups of eight, `width` bits each.
    A run is written as one only where a stretch of packed groups may end:
    at its start or after a whole group. Encoding takes the runs `Runs`
    finds and lays them out so; decoding reads them a block at a time, so
    as a cascade's store -- `Cascade[Dictionary[], RleEncoding]` -- it feeds
    the codecs before it. It takes unsigned integers of `width` bits."""

    comptime GROUP = 8

    var width: Int
    var values: Buffer[mut=True]
    """The values being encoded."""
    var taken: Int
    var node: Encoded
    """The runs being decoded."""
    var runs: RleRuns
    var group: PackedGroup
    var unpacked: Unpacked
    var bad: Bool
    """Whether the runs ended before the values."""

    def __init__(out self):
        self = Self(0)

    def __init__(out self, width: Int):
        self.width = width
        self.values = Buffer.alloc_uninit(0)
        self.taken = 0
        self.node = Encoded()
        self.runs = RleRuns()
        self.group = PackedGroup(width)
        self.unpacked = Unpacked()
        self.bad = False

    def start[T: DType](mut self, count: Int) raises:
        self.values = Buffer.alloc_uninit[T](count)
        self.taken = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        self.values.extend(src, self.taken, len(src))
        self.taken += len(src)

    def finish[T: DType](mut self) raises -> Encoded:
        var values = self.values^.to_immutable()
        self.values = Buffer.alloc_uninit(0)
        var out = List[UInt8]()
        Self.write_runs(
            values.view[T](0, self.taken).as_span(), self.width, out
        )
        return Encoded(bytes=Span(out))

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        self.node = node.copy()
        self.runs = RleRuns()
        self.unpacked.start(count)
        self.bad = False

    def _unpack[T: DType](mut self):
        var out = self.unpacked.next[T]()
        var n = self.unpacked.size()
        if self.width == 0:
            out.as_span()[:n].fill(0)
        elif not self.runs.fill[T](
            BufferView(self.node.bytes()), self.group, out[:n]
        ):
            self.bad = True

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        if self.unpacked.due(i):
            self._unpack[T]()
        return self.unpacked.load[T, W](i)

    def check(self) raises:
        if self.bad:
            raise CorruptError("parquet: truncated RLE runs")

    @staticmethod
    def write_runs[
        T: DType
    ](values: Span[Scalar[T], _], width: Int, mut out: List[UInt8]) raises:
        """Append `values` at `width` bits; a `width` of 0 writes nothing."""
        comptime G = Self.GROUP
        comptime U = Unsigned[T]
        if width == 0:
            return
        var codec = Runs[]()
        var runs = codec.encode_view(BufferView(values))
        var run_values = runs[Runs[].VALUES].values[T]()
        var run_lengths = runs[Runs[].LENGTHS].values[DType.uint64]()
        var byte_width = (width + 7) // 8
        var literals = List[UInt64]()
        for r in range(len(run_values)):
            var v = bitcast[U](run_values[r]).cast[DType.uint64]()
            var n = Int(run_lengths[r])
            # A run's first values finish the open group of literals.
            var fill = min((G - len(literals) % G) % G, n)
            for _ in range(fill):
                literals.append(v)
            n -= fill
            if n >= G:
                Self._literals(literals, width, out)
                Leb128.write(out, UInt64(n) << 1)
                LittleEndian.append(out, v, byte_width)
            else:
                for _ in range(n):
                    literals.append(v)
        Self._literals(literals, width, out)

    @staticmethod
    def _literals(
        mut literals: List[UInt64], width: Int, mut out: List[UInt8]
    ) raises:
        """Write `literals` as a bit-packed run, the last group padded with
        zeros, and clear them."""
        comptime G = Self.GROUP
        if len(literals) == 0:
            return
        var groups = (len(literals) + G - 1) // G
        Leb128.write(out, (UInt64(groups) << 1) | 1)
        var start = len(out)
        out.resize(start + groups * width, 0)
        var bits = BitmapView(Span(out)[start:])
        var values = BufferView(Span(literals))
        for g in range(groups):
            var lo = g * G
            PackedGroup.pack(
                bits, lo * width, values, lo, min(lo + G, len(literals)), width
            )
        literals.clear()

    @staticmethod
    def read_runs[
        T: DType
    ](
        data: Span[UInt8, _], width: Int, count: Int, mut out: List[Scalar[T]]
    ) raises -> Int:
        """Append the first `count` values of the runs in `data`, `width`
        bits each; return where the runs read end. A run past `count` is
        read whole, as a packed run's padding is."""
        var base = len(out)
        out.resize(unsafe_uninit_length=base + count)
        var dst = BufferView(Span(out))[base:]
        if width == 0:
            dst.as_span().fill(0)
            return 0
        var runs = RleRuns()
        if not runs.fill[T](BufferView(data), PackedGroup(width), dst):
            raise CorruptError("parquet: truncated RLE runs")
        return runs.pos

    @staticmethod
    def count(
        data: Span[UInt8, _], width: Int, count: Int, target: UInt64
    ) raises -> Int:
        """How many of the first `count` values are `target` -- without
        decoding them when the runs are one run, as a column with no nulls'
        levels are."""
        if count == 0:
            return 0
        if width == 0:
            return count if target == 0 else 0
        var header, pos = Leb128.read(data, 0)
        var byte_width = (width + 7) // 8
        if (header & 1) == 0 and header >> 1 >= UInt64(count):
            if pos + byte_width > len(data):
                raise CorruptError("parquet: truncated RLE run")
            var v = LittleEndian.partial[DType.uint64](
                data[pos : pos + byte_width], 0
            )
            return count if v == target else 0
        var values = List[UInt32](capacity=count)
        _ = Self.read_runs[DType.uint32](data, width, count, values)
        var matches = 0
        for v in values:
            if UInt64(v) == target:
                matches += 1
        return matches


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
        var low = encoded[2][FrameOfReference[].LOWS].values[U]()
        var w = encoded[3][BitPack[].WIDTHS].values[DType.uint8]()
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
        T: DType
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
        if UInt64(count) > total:
            raise CorruptError(
                t"parquet: {count} values from a stream of {total}"
            )
        if (
            miniblocks == 0
            or miniblocks > block_size
            or block_size > UInt64(1 << 31)
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
        out.extend(encoded[0][Lengths[].BYTES].bytes())

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
        var pos = DeltaBinaryPackedEncoding.decode[DType.int32](
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
        ref suffixes = encoded[0][Prefixes[].SUFFIXES]
        DeltaBinaryPackedEncoding.encode(
            encoded[1].values[DType.int32]().as_span(), out
        )
        DeltaBinaryPackedEncoding.encode(
            suffixes[1].values[DType.int32]().as_span(), out
        )
        out.extend(suffixes[0][Lengths[].BYTES].bytes())

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
        var pos = DeltaBinaryPackedEncoding.decode[DType.int32](
            src, count, prefixes
        )
        var lengths = List[Int32](capacity=count)
        pos += DeltaBinaryPackedEncoding.decode[DType.int32](
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
            raise CorruptError("parquet: a dictionary page with no codes")
        # A `Dictionary` node, its distinct values `Plain`'s slot, then
        # the store's.
        var tree = Encoded(
            count, [Encoded(0, [dict.copy()]), Encoded(bytes=values[1:])]
        )
        var cascade = Cascade((Dictionary(), RleEncoding(Int(values[0]))))
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
        var distinct = encoded[0][Dictionary[].DISTINCT].values[store.native]()
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
        ref strings = encoded[0][Dictionary[].DISTINCT]
        var lengths = strings[1].values[DType.int32]()
        var bytes = strings[0][Lengths[].BYTES].bytes()
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
            var length = Int(LittleEndian.fixed[DType.uint32](values, 0))
            var decoded = List[Int32](capacity=num_present)
            _ = RleEncoding.read_runs[DType.int32](
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
