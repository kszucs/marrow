# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Stores, the codec that ends a cascade and holds its values' bytes:
`Plain` and `ByteStreamSplit` their bytes as they are, `BitPack` the bits
each frame needs, `Runs` each run once, `Frequency` each frame's commonest
value once, `Buckets` and `Gorilla` each value in as many bits as it needs,
and `Compressed[C]` `Plain`'s bytes compressed by a `Compressor`."""

from std.bit import bit_width, count_leading_zeros, count_trailing_zeros
from std.memory import bitcast
from std.sys import bit_width_of, size_of

from ..buffers import Buffer
from ..errors import CorruptError
from ..views import BitmapView, BufferView, PackedGroup
from .core import (
    Cascade,
    Codec,
    Encoded,
    Exceptions,
    FrameValues,
    Unpacked,
)
from ..compress import Compressor
from ..utils.bits import BitCursor
from ..utils.integers import Unsigned


struct Plain(Codec):
    """The values' bytes, little-endian -- Parquet's PLAIN.

    [1, 258] as uint16  ->  01 00 02 01
    """

    comptime STORES_VALUES = True

    var bytes: Buffer[mut=True]
    """The bytes being encoded."""
    var node: Encoded
    """The node being decoded."""
    var at: Int
    """Values encoded."""

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self.node = Encoded()
        self.at = 0

    def start[T: DType](mut self, count: Int) raises:
        self.bytes = Buffer.alloc_uninit[T](count)
        self.at = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        self.bytes.extend(src, self.at, len(src))
        self.at += len(src)

    def finish[T: DType](mut self) raises -> Encoded:
        var frozen = self.bytes^.to_immutable()
        self.bytes = Buffer.alloc_uninit(0)
        return Encoded(
            frozen^,
            self.at * size_of[Scalar[T]](),
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        if node.size != count * size_of[Scalar[T]]():
            raise CorruptError(
                t"codecs: {node.size} plain bytes for {count} {T} values"
            )
        self.node = node.copy()

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        return self.node.values[T]().load_unaligned[W](i)


struct ByteStreamSplit(Codec):
    """The values' bytes as planes, every first byte, then every second --
    Parquet's BYTE_STREAM_SPLIT.

        [0x0102, 0x0304] as uint16  ->  02 04 01 03
    """

    var bytes: Buffer[mut=True]
    """The planes being encoded."""
    var node: Encoded
    """The planes being decoded."""
    var count: Int
    var at: Int
    """Values encoded."""

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self.node = Encoded()
        self.count = 0
        self.at = 0

    def start[T: DType](mut self, count: Int) raises:
        self.bytes = Buffer.alloc_uninit[T](count)
        self.count = count
        self.at = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime B = size_of[Scalar[T]]()
        comptime W = 16
        var planes = self.bytes.view[DType.uint8](0, self.count * B)
        var n = len(src)
        var i = 0
        while i + W <= n:
            var bits = bitcast[U, W](src.load[W](i))
            comptime for k in range(B):
                planes.store[W](
                    k * self.count + self.at + i,
                    (bits >> Scalar[U](8 * k)).cast[DType.uint8](),
                )
            i += W
        while i < n:
            var bits = bitcast[U, 1](src.load[1](i))
            comptime for k in range(B):
                planes.store[1](
                    k * self.count + self.at + i,
                    (bits >> Scalar[U](8 * k)).cast[DType.uint8](),
                )
            i += 1
        self.at += n

    def finish[T: DType](mut self) raises -> Encoded:
        var frozen = self.bytes^.to_immutable()
        self.bytes = Buffer.alloc_uninit(0)
        return Encoded(
            frozen^,
            self.count * size_of[Scalar[T]](),
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        if node.size != count * size_of[Scalar[T]]():
            raise CorruptError(
                t"codecs: {node.size} plane bytes for {count} {T} values"
            )
        self.node = node.copy()
        self.count = count

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        comptime U = Unsigned[T]
        var planes = self.node.values[DType.uint8]()
        var bits = SIMD[U, W](0)
        comptime for k in range(size_of[Scalar[T]]()):
            bits |= planes.load[W](k * self.count + i).cast[U]() << Scalar[U](
                8 * k
            )
        return bitcast[T, W](bits)


struct BitPack[Widths: Codec = Plain](Codec):
    """Each value at the width the largest of its frame needs, least
    significant bit first, in groups of eight -- so a group is `width`
    whole bytes. The widths are the `Widths` slot.

        [0, 1, 3] as uint8  ->  34 00, widths [2]
    """

    comptime GROUP = 8
    comptime SLOTS = 1
    comptime WIDTHS = 0
    """The slot of the widths."""

    var widths: FrameValues[Self.Widths]
    var bytes: Buffer[mut=True]
    """The bytes being encoded."""
    var node: Encoded
    """The bytes being decoded."""
    var at: Int
    """The bit the next group is packed at."""
    var starts: List[Int]
    """By frame decoded, the bit value 0 would start at, so value `v` of
    frame `b` starts at `starts[b] + v * width`."""
    var groups: List[PackedGroup]
    """By frame decoded, how a group of it holds its values."""

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Widths())

    def __init__(out self, size: Int, var widths: Self.Widths):
        self.widths = FrameValues(size, Self.GROUP, widths^)
        self.bytes = Buffer.alloc_uninit(0)
        self.node = Encoded()
        self.at = 0
        self.starts = List[Int]()
        self.groups = List[PackedGroup]()

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "BitPack takes unsigned integers"
        self.widths.start[DType.uint8](count)
        var groups = (count + Self.GROUP - 1) // Self.GROUP
        self.bytes = Buffer.alloc_uninit[DType.uint8](
            groups * bit_width_of[T]()
        )
        self.at = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime G = Self.GROUP
        comptime W = 64 // size_of[Scalar[T]]()
        var bits = BitmapView(self.bytes.view[DType.uint8]().as_span())
        var f = self.widths.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            var any = SIMD[T, W](0)
            var i = lo
            while i + W <= hi:
                any |= src.load[W](i)
                i += W
            var top = any.reduce_or()
            while i < hi:
                top |= src[i]
                i += 1
            var width = Int(bit_width(top))
            self.widths.put(UInt8(width))
            i = lo
            while i < hi:
                PackedGroup.pack(bits, self.at, src, i, min(i + G, hi), width)
                self.at += G * width
                i += G
            lo = hi

    def finish[T: DType](mut self) raises -> Encoded:
        var frozen = self.bytes^.to_immutable()
        self.bytes = Buffer.alloc_uninit(0)
        return Encoded(
            frozen^,
            self.at // 8,
            [self.widths.finish[DType.uint8]()],
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "BitPack takes unsigned integers"
        comptime G = Self.GROUP
        self.widths.start[DType.uint8](node[Self.WIDTHS], count)
        var f = self.widths.frames.size
        var frames = len(self.widths)
        self.starts = List[Int](capacity=frames)
        self.groups = List[PackedGroup](capacity=frames)
        var size = 0
        for b in range(frames):
            var width = Int(self.widths.get[DType.uint8](b))
            if width > bit_width_of[T]():
                raise CorruptError(t"codecs: BitPack width {width} for {T}")
            # Frames hold whole groups, so each starts at a byte.
            self.starts.append(8 * size - b * f * width)
            self.groups.append(PackedGroup(width))
            size += (min(f, count - b * f) + G - 1) // G * width
        if node.size != size:
            raise CorruptError(t"codecs: {node.size} BitPack bytes, not {size}")
        self.node = node.copy()

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        comptime G = Self.GROUP
        comptime if W < G:
            var b = self.widths.frames.of(i)
            var width = self.groups.unsafe_get(b).width
            return (
                BitmapView(self.node.bytes())
                .load_packed[W](self.starts.unsafe_get(b) + i * width, width)
                .cast[T]()
            )
        else:
            # A register may span frames; a group of `G` never does.
            var bytes = BufferView(self.node.bytes())
            var out = SIMD[T, W]()
            comptime for g in range(W // G):
                var v = i + g * G
                var b = self.widths.frames.of(v)
                ref group = self.groups.unsafe_get(b)
                var byte = (self.starts.unsafe_get(b) + v * group.width) >> 3
                out = out.insert[offset=g * G](
                    group.unpack(bytes, byte).cast[T]()
                )
            return out


struct Runs[
    Values: Codec = Plain,
    Lengths: Codec = Plain,
](Codec):
    """Each run of equal values as its value and its length, the `Values`
    and `Lengths` slots -- lengths as `uint64`s.

        [5, 5, 5, 1, 2, 2]  ->  values [5, 1, 2], lengths [3, 1, 2]
    """

    comptime SLOTS = 2
    comptime VALUES = 0
    """The slot of the runs' values."""
    comptime LENGTHS = 1
    """The slot of the runs' lengths."""

    var values: Self.Values
    var lengths: Self.Lengths
    var found: Buffer[mut=True]
    """Each run's value."""
    var counts: List[UInt64]
    """Each run's length."""
    var runs: Int
    """Runs found, or the run being decoded."""
    var value: UInt64
    """The bits of the run being found."""
    var count: Int
    """Values in the run being found; values left of the run being
    decoded."""
    var unpacked: Unpacked

    def __init__(out self):
        self = Self(Self.Values(), Self.Lengths())

    def __init__(out self, var values: Self.Values, var lengths: Self.Lengths):
        self.values = values^
        self.lengths = lengths^
        self.found = Buffer.alloc_uninit(0)
        self.counts = List[UInt64]()
        self.runs = 0
        self.value = 0
        self.count = 0
        self.unpacked = Unpacked()

    # --- encoding ---

    def start[T: DType](mut self, count: Int) raises:
        self.found = Buffer.alloc_uninit[T](count)
        self.counts = List[UInt64]()
        self.runs = 0
        self.count = 0

    def _run[T: DType](mut self):
        self.found.view[T]().unsafe_set(
            self.runs, bitcast[T](self.value.cast[Unsigned[T]]())
        )
        self.counts.append(UInt64(self.count))
        self.runs += 1

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var n = len(src)
        var i = 0
        while i < n:
            var bits = bitcast[U](src.unsafe_get(i)).cast[DType.uint64]()
            i += 1
            if self.count > 0 and bits == self.value:
                self.count += 1
            else:
                if self.count > 0:
                    self._run[T]()
                self.value = bits
                self.count = 1
            # The run goes on while the values equal it, a register at a
            # time.
            var v = SIMD[U, W](self.value.cast[U]())
            while (
                i + W <= n and bitcast[U, W](src.load[W](i)).eq(v).reduce_and()
            ):
                self.count += W
                i += W

    def finish[T: DType](mut self) raises -> Encoded:
        if self.count > 0:
            self._run[T]()
        var found = self.found^.to_immutable()
        self.found = Buffer.alloc_uninit(0)
        return Encoded(
            0,
            [
                self.values.encode_view(
                    BufferView(found.view[T](0, self.runs).as_span())
                ),
                self.lengths.encode_view(BufferView(Span(self.counts))),
            ],
        )

    # --- decoding ---

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        var runs = node[Self.LENGTHS].count
        var stored = node[Self.VALUES].count
        if stored != runs:
            raise CorruptError(t"codecs: {stored} run values for {runs} runs")
        self.counts = List[UInt64](length=runs, fill=0)
        self.lengths.decode_into[DType.uint64](
            node[Self.LENGTHS], BufferView(Span(self.counts))
        )
        var total = UInt64(0)
        for c in self.counts:
            total += c
        if total != UInt64(count):
            raise CorruptError(t"codecs: runs of {total} values, not {count}")
        self.found = Buffer.alloc_uninit[T](runs)
        self.values.decode_into[T](
            node[Self.VALUES], self.found.view[T](0, runs)
        )
        self.runs = -1
        self.count = 0
        self.unpacked.start(count)

    def _unpack[T: DType](mut self):
        # A run is written a cache line at a time, past its end into what
        # the runs after it overwrite.
        comptime W = 64 // size_of[Scalar[T]]()
        var out = self.unpacked.next[T]()
        var n = self.unpacked.size()
        var found = self.found.view[T]()
        var i = 0
        while i < n:
            if self.count == 0:
                self.runs += 1
                self.count = Int(self.counts[self.runs])
            var take = min(self.count, n - i)
            var v = SIMD[T, W](found.unsafe_get(self.runs))
            var j = 0
            while j < take:
                out.store[W](i + j, v)
                j += W
            self.count -= take
            i += take

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        if self.unpacked.due(i):
            self._unpack[T]()
        return self.unpacked.load[T, W](i)


struct Buckets(Codec):
    """Each value in the narrowest of a few widths it fits, behind a prefix
    saying which: `0` for zero, then `10` and 7 bits, `110` and 9, `1110`
    and 12, `11110` and 32, `11111` and 64 -- ClickHouse's DoubleDelta is
    `Delta, Delta, Zigzag, Buckets`. It takes unsigned integers.

        [0, 5, 300] as uint16  ->  0 | 10 0000101 | 1110 000100101100
    """

    comptime WIDTHS = SIMD[DType.uint8, 8](0, 7, 9, 12, 32, 64, 0, 0)
    """The payload width of each bucket, the bucket being how many ones the
    prefix holds."""

    var bytes: Buffer[mut=True]
    """The bits being encoded."""
    var node: Encoded
    """The bits being decoded."""
    var at: Int
    """The next bit written or read."""
    var unpacked: Unpacked

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self.node = Encoded()
        self.at = 0
        self.unpacked = Unpacked()

    @staticmethod
    @always_inline
    def _bucket(v: UInt64) -> Int:
        var w = Int(bit_width(v))
        if v == 0:
            return 0
        elif w <= 7:
            return 1
        elif w <= 9:
            return 2
        elif w <= 12:
            return 3
        elif w <= 32:
            return 4
        return 5

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "Buckets takes unsigned integers"
        self.bytes = Buffer.alloc_uninit[DType.uint8]((count * 69 + 7) // 8 + 8)
        self.at = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        var out = BitCursor(self.bytes.view[DType.uint8]().as_span(), self.at)
        for i in range(len(src)):
            var v = src.unsafe_get(i).cast[DType.uint64]()
            var b = Self._bucket(v)
            # `b` ones, then a zero unless the prefix is the longest.
            out.put(UInt64((1 << b) - 1), b + 1 if b < 5 else 5)
            out.put(v, Int(Self.WIDTHS[b]))
        self.at = out.at

    def finish[T: DType](mut self) raises -> Encoded:
        var frozen = self.bytes^.to_immutable()
        self.bytes = Buffer.alloc_uninit(0)
        return Encoded(frozen^, (self.at + 7) // 8)

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "Buckets takes unsigned integers"
        self.node = node.copy()
        self.at = 0
        self.unpacked.start(count)

    def _unpack[T: DType](mut self):
        # Bits past the node read as zero, which `check` catches.
        var src = BitCursor(self.node.bytes(), self.at)
        var out = self.unpacked.next[T]()
        for i in range(self.unpacked.size()):
            var b = Int(count_trailing_zeros(~src.peek(5)))
            src.at += b + 1 if b < 5 else 5
            out.unsafe_set(i, src.take(Int(Self.WIDTHS[b])).cast[T]())
        self.at = src.at

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        if self.unpacked.due(i):
            self._unpack[T]()
        return self.unpacked.load[T, W](i)

    def check(self) raises:
        if (self.at + 7) // 8 != self.node.size:
            raise CorruptError(
                t"codecs: {self.node.size} bucket bytes, not"
                t" {(self.at + 7) // 8}"
            )


struct Gorilla(Codec):
    """Values after `Xor`, as Gorilla writes them (Pelkonen et al.): `0` for
    zero; otherwise `1`, then `0` and the bits inside the window of the
    value before when they fit it, or `1`, the leading zeros in 6 bits, the
    meaningful bits' count less one in 6 bits, and those bits. Gorilla's
    floats are `Xor, Gorilla`. It takes unsigned integers.

        [0, 0x80] as uint8  ->  0 | 1 1 000000 000000 1
    """

    var bytes: Buffer[mut=True]
    """The bits being encoded."""
    var node: Encoded
    """The bits being decoded."""
    var at: Int
    """The next bit written or read."""
    var lead: Int
    """The window's leading zeros."""
    var width: Int
    """The window's meaningful bits; 0 before the first window."""
    var bad: Bool
    """Whether a window read lies outside the value's bits."""
    var unpacked: Unpacked

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self.node = Encoded()
        self.at = 0
        self.lead = 0
        self.width = 0
        self.bad = False
        self.unpacked = Unpacked()

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "Gorilla takes unsigned integers"
        comptime B = bit_width_of[T]()
        self.bytes = Buffer.alloc_uninit[DType.uint8](
            (count * (B + 14) + 7) // 8 + 8
        )
        self.at = 0
        self.lead = 0
        self.width = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime B = bit_width_of[T]()
        var out = BitCursor(self.bytes.view[DType.uint8]().as_span(), self.at)
        for i in range(len(src)):
            var v = src.unsafe_get(i)
            if v == 0:
                out.put(0, 1)
                continue
            var lead = Int(count_leading_zeros(v))
            var trail = Int(count_trailing_zeros(v))
            var x = v.cast[DType.uint64]()
            if (
                self.width > 0
                and lead >= self.lead
                and trail >= B - self.lead - self.width
            ):
                # `1`, `0`: inside the window of the value before.
                out.put(1, 2)
                out.put(x >> UInt64(B - self.lead - self.width), self.width)
            else:
                self.lead = lead
                self.width = B - lead - trail
                out.put(3, 2)
                out.put(UInt64(lead), 6)
                out.put(UInt64(self.width - 1), 6)
                out.put(x >> UInt64(trail), self.width)
        self.at = out.at

    def finish[T: DType](mut self) raises -> Encoded:
        var frozen = self.bytes^.to_immutable()
        self.bytes = Buffer.alloc_uninit(0)
        return Encoded(frozen^, (self.at + 7) // 8)

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "Gorilla takes unsigned integers"
        self.node = node.copy()
        self.at = 0
        self.lead = 0
        self.width = 0
        self.bad = False
        self.unpacked.start(count)

    def _unpack[T: DType](mut self):
        comptime B = bit_width_of[T]()
        var src = BitCursor(self.node.bytes(), self.at)
        var out = self.unpacked.next[T]()
        for i in range(self.unpacked.size()):
            var head = src.peek(2)
            if head & 1 == 0:
                src.at += 1
                out.unsafe_set(i, 0)
                continue
            src.at += 2
            if head == 3:
                # A window past the value's bits is noted for `check` and
                # read clipped to them.
                var lead = Int(src.take(6))
                var width = Int(src.take(6)) + 1
                self.bad |= lead + width > B
                self.lead = min(lead, B)
                self.width = min(width, B - self.lead)
            else:
                self.bad |= self.width == 0
            var x = src.take(self.width)
            # An empty window reads as zero.
            var shift = min(B - self.lead - self.width, B - 1)
            out.unsafe_set(i, (x << UInt64(shift)).cast[T]())
        self.at = src.at

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        if self.unpacked.due(i):
            self._unpack[T]()
        return self.unpacked.load[T, W](i)

    def check(self) raises:
        if self.bad:
            raise CorruptError("codecs: a Gorilla window past the value's bits")
        if (self.at + 7) // 8 != self.node.size:
            raise CorruptError(
                t"codecs: {self.node.size} Gorilla bytes, not"
                t" {(self.at + 7) // 8}"
            )


struct Compressed[C: Compressor](Codec):
    """The values' bytes, as `Plain` stores them, compressed whole by `C`:
    `Cascade[Delta, Compressed[Zstd]]` is ClickHouse's `CODEC(Delta, ZSTD)`.

        [1, 1, 1, 1] as uint32 -> Zstd's frame of 01 00 00 00 01 00 00 00 ...
    """

    var plain: Plain

    def __init__(out self):
        self.plain = Plain()

    def start[T: DType](mut self, count: Int) raises:
        self.plain.start[T](count)

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        self.plain.store(src)

    def finish[T: DType](mut self) raises -> Encoded:
        var raw = self.plain.finish[T]()
        var out = List[UInt8]()
        Self.C.compress(raw.bytes(), out)
        return Encoded(bytes=Span(out))

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        var n = count * size_of[Scalar[T]]()
        var raw = Buffer.alloc_uninit[DType.uint8](n)
        Self.C.decompress(node.bytes(), raw.view[DType.uint8](0, n).as_span())
        self.plain.start[T](Encoded(raw^.to_immutable(), n), count)

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        return self.plain.load[T, W](i)


struct Frequency[
    Tops: Codec = Plain,
    Positions: Codec = Plain,
    Values: Codec = Plain,
](Codec):
    """For a column that is mostly one value (BtrBlocks' Frequency): each
    frame's commonest value, the `Tops` slot, and every other value an
    exception with its position. The commonest is found by majority vote,
    so a frame with no majority keeps more exceptions than it needs.

        [7, 7, 3, 7] as uint8  ->  tops [7], exceptions [3] at [2]
    """

    comptime SLOTS = 3
    comptime TOPS = 0
    """The slot of each frame's commonest value."""
    comptime POSITIONS = 1
    """The slot of the exceptions' positions."""
    comptime VALUES = 2
    """The slot of the exceptions."""

    var tops: FrameValues[Self.Tops]
    var exceptions: Exceptions[Self.Positions, Self.Values]

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Tops(), Self.Positions(), Self.Values())

    def __init__(
        out self,
        size: Int,
        var tops: Self.Tops,
        var positions: Self.Positions,
        var values: Self.Values,
    ):
        self.tops = FrameValues(size, 64, tops^)
        self.exceptions = Exceptions(positions^, values^)

    def start[T: DType](mut self, count: Int) raises:
        self.tops.start[T](count)
        self.exceptions.start[T](count)

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        var f = self.tops.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            # Boyer and Moore's vote, by bits.
            var top = bitcast[U](src.unsafe_get(lo))
            var lead = 0
            for i in range(lo, hi):
                var v = bitcast[U](src.unsafe_get(i))
                if lead == 0:
                    top = v
                    lead = 1
                elif v == top:
                    lead += 1
                else:
                    lead -= 1
            self.tops.put(bitcast[T](top))
            for i in range(lo, hi):
                if bitcast[U](src.unsafe_get(i)) != top:
                    self.exceptions.keep[T](i, src.unsafe_get(i))
            lo = hi
        self.exceptions.passed(len(src))

    def finish[T: DType](mut self) raises -> Encoded:
        var slots = List[Encoded](capacity=3)
        slots.append(self.tops.finish[T]())
        self.exceptions.finish[T](slots)
        return Encoded(0, slots^)

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        self.tops.start[T](node[Self.TOPS], count)
        self.exceptions.start[T](node[Self.POSITIONS], node[Self.VALUES], count)

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        return self.exceptions.put[T, W](i, SIMD[T, W](self.tops.at[T](i)))
