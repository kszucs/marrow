# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Stores, the codec that ends a cascade and holds its values' bytes:
`Plain` and `ByteStreamSplit` their bytes as they are, `BitPack` the bits
each frame needs, `Runs` each run once, and `Rle` Parquet's RLE / bit-packed
hybrid of runs and packed groups."""

from std.bit import bit_width
from std.memory import bitcast
from std.sys import bit_width_of, size_of

from ..buffers import Buffer
from ..errors import CorruptError
from ..views import BitmapView, BufferView, PackedGroup
from .core import (
    Codec,
    Encoded,
    FrameValues,
    Unpacked,
)
from ..utils.byteorder import Leb128, LittleEndian
from ..utils.integers import Unsigned


struct Plain(Codec):
    """The values' bytes, little-endian -- Parquet's PLAIN.

    [1, 258] as uint16  ->  01 00 02 01
    """

    comptime STORES_VALUES = True

    var bytes: Buffer[mut=True]
    """The bytes being encoded."""
    var _node: Encoded
    """The node being decoded."""
    var at: Int
    """Values encoded."""

    @staticmethod
    def node[T: DType](values: Span[Scalar[T], _]) -> Encoded:
        """`Plain`'s node of `values`, a copy."""
        return Encoded(values)

    @staticmethod
    def node(*, bytes: Span[UInt8, _]) -> Encoded:
        """`Plain`'s node of values laid out in `bytes`, a copy."""
        return Encoded(bytes=bytes)

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self._node = Encoded()
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
        self._node = node.copy()

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        return self._node.values[T]().load_unaligned[W](i)


struct ByteStreamSplit(Codec):
    """The values' bytes as planes, every first byte, then every second --
    Parquet's BYTE_STREAM_SPLIT.

        [0x0102, 0x0304] as uint16  ->  02 04 01 03
    """

    var bytes: Buffer[mut=True]
    """The planes being encoded."""
    var _node: Encoded
    """The planes being decoded."""
    var count: Int
    var at: Int
    """Values encoded."""

    @staticmethod
    def node(planes: Span[UInt8, _]) -> Encoded:
        """`ByteStreamSplit`'s node of `planes`, a copy."""
        return Encoded(bytes=planes)

    def __init__(out self):
        self.bytes = Buffer.alloc_uninit(0)
        self._node = Encoded()
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
        self._node = node.copy()
        self.count = count

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        comptime U = Unsigned[T]
        var planes = self._node.values[DType.uint8]()
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

    var _widths: FrameValues[Self.Widths]
    var bytes: Buffer[mut=True]
    """The bytes being encoded."""
    var _node: Encoded
    """The bytes being decoded."""
    var at: Int
    """The bit the next group is packed at."""
    var starts: List[Int]
    """By frame decoded, the bit value 0 would start at, so value `v` of
    frame `b` starts at `starts[b] + v * width`."""
    var groups: List[PackedGroup]
    """By frame decoded, how a group of it holds its values."""

    @staticmethod
    def node(packed: Span[UInt8, _], var widths: Encoded) -> Encoded:
        """`BitPack`'s node: a copy of `packed`, the frames' groups, and
        `widths`, the node of its `Widths` slot."""
        var node = Encoded(bytes=packed)
        node.children.append(widths^)
        return node^

    @staticmethod
    def widths(
        node: Encoded,
    ) -> BufferView[DType.uint8, origin_of(node.children[Self.WIDTHS].buffer)]:
        """The width of each frame `node` holds, read where they lie: the
        `Widths` slot keeps them as they are."""
        comptime assert (
            Self.Widths.STORES_VALUES
        ), "the Widths slot does not keep its values as they are"
        return node[Self.WIDTHS].values[DType.uint8]()

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Widths())

    def __init__(out self, size: Int, var widths: Self.Widths):
        self._widths = FrameValues(size, Self.GROUP, widths^)
        self.bytes = Buffer.alloc_uninit(0)
        self._node = Encoded()
        self.at = 0
        self.starts = List[Int]()
        self.groups = List[PackedGroup]()

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "BitPack takes unsigned integers"
        self._widths.start[DType.uint8](count)
        var groups = (count + Self.GROUP - 1) // Self.GROUP
        self.bytes = Buffer.alloc_uninit[DType.uint8](
            groups * bit_width_of[T]()
        )
        self.at = 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime G = Self.GROUP
        comptime W = 64 // size_of[Scalar[T]]()
        var bits = BitmapView(self.bytes.view[DType.uint8]().as_span())
        var f = self._widths.frames.size
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
            self._widths.put(UInt8(width))
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
            [self._widths.finish[DType.uint8]()],
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "BitPack takes unsigned integers"
        comptime G = Self.GROUP
        self._widths.start[DType.uint8](node[Self.WIDTHS], count)
        var f = self._widths.frames.size
        var frames = len(self._widths)
        self.starts = List[Int](capacity=frames)
        self.groups = List[PackedGroup](capacity=frames)
        var size = 0
        for b in range(frames):
            var width = Int(self._widths.get[DType.uint8](b))
            if width > bit_width_of[T]():
                raise CorruptError(t"codecs: BitPack width {width} for {T}")
            # Frames hold whole groups, so each starts at a byte.
            self.starts.append(8 * size - b * f * width)
            self.groups.append(PackedGroup(width))
            size += (min(f, count - b * f) + G - 1) // G * width
        if node.size != size:
            raise CorruptError(t"codecs: {node.size} BitPack bytes, not {size}")
        self._node = node.copy()

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        comptime G = Self.GROUP
        comptime if W < G:
            var b = self._widths.frames.of(i)
            var width = self.groups.unsafe_get(b).width
            return (
                BitmapView(self._node.bytes())
                .load_packed[W](self.starts.unsafe_get(b) + i * width, width)
                .cast[T]()
            )
        else:
            # A register may span frames; a group of `G` never does.
            var bytes = BufferView(self._node.bytes())
            var out = SIMD[T, W]()
            comptime for g in range(W // G):
                var v = i + g * G
                var b = self._widths.frames.of(v)
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
        comptime G = Rle.GROUP
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


struct Rle(Codec):
    """Parquet's RLE encoding: runs of one value -- a header, then the value
    in its whole bytes -- or bit-packed groups of eight, `width` bits each.
    A run is written as one only where a stretch of packed groups may end:
    at its start or after a whole group. Encoding takes the runs `Runs`
    finds and lays them out so; decoding reads them a block at a time, so
    as a cascade's store -- `Cascade[Dictionary[], Rle]` -- it feeds
    the codecs before it. It takes unsigned integers of `width` bits."""

    comptime GROUP = 8

    var width: Int
    var values: Buffer[mut=True]
    """The values being encoded."""
    var taken: Int
    var _node: Encoded
    """The runs being decoded."""
    var runs: RleRuns
    var group: PackedGroup
    var unpacked: Unpacked
    var bad: Bool
    """Whether the runs ended before the values."""

    @staticmethod
    def node(runs: Span[UInt8, _]) -> Encoded:
        """`Rle`'s node of `runs`, a copy."""
        return Encoded(bytes=runs)

    def __init__(out self):
        self = Self(0)

    def __init__(out self, width: Int):
        self.width = width
        self.values = Buffer.alloc_uninit(0)
        self.taken = 0
        self._node = Encoded()
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
        return Self.node(Span(out))

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        if self.width > bit_width_of[T]():
            raise CorruptError(t"rle: {count} values of {self.width} bits")
        self._node = node.copy()
        self.runs = RleRuns()
        self.unpacked.start(count)
        self.bad = False

    def _unpack[T: DType](mut self):
        var out = self.unpacked.next[T]()
        var n = self.unpacked.size()
        if self.width == 0:
            out.as_span()[:n].fill(0)
        elif not self.runs.fill[T](
            BufferView(self._node.bytes()), self.group, out[:n]
        ):
            self.bad = True

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        if self.unpacked.due(i):
            self._unpack[T]()
        return self.unpacked.load[T, W](i)

    def check(self) raises:
        if self.bad:
            raise CorruptError("rle: truncated runs")

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
        if width > bit_width_of[T]():
            raise CorruptError(t"rle: {count} values of {width} bits")
        var base = len(out)
        out.resize(unsafe_uninit_length=base + count)
        var dst = BufferView(Span(out))[base:]
        if width == 0:
            dst.as_span().fill(0)
            return 0
        var runs = RleRuns()
        if not runs.fill[T](BufferView(data), PackedGroup(width), dst):
            raise CorruptError("rle: truncated runs")
        return runs.pos

    @staticmethod
    def read_codes[
        T: DType
    ](data: Span[UInt8, _], count: Int, mut out: List[Scalar[T]]) raises:
        """Append a data page's first `count` dictionary codes: their width
        in a byte, then their runs."""
        if len(data) == 0:
            if count > 0:
                raise CorruptError(t"rle: no codes for {count} values")
            return
        _ = Self.read_runs[T](data[1:], Int(data[0]), count, out)

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
                raise CorruptError("rle: truncated run")
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
