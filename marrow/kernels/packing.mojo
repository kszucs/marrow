# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Packers: values in, a typed `Packed` out, and back a register at a time.

Each `*Pack` has its `*Unpack` and the `*Packed` a column is packed as.
`PlainPack` keeps the values' bytes as they are, `SplitPack` as byte planes,
`BitPack` at the bits each segment needs, and `RlePack` as Parquet's RLE /
bit-packed hybrid. `FittedPack` fits one value per segment with a fold and
packs each value against its segment's fit with an inner packer; frame of
reference, each value less its segment's least, is `ReferencePack`.
`RunsPack` (`runs.mojo`) stores each run of equal values once.

Byte strings go in as offsets and bytes and come out the same way:
`LengthsPack` keeps their bytes end to end and packs their lengths,
`PrefixesPack` packs the length of the prefix each shares with the one
before and keeps the rest of each as `LengthsPack` does. Each takes the value
packers of its `int32` streams, and has an `*Unpack` reading them back a
register at a time.

A packer that needs more than its values wraps an inner packer, and its
`Packed` holds the inner one's: `FittedPacked[BitPacked]` is the fits plus
the bits.

Segments: a packer that keeps a value per segment takes, encoding, the
segment sizes that divide `BLOCK`, so a block holds whole segments, and,
decoding, the multiples of its step -- a group of eight for `BitPack`, 64
values for `FittedPack` -- so a register never spans two segments.
"""

from std.algorithm import vectorize
from std.bit import bit_width
from std.builtin.rebind import rebind_var
from std.math import ceildiv
from std.memory import bitcast
from std.sys import bit_width_of, size_of

from ..buffers import Buffer
from ..compress.lz77 import match_length
from ..errors import CorruptError, InvalidError
from ..utils.byteorder import Leb128, LittleEndian
from ..utils.integers import Unsigned
from ..views import BitmapView, BufferView, PackedGroup
from .aggregate import FoldKernel, MinFold
from .compose import Codec
from .core import BinaryKernel, Kernel, PackKernel, UnpackKernel
from .numeric import AddKernel, SubKernel
from .runs import RunsPack, RunsUnpack
from .segments import BySize, PerSegment


comptime BLOCK = 1024
"""The values a codec carries through its stages at once: a block, in each
stage's dtype, stays in the L1 cache from stage to stage."""


struct Unpacked(Movable):
    """The block of values an unpacker decodes a block at a time -- runs, or
    the hybrid's runs and groups -- for the registers to load: the values
    from `first` on, with a cache line to spare past them. Values of at most
    eight bytes."""

    var bytes: List[UInt8]
    var first: Int
    var count: Int
    """Values in the column."""

    def __init__(out self, count: Int):
        """Before the first block of `count` values."""
        self.bytes = List[UInt8](length=8 * BLOCK + 64, fill=0)
        self.first = -BLOCK
        self.count = count

    @always_inline
    def due(self, i: Int) -> Bool:
        """Whether value `i` is past the block held."""
        return i >= self.first + BLOCK

    def next[T: DType](mut self) -> BufferView[T, origin_of(self.bytes)]:
        """The block after the one held, to write."""
        comptime assert (
            size_of[Scalar[T]]() <= 8
        ), "Unpacked holds values of at most 8 bytes"
        self.first += BLOCK
        return BufferView[T, origin_of(self.bytes)].from_bytes(Span(self.bytes))

    def size(self) -> Int:
        """The values of the block held."""
        return min(BLOCK, self.count - self.first)

    @always_inline
    def load[T: DType, W: Int](self, i: Int) -> SIMD[T, W]:
        comptime assert (
            size_of[Scalar[T]]() <= 8
        ), "Unpacked holds values of at most 8 bytes"
        return (
            BufferView[T, origin_of(self.bytes)]
            .from_bytes(Span(self.bytes))
            .load[W](i - self.first)
        )


struct _Filling(Movable):
    """A buffer being filled: `at` values, or bits, of room for `count`."""

    var buffer: Buffer[mut=True]
    var at: Int
    var count: Int

    def __init__(out self, var buffer: Buffer[mut=True], count: Int):
        self.buffer = buffer^
        self.at = 0
        self.count = count

    def frozen(deinit self) -> Buffer[mut=False]:
        """The buffer, filled."""
        return self.buffer^.to_immutable()


# --- Plain ---------------------------------------------------------------------


@fieldwise_init
struct PlainPacked(Copyable, Movable):
    """The values' bytes as they are, little-endian: Parquet's PLAIN.

    [1, 258] as uint16  ->  01 00 02 01
    """

    var bytes: Buffer[mut=False]
    var size: Int
    """How many of the bytes are the values'."""


@fieldwise_init
struct PlainPack(Copyable, Defaultable, PackKernel):
    """Values as `PlainPacked`."""

    comptime name = "plain_pack"
    comptime Packed = PlainPacked
    comptime State[T: DType]: Movable & Deinitable = _Filling

    def start[T: DType](self, count: Int) raises -> _Filling:
        return _Filling(Buffer.alloc_uninit[T](count), count)

    def store[
        T: DType
    ](self, mut state: _Filling, block: BufferView[T, _]) raises:
        state.buffer.extend(block, state.at, len(block))
        state.at += len(block)

    def finish[T: DType](self, var state: _Filling) raises -> PlainPacked:
        var size = state.at * size_of[Scalar[T]]()
        return PlainPacked(state^.frozen(), size)


@fieldwise_init
struct PlainUnpack(Copyable, Defaultable, UnpackKernel):
    """`PlainPacked` read where it lies."""

    comptime name = "plain_unpack"
    comptime Packed = PlainPacked
    comptime State[T: DType]: Movable & Deinitable = PlainPacked

    def start[
        T: DType
    ](self, var packed: PlainPacked, count: Int) raises -> PlainPacked:
        var size = count * size_of[Scalar[T]]()
        if packed.size != size or size > len(packed.bytes):
            raise Self.error[CorruptError](
                t"{packed.size} bytes in a buffer of {len(packed.bytes)} for"
                t" {count} {T} values"
            )
        return packed^

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: PlainPacked, i: Int) -> SIMD[T, W]:
        return state.bytes.view[T]().load_unaligned[W](i)


# --- Split ---------------------------------------------------------------------


@fieldwise_init
struct SplitPacked(Copyable, Movable):
    """The values' bytes as planes, every first byte, then every second:
    Parquet's BYTE_STREAM_SPLIT.

        [0x0102, 0x0304] as uint16  ->  02 04 01 03
    """

    var bytes: Buffer[mut=False]
    var size: Int
    """How many of the bytes are the planes'."""


@fieldwise_init
struct SplitPack(Copyable, Defaultable, PackKernel):
    """Values as `SplitPacked`."""

    comptime name = "split_pack"
    comptime Packed = SplitPacked
    comptime State[T: DType]: Movable & Deinitable = _Filling

    def start[T: DType](self, count: Int) raises -> _Filling:
        return _Filling(Buffer.alloc_uninit[T](count), count)

    def store[
        T: DType
    ](self, mut state: _Filling, block: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime B = size_of[Scalar[T]]()
        var count = state.count
        var at = state.at
        var planes = state.buffer.view[DType.uint8](0, count * B)

        @always_inline
        def split[W: Int](i: Int) {imm}:
            var bits = bitcast[U, W](block.load[W](i))
            comptime for k in range(B):
                planes.store[W](
                    k * count + at + i,
                    (bits >> Scalar[U](8 * k)).cast[DType.uint8](),
                )

        vectorize[16](len(block), split)
        state.at += len(block)

    def finish[T: DType](self, var state: _Filling) raises -> SplitPacked:
        var size = state.count * size_of[Scalar[T]]()
        return SplitPacked(state^.frozen(), size)


@fieldwise_init
struct SplitUnpack(Copyable, Defaultable, UnpackKernel):
    """`SplitPacked`'s planes joined back into values."""

    comptime name = "split_unpack"
    comptime Packed = SplitPacked
    comptime State[T: DType]: Movable & Deinitable = SplitPacked

    def start[
        T: DType
    ](self, var packed: SplitPacked, count: Int) raises -> SplitPacked:
        var size = count * size_of[Scalar[T]]()
        if packed.size != size or size > len(packed.bytes):
            raise Self.error[CorruptError](
                t"{packed.size} plane bytes in a buffer of {len(packed.bytes)}"
                t" for {count} {T} values"
            )
        return packed^

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: SplitPacked, i: Int) -> SIMD[T, W]:
        comptime U = Unsigned[T]
        comptime B = size_of[Scalar[T]]()
        var count = state.size // B
        var planes = state.bytes.view[DType.uint8]()
        var bits = SIMD[U, W](0)
        comptime for k in range(B):
            bits |= planes.load[W](k * count + i).cast[U]() << Scalar[U](8 * k)
        return bitcast[T, W](bits)


# --- BitPack -------------------------------------------------------------------


@fieldwise_init
struct BitPacked(Copyable, Movable):
    """Each value at the width the largest of its segment needs, least
    significant bit first, in groups of eight, so a group is `width` whole
    bytes; and each segment's width.

        [0, 1, 3] as uint8  ->  34 00, widths [2]
    """

    var bytes: Buffer[mut=False]
    var size: Int
    """How many of the bytes are the groups'."""
    var widths: PerSegment
    """Each segment's width, a `uint8`."""


struct _BitPacking(Movable):
    """The groups packed so far, `at` bits, and each segment's width, for
    the values `stored`."""

    var bytes: Buffer[mut=True]
    var at: Int
    var widths: Buffer[mut=True]
    var segments: Int
    var stored: Int

    def __init__(out self, groups: Int, bits: Int, segments: Int):
        self.bytes = Buffer.alloc_uninit[DType.uint8](groups * bits)
        self.at = 0
        self.widths = Buffer.alloc_uninit[DType.uint8](segments)
        self.segments = 0
        self.stored = 0

    def packed(deinit self) -> BitPacked:
        """What was packed; the bytes taken over, without a copy."""
        return BitPacked(
            self.bytes^.to_immutable(),
            self.at // 8,
            PerSegment(self.widths^.to_immutable(), self.segments),
        )


@fieldwise_init
struct BitPack(Copyable, Defaultable, PackKernel):
    """Unsigned integers as `BitPacked`, in segments of `size` values."""

    comptime name = "bit_pack"
    comptime Packed = BitPacked
    comptime State[T: DType]: Movable & Deinitable = _BitPacking
    comptime GROUP = 8

    var size: Int

    def __init__(out self):
        self.size = BLOCK

    def start[T: DType](self, count: Int) raises -> _BitPacking:
        comptime assert T.is_unsigned(), "BitPack takes unsigned integers"
        var f = self.size
        if f <= 0 or f % Self.GROUP != 0 or BLOCK % f != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        return _BitPacking(
            ceildiv(count, Self.GROUP), bit_width_of[T](), ceildiv(count, f)
        )

    def store[
        T: DType
    ](self, mut state: _BitPacking, block: BufferView[T, _]) raises:
        comptime G = Self.GROUP
        comptime W = 64 // size_of[Scalar[T]]()
        var m = len(block)
        if m > BLOCK or state.stored % BLOCK != 0:
            raise Self.block_error(m)
        var bits = BitmapView(state.bytes.view[DType.uint8]().as_span())
        var widths = state.widths.view[DType.uint8]()
        var f = self.size
        var lo = 0
        while lo < m:
            var hi = min(lo + f, m)
            var top = Scalar[T](0)
            var i = lo
            while i + W <= hi:
                top |= block.load[W](i).reduce_or()
                i += W
            while i < hi:
                top |= block.load[1](i)
                i += 1
            var width = Int(bit_width(top))
            widths.unsafe_set(state.segments, UInt8(width))
            state.segments += 1
            i = lo
            while i < hi:
                PackedGroup.pack(
                    bits, state.at, block, i, min(i + G, hi), width
                )
                state.at += G * width
                i += G
            lo = hi
        state.stored += m

    def finish[T: DType](self, var state: _BitPacking) raises -> BitPacked:
        return state^.packed()


struct _BitUnpacking(Movable):
    """What `BitUnpack` reads: the packed bytes, by segment the bit value 0
    would start at -- value `v` of segment `s` starts at
    `starts[s] + v * width` -- and by width how a group holds its values;
    and the segment being read."""

    var packed: BitPacked
    var segments: BySize
    var starts: List[Int]
    var groups: List[PackedGroup]
    """Group `w` holds values of `w` bits, from 0 to the widest used."""
    var end: Int
    """One past the last value of the segment being read."""
    var start: Int
    """Its `starts` entry."""
    var width: Int
    """Its width."""

    def __init__(out self, var packed: BitPacked, segments: BySize):
        self.packed = packed^
        self.segments = segments
        self.starts = List[Int](capacity=segments.count())
        self.groups = List[PackedGroup]()
        self.end = 0
        self.start = 0
        self.width = 0

    def enter(mut self, v: Int):
        """Value `v`'s segment, as the one being read: the values are read
        in order, so a segment is looked up once."""
        var s = self.segments.of(v)
        self.end = self.segments.end(s)
        self.start = self.starts.unsafe_get(s)
        self.width = Int(self.packed.widths.get[DType.uint8](s))


@fieldwise_init
struct BitUnpack(Copyable, Defaultable, UnpackKernel):
    """`BitPacked`'s values, in segments of `size` values."""

    comptime name = "bit_unpack"
    comptime Packed = BitPacked
    comptime State[T: DType]: Movable & Deinitable = _BitUnpacking
    comptime GROUP = 8

    var size: Int

    def __init__(out self):
        self.size = BLOCK

    def start[
        T: DType
    ](self, var packed: BitPacked, count: Int) raises -> _BitUnpacking:
        comptime assert T.is_unsigned(), "BitUnpack yields unsigned integers"
        comptime G = Self.GROUP
        var f = self.size
        if f <= 0 or f % G != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        var segments = BySize(count, f)
        if (
            len(packed.widths) != segments.count()
            or not packed.widths.holds[DType.uint8]()
        ):
            raise Self.error[CorruptError](
                t"{len(packed.widths)} widths in a buffer of"
                t" {len(packed.widths.values)} bytes for {segments.count()}"
                t" segments"
            )
        var state = _BitUnpacking(packed^, segments)
        var widest = 0
        var size = 0
        for s in range(segments.count()):
            var width = Int(state.packed.widths.get[DType.uint8](s))
            if width > bit_width_of[T]():
                raise Self.error[CorruptError](t"width {width} for {T}")
            widest = max(widest, width)
            # Segments hold whole groups, so each starts at a byte.
            state.starts.append(8 * size - s * f * width)
            size += ceildiv(segments.end(s) - segments.start(s), G) * width
        if state.packed.size != size or size > len(state.packed.bytes):
            raise Self.error[CorruptError](
                t"{state.packed.size} bytes in a buffer of"
                t" {len(state.packed.bytes)}, not {size}"
            )
        # One a width, not one a segment: segments far outnumber widths.
        state.groups.reserve(widest + 1)
        for width in range(widest + 1):
            state.groups.append(PackedGroup(width))
        return state^

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: _BitUnpacking, i: Int) -> SIMD[T, W]:
        comptime G = Self.GROUP
        var bytes = state.packed.bytes.view[DType.uint8](0, state.packed.size)
        comptime if W <= G:
            # A register within a group lies in one segment, the one being
            # read. A wider one looks up each group's: sharing one width
            # across its groups has the compiler set up their fallback path
            # on every group.
            if i >= state.end:
                state.enter(i)
            var width = state.width
            comptime if W < G:
                return (
                    BitmapView(bytes.as_span())
                    .load_packed[W](state.start + i * width, width)
                    .cast[T]()
                )
            else:
                return rebind[SIMD[T, W]](
                    state.groups.unsafe_get(width)
                    .unpack(bytes, (state.start + i * width) >> 3)
                    .cast[T]()
                )
        else:
            # A register may span segments; a group of `G` never does.
            var out = SIMD[T, W]()
            comptime for g in range(W // G):
                var v = i + g * G
                var s = state.segments.of(v)
                ref group = state.groups.unsafe_get(
                    Int(state.packed.widths.get[DType.uint8](s))
                )
                var byte = (state.starts.unsafe_get(s) + v * group.width) >> 3
                out = out.insert[offset=g * G](
                    group.unpack(bytes, byte).cast[T]()
                )
            return out


# --- Fitted --------------------------------------------------------------------


@fieldwise_init
struct FittedPacked[P: Copyable & Deinitable](Copyable, Movable):
    """Each segment's fit, of the column's dtype, and the values packed
    against it by the inner packer."""

    var fits: PerSegment
    var inner: Self.P


@fieldwise_init
struct FittedPack[F: FoldKernel, Enc: BinaryKernel, Inner: PackKernel](
    PackKernel
):
    """Fits one value per segment of `size` values with the fold `F`, and
    packs `Enc(v, fit)` with `Inner`. Integers only.

        ReferencePack over [100, 101, 103] as uint8  ->  [0, 1, 3], fits [100]
    """

    comptime name = "fitted_pack"
    comptime Packed = FittedPacked[Self.Inner.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        _Filling, Self.Inner.State[T]
    ]
    comptime STEP = 64

    var size: Int
    var inner: Self.Inner

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        comptime assert T.is_integral(), "FittedPack takes integers"
        var f = self.size
        if f <= 0 or f % Self.STEP != 0 or BLOCK % f != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        # The buffer holds one fit a segment, while `at` counts the values
        # stored, so the fits written are `ceildiv(at, size)`.
        return (
            _Filling(Buffer.alloc_uninit[T](ceildiv(count, f)), count),
            self.inner.start[T](count),
        )

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        comptime W = 64 // size_of[Scalar[T]]()
        var m = len(block)
        if m > BLOCK or state[0].at % BLOCK != 0:
            raise Self.block_error(m)
        var scratch = Array[UInt8, BLOCK * size_of[Scalar[T]]()](
            uninitialized=True
        )
        var fitted = BufferView[T].from_bytes(Span(scratch))[:m]
        var fits = state[0].buffer.view[T]()
        var f = self.size
        # The segment the block starts, as blocks hold whole segments.
        var s = state[0].at // f
        var lo = 0
        while lo < m:
            var hi = min(lo + f, m)
            # Each register folded across its lanes, then into the fit.
            var fit = Self.F.identity[T]()
            var i = lo
            while i + W <= hi:
                fit = Self.F.combine[T, 1](
                    fit, Self.F.fold_lanes[T, W](block.load[W](i))
                )
                i += W
            while i < hi:
                fit = Self.F.combine[T, 1](fit, block.load[1](i))
                i += 1
            fits.unsafe_set(s, fit)
            s += 1
            var base = SIMD[T, W](fit)
            i = lo
            while i + W <= hi:
                fitted.store[W](
                    i, Self.Enc.same_core[T, W](block.load[W](i), base)
                )
                i += W
            while i < hi:
                fitted.store[1](
                    i, Self.Enc.same_core[T, 1](block.load[1](i), fit)
                )
                i += 1
            lo = hi
        state[0].at += m
        self.inner.store[T](state[1], fitted)

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        var fits = Optional[_Filling](None)
        var inner = Optional[Self.Inner.State[T]](None)

        def take[k: Int](var part: Self.State[T].Ts[k]) {mut fits, mut inner}:
            comptime if k == 0:
                fits = rebind_var[_Filling](part^)
            else:
                inner = rebind_var[Self.Inner.State[T]](part^)

        state^.consume_elements(take)
        var filled = fits.take()
        var count = ceildiv(filled.at, self.size)
        return Self.Packed(
            PerSegment(filled^.frozen(), count),
            self.inner.finish[T](inner.take()),
        )


@fieldwise_init
struct FittedUnpack[Dec: BinaryKernel, Inner: UnpackKernel](UnpackKernel):
    """`Dec(v, fit)` of each value `Inner` loads and its segment's fit, in
    segments of `size` values."""

    comptime name = "fitted_unpack"
    comptime Packed = FittedPacked[Self.Inner.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        PerSegment, BySize, Self.Inner.State[T]
    ]
    comptime STEP = 64

    var size: Int
    var inner: Self.Inner

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        comptime assert T.is_integral(), "FittedUnpack yields integers"
        var f = self.size
        if f <= 0 or f % Self.STEP != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        var segments = BySize(count, f)
        if len(packed.fits) != segments.count() or not packed.fits.holds[T]():
            raise Self.error[CorruptError](
                t"{len(packed.fits)} fits in a buffer of"
                t" {len(packed.fits.values)} bytes for {segments.count()}"
                t" segments"
            )
        return (
            packed.fits.copy(),
            segments,
            self.inner.start[T](packed.inner.copy(), count),
        )

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        var fit = state[0].get[T](state[1].of(i))
        var v = self.inner.load[T, W](state[2], i)
        return Self.Dec.same_core[T, W](v, SIMD[T, W](fit))

    def check[T: DType](self, state: Self.State[T]) raises:
        self.inner.check[T](state[2])


comptime ReferencePack[Inner: PackKernel] = FittedPack[
    MinFold, SubKernel, Inner
]
"""Frame of reference: each value less the least of its segment, so a column
clustered far from zero leaves small integers for the inner packer."""

comptime ReferenceUnpack[Inner: UnpackKernel] = FittedUnpack[AddKernel, Inner]
"""`ReferencePack`'s values: each segment's least added back."""

comptime ReferencePacked[P: Copyable & Deinitable] = FittedPacked[P]
"""What `ReferencePack` packs: each segment's least, and the inner `Packed`."""


# --- Rle -----------------------------------------------------------------------


@fieldwise_init
struct RlePacked(Copyable, Movable):
    """Parquet's RLE / bit-packed hybrid: runs of one value -- a header, then
    the value in its whole bytes -- or bit-packed groups of eight."""

    var bytes: Buffer[mut=False]
    var size: Int
    """How many of the bytes are the runs'."""


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
        comptime G = RlePack.GROUP
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


@fieldwise_init
struct RlePack(Copyable, PackKernel):
    """Unsigned integers of `width` bits as `RlePacked`. A run is written as
    one only where a stretch of packed groups may end: at its start or after
    a whole group."""

    comptime name = "rle_pack"
    comptime Packed = RlePacked
    comptime State[T: DType]: Movable & Deinitable = _Filling
    comptime GROUP = 8

    var width: Int

    def start[T: DType](self, count: Int) raises -> _Filling:
        return _Filling(Buffer.alloc_uninit[T](count), count)

    def store[
        T: DType
    ](self, mut state: _Filling, block: BufferView[T, _]) raises:
        state.buffer.extend(block, state.at, len(block))
        state.at += len(block)

    def finish[T: DType](self, var state: _Filling) raises -> RlePacked:
        var n = state.at
        var values = state^.frozen()
        var out = List[UInt8]()
        Self.write_runs(values.view[T](0, n).as_span(), self.width, out)
        var bytes = Buffer.alloc_uninit(len(out))
        bytes.extend(BufferView(Span(out)), 0, len(out))
        return RlePacked(bytes^.to_immutable(), len(out))

    @staticmethod
    def write_runs[
        T: DType
    ](values: Span[Scalar[T], _], width: Int, mut out: List[UInt8]) raises:
        """Append `values` at `width` bits; a `width` of 0 writes nothing."""
        comptime G = Self.GROUP
        comptime U = Unsigned[T]
        if width == 0:
            return
        var codec = Codec(
            RunsPack(PlainPack(), PlainPack()),
            RunsUnpack(PlainUnpack(), PlainUnpack()),
        )
        var runs = codec.encode(BufferView(values))
        var run_values = runs.values.bytes.view[T](0, runs.runs)
        var run_ends = runs.ends.bytes.view[DType.uint64](0, runs.runs)
        var byte_width = (width + 7) // 8
        var literals = List[UInt64]()
        var start = 0
        for r in range(runs.runs):
            var v = bitcast[U](run_values[r]).cast[DType.uint64]()
            var end = Int(run_ends[r])
            var n = end - start
            start = end
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


struct _RleReading(Movable):
    """What `RleUnpack` reads: the runs, where it is in them, and the block
    decoded; whether the runs ended before the values."""

    var packed: RlePacked
    var runs: RleRuns
    var group: PackedGroup
    var unpacked: Unpacked
    var bad: Bool

    def __init__(out self, var packed: RlePacked, width: Int, count: Int):
        self.packed = packed^
        self.runs = RleRuns()
        self.group = PackedGroup(width)
        self.unpacked = Unpacked(count)
        self.bad = False


@fieldwise_init
struct RleUnpack(Copyable, UnpackKernel):
    """`RlePacked`'s values of `width` bits, a block at a time."""

    comptime name = "rle_unpack"
    comptime Packed = RlePacked
    comptime State[T: DType]: Movable & Deinitable = _RleReading

    var width: Int

    def start[
        T: DType
    ](self, var packed: RlePacked, count: Int) raises -> _RleReading:
        if (
            self.width > bit_width_of[T]()
            or packed.size < 0
            or packed.size > len(packed.bytes)
        ):
            raise Self.error[CorruptError](
                t"{count} values of {self.width} bits in {packed.size} bytes"
                t" of a buffer of {len(packed.bytes)}"
            )
        return _RleReading(packed^, self.width, count)

    def _unpack[T: DType](self, mut state: _RleReading):
        var out = state.unpacked.next[T]()
        var n = state.unpacked.size()
        if self.width == 0:
            out.as_span()[:n].fill(0)
        elif not state.runs.fill[T](
            state.packed.bytes.view[DType.uint8](0, state.packed.size),
            state.group,
            out[:n],
        ):
            state.bad = True

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: _RleReading, i: Int) -> SIMD[T, W]:
        if state.unpacked.due(i):
            self._unpack[T](state)
        return state.unpacked.load[T, W](i)

    def check[T: DType](self, state: _RleReading) raises:
        if state.bad:
            raise Self.error[CorruptError]("truncated runs")

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
            raise Self.error[CorruptError](t"{count} values of {width} bits")
        var base = len(out)
        out.resize(unsafe_uninit_length=base + count)
        var dst = BufferView(Span(out))[base:]
        if width == 0:
            dst.as_span().fill(0)
            return 0
        var runs = RleRuns()
        if not runs.fill[T](BufferView(data), PackedGroup(width), dst):
            raise Self.error[CorruptError]("truncated runs")
        return runs.pos

    @staticmethod
    def read_codes[
        T: DType
    ](data: Span[UInt8, _], count: Int, mut out: List[Scalar[T]]) raises:
        """Append a data page's first `count` dictionary codes: their width
        in a byte, then their runs."""
        if len(data) == 0:
            if count > 0:
                raise Self.error[CorruptError](t"no codes for {count} values")
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
                raise Self.error[CorruptError]("truncated run")
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


# --- Lengths -------------------------------------------------------------------


@fieldwise_init
struct LengthsPacked[L: Copyable & Deinitable](Copyable, Movable):
    """Byte strings as their bytes end to end, and their lengths, `int32`s
    packed by a value packer: Parquet's DELTA_LENGTH_BYTE_ARRAY when the
    lengths are delta-packed.

        ["ab", "abc"]  ->  61 62 61 62 63, lengths [2, 3]
    """

    var bytes: Buffer[mut=False]
    var size: Int
    """How many of the bytes are the strings'."""
    var lengths: Self.L


@fieldwise_init
struct LengthsPack[L: PackKernel](Kernel, Movable):
    """Byte strings as `LengthsPacked`, the lengths packed by `L`."""

    comptime name = "lengths_pack"

    var lengths: Self.L

    def pack(
        self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> LengthsPacked[Self.L.Packed]:
        """String `i` is `data[offsets[i] : offsets[i + 1]]`."""
        var n = max(len(offsets) - 1, 0)
        var ends = BufferView(offsets)
        var state = self.lengths.start[DType.int32](n)
        var scratch = Array[UInt8, BLOCK * 4](uninitialized=True)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            var lengths = BufferView[DType.int32].from_bytes(Span(scratch))[:m]

            @always_inline
            def length[W: Int](i: Int) {imm}:
                lengths.store[W](
                    i, ends.load[W](lo + i + 1) - ends.load[W](lo + i)
                )

            vectorize[16](m, length)
            self.lengths.store[DType.int32](state, lengths)
            lo += m
        var first = Int(offsets[0]) if n > 0 else 0
        var size = Int(offsets[n]) - first if n > 0 else 0
        var bytes = Buffer.alloc_uninit(size)
        bytes.extend(BufferView(data[first : first + size]), 0, size)
        return LengthsPacked(
            bytes^.to_immutable(),
            size,
            self.lengths.finish[DType.int32](state^),
        )


@fieldwise_init
struct LengthsUnpack[L: UnpackKernel](Kernel, Movable):
    """`LengthsPacked`'s byte strings, the lengths unpacked by `L` a register
    at a time."""

    comptime name = "lengths_unpack"

    var lengths: Self.L

    def unpack(
        self,
        var packed: LengthsPacked[Self.L.Packed],
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """The `count` strings `packed` holds: their bytes appended to
        `data`, and where each ends to `offsets`."""
        if packed.size < 0 or packed.size > len(packed.bytes):
            raise Self.error[CorruptError](
                t"{packed.size} bytes in a buffer of {len(packed.bytes)}"
            )
        # The bytes go where they belong at once; the lengths then only say
        # where each string ends.
        var end = len(data)
        data.extend(packed.bytes.view[DType.uint8](0, packed.size).as_span())
        var last = len(data)
        var state = self.lengths.start[DType.int32](
            packed.lengths.copy(), count
        )
        var base = len(offsets)
        offsets.resize(unsafe_uninit_length=base + count)
        var out = BufferView(Span(offsets))
        # Checked once every length is read: with none negative, the ends
        # only grow, so a last end at the bytes' end means all are within.
        var signs = Int32(0)
        comptime W = 16
        var i = 0
        while i + W <= count:
            var v = self.lengths.load[DType.int32, W](state, i)
            signs |= v.reduce_or()
            comptime for k in range(W):
                end += Int(v[k])
                out.unsafe_set(base + i + k, Int32(end))
            i += W
        while i < count:
            var v = self.lengths.load[DType.int32, 1](state, i)
            signs |= v
            end += Int(v)
            out.unsafe_set(base + i, Int32(end))
            i += 1
        self.lengths.check[DType.int32](state)
        if signs < 0 or end != last:
            raise Self.error[CorruptError]("lengths do not cover the bytes")


# --- Prefixes ------------------------------------------------------------------


@fieldwise_init
struct PrefixesPacked[P: Copyable & Deinitable, L: Copyable & Deinitable](
    Copyable, Movable
):
    """Byte strings as the length of the prefix each shares with the one
    before, `int32`s packed by a value packer, and what follows each prefix,
    its suffix, as `LengthsPacked`: Parquet's DELTA_BYTE_ARRAY.

        ["ab", "abc"]  ->  prefixes [0, 2], suffixes ["ab", "c"]
    """

    var prefixes: Self.P
    var suffixes: LengthsPacked[Self.L]


struct PrefixesPack[P: PackKernel, L: PackKernel](Kernel, Movable):
    """Byte strings as `PrefixesPacked`: the prefix lengths packed by `P`,
    the suffixes' lengths by `L`."""

    comptime name = "prefixes_pack"

    var prefixes: Self.P
    var suffixes: LengthsPack[Self.L]

    def __init__(out self, var prefixes: Self.P, var lengths: Self.L):
        self.prefixes = prefixes^
        self.suffixes = LengthsPack(lengths^)

    def pack(
        self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> PrefixesPacked[Self.P.Packed, Self.L.Packed]:
        """String `i` is `data[offsets[i] : offsets[i + 1]]`."""
        var n = max(len(offsets) - 1, 0)
        var state = self.prefixes.start[DType.int32](n)
        var scratch = Array[UInt8, BLOCK * 4](uninitialized=True)
        # The suffixes take at most the strings' bytes.
        var bytes = List[UInt8]()
        bytes.resize(unsafe_uninit_length=len(data))
        var ends = List[Int32]()
        ends.resize(unsafe_uninit_length=n + 1)
        var suffixes = BufferView(Span(ends))
        suffixes.unsafe_set(0, 0)
        var bounds = BufferView(offsets)
        var out = Span(bytes)
        var at = 0
        # The string before: none, so the first shares nothing.
        var before = 0
        var after = 0
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            var prefixes = BufferView[DType.int32].from_bytes(Span(scratch))[:m]
            for i in range(m):
                var start = Int(bounds.unsafe_get(lo + i))
                var end = Int(bounds.unsafe_get(lo + i + 1))
                var most = min(after - before, end - start)
                var p = match_length(data, start, before, start + most)
                prefixes.unsafe_set(i, Int32(p))
                out[at : at + end - start - p].copy_from(data[start + p : end])
                at += end - start - p
                suffixes.unsafe_set(lo + i + 1, Int32(at))
                before = start
                after = end
            self.prefixes.store[DType.int32](state, prefixes)
            lo += m
        return PrefixesPacked(
            self.prefixes.finish[DType.int32](state^),
            self.suffixes.pack(Span(ends), Span(bytes)[:at]),
        )


struct PrefixesUnpack[P: UnpackKernel, L: UnpackKernel](Kernel, Movable):
    """`PrefixesPacked`'s byte strings: the prefix lengths unpacked by `P` a
    register at a time, the suffixes' lengths by `L`."""

    comptime name = "prefixes_unpack"

    var prefixes: Self.P
    var suffixes: LengthsUnpack[Self.L]

    def __init__(out self, var prefixes: Self.P, var lengths: Self.L):
        self.prefixes = prefixes^
        self.suffixes = LengthsUnpack(lengths^)

    def unpack(
        self,
        var packed: PrefixesPacked[Self.P.Packed, Self.L.Packed],
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """The `count` strings `packed` holds: their bytes appended to
        `data`, and where each ends to `offsets`."""
        var ends: List[Int32] = [0]
        var bytes = List[UInt8]()
        self.suffixes.unpack(packed.suffixes.copy(), count, ends, bytes)
        var suffixes = BufferView(Span(ends))
        var state = self.prefixes.start[DType.int32](
            packed.prefixes.copy(), count
        )
        offsets.reserve(len(offsets) + count)
        # The string before: none, so the first shares nothing.
        var before = len(data)
        var after = before
        var bad = False
        # A block of prefixes at a time, pulled a register at a time, so the
        # strings are put together in a loop of their own.
        comptime W = 16
        var scratch = Array[UInt8, BLOCK * 4](uninitialized=True)
        var lo = 0
        while lo < count:
            var m = min(BLOCK, count - lo)
            var prefixes = BufferView[DType.int32].from_bytes(Span(scratch))[:m]
            var i = 0
            while i + W <= m:
                prefixes.store[W](
                    i, self.prefixes.load[DType.int32, W](state, lo + i)
                )
                i += W
            while i < m:
                prefixes.store[1](
                    i, self.prefixes.load[DType.int32, 1](state, lo + i)
                )
                i += 1
            for k in range(m):
                var p = Int(prefixes.unsafe_get(k))
                if p < 0 or p > after - before:
                    bad = True
                    p = 0
                var first = Int(suffixes.unsafe_get(lo + k))
                var last = Int(suffixes.unsafe_get(lo + k + 1))
                var start = len(data)
                data.resize(unsafe_uninit_length=start + p + last - first)
                # The prefix is the start of the string before, which ends
                # where this one starts, so the two never overlap.
                var out = BufferView(Span(data))
                for j in range(p):
                    out.unsafe_set(start + j, out.unsafe_get(before + j))
                out.as_span()[start + p : start + p + last - first].copy_from(
                    Span(bytes)[first:last]
                )
                offsets.append(Int32(len(data)))
                before = start
                after = len(data)
            lo += m
        self.prefixes.check[DType.int32](state)
        if bad:
            raise Self.error[CorruptError](
                "a prefix longer than the string before it"
            )
