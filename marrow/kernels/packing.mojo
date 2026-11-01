# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

# `FsstPack`'s symbol table and coding, and `FsstUnpack`'s decoding, are ported
# from FSST (https://github.com/cwida/fsst): `libfsst.cpp`
# (`buildSymbolTable`, `compressBulk`, `makeSample`), `libfsst.hpp` (`Symbol`,
# `SymbolTable`, `finalize`) and `fsst.h` (`fsst_decompress`). Copyright
# 2018-2020 CWI, TU Munich, FSU Jena, MIT License, and modified; see
# NOTICE.txt.

"""Packers: values in, a typed `Packed` out, and back a register at a time.

Each `*Pack` has its `*Unpack` and the `*Packed` a column is packed as.
`PlainPack` keeps the values' bytes as they are, `SplitPack` as byte planes,
`BitPack` at the bits each segment needs, and `RlePack` as Parquet's RLE /
bit-packed hybrid. `FittedPack` fits one value per segment with a fold and
packs each value against its segment's fit with an inner packer; frame of
reference, each value less its segment's least, is `ReferencePack`.
`ConstantPack` keeps a column that is one value as the value. `PatchPack`
fits each segment with a `FitKernel`, packs the values encoded under the fit
with an inner packer, and keeps apart, with their positions, the values
decoding does not give back -- the exceptions: `PforPack` packs at a width
fitted per segment and patches the values wider, `FrequencyPack` keeps each
segment's most common value once and patches the rest. `RunsPack`
(`runs.mojo`) stores each run of equal values once.

Byte strings go in as offsets and bytes and come out the same way:
`LengthsPack` keeps their bytes end to end and packs their lengths,
`PrefixesPack` packs the length of the prefix each shares with the one
before and keeps the rest of each as `LengthsPack` does, and `FsstPack` codes
each by a table of up to 255 symbols of one to eight bytes, built from a
sample of them, and keeps the code strings as `LengthsPack` does. Each takes
the value packers of its `int32` streams, and has an `*Unpack` reading them
back a register at a time.

A packer that needs more than its values wraps an inner packer, and its
`Packed` holds the inner one's: `FittedPacked[BitPacked]` is the fits plus
the bits.

Segments: a packer that keeps a value per segment takes, encoding, the
segment sizes that divide `BLOCK`, so a block holds whole segments, and,
decoding, the multiples of its step -- a group of eight for `BitPack`, 64
values for `FittedPack` and `PatchPack` -- so a register never spans two
segments.
"""

from std.algorithm import vectorize
from std.bit import bit_width
from std.builtin.rebind import rebind_var
from std.builtin.sort import sort
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
from .compose import Codec, NoState
from .core import (
    BinaryKernel,
    FitKernel,
    Kernel,
    PackKernel,
    UnpackKernel,
)
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

    def nbytes[T: DType](self, packed: PlainPacked) -> Int:
        return packed.size


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

    def nbytes[T: DType](self, packed: SplitPacked) -> Int:
        return packed.size


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

    def nbytes[T: DType](self, packed: BitPacked) -> Int:
        # A `uint8` width a segment.
        return packed.size + len(packed.widths)


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

    def nbytes[T: DType](self, packed: Self.Packed) -> Int:
        return len(packed.fits) * size_of[Scalar[T]]() + self.inner.nbytes[T](
            packed.inner
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


# --- Constant ------------------------------------------------------------------


@fieldwise_init
struct ConstantPacked(Copyable, Movable):
    """One value standing for every value: its bits, and how many.

    [42, 42, 42] as int64  ->  value 42, count 3
    """

    var value: UInt64
    """The value's bits."""
    var count: Int


@fieldwise_init
struct ConstantPack(Copyable, Defaultable, PackKernel):
    """Values that are all one value, of at most eight bytes, as
    `ConstantPacked`. Equal means the same bits; a second value is
    refused."""

    comptime name = "constant_pack"
    comptime Packed = ConstantPacked
    comptime State[T: DType]: Movable & Deinitable = ConstantPacked

    def start[T: DType](self, count: Int) raises -> ConstantPacked:
        comptime assert (
            size_of[Scalar[T]]() <= 8
        ), "ConstantPack takes values of at most 8 bytes"
        return ConstantPacked(0, 0)

    def store[
        T: DType
    ](self, mut state: ConstantPacked, block: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var n = len(block)
        var i = 0
        while i < n:
            var bits = bitcast[U](block.load[1](i)).cast[DType.uint64]()
            if state.count == 0:
                state.value = bits
            elif bits != state.value:
                raise Self.error[InvalidError](
                    t"value {state.count} is not the one before it"
                )
            state.count += 1
            i += 1
            # The values after it, a register at a time while they are it.
            var v = SIMD[U, W](state.value.cast[U]())
            while (
                i + W <= n
                and bitcast[U, W](block.load[W](i)).eq(v).reduce_and()
            ):
                state.count += W
                i += W

    def finish[
        T: DType
    ](self, var state: ConstantPacked) raises -> ConstantPacked:
        return state^

    def nbytes[T: DType](self, packed: ConstantPacked) -> Int:
        return size_of[Scalar[T]]() if packed.count > 0 else 0


@fieldwise_init
struct ConstantUnpack(Copyable, Defaultable, UnpackKernel):
    """`ConstantPacked`'s value, as every value."""

    comptime name = "constant_unpack"
    comptime Packed = ConstantPacked
    comptime State[T: DType]: Movable & Deinitable = Scalar[T]

    def start[
        T: DType
    ](self, var packed: ConstantPacked, count: Int) raises -> Scalar[T]:
        comptime B = bit_width_of[T]()
        comptime assert (
            B <= 64
        ), "ConstantUnpack yields values of at most 8 bytes"
        var fits = True
        comptime if B < 64:
            fits = packed.value >> UInt64(B) == 0
        if packed.count != count or not fits:
            raise Self.error[CorruptError](
                t"{packed.count} values of bits {packed.value} for {count}"
                t" {T} values"
            )
        return bitcast[T](packed.value.cast[Unsigned[T]]())

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Scalar[T], i: Int) -> SIMD[T, W]:
        return SIMD[T, W](state)


# --- Patch ---------------------------------------------------------------------


@fieldwise_init
struct PatchPacked[P: Copyable & Deinitable](Copyable, Movable):
    """Each segment's fit, the values whose round trip under it fails -- the
    exceptions, each with its position -- and every value encoded under its
    fit, an exception holding a filler, packed by the inner packer.

        Frequency over [7, 7, 3, 7, 9] as int64  ->  fits [7], ends [2],
        positions [2, 4], values [3, 9], the inner packer five zeros
    """

    var fits: PerSegment
    """Each segment's fit, a `uint64`."""
    var ends: PerSegment
    """The exceptions up to each segment's end, a `uint32`."""
    var positions: PlainPacked
    """Each exception's position within its segment, a `uint16`, ascending
    within the segment."""
    var values: PlainPacked
    """The exceptions, in the column's dtype."""
    var inner: Self.P


struct _Patching[T: DType](Movable):
    """The fits and ends of the segments packed, the exceptions found in
    them, and the values stored."""

    var fits: Buffer[mut=True]
    var ends: Buffer[mut=True]
    var segments: Int
    var positions: List[UInt16]
    var values: List[Scalar[Self.T]]
    var stored: Int

    def __init__(out self, segments: Int):
        self.fits = Buffer.alloc_uninit[DType.uint64](segments)
        self.ends = Buffer.alloc_uninit[DType.uint32](segments)
        self.segments = 0
        self.positions = List[UInt16]()
        self.values = List[Scalar[Self.T]]()
        self.stored = 0

    def packed[
        P: Copyable & Deinitable
    ](deinit self, var inner: P) -> PatchPacked[P]:
        """What was packed, `inner` the values encoded."""
        var n = len(self.positions)
        var positions = Buffer.alloc_uninit[DType.uint16](n)
        positions.extend(BufferView(Span(self.positions)), 0, n)
        var values = Buffer.alloc_uninit[Self.T](n)
        values.extend(BufferView(Span(self.values)), 0, n)
        return PatchPacked(
            PerSegment(self.fits^.to_immutable(), self.segments),
            PerSegment(self.ends^.to_immutable(), self.segments),
            PlainPacked(positions^.to_immutable(), 2 * n),
            PlainPacked(values^.to_immutable(), n * size_of[Scalar[Self.T]]()),
            inner^,
        )


@fieldwise_init
struct PatchPack[Fit: FitKernel, Inner: PackKernel](PackKernel):
    """Fits each segment of `size` values with `Fit`, packs each value
    encoded under the fit with `Inner`, and keeps each value its decoding
    does not give back bit for bit.

    An exception's slot holds the encoding of the segment's first value
    that does come back, or zero when none does, so it widens nothing the
    inner packer packs.
    """

    comptime name = "patch_pack"
    comptime Packed = PatchPacked[Self.Inner.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        _Patching[T], Self.Fit.State[T], Self.Inner.State[Self.Fit.Encoded[T]]
    ]
    comptime STEP = 64

    var size: Int
    var inner: Self.Inner

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        var f = self.size
        if f <= 0 or f % Self.STEP != 0 or BLOCK % f != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        # The ends count exceptions in 32 bits.
        if count > Int(UInt32.MAX):
            raise Self.error[InvalidError](t"{count} values")
        return (
            _Patching[T](ceildiv(count, f)),
            Self.Fit.start[T](),
            self.inner.start[Self.Fit.Encoded[T]](count),
        )

    @always_inline
    @staticmethod
    def _encode[
        T: DType, W: Int
    ](
        mut patching: _Patching[T],
        block: BufferView[T, _],
        encoded: BufferView[mut=True, Self.Fit.Encoded[T], _],
        i: Int,
        lo: Int,
        fit: UInt64,
        mut filler: Optional[Scalar[Self.Fit.Encoded[T]]],
    ):
        """The register at `i` encoded under `fit`, into `encoded`; each of
        its values that does not come back kept, at its position in the
        segment from `lo`, and the first that does the filler."""
        comptime U = Unsigned[T]
        var v = block.load[W](i)
        var e = Self.Fit.encode[T, W](v, fit)
        encoded.store[W](i, e)
        var back = Self.Fit.decode[T, W](e, fit)
        var lost = bitcast[U, W](back).ne(bitcast[U, W](v))
        if lost.reduce_or():
            for k in range(W):
                if lost[k]:
                    patching.positions.append(UInt16(i + k - lo))
                    patching.values.append(v[k])
                elif not filler:
                    filler = e[k]
        elif not filler:
            filler = e[0]

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        comptime O = Self.Fit.Encoded[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var m = len(block)
        if m > BLOCK or state[0].stored % BLOCK != 0:
            raise Self.block_error(m)
        var scratch = Array[UInt8, BLOCK * size_of[Scalar[O]]()](
            uninitialized=True
        )
        var encoded = BufferView[O].from_bytes(Span(scratch))[:m]
        var lo = 0
        while lo < m:
            var hi = min(lo + self.size, m)
            var fit = Self.Fit.fit[T](state[1], block[lo:hi])
            var first = len(state[0].positions)
            var filler = Optional[Scalar[O]](None)
            var i = lo
            while i + W <= hi:
                Self._encode[T, W](state[0], block, encoded, i, lo, fit, filler)
                i += W
            while i < hi:
                Self._encode[T, 1](state[0], block, encoded, i, lo, fit, filler)
                i += 1
            var fill = Scalar[O](0)
            if filler:
                fill = filler.value()
            for k in range(first, len(state[0].positions)):
                encoded.unsafe_set(lo + Int(state[0].positions[k]), fill)
            var s = state[0].segments
            state[0].fits.view[DType.uint64]().unsafe_set(s, fit)
            state[0].ends.view[DType.uint32]().unsafe_set(
                s, UInt32(len(state[0].positions))
            )
            state[0].segments += 1
            lo = hi
        state[0].stored += m
        self.inner.store[O](state[2], encoded)

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        comptime O = Self.Fit.Encoded[T]
        var patching = Optional[_Patching[T]](None)
        var inner = Optional[Self.Inner.State[O]](None)

        # The fitting state is spent.
        def take[
            k: Int
        ](var part: Self.State[T].Ts[k]) {mut patching, mut inner}:
            comptime if k == 0:
                patching = rebind_var[_Patching[T]](part^)
            elif k == 1:
                _ = rebind_var[Self.Fit.State[T]](part^)
            else:
                inner = rebind_var[Self.Inner.State[O]](part^)

        state^.consume_elements(take)
        return patching.take().packed(self.inner.finish[O](inner.take()))

    def nbytes[T: DType](self, packed: Self.Packed) -> Int:
        # A `uint64` fit and a `uint32` end a segment.
        return (
            8 * len(packed.fits)
            + 4 * len(packed.ends)
            + packed.positions.size
            + packed.values.size
            + self.inner.nbytes[Self.Fit.Encoded[T]](packed.inner)
        )


struct _PatchReading(Movable):
    """What `PatchUnpack` reads: each segment's fit, the exceptions and
    where each lies in the column -- then a position past every value --
    and the next exception to put back."""

    var fits: PerSegment
    var segments: BySize
    var values: Buffer[mut=False]
    var at: List[Int]
    var next: Int
    var ahead: Int
    """The next exception's position."""

    def __init__(
        out self,
        var fits: PerSegment,
        segments: BySize,
        var values: Buffer[mut=False],
        var at: List[Int],
    ):
        self.fits = fits^
        self.segments = segments
        self.values = values^
        self.ahead = at[0]
        self.at = at^
        self.next = 0


@fieldwise_init
struct PatchUnpack[Fit: FitKernel, Inner: UnpackKernel](UnpackKernel):
    """Each value `Inner` loads decoded by `Fit` under the fit of its
    segment of `size` values; and each exception put back where it was."""

    comptime name = "patch_unpack"
    comptime Packed = PatchPacked[Self.Inner.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        _PatchReading, Self.Inner.State[Self.Fit.Encoded[T]]
    ]
    comptime STEP = 64

    var size: Int
    var inner: Self.Inner

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        var f = self.size
        if f <= 0 or f % Self.STEP != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        var segments = BySize(count, f)
        var s = segments.count()
        if len(packed.fits) != s or not packed.fits.holds[DType.uint64]():
            raise Self.error[CorruptError](
                t"{len(packed.fits)} fits in a buffer of"
                t" {len(packed.fits.values)} bytes for {s} segments"
            )
        if len(packed.ends) != s or not packed.ends.holds[DType.uint32]():
            raise Self.error[CorruptError](
                t"{len(packed.ends)} ends in a buffer of"
                t" {len(packed.ends.values)} bytes for {s} segments"
            )
        for g in range(s):
            var fit = packed.fits.get[DType.uint64](g)
            if not Self.Fit.valid[T](fit):
                raise Self.error[CorruptError](t"fit {fit} for {T}")
        # As many exceptions as the last end counts.
        var n = 0
        if s > 0:
            n = Int(packed.ends.get[DType.uint32](s - 1))
        var p = packed.positions.size
        var v = packed.values.size
        if (
            p != 2 * n
            or p > len(packed.positions.bytes)
            or v != n * size_of[Scalar[T]]()
            or v > len(packed.values.bytes)
        ):
            raise Self.error[CorruptError](
                t"{p} bytes of positions and {v} of values for {n} exceptions"
            )
        var positions = packed.positions.bytes.view[DType.uint16](0, n)
        var at = List[Int](capacity=n + 1)
        var k = 0
        for g in range(s):
            var end = Int(packed.ends.get[DType.uint32](g))
            if end < k or end > n:
                raise Self.error[CorruptError](
                    t"segment {g}'s exceptions end at {end}, after {k} of {n}"
                )
            var first = segments.start(g)
            var length = segments.end(g) - first
            var last = -1
            while k < end:
                var q = Int(positions[k])
                if q <= last or q >= length:
                    raise Self.error[CorruptError](
                        t"an exception at {q}, after {last}, in segment {g}"
                        t" of {length} values"
                    )
                at.append(first + q)
                last = q
                k += 1
        at.append(Int.MAX)
        return (
            _PatchReading(
                packed.fits.copy(), segments, packed.values.bytes, at^
            ),
            self.inner.start[Self.Fit.Encoded[T]](packed.inner.copy(), count),
        )

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        var fit = state[0].fits.get[DType.uint64](state[0].segments.of(i))
        var v = self.inner.load[Self.Fit.Encoded[T], W](state[1], i)
        var out = Self.Fit.decode[T, W](v, fit)
        # One compare a register; the exceptions it covers put back.
        if state[0].ahead < i + W:
            var kept = state[0].values.view[T]()
            while state[0].ahead < i + W:
                out[state[0].ahead - i] = kept.unsafe_get(state[0].next)
                state[0].next += 1
                state[0].ahead = state[0].at.unsafe_get(state[0].next)
        return out

    def check[T: DType](self, state: Self.State[T]) raises:
        self.inner.check[Self.Fit.Encoded[T]](state[1])


struct Pfor(FitKernel):
    """Unsigned integers at a width fitted per segment: each value's low
    `fit` bits, decoded as they are. The width is the one that packs the
    segment smallest, a value wider than it costing its own bits and a
    16-bit position too."""

    comptime name = "pfor"
    comptime State[T: DType]: Movable & Deinitable = NoState

    @staticmethod
    def start[T: DType]() -> NoState:
        return NoState()

    @staticmethod
    def fit[T: DType](mut state: NoState, segment: BufferView[T, _]) -> UInt64:
        comptime assert T.is_unsigned(), "Pfor takes unsigned integers"
        comptime B = bit_width_of[T]()
        var n = len(segment)
        # How many values need each width; then, from the widest down, the
        # width that costs least, ties to the wider.
        var needs = Array[Int, B + 1](fill=0)
        for i in range(n):
            needs[Int(bit_width(segment.unsafe_get(i)))] += 1
        var best = B
        var cost = n * B
        var wider = 0
        var w = B
        while w > 0:
            wider += needs[w]
            w -= 1
            var c = w * n + wider * (16 + B)
            if c < cost:
                cost = c
                best = w
        return UInt64(best)

    @always_inline
    @staticmethod
    def encode[T: DType, W: Int](v: SIMD[T, W], fit: UInt64) -> SIMD[T, W]:
        comptime assert T.is_unsigned(), "Pfor takes unsigned integers"
        var top = Scalar[T](0)
        if fit > 0:
            top = Scalar[T].MAX >> Scalar[T](UInt64(bit_width_of[T]()) - fit)
        return v & SIMD[T, W](top)

    @always_inline
    @staticmethod
    def decode[T: DType, W: Int](v: SIMD[T, W], fit: UInt64) -> SIMD[T, W]:
        # The width says only which values were too wide to keep.
        return v

    @staticmethod
    def valid[T: DType](fit: UInt64) -> Bool:
        return fit <= UInt64(bit_width_of[T]())


struct Frequency(FitKernel):
    """Values of at most eight bytes under their segment's most common
    value, the fit -- the one held longest, ties the smaller bits -- each
    encoded as zero and decoded as the fit, so only the values that are not
    it are kept."""

    comptime name = "frequency"
    comptime State[T: DType]: Movable & Deinitable = NoState

    @staticmethod
    def start[T: DType]() -> NoState:
        return NoState()

    @staticmethod
    def fit[T: DType](mut state: NoState, segment: BufferView[T, _]) -> UInt64:
        comptime assert (
            size_of[Scalar[T]]() <= 8
        ), "Frequency takes values of at most 8 bytes"
        comptime U = Unsigned[T]
        var n = len(segment)
        debug_assert(n <= BLOCK, "Frequency: a segment of a block")
        # The bits sorted, so each value's copies are a run.
        var scratch = Array[UInt64, BLOCK](uninitialized=True)
        var bits = Span(scratch)[:n]
        for i in range(n):
            bits[i] = bitcast[U](segment.unsafe_get(i)).cast[DType.uint64]()
        sort(bits)
        var top = UInt64(0)
        var most = 0
        var run = 0
        for i in range(n):
            if i > 0 and bits[i] != bits[i - 1]:
                run = 0
            run += 1
            if run > most:
                most = run
                top = bits[i]
        return top

    @always_inline
    @staticmethod
    def encode[T: DType, W: Int](v: SIMD[T, W], fit: UInt64) -> SIMD[T, W]:
        return SIMD[T, W](0)

    @always_inline
    @staticmethod
    def decode[T: DType, W: Int](v: SIMD[T, W], fit: UInt64) -> SIMD[T, W]:
        return SIMD[T, W](bitcast[T](fit.cast[Unsigned[T]]()))

    @staticmethod
    def valid[T: DType](fit: UInt64) -> Bool:
        comptime B = bit_width_of[T]()
        comptime if B < 64:
            return fit >> UInt64(B) == 0
        else:
            return True


comptime PforPack = PatchPack[Pfor, BitPack]
"""Patched bit packing: each segment at the width that packs it smallest,
the values wider than it kept apart. Under `ReferencePack`, patched frame of
reference."""

comptime PforUnpack = PatchUnpack[Pfor, BitUnpack]
"""`PforPack`'s values."""

comptime FrequencyPack = PatchPack[Frequency, ConstantPack]
"""Each segment's most common value once, and the values that are not it
with their positions."""

comptime FrequencyUnpack = PatchUnpack[Frequency, ConstantUnpack]
"""`FrequencyPack`'s values."""


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

    def nbytes[T: DType](self, packed: RlePacked) -> Int:
        return packed.size

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

    def nbytes(self, packed: LengthsPacked[Self.L.Packed]) -> Int:
        """The bytes `packed` holds: the strings', and their lengths'."""
        return packed.size + self.lengths.nbytes[DType.int32](packed.lengths)


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

    def nbytes(
        self, packed: PrefixesPacked[Self.P.Packed, Self.L.Packed]
    ) -> Int:
        """The bytes `packed` holds: the prefix lengths', and the
        suffixes'."""
        return self.prefixes.nbytes[DType.int32](
            packed.prefixes
        ) + self.suffixes.nbytes(packed.suffixes)


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


# --- Fsst ----------------------------------------------------------------------


@fieldwise_init
struct FsstPacked[L: Copyable & Deinitable](Copyable, Movable):
    """Byte strings as codes into a table of up to 255 symbols of one to
    eight bytes: each string a code a symbol, and for a byte no symbol
    starts with, the escape 255 and the byte; the table, and the code
    strings as `LengthsPacked`.

        ["abab", "ab"], one symbol "ab"  ->  codes ["\\x00\\x00", "\\x00"]
    """

    var symbols: Buffer[mut=False]
    """Each symbol's bytes in a `uint64`, little-endian."""
    var lengths: Buffer[mut=False]
    """Each symbol's length, a `uint8` from 1 to 8."""
    var count: Int
    """The symbols, at most 255."""
    var codes: LengthsPacked[Self.L]


struct _SymbolTable(Copyable, Movable):
    """The symbols an `FsstPack` codes by, and the tables it finds the
    longest symbol a string goes on with by: a symbol of three bytes or more
    by a hash of its first three, of two by its two, of one by its byte.

    While the table is built, codes 0-255 are the bytes themselves, escaped,
    and a symbol's code is 256 on; once `finalize`d, the symbols are codes
    0-254 and every escape is 511 in the lookups, written as 255. A symbol's
    `icl` packs its length (bits 28-31), its code (16-27) and the bits past
    its bytes in a `uint64` (0-15).
    """

    comptime BASE = 256
    """The first code a symbol takes while the table is built."""
    comptime CODES = 512
    comptime MASK = UInt64(511)
    comptime HASHED = 1024
    comptime FREE = UInt64((15 << 28) | (511 << 16))
    """The `icl` of a free hash bucket, past every symbol's."""
    comptime ESCAPE = 255
    comptime SAMPLE = 1 << 14
    """How many bytes of the strings a table is built from, at least."""
    comptime LINE = 512
    """The most bytes of one string a sample takes at once."""

    var nums: List[UInt64]
    """Each code's bytes, little-endian, and nothing past them."""
    var icls: List[UInt64]
    """Each code's `icl`."""
    var hash_nums: List[UInt64]
    """The symbols of three bytes or more, by a hash of their first three."""
    var hash_icls: List[UInt64]
    """Their `icl`s, `FREE` where none is."""
    var shorts: List[UInt16]
    """Each two bytes' code: a two-byte symbol's, else the first byte's."""
    var bytes: List[UInt16]
    """Each byte's code, a one-byte symbol's or its escape."""
    var count: Int
    """The symbols, at most 255."""
    var terminator: Int
    """A byte no symbol of two bytes or more holds, so a string can end in
    it while a register reads past its end."""
    var histogram: List[Int]
    """How many symbols are 1, 2, ... 8 bytes long."""

    def __init__(out self, terminator: Int):
        """No symbols: every byte escaped."""
        self.nums = List[UInt64](length=Self.CODES, fill=0)
        self.icls = List[UInt64](
            length=Self.CODES, fill=Self.icl(Int(Self.MASK), 1)
        )
        for i in range(Self.BASE):
            self.nums[i] = UInt64(i)
            self.icls[i] = Self.icl(i, 1)
        self.hash_nums = List[UInt64](length=Self.HASHED, fill=0)
        self.hash_icls = List[UInt64](length=Self.HASHED, fill=Self.FREE)
        self.shorts = List[UInt16](length=1 << 16, fill=0)
        for i in range(1 << 16):
            self.shorts[i] = UInt16((1 << 12) | (i & 255))
        self.bytes = List[UInt16](length=256, fill=0)
        for i in range(256):
            self.bytes[i] = UInt16((1 << 12) | i)
        self.count = 0
        self.terminator = terminator
        self.histogram = List[Int](length=8, fill=0)

    @always_inline
    @staticmethod
    def hash(w: UInt64) -> UInt64:
        var h = w * 2971215073
        return h ^ (h >> 15)

    @always_inline
    @staticmethod
    def icl(code: Int, length: Int) -> UInt64:
        return UInt64((length << 28) | (code << 16) | ((8 - length) * 8))

    @always_inline
    def length(self, code: Int) -> Int:
        return Int(self.icls[code] >> 28)

    def add(mut self, num: UInt64, length: Int) -> Bool:
        """Symbol `num` of `length` bytes as the next code; false when its
        hash bucket is taken."""
        var code = Self.BASE + self.count
        var icl = Self.icl(code, length)
        var added = True
        if length == 1:
            self.bytes[Int(num & 0xFF)] = UInt16(code + (1 << 12))
        elif length == 2:
            self.shorts[Int(num & 0xFFFF)] = UInt16(code + (2 << 12))
        else:
            var at = Int(Self.hash(num & 0xFFFFFF) & UInt64(Self.HASHED - 1))
            if self.hash_icls[at] < Self.FREE:
                added = False
            else:
                self.hash_icls[at] = icl
                self.hash_nums[at] = num & (UInt64.MAX >> (icl & 0xFF))
        if added:
            self.nums[code] = num
            self.icls[code] = icl
            self.histogram[length - 1] += 1
            self.count += 1
        return added

    def longest(self, num: UInt64, length: Int) -> Int:
        """The code of the longest symbol the `length` bytes of `num` start
        with, an escape when none does."""
        var at = Int(Self.hash(num & 0xFFFFFF) & UInt64(Self.HASHED - 1))
        var hashed = self.hash_icls[at]
        var short = UInt64(self.shorts[Int(num & 0xFFFF)]) & Self.MASK
        var code: UInt64
        if hashed <= Self.icl(Self.CODES, length) and self.hash_nums[at] == (
            num & (UInt64.MAX >> (hashed & 0xFF))
        ):
            code = (hashed >> 16) & Self.MASK
        elif length >= 2 and short >= UInt64(Self.BASE):
            code = short
        else:
            code = UInt64(self.bytes[Int(num & 0xFF)]) & Self.MASK
        return Int(code)

    def counted(
        self,
        lines: Span[Int, _],
        sample: Span[UInt8, _],
        fraction: Int,
        mut singles: List[UInt16],
        mut pairs: List[UInt16],
    ) -> Int:
        """The bytes coding the sample by this table saves, counting each
        code, and below the last round each pair of codes, as it goes. A
        round below the last skips lines at random, the more the lower
        `fraction` is, out of 128."""
        var gain = 0
        for i in range(len(lines) - 1):
            var cur = lines[i]
            var end = lines[i + 1]
            var skip = (
                fraction < 128
                and Int(1 + (Self.hash(UInt64((i + 1) * fraction)) & 127))
                > fraction
            )
            if cur < end and not skip:
                var start = cur
                var code1 = self.longest(
                    LittleEndian.partial[DType.uint64](sample[:end], cur),
                    min(end - cur, 8),
                )
                cur += self.length(code1)
                gain += self.length(code1) - (2 if code1 < Self.BASE else 1)
                var more = True
                while more:
                    singles[code1] += 1
                    # The first byte on its own, as an alternative.
                    if self.length(code1) != 1:
                        singles[Int(sample[start])] += 1
                    if cur == end:
                        more = False
                    else:
                        start = cur
                        var code2 = self.longest(
                            LittleEndian.partial[DType.uint64](
                                sample[:end], cur
                            ),
                            min(end - cur, 8),
                        )
                        cur += self.length(code2)
                        gain += (cur - start) - (2 if code2 < Self.BASE else 1)
                        if fraction < 128:
                            pairs[code1 * Self.CODES + code2] += 1
                            # Extended by the next byte alone, as an
                            # alternative.
                            if cur - start > 1:
                                pairs[
                                    code1 * Self.CODES + Int(sample[start])
                                ] += 1
                        code1 = code2
        return gain

    def made(
        self,
        mut singles: List[UInt16],
        pairs: List[UInt16],
        fraction: Int,
    ) -> Self:
        """The table of the 255 symbols, of this one's and of pairs of them
        joined, that save the most by the counts; below the last round
        `fraction`, rarer candidates are dropped."""
        # The terminator counted the most often, so it stays a symbol.
        if self.count > 0:
            singles[Self.BASE] = UInt16.MAX
        else:
            singles[self.terminator] = UInt16.MAX
        var least = (5 * fraction) // 128
        # A candidate: its bytes, its length, and what it saves.
        var candidates = List[SIMD[DType.uint64, 4]]()
        for pos1 in range(Self.BASE + self.count):
            var count1 = Int(singles[pos1])
            if count1 > 0:
                var num1 = self.nums[pos1]
                var length1 = self.length(pos1)
                # Single bytes promoted: fewer escapes.
                var weighted = count1 * (8 if length1 == 1 else 1)
                if weighted >= least:
                    candidates.append(
                        SIMD[DType.uint64, 4](
                            num1,
                            UInt64(length1),
                            UInt64(weighted * length1),
                            0,
                        )
                    )
                if (
                    fraction < 128
                    and length1 < 8
                    and Int(num1 & 0xFF) != self.terminator
                ):
                    for pos2 in range(Self.BASE + self.count):
                        var count2 = Int(pairs[pos1 * Self.CODES + pos2])
                        var num2 = self.nums[pos2]
                        if (
                            count2 >= max(least, 1)
                            and Int(num2 & 0xFF) != self.terminator
                        ):
                            var length = min(length1 + self.length(pos2), 8)
                            candidates.append(
                                SIMD[DType.uint64, 4](
                                    (num2 << UInt64(8 * length1)) | num1,
                                    UInt64(length),
                                    UInt64(count2 * length),
                                    0,
                                )
                            )

        # Equal candidates merged, their savings summed.
        def by_symbol(
            a: SIMD[DType.uint64, 4], b: SIMD[DType.uint64, 4]
        ) -> Bool:
            return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

        sort(candidates, by_symbol)
        var merged = List[SIMD[DType.uint64, 4]]()
        for c in candidates:
            var last = len(merged) - 1
            if (
                last >= 0
                and merged[last][0] == c[0]
                and merged[last][1] == c[1]
            ):
                merged[last][2] += c[2]
            else:
                merged.append(c)

        # The most saving first; ties to the smaller bytes, then the
        # shorter symbol.
        def by_gain(a: SIMD[DType.uint64, 4], b: SIMD[DType.uint64, 4]) -> Bool:
            return a[2] > b[2] or (
                a[2] == b[2] and (a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]))
            )

        sort(merged, by_gain)
        var table = Self(self.terminator)
        for c in merged:
            if table.count < 255:
                _ = table.add(c[0], Int(c[1]))
        return table^

    @staticmethod
    def build(lines: Span[Int, _], sample: Span[UInt8, _]) -> Self:
        """The table for the strings `lines` cuts `sample` into: five
        rounds, each counting the codes the table before it gives and making
        the next from the counts; the best table's counts make the last."""
        # The terminator: the least common byte, the lowest of those.
        var histogram = List[Int](length=256, fill=0)
        for b in sample:
            histogram[Int(b)] += 1
        var terminator = 0
        var least = 2 * Self.SAMPLE
        for i in reversed(range(256)):
            if histogram[i] <= least:
                terminator = i
                least = histogram[i]
        var table = Self(terminator)
        var best = table.copy()
        var gained = -2 * Self.SAMPLE
        var singles = List[UInt16](length=Self.CODES, fill=0)
        var pairs = List[UInt16](length=Self.CODES * Self.CODES, fill=0)
        var kept = singles.copy()
        for fraction in [8, 38, 68, 98, 128]:
            for i in range(Self.CODES):
                singles[i] = 0
            for i in range(Self.CODES * Self.CODES):
                pairs[i] = 0
            var gain = table.counted(lines, sample, fraction, singles, pairs)
            if gain >= gained:
                kept = singles.copy()
                best = table.copy()
                gained = gain
            if fraction < 128:
                table = table.made(singles, pairs, fraction)
        best = best.made(kept, pairs, 128)
        best.finalize()
        return best^

    def finalize(mut self):
        """Renumber the symbols 0-254 by length -- two bytes, those no longer
        symbol starts like first, then three to eight, then one -- and point
        every lookup at the new codes, single bytes in the two-byte lookup
        too."""
        var codes = List[Int](length=256, fill=0)
        # Where each length's codes start.
        var next = List[Int](length=8, fill=0)
        next[0] = self.count - self.histogram[0]
        for i in range(1, 7):
            next[i + 1] = next[i] + self.histogram[i]
        var alone = 0
        var shared = next[2]
        for i in range(self.count):
            var num = self.nums[Self.BASE + i]
            var length = self.length(Self.BASE + i)
            var code: Int
            if length == 2:
                var longer = False
                for k in range(self.count):
                    if (
                        k != i
                        and self.length(Self.BASE + k) > 1
                        and (self.nums[Self.BASE + k] & 0xFFFF)
                        == (num & 0xFFFF)
                    ):
                        longer = True
                if longer:
                    shared -= 1
                    code = shared
                else:
                    code = alone
                    alone += 1
            else:
                code = next[length - 1]
                next[length - 1] += 1
            codes[i] = code
            self.nums[code] = num
            self.icls[code] = Self.icl(code, length)
        var escape = UInt16(511 + (1 << 12))
        for i in range(256):
            var b = self.bytes[i]
            if (UInt64(b) & Self.MASK) >= UInt64(Self.BASE):
                self.bytes[i] = UInt16(codes[Int(b & 0xFF)] + (1 << 12))
            else:
                self.bytes[i] = escape
        for i in range(1 << 16):
            var s = self.shorts[i]
            if (UInt64(s) & Self.MASK) >= UInt64(Self.BASE):
                self.shorts[i] = UInt16(codes[Int(s & 0xFF)]) + (s & (15 << 12))
            else:
                self.shorts[i] = self.bytes[i & 0xFF]
        for i in range(Self.HASHED):
            if self.hash_icls[i] < Self.FREE:
                var code = codes[Int((self.hash_icls[i] >> 16) & 0xFF)]
                self.hash_nums[i] = self.nums[code]
                self.hash_icls[i] = self.icls[code]

    @always_inline
    def coded(
        self,
        src: Span[UInt8, _],
        var cur: Int,
        end: Int,
        dst: BufferView[mut=True, DType.uint8, _],
        mut at: Int,
    ) -> Int:
        """The bytes of `src` from `cur` to at least `end`, coded into `dst`
        from `at` on; where it stopped, at most seven bytes past `end`.
        Reads eight bytes from each position, and writes a byte past
        `at`."""
        while cur < end:
            var word = LittleEndian.fixed[DType.uint64](src, cur)
            var short = UInt64(self.shorts[Int(word & 0xFFFF)])
            var hashed_at = Int(
                Self.hash(word & 0xFFFFFF) & UInt64(Self.HASHED - 1)
            )
            var hashed = self.hash_icls[hashed_at]
            # The byte an escape writes, before knowing there is one.
            dst.unsafe_set(at + 1, UInt8(word & 0xFF))
            if hashed < Self.FREE and self.hash_nums[hashed_at] == (
                word & (UInt64.MAX >> (hashed & 0xFF))
            ):
                dst.unsafe_set(at, UInt8((hashed >> 16) & 0xFF))
                at += 1
                cur += Int(hashed >> 28)
            else:
                dst.unsafe_set(at, UInt8(short & 0xFF))
                at += 1 + Int((short & 256) >> 8)
                cur += Int(short >> 12)
        return cur


struct FsstPack[L: PackKernel](Kernel, Movable):
    """Byte strings as `FsstPacked`: a symbol table built from a sample of
    them, and each string coded by it, the code strings' lengths packed by
    `L`. Equal input packs to equal bytes."""

    comptime name = "fsst_pack"

    var codes: LengthsPack[Self.L]

    def __init__(out self, var lengths: Self.L):
        self.codes = LengthsPack(lengths^)

    def pack(
        self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> FsstPacked[Self.L.Packed]:
        """String `i` is `data[offsets[i] : offsets[i + 1]]`."""
        comptime T = _SymbolTable
        var n = max(len(offsets) - 1, 0)
        var bounds = BufferView(offsets)
        var total = Int(offsets[n]) - Int(offsets[0]) if n > 0 else 0
        # The sample: every string when they are small; else pieces of at
        # most a line, from strings chosen at random, by a fixed seed.
        var sample = List[UInt8]()
        var lines: List[Int] = [0]
        if total < T.SAMPLE:
            for i in range(n):
                sample.extend(
                    data[
                        Int(bounds.unsafe_get(i)) : Int(
                            bounds.unsafe_get(i + 1)
                        )
                    ]
                )
                lines.append(len(sample))
        else:
            var seed = T.hash(4637947)
            while len(sample) < T.SAMPLE:
                seed = T.hash(seed)
                var line = Int(seed % UInt64(n))
                # The next string that is not empty.
                while bounds.unsafe_get(line + 1) == bounds.unsafe_get(line):
                    line = (line + 1) % n
                var start = Int(bounds.unsafe_get(line))
                var size = Int(bounds.unsafe_get(line + 1)) - start
                seed = T.hash(seed)
                var piece = T.LINE * Int(
                    seed % UInt64(1 + (size - 1) // T.LINE)
                )
                sample.extend(
                    data[
                        start
                        + piece : start
                        + piece
                        + min(size - piece, T.LINE)
                    ]
                )
                lines.append(len(sample))
        var table = T.build(Span(lines), Span(sample))

        # Each code at most two bytes a byte, and one written past the end.
        var coded = List[UInt8]()
        coded.resize(unsafe_uninit_length=2 * total + 8)
        var out = BufferView(Span(coded))
        var ends = List[Int32](capacity=n + 1)
        ends.append(0)
        # A string's last bytes, then the terminator, so the eight-byte
        # reads stay within; no symbol longer than a byte holds it, so no
        # code reaches past the string.
        var tail = Array[UInt8, 17](fill=0)
        var at = 0
        for i in range(n):
            var start = Int(bounds.unsafe_get(i))
            var end = Int(bounds.unsafe_get(i + 1))
            var stop = end - 8 if end - start > 8 else start
            var cur = table.coded(data, start, stop, out, at)
            var left = end - cur
            for k in range(left):
                tail[k] = data[cur + k]
            tail[left] = UInt8(table.terminator)
            _ = table.coded(Span(tail), 0, left, out, at)
            ends.append(Int32(at))

        var symbols = Buffer.alloc_uninit[DType.uint64](table.count)
        var lengths = Buffer.alloc_uninit(table.count)
        for c in range(table.count):
            LittleEndian.store[DType.uint64](
                symbols.view().as_span(), 8 * c, table.nums[c]
            )
            lengths.unsafe_set(c, UInt8(table.length(c)))
        return FsstPacked(
            symbols^.to_immutable(),
            lengths^.to_immutable(),
            table.count,
            self.codes.pack(Span(ends), Span(coded)[:at]),
        )

    def nbytes(self, packed: FsstPacked[Self.L.Packed]) -> Int:
        """The bytes `packed` holds: the symbols', eight each, their
        lengths', one each, and the code strings'."""
        return 9 * packed.count + self.codes.nbytes(packed.codes)


struct FsstUnpack[L: UnpackKernel](Kernel, Movable):
    """`FsstPacked`'s byte strings: each code as its symbol, eight bytes
    copied at once, the code strings' lengths unpacked by `L`."""

    comptime name = "fsst_unpack"

    var codes: LengthsUnpack[Self.L]

    def __init__(out self, var lengths: Self.L):
        self.codes = LengthsUnpack(lengths^)

    def unpack(
        self,
        var packed: FsstPacked[Self.L.Packed],
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """The `count` strings `packed` holds: their bytes appended to
        `data`, and where each ends to `offsets`."""
        var symbols = packed.count
        if symbols < 0 or symbols > 255:
            raise Self.error[CorruptError](t"{symbols} symbols, past 255")
        if len(packed.symbols) < 8 * symbols or len(packed.lengths) < symbols:
            raise Self.error[CorruptError](
                t"{symbols} symbols in buffers of {len(packed.symbols)} and"
                t" {len(packed.lengths)} bytes"
            )
        # A code past the table has no bytes, and is caught by that.
        var table = Array[UInt64, 256](fill=0)
        var lengths = Array[UInt8, 256](fill=0)
        for c in range(symbols):
            var length = packed.lengths.unsafe_get(c)
            if length < 1 or length > 8:
                raise Self.error[CorruptError](t"symbol {c} of {length} bytes")
            lengths[c] = length
            table[c] = LittleEndian.fixed[DType.uint64](
                packed.symbols.view().as_span(), 8 * c
            )
        var ends: List[Int32] = [0]
        var coded = List[UInt8]()
        self.codes.unpack(packed.codes.copy(), count, ends, coded)
        var codes = BufferView(Span(coded))
        # Each code at most eight bytes, and eight written at once.
        var at = len(data)
        data.resize(unsafe_uninit_length=at + 8 * len(coded) + 8)
        var out = Span(data)
        offsets.reserve(len(offsets) + count)
        var unknown = False
        var dangling = False
        var p = 0
        for s in range(count):
            var last = Int(ends[s + 1])
            while p < last:
                var c = Int(codes.unsafe_get(p))
                if c != _SymbolTable.ESCAPE:
                    LittleEndian.store[DType.uint64](out, at, table[c])
                    at += Int(lengths[c])
                    unknown = unknown or lengths[c] == 0
                    p += 1
                elif p + 1 < last:
                    out[at] = codes.unsafe_get(p + 1)
                    at += 1
                    p += 2
                else:
                    dangling = True
                    p = last
            offsets.append(Int32(at))
        data.shrink(at)
        if unknown:
            raise Self.error[CorruptError](t"a code past {symbols} symbols")
        elif dangling:
            raise Self.error[CorruptError]("an escape ends a string")
