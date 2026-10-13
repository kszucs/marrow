# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Lanes, the codecs of a cascade but the last, each turning a value into
one value: `Delta`, `Zigzag`, `Ordered`, `Cast` and `ByteSwap` with nothing
kept; `FrameOfReference` with a parameter a frame; and `Dictionary` with the
distinct values. Each is also usable alone, a register at a time --
`Zigzag().forward[DType.int64, 1](v)`."""

from std.algorithm import vectorize
from std.bit import byte_swap
from std.memory import bitcast
from std.sys import size_of

from ..buffers import Buffer
from ..errors import CorruptError
from ..utils.hashing import Fmix64, RapidHash64, SwissHashTable
from ..utils.integers import Unsigned, ordered, unordered, unzigzag, zigzag
from ..views import BufferView, prefix_sum
from .core import (
    BLOCK,
    Cascade,
    Codec,
    Encoded,
    FrameValues,
)
from .stores import Plain
from .strings import Lengths


struct Delta(Codec):
    """Each integer minus the one before it -- `first` before the first,
    zero unless given -- wrapping on overflow.

        [10, 12, 15, 15]  ->  [10, 2, 3, 0]
    """

    var first: UInt64
    """The bits of the value before the first."""
    var last: UInt64
    """The bits of the value before the next one."""

    def __init__(out self):
        self.first = 0
        self.last = 0

    def __init__[T: DType](out self, first: Scalar[T]):
        self.first = bitcast[Unsigned[T]](first).cast[DType.uint64]()
        self.last = self.first

    def start[T: DType](mut self, count: Int) raises:
        self.last = self.first

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        self.last = self.first

    @always_inline
    def forward[T: DType, W: Int](mut self, v: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_integral(), "Delta takes integers"
        comptime U = Unsigned[T]
        var bits = bitcast[U, W](v)
        var prev = bits.shift_right[1]()
        prev[0] = self.last.cast[U]()
        self.last = bits[W - 1].cast[DType.uint64]()
        return bitcast[T, W](bits - prev)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        comptime U = Unsigned[T]
        var out = prefix_sum(bitcast[U, W](v), self.last.cast[U]())
        self.last = out[W - 1].cast[DType.uint64]()
        return bitcast[T, W](out)


@fieldwise_init
struct Zigzag(Codec):
    """Signed integers as unsigned ones, small magnitudes staying small:
    `zigzag`.

    [0, -1, 1, -2]  ->  [0, 1, 2, 3]
    """

    comptime Out[T: DType]: DType = Unsigned[T]

    @always_inline
    def forward[T: DType, W: Int](self, v: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        return zigzag(v)

    @always_inline
    def backward[
        T: DType, W: Int
    ](self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        return unzigzag[T](v)


@fieldwise_init
struct Ordered(Codec):
    """Numbers as the unsigned integers of their width that order as they
    do: `ordered`.

        [-1, 0, 1] as int8  ->  [0x7f, 0x80, 0x81]
    """

    comptime Out[T: DType]: DType = Unsigned[T]

    @always_inline
    def forward[T: DType, W: Int](self, v: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        return ordered(v)

    @always_inline
    def backward[
        T: DType, W: Int
    ](self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        return unordered[T](v)


@fieldwise_init
struct Cast[To: DType](Codec):
    """Each value as a `To` -- what a format keeps a type as when it has no
    narrower one: Parquet's INT32 for 8- and 16-bit integers. A value comes
    back as it went in when `To` holds it; a NaN comes back a NaN, its
    payload not kept.

        [1, -2] as int8, Cast[DType.int32]  ->  [1, -2] as int32
    """

    comptime Out[T: DType]: DType = Self.To

    @always_inline
    def forward[T: DType, W: Int](self, v: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        return v.cast[Self.To]()

    @always_inline
    def backward[
        T: DType, W: Int
    ](self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        return v.cast[T]()


@fieldwise_init
struct ByteSwap(Codec):
    """Each value's bytes in the other order: big-endian values -- Parquet's
    wide decimals -- from little-endian ones, and back.

        [0x0102] as uint16  ->  [0x0201]
    """

    @always_inline
    def forward[T: DType, W: Int](self, v: SIMD[T, W]) -> SIMD[T, W]:
        return byte_swap(v)

    @always_inline
    def backward[T: DType, W: Int](self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        return byte_swap(v)


struct FrameOfReference[Lows: Codec = Plain](Codec):
    """Frame of reference: each value less the smallest of its frame, so a
    column clustered far from zero leaves small integers for `BitPack`. The
    smallest values are the `Lows` slot. It takes unsigned integers:
    `Ordered` before it makes signed integers and floats such.

        [100, 101, 103] as uint8  ->  [0, 1, 3], lows [100]
    """

    comptime SLOTS = 1
    var _lows: FrameValues[Self.Lows]

    @staticmethod
    def lows[
        U: DType
    ](node: Encoded) -> BufferView[U, origin_of(node.children[0].buffer)]:
        """The smallest value of each frame `node` holds, as `U`s, read where
        they lie: the `Lows` slot keeps them as they are."""
        comptime assert (
            Self.Lows.STORES_VALUES
        ), "the Lows slot does not keep its values as they are"
        return node[0].values[U]()

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Lows())

    def __init__(out self, size: Int, var lows: Self.Lows):
        self._lows = FrameValues(size, 64, lows^)

    def start[T: DType](mut self, count: Int) raises:
        comptime assert (
            T.is_unsigned()
        ), "FrameOfReference takes unsigned integers"
        self._lows.start[T](count)

    def forward[
        T: DType
    ](mut self, src: BufferView[T, _], dst: BufferView[mut=True, T, _]) raises:
        var f = self._lows.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            var low = Scalar[T].MAX

            @always_inline
            def least[W: Int](i: Int) {mut low, imm}:
                low = min(low, src.load[W](lo + i).reduce_min())

            vectorize[64 // size_of[Scalar[T]]()](hi - lo, least)
            self._lows.put(low)

            @always_inline
            def lower[W: Int](i: Int) {imm}:
                dst.store[W](lo + i, src.load[W](lo + i) - low)

            vectorize[64 // size_of[Scalar[T]]()](hi - lo, lower)
            lo = hi

    def finish[T: DType](mut self) raises -> Encoded:
        return Encoded(0, [self._lows.finish[T]()])

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert (
            T.is_unsigned()
        ), "FrameOfReference takes unsigned integers"
        self._lows.start[T](node[0], count)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        return v + self._lows.at[T](i)


struct Dictionary[
    Values: Codec = Plain,
    Strings: Codec = Cascade[Lengths[], Plain],
](Codec):
    """Each value as its code, a `uint32`, into the distinct values in the
    order they first appear: the `Values` slot -- or, for byte strings, the
    `Strings` slot. Values compare by their bits, byte strings by their
    bytes.

        [5, 5, 9, 5]  ->  [0, 0, 1, 0], values [5, 9]
    """

    comptime Out[T: DType]: DType = DType.uint32
    comptime BytesOut: DType = DType.uint32
    comptime SLOTS = 1

    var values: Self.Values
    var strings: Self.Strings
    var table: SwissHashTable
    """Codes by their values' or byte strings' hashes."""
    var _distinct: Buffer[mut=True]
    """The distinct values found while encoding."""
    var decoded: Encoded
    """The distinct values decoded."""
    var ends: List[Int32]
    """Where each distinct byte string ends in `bytes`, after a 0."""
    var bytes: List[UInt8]
    """The distinct byte strings' bytes."""
    var known: Int
    """Distinct values or byte strings decoded."""
    var seen: Int
    """One past the largest code decoded."""
    var hashes: List[UInt64]
    var firsts: List[Int32]
    """A block's hashes and first positions of new values, reused from
    block to block."""

    @staticmethod
    def distinct(node: Encoded) -> ref[node.children[0]] Encoded:
        """The node of the distinct values or byte strings `node` holds."""
        return node[0]

    def __init__(out self):
        self = Self(Self.Values(), Self.Strings())

    def __init__(out self, var values: Self.Values, var strings: Self.Strings):
        self.values = values^
        self.strings = strings^
        self.table = SwissHashTable()
        self._distinct = Buffer.alloc_uninit(0)
        self.decoded = Encoded()
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self.known = 0
        self.seen = 0
        self.hashes = List[UInt64]()
        self.firsts = List[Int32]()

    def check(self) raises:
        if self.seen > self.known:
            raise CorruptError(
                t"codecs: code {self.seen - 1} past {self.known} values"
            )

    # --- values ---

    def start[T: DType](mut self, count: Int) raises:
        self.table = SwissHashTable()
        self._distinct = Buffer.alloc_uninit[T](count)
        self.hashes.resize(BLOCK, 0)
        self.firsts.resize(BLOCK, 0)

    def forward[
        T: DType
    ](
        mut self,
        src: BufferView[T, _],
        dst: BufferView[mut=True, DType.uint32, _],
    ) raises:
        """A value's hash is `Fmix64` of its bits, a bijection, so equal
        hashes are equal values and none is compared."""
        comptime U = Unsigned[T]
        var n = len(src)
        var hv = BufferView(Span(self.hashes))[:n]

        @always_inline
        def hash[W: Int](i: Int) {imm}:
            hv.store[W](
                i,
                Fmix64.mix[W](
                    bitcast[U, W](src.load[W](i)).cast[DType.uint64]()
                ),
            )

        vectorize[64 // size_of[Scalar[T]]()](n, hash)
        var fv = BufferView(Span(self.firsts))[:n]
        var before = len(self.table)
        # A code is never negative, so its bits as an `int32` are a
        # `uint32`'s: the table writes the codes straight into `dst`.
        var k = self.table.insert_all(
            hv, BufferView[DType.int32].from_bytes(dst.as_bytes().as_span()), fv
        )
        var distinct = self._distinct.view[T]()
        for j in range(k):
            distinct.unsafe_set(before + j, src[Int(fv[j])])

    def finish[T: DType](mut self) raises -> Encoded:
        var distinct = self._distinct^.to_immutable()
        self._distinct = Buffer.alloc_uninit(0)
        var k = len(self.table)
        return Encoded(
            0,
            [
                self.values.encode_view(
                    BufferView(distinct.view[T](0, k).as_span())
                )
            ],
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        var k = node[0].count
        if count > 0 and k == 0:
            raise CorruptError(t"codecs: {count} codes into no values")
        self.decoded = self.values.held[T](node[0])
        self.known = k
        self.seen = 0

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[DType.uint32, W]) -> SIMD[T, W]:
        # A code past the values is clamped to the last and noted, so the
        # gather takes no branch and `check` raises after.
        self.seen = max(self.seen, Int(v.reduce_max()) + 1)
        var last = UInt32(self.known - 1)
        return self.decoded.values[T]().gather[W](
            min(v, last).cast[DType.int64]()
        )

    # --- byte strings ---

    def forward(
        mut self,
        offsets: Span[Int32, _],
        data: Span[UInt8, _],
        dst: BufferView[mut=True, DType.uint32, _],
    ) raises:
        for i in range(len(offsets) - 1):
            var v = data[Int(offsets[i]) : Int(offsets[i + 1])]
            var h = RapidHash64.hash(v)

            def same(code: Int) raises {imm} -> Bool:
                var lo = Int(self.ends[code])
                return Span(self.bytes)[lo : Int(self.ends[code + 1])] == v

            var code = self.table.find_one(h, same)
            if code < 0:
                code = self.table.insert_new(h)
                self.bytes.extend(v)
                self.ends.append(Int32(len(self.bytes)))
            dst.store[1](i, UInt32(code))

    def finish(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        var node = Encoded(
            0, [self.strings.encode(Span(self.ends), Span(self.bytes))]
        )
        self.table = SwissHashTable()
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        return node^

    def start(mut self, ref node: Encoded, mut data: List[UInt8]) raises:
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self.strings.decode(node[0], self.ends, self.bytes)
        self.known = len(self.ends) - 1
        self.seen = 0

    def decode(
        mut self,
        src: BufferView[DType.uint32, _],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        for i in range(len(src)):
            var c = Int(src[i])
            self.seen = max(self.seen, c + 1)
            if c >= self.known:
                continue
            data.extend(
                Span(self.bytes)[Int(self.ends[c]) : Int(self.ends[c + 1])]
            )
            offsets.append(Int32(len(data)))
