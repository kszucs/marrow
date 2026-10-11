# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Lanes, the codecs of a cascade but the last, each turning a value into
one value: `Delta`, `Xor`, `Zigzag`, `Ordered`, `Cast` and `ByteSwap` with
nothing kept; `FrameOfReference` and `Gcd` with a parameter a frame;
`Dictionary` with the distinct values; and `Patch` and `Alp` with the
exceptions they set apart. Each is also usable alone, a register at a time --
`Zigzag().forward[DType.int64, 1](v)`."""

from std.bit import bit_width, byte_swap
from std.memory import bitcast
from std.sys import bit_width_of, size_of

from ..buffers import Buffer
from ..errors import CorruptError
from ..utils.hashing import Fmix64, RapidHash64, SwissHashTable
from ..utils.integers import Unsigned, ordered, unordered, unzigzag, zigzag
from ..views import BufferView
from .core import (
    BLOCK,
    Cascade,
    Codec,
    Encoded,
    Exceptions,
    FrameValues,
    Frames,
)
from .stores import Plain
from .strings import Lengths


struct Prior[xor: Bool](Codec):
    """Each value against the one before it -- `first` before the first,
    zero unless given. `Delta` takes it away from an integer, wrapping on
    overflow; `Xor` xors any value's bits with its bits, so a value that
    repeats becomes 0 and one that changes a little keeps only its low bits
    -- Gorilla's first step -- and makes unsigned integers.

        Delta  [10, 12, 15, 15]  ->  [10, 2, 3, 0]
        Xor    [5, 5, 7] as uint8  ->  [5, 0, 2]
    """

    comptime Out[T: DType]: DType = Unsigned[T] if Self.xor else T

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
    def forward[
        T: DType, W: Int
    ](mut self, v: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        comptime assert Self.xor or T.is_integral(), "Delta takes integers"
        comptime U = Unsigned[T]
        var bits = bitcast[U, W](v)
        var prev = bits.shift_right[1]()
        prev[0] = self.last.cast[U]()
        self.last = bits[W - 1].cast[DType.uint64]()
        comptime if Self.xor:
            return bitcast[Self.Out[T], W](bits ^ prev)
        else:
            return bitcast[Self.Out[T], W](bits - prev)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        # A running sum lane by lane: shifted adds measured slower, the
        # zero-filling shift lowering to a shuffle.
        comptime U = Unsigned[T]
        var bits = bitcast[U, W](v)
        var out = SIMD[U, W]()
        var p = self.last.cast[U]()
        comptime for j in range(W):
            comptime if Self.xor:
                p ^= bits[j]
            else:
                p += bits[j]
            out[j] = p
        self.last = p.cast[DType.uint64]()
        return bitcast[T, W](out)


comptime Delta = Prior[False]
"""Each value minus the one before it."""

comptime Xor = Prior[True]
"""Each value's bits xor the bits of the one before it."""


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
    comptime LOWS = 0
    """The slot of the smallest values."""

    var lows: FrameValues[Self.Lows]

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Lows())

    def __init__(out self, size: Int, var lows: Self.Lows):
        self.lows = FrameValues(size, 64, lows^)

    def start[T: DType](mut self, count: Int) raises:
        comptime assert (
            T.is_unsigned()
        ), "FrameOfReference takes unsigned integers"
        self.lows.start[T](count)

    def forward[
        T: DType
    ](mut self, src: BufferView[T, _], dst: BufferView[mut=True, T, _]) raises:
        comptime W = 64 // size_of[Scalar[T]]()
        var f = self.lows.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            var lows = SIMD[T, W](Scalar[T].MAX)
            var i = lo
            while i + W <= hi:
                lows = min(lows, src.load[W](i))
                i += W
            var low = lows.reduce_min()
            while i < hi:
                low = min(low, src[i])
                i += 1
            self.lows.put(low)
            i = lo
            while i + W <= hi:
                dst.store[W](i, src.load[W](i) - low)
                i += W
            while i < hi:
                dst.store[1](i, src.load[1](i) - low)
                i += 1
            lo = hi

    def finish[T: DType](mut self) raises -> Encoded:
        return Encoded(0, [self.lows.finish[T]()])

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert (
            T.is_unsigned()
        ), "FrameOfReference takes unsigned integers"
        self.lows.start[T](node[Self.LOWS], count)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        return v + self.lows.at[T](i)


struct Gcd[Divisors: Codec = Plain](Codec):
    """Each value divided by the greatest common divisor of its frame, so a
    column of multiples -- milliseconds in whole seconds, prices in whole
    cents -- leaves small integers. The divisors are the `Divisors` slot.
    It takes unsigned integers; after `FrameOfReference` it divides what the
    values differ by.

        [0, 1000, 3000] as uint16  ->  [0, 1, 3], divisors [1000]
    """

    comptime SLOTS = 1
    comptime DIVISORS = 0
    """The slot of the divisors."""

    var divisors: FrameValues[Self.Divisors]

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Divisors())

    def __init__(out self, size: Int, var divisors: Self.Divisors):
        self.divisors = FrameValues(size, 64, divisors^)

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "Gcd takes unsigned integers"
        self.divisors.start[T](count)

    def forward[
        T: DType
    ](mut self, src: BufferView[T, _], dst: BufferView[mut=True, T, _]) raises:
        comptime W = 64 // size_of[Scalar[T]]()
        var f = self.divisors.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            # Euclid's, value by value, stopping once it reaches 1.
            var g = Scalar[T](0)
            var i = lo
            while i < hi and g != 1:
                var a = g
                var b = src.unsafe_get(i)
                while b != 0:
                    var r = a % b
                    a = b
                    b = r
                g = a
                i += 1
            g = max(g, 1)
            self.divisors.put(g)
            i = lo
            if g == 1:
                dst.as_span()[lo:hi].copy_from(src.as_span()[lo:hi])
            else:
                while i + W <= hi:
                    dst.store[W](i, src.load[W](i) // g)
                    i += W
                while i < hi:
                    dst.store[1](i, src.load[1](i) // g)
                    i += 1
            lo = hi

    def finish[T: DType](mut self) raises -> Encoded:
        return Encoded(0, [self.divisors.finish[T]()])

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "Gcd takes unsigned integers"
        self.divisors.start[T](node[Self.DIVISORS], count)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        return v * self.divisors.at[T](i)


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
    comptime DISTINCT = 0
    """The slot of the distinct values or byte strings."""

    var values: Self.Values
    var strings: Self.Strings
    var table: SwissHashTable
    """Codes by their values' or byte strings' hashes."""
    var distinct: Buffer[mut=True]
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
    var ids: List[Int32]
    var firsts: List[Int32]
    """A block's hashes, codes and first positions of new values, reused
    from block to block."""

    def __init__(out self):
        self = Self(Self.Values(), Self.Strings())

    def __init__(out self, var values: Self.Values, var strings: Self.Strings):
        self.values = values^
        self.strings = strings^
        self.table = SwissHashTable()
        self.distinct = Buffer.alloc_uninit(0)
        self.decoded = Encoded()
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self.known = 0
        self.seen = 0
        self.hashes = List[UInt64]()
        self.ids = List[Int32]()
        self.firsts = List[Int32]()

    def check(self) raises:
        if self.seen > self.known:
            raise CorruptError(
                t"codecs: code {self.seen - 1} past {self.known} values"
            )

    # --- values ---

    def start[T: DType](mut self, count: Int) raises:
        self.table = SwissHashTable()
        self.distinct = Buffer.alloc_uninit[T](count)
        self.hashes.resize(BLOCK, 0)
        self.ids.resize(BLOCK, 0)
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
        comptime W = 64 // size_of[Scalar[T]]()
        var n = len(src)
        var hv = BufferView(Span(self.hashes))[:n]
        var i = 0
        while i + W <= n:
            hv.store[W](
                i,
                Fmix64.mix[W](
                    bitcast[U, W](src.load[W](i)).cast[DType.uint64]()
                ),
            )
            i += W
        while i < n:
            hv.store[1](
                i,
                Fmix64.mix[1](
                    bitcast[U, 1](src.load[1](i)).cast[DType.uint64]()
                ),
            )
            i += 1
        var iv = BufferView(Span(self.ids))[:n]
        var fv = BufferView(Span(self.firsts))[:n]
        var before = len(self.table)
        var k = self.table.insert_all(hv, iv, fv)
        var distinct = self.distinct.view[T]()
        for j in range(k):
            distinct.unsafe_set(before + j, src[Int(fv[j])])
        # A code is never negative: its bits are a `uint32`'s.
        i = 0
        while i + W <= n:
            dst.store[W](i, bitcast[DType.uint32, W](iv.load[W](i)))
            i += W
        while i < n:
            dst.store[1](i, bitcast[DType.uint32, 1](iv.load[1](i)))
            i += 1

    def finish[T: DType](mut self) raises -> Encoded:
        var distinct = self.distinct^.to_immutable()
        self.distinct = Buffer.alloc_uninit(0)
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
        var k = node[Self.DISTINCT].count
        if count > 0 and k == 0:
            raise CorruptError(t"codecs: {count} codes into no values")
        self.decoded = self.values.held[T](node[Self.DISTINCT])
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
        self.strings.decode(node[Self.DISTINCT], self.ends, self.bytes)
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


struct Patch[Positions: Codec = Plain, Values: Codec = Plain](Codec):
    """In each frame, the values wider than the width that packs the frame
    smallest -- counting each exception at its own width and its position --
    are set apart and replaced by 0, so `BitPack` after it packs the rest
    narrow. PFOR is `FrameOfReference, Patch, BitPack`. It takes unsigned
    integers.

        [1, 2, 900, 3] as uint16  ->  [1, 2, 0, 3], exceptions [900] at [2]
    """

    comptime SLOTS = 2
    comptime POSITIONS = 0
    """The slot of the exceptions' positions."""
    comptime VALUES = 1
    """The slot of the exceptions."""

    var frames: Frames
    var exceptions: Exceptions[Self.Positions, Self.Values]

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Positions(), Self.Values())

    def __init__(
        out self,
        size: Int,
        var positions: Self.Positions,
        var values: Self.Values,
    ):
        self.frames = Frames(size, 64)
        self.exceptions = Exceptions(positions^, values^)

    def start[T: DType](mut self, count: Int) raises:
        comptime assert T.is_unsigned(), "Patch takes unsigned integers"
        _ = self.frames.over[True](count)
        self.exceptions.start[T](count)

    def forward[
        T: DType
    ](mut self, src: BufferView[T, _], dst: BufferView[mut=True, T, _]) raises:
        comptime B = bit_width_of[T]()
        var f = self.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            # How many values need each width; then the width that packs the
            # frame smallest, an exception costing its value and position.
            var needs = Array[Int, B + 1](fill=0)
            for i in range(lo, hi):
                needs.unsafe_get(Int(bit_width(src.unsafe_get(i)))) += 1
            var best = B
            var cost = (hi - lo) * B
            var wider = 0
            var w = B
            while w > 0:
                wider += needs.unsafe_get(w)
                w -= 1
                var c = (hi - lo) * w + wider * (B + 64)
                if c < cost:
                    cost = c
                    best = w
            var top = (
                Scalar[T](1) << Scalar[T](best)
            ) - 1 if best < B else Scalar[T].MAX
            for i in range(lo, hi):
                var v = src[i]
                if v > top:
                    self.exceptions.keep[T](i, v)
                    v = 0
                dst.unsafe_set(i, v)
            lo = hi
        self.exceptions.passed(len(src))

    def finish[T: DType](mut self) raises -> Encoded:
        var slots = List[Encoded](capacity=2)
        self.exceptions.finish[T](slots)
        return Encoded(0, slots^)

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert T.is_unsigned(), "Patch takes unsigned integers"
        self.exceptions.start[T](node[Self.POSITIONS], node[Self.VALUES], count)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        return self.exceptions.put[T, W](i, v)


struct Alp[
    Scales: Codec = Plain,
    Positions: Codec = Plain,
    Values: Codec = Plain,
](Codec):
    """Floats as integers: each frame's values times `10^e`, divided by
    `10^f`, rounded -- for the `e` and `f` that turn most of the frame into
    integers that turn back into exactly its values (ALP, Afroozeh and
    Boncz). The rest are exceptions, their integers the frame's first good
    one, so the integers stay close for `FrameOfReference`. Each frame's
    `e` and `f` are the `Scales` slot, `e << 8 | f`.

        [1.5, 2.25, 0.5] as float64  ->  [150, 225, 50], scales [2 << 8]
    """

    comptime Out[T: DType]: DType = DType.int64 if T == DType.float64 else (
        DType.int32
    )
    comptime SLOTS = 3
    comptime SCALES = 0
    """The slot of the scales."""
    comptime POSITIONS = 1
    """The slot of the exceptions' positions."""
    comptime VALUES = 2
    """The slot of the exceptions."""
    comptime SAMPLE = 32
    """Values a frame's scale is chosen on."""
    comptime CANDIDATES = 5
    """Scales the frames after the first choose among."""

    var scales: FrameValues[Self.Scales]
    var exceptions: Exceptions[Self.Positions, Self.Values]
    var candidates: List[UInt16]
    """The scales the first frame found best."""
    var ups: List[Float64]
    """`10^f`, by frame decoded."""
    var downs: List[Float64]
    """`10^-e`, by frame decoded."""

    def __init__(out self):
        self = Self(0)

    def __init__(out self, size: Int):
        self = Self(size, Self.Scales(), Self.Positions(), Self.Values())

    def __init__(
        out self,
        size: Int,
        var scales: Self.Scales,
        var positions: Self.Positions,
        var values: Self.Values,
    ):
        self.scales = FrameValues(size, 64, scales^)
        self.exceptions = Exceptions(positions^, values^)
        self.candidates = List[UInt16]()
        self.ups = List[Float64]()
        self.downs = List[Float64]()

    # --- the arithmetic, in `T` ---

    @staticmethod
    def _most[T: DType]() -> Int:
        """The largest `e` or `f`: every power of ten up to it is exact."""
        return 18 if T == DType.float64 else 10

    @staticmethod
    def _limit[T: DType]() -> Scalar[T]:
        """The largest integer `_round` rounds to exactly."""
        return Scalar[T](2.0**51) if T == DType.float64 else Scalar[T](
            2.0**22
        )

    @staticmethod
    def _tens[T: DType]() -> SIMD[T, 32]:
        """`10^e` for each `e` up to `_most`, exact."""
        var tens = SIMD[T, 32](1)
        var p = Scalar[T](1)
        for e in range(Self._most[T]() + 1):
            tens[e] = p
            p *= 10
        return tens

    @staticmethod
    @always_inline
    def _ten[T: DType](e: Int) -> Scalar[T]:
        comptime tens = Self._tens[T]()
        return tens[e]

    @staticmethod
    @always_inline
    def _tenth[T: DType](e: Int) -> Scalar[T]:
        """`10^-e`, rounded once."""
        comptime tenths = 1 / Self._tens[T]()
        return tenths[e]

    @staticmethod
    @always_inline
    def _round[T: DType, W: Int](x: SIMD[T, W]) -> SIMD[T, W]:
        """`x` rounded to the nearest integer, ties to even, by adding and
        taking away 1.5 times the weight of the mantissa's top bit."""
        var magic = Scalar[T](6755399441055744.0) if T == DType.float64 else (
            Scalar[T](12582912.0)
        )
        return (x + magic) - magic

    @staticmethod
    @always_inline
    def _scaled[
        T: DType, W: Int
    ](v: SIMD[T, W], scale: UInt16) -> Tuple[SIMD[T, W], SIMD[DType.bool, W]]:
        """`v` as integers at `scale`, still as floats, and which turn back
        into exactly `v`."""
        comptime U = Unsigned[T]
        var e = Int(scale >> 8)
        var f = Int(scale & 0xFF)
        var x = v * Self._ten[T](e) * Self._tenth[T](f)
        var limit = Self._limit[T]()
        var fits = x.le(limit) & x.ge(-limit)
        var n = Self._round(fits.select(x, 0))
        var back = n * Self._ten[T](f) * Self._tenth[T](e)
        return (n, fits & bitcast[U, W](back).eq(bitcast[U, W](v)))

    def _best[
        T: DType
    ](
        self, src: BufferView[T, _], lo: Int, hi: Int, scales: List[UInt16]
    ) -> UInt16:
        """The scale among `scales` that leaves `src[lo:hi]` smallest, judged
        on a sample: the bits its integers' range takes, and each
        exception's value and position."""
        comptime B = size_of[Scalar[T]]() * 8
        # Scattered rather than evenly spaced, so a column whose values
        # repeat with a period cannot fool the sample.
        var sampled = min(Self.SAMPLE, hi - lo)
        var best = scales[0]
        var least = Int.MAX
        for scale in scales:
            var low = Scalar[T].MAX
            var high = -Scalar[T].MAX
            var missed = 0
            for k in range(sampled):
                var i = lo + (k * 7919) % (hi - lo)
                var n, ok = Self._scaled[T, 1](src.unsafe_get(i), scale)
                if ok[0]:
                    low = min(low, n[0])
                    high = max(high, n[0])
                else:
                    missed += 1
            var range = Int(high - low) if high >= low else 0
            var cost = sampled * Int(bit_width(UInt64(range))) + missed * (
                B + 64
            )
            if cost < least:
                least = cost
                best = scale
        return best

    # --- encoding ---

    def start[T: DType](mut self, count: Int) raises:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "Alp takes floats"
        self.scales.start[DType.uint16](count)
        self.candidates = List[UInt16]()
        self.exceptions.start[T](count)

    def forward[
        T: DType
    ](
        mut self,
        src: BufferView[T, _],
        dst: BufferView[mut=True, Self.Out[T], _],
    ) raises:
        comptime O = Self.Out[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var f = self.scales.frames.size
        var lo = 0
        while lo < len(src):
            var hi = min(lo + f, len(src))
            if len(self.candidates) == 0:
                # Every scale on the first frame; its best few after.
                var every = List[UInt16]()
                for e in range(Self._most[T]() + 1):
                    for g in range(e + 1):
                        every.append(UInt16(e << 8 | g))
                for _ in range(Self.CANDIDATES):
                    var best = self._best(src, lo, hi, every)
                    self.candidates.append(best)
                    for k in range(len(every)):
                        if every[k] == best:
                            _ = every.pop(k)
                            break
            var scale = self._best(src, lo, hi, self.candidates)
            self.scales.put(scale)
            var missed = self.exceptions.count()
            var fill = Scalar[O](0)
            var found = False
            var i = lo
            while i < hi:
                var w = min(W, hi - i)
                var n: SIMD[T, W]
                var ok: SIMD[DType.bool, W]
                if w == W:
                    n, ok = Self._scaled[T, W](src.load[W](i), scale)
                else:
                    var v = SIMD[T, W](0)
                    for j in range(w):
                        v[j] = src.unsafe_get(i + j)
                    n, ok = Self._scaled[T, W](v, scale)
                if w == W and ok.reduce_and():
                    # Every value turns back exactly: the register at once.
                    dst.store[W](i, n.cast[O]())
                    if not found:
                        fill = n[0].cast[O]()
                        found = True
                    i += W
                    continue
                for j in range(w):
                    if ok[j]:
                        if not found:
                            fill = n[j].cast[O]()
                            found = True
                        dst.unsafe_set(i + j, n[j].cast[O]())
                    else:
                        self.exceptions.keep[T](i + j, src.unsafe_get(i + j))
                i += w
            # An exception's integer is the frame's first good one.
            self.exceptions.fill(missed, dst, fill)
            lo = hi
        self.exceptions.passed(len(src))

    def finish[T: DType](mut self) raises -> Encoded:
        var slots = List[Encoded](capacity=3)
        slots.append(self.scales.finish[DType.uint16]())
        self.exceptions.finish[T](slots)
        return Encoded(0, slots^)

    # --- decoding ---

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "Alp takes floats"
        self.scales.start[DType.uint16](node[Self.SCALES], count)
        var frames = len(self.scales)
        self.ups = List[Float64](capacity=frames)
        self.downs = List[Float64](capacity=frames)
        for b in range(frames):
            var scale = self.scales.get[DType.uint16](b)
            var e = Int(scale >> 8)
            var f = Int(scale & 0xFF)
            if e > Self._most[T]() or f > e:
                raise CorruptError(t"codecs: an Alp scale of {scale}")
            self.ups.append(Float64(Self._ten[T](f)))
            self.downs.append(Float64(Self._tenth[T](e)))
        self.exceptions.start[T](node[Self.POSITIONS], node[Self.VALUES], count)

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        var b = self.scales.frames.of(i)
        var up = self.ups.unsafe_get(b).cast[T]()
        var down = self.downs.unsafe_get(b).cast[T]()
        var x = v.cast[T]() * up * down
        return self.exceptions.put[T, W](i, x)
