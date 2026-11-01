# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

# The exponent-pair search follows ALP (https://github.com/cwida/ALP),
# `include/alp/encoder.hpp` (`find_top_k_combinations`,
# `find_best_exponent_factor_from_combinations`): its size estimate, the
# order pairs are compared in, the five kept (`MAX_K_COMBINATIONS`) and the
# early exit after two no better (`SAMPLING_EARLY_EXIT_THRESHOLD`); and
# `AlpRdPack` takes ALP-RD's limit of 16 left bits (`CUTTING_LIMIT`,
# `include/alp/config.hpp`). Copyright (c) 2023 CWI, Azim Afroozeh, Leonardo
# Xavier Kuffo Rivero, MIT License, and modified; see NOTICE.txt.

"""Floats packed as the integers of their decimal digits.

`Alp` makes each float of a segment the integer `round(v * 10^e * 10^-f)`,
under the exponent pair `(e, f)` its segment is fitted with, and the integer
the float again, `n * 10^f * 10^-e`. Under `PatchPack` a value that does not
come back bit for bit -- NaN, the infinities, `-0.0`, a value with more
digits than the integer holds -- is kept apart as an exception: `AlpPack`
packs the integers by frame of reference, and `AlpUnpack` reads them back.

A segment's pair is chosen on 32 of its values, spread across it: on the
first segment of each run of 100, every pair, the five that pack those values
smallest kept; on each segment, the best of the five.

The floats no decimal integer gives back are packed by `AlpRdPack` instead:
each float's bits cut in two, the left bits -- the sign, the exponent and the
top of the mantissa, few distinct among floats of one magnitude -- packed by
one packer, usually a dictionary, and the right bits by another; the cut is
chosen per segment. `AlpRdUnpack` joins them again.
"""

from std.algorithm import vectorize
from std.bit import bit_width
from std.builtin.globals import global_constant
from std.builtin.rebind import rebind_var
from std.builtin.sort import sort
from std.math import ceildiv
from std.memory import bitcast
from std.sys import bit_width_of, size_of

from ..buffers import Buffer
from ..errors import CorruptError, InvalidError
from ..utils.integers import Unsigned
from ..views import BufferView
from .bitwise import OrderedDecode, OrderedEncode
from .compose import Compose, Decoder, Encoder
from .core import FitKernel, PackKernel, UnpackKernel
from .packing import (
    BLOCK,
    BitPack,
    BitUnpack,
    PatchPack,
    PatchUnpack,
    ReferencePack,
    ReferenceUnpack,
)
from .segments import BySize, PerSegment


def _most[T: DType]() -> Int:
    """The largest `e` or `f` for floats of `T`: every power of ten up to it
    is exact in `T`."""
    comptime assert (
        T == DType.float64 or T == DType.float32
    ), "ALP takes float32 and float64"
    return 18 if T == DType.float64 else 10


def _tens[T: DType]() -> Array[Scalar[T], 19]:
    """`10^k` for each `k` up to `_most[T]()`, exact."""
    var tens = Array[Scalar[T], 19](fill=1)
    var p = Scalar[T](1)
    for k in range(_most[T]() + 1):
        tens[k] = p
        p *= 10
    return tens^


def _tenths[T: DType]() -> Array[Scalar[T], 19]:
    """`10^-k` for each `k` up to `_most[T]()`, rounded once."""
    var tens = _tens[T]()
    var tenths = Array[Scalar[T], 19](fill=1)
    for k in range(_most[T]() + 1):
        tenths[k] = 1 / tens[k]
    return tenths^


# Read from memory: a table held as a value is stored to the stack whole at
# each lookup.
@always_inline
def _ten[T: DType](k: Int) -> Scalar[T]:
    comptime tens = _tens[T]()
    return global_constant[tens]().unsafe_get(k)


@always_inline
def _tenth[T: DType](k: Int) -> Scalar[T]:
    comptime tenths = _tenths[T]()
    return global_constant[tenths]().unsafe_get(k)


def _valid[T: DType](fit: UInt64) -> Bool:
    """Whether `fit` is a pair `e << 8 | f` with `f <= e <= _most[T]()`."""
    var e = fit >> 8
    return (fit & 0xFF) <= e and e <= UInt64(_most[T]())


struct _AlpFitting(Movable):
    """The pairs the last search over every pair kept, best first, and the
    segments left before the next."""

    var pairs: List[UInt64]
    var left: Int

    def __init__(out self):
        self.pairs = List[UInt64]()
        self.left = 0


struct Alp(FitKernel):
    """Floats as the integers of their decimal digits: `round(v * 10^e *
    10^-f)` for the exponent pair `(e, f)` its segment is fitted with, the
    fit `e << 8 | f`, `f <= e <= 18` (10 for `float32`); decoded `n * 10^f *
    10^-e`, by two multiplications, `10^-e` correctly rounded. NaN, the
    infinities and a value scaled past `2^51` (`2^22` for `float32`) are
    made 0, which does not give them back.

        [1.5, 2.25, 0.5] as float64, fit 2 << 8  ->  [150, 225, 50]
    """

    comptime name = "alp"
    comptime Encoded[T: DType]: DType = DType.int64 if T == DType.float64 else (
        DType.int32 if T == DType.float32 else T
    )
    comptime State[T: DType]: Movable & Deinitable = _AlpFitting
    comptime SAMPLE = 32
    """The values of a segment its pair is chosen on."""
    comptime KEPT = 5
    """The pairs a search over every pair keeps."""
    comptime EVERY = 100
    """The segments one search over every pair serves."""

    @staticmethod
    def start[T: DType]() -> _AlpFitting:
        return _AlpFitting()

    @staticmethod
    def _cost[
        T: DType
    ](
        sample: Array[Scalar[T], Self.SAMPLE],
        m: Int,
        fit: UInt64,
        penalize: Bool,
    ) -> Int:
        """The bits the first `m` values of `sample` pack in under `fit`:
        each at the width of the range of the integers that come back, and
        each that does not at its own bits and a 16-bit position. `Int.MAX`
        when `penalize` and fewer than two come back."""
        comptime O = Self.Encoded[T]
        comptime U = Unsigned[T]
        var low = Scalar[O].MAX
        var high = Scalar[O].MIN
        var back = 0
        for k in range(m):
            var v = SIMD[T, 1](sample[k])
            var n = Self.encode[T, 1](v, fit)
            var x = Self.decode[T, 1](n, fit)
            if bitcast[U](x) == bitcast[U](v):
                back += 1
                low = min(low, n[0])
                high = max(high, n[0])
        var cost = Int.MAX
        if not penalize or back >= 2:
            var width = 0
            if back > 0:
                width = Int(
                    bit_width(
                        high.cast[DType.int64]() - low.cast[DType.int64]()
                    )
                )
            cost = m * width + (m - back) * (bit_width_of[T]() + 16)
        return cost

    @staticmethod
    def fit[
        T: DType
    ](mut state: _AlpFitting, segment: BufferView[T, _]) -> UInt64:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "Alp takes float32 and float64"
        var n = len(segment)
        # Spread by a prime stride: equidistant values sample one phase of
        # a column periodic in the distance.
        var m = min(Self.SAMPLE, n)
        var sample = Array[Scalar[T], Self.SAMPLE](fill=0)
        for k in range(m):
            sample[k] = segment.unsafe_get((k * 7919) % n)
        # Every pair, from the largest `e` and `f` down, the five that pack
        # smallest kept, ties to the larger; again once those served their
        # segments, or found none.
        if state.left == 0 or len(state.pairs) == 0:
            var pairs = List[UInt64]()
            var costs = List[Int]()
            for e in range(_most[T](), -1, -1):
                for f in range(e, -1, -1):
                    var pair = UInt64(e << 8 | f)
                    var c = Self._cost[T](sample, m, pair, True)
                    if c < Int.MAX:
                        pairs.append(pair)
                        costs.append(c)
                        var at = len(costs) - 1
                        while at > 0 and costs[at] < costs[at - 1]:
                            costs[at] = costs[at - 1]
                            pairs[at] = pairs[at - 1]
                            costs[at - 1] = c
                            pairs[at - 1] = pair
                            at -= 1
                        if len(pairs) > Self.KEPT:
                            _ = pairs.pop()
                            _ = costs.pop()
            state.pairs = pairs^
            state.left = Self.EVERY
        state.left -= 1
        # The best of those kept, stopping after two in a row no better.
        var best = UInt64(0)
        var least = Int.MAX
        var worse = 0
        for k in range(len(state.pairs)):
            var c = Self._cost[T](sample, m, state.pairs[k], False)
            if c < least:
                least = c
                best = state.pairs[k]
                worse = 0
            else:
                worse += 1
                if worse == 2:
                    break
        return best

    @always_inline
    @staticmethod
    def encode[
        T: DType, W: Int
    ](v: SIMD[T, W], fit: UInt64) -> SIMD[Self.Encoded[T], W]:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "Alp takes float32 and float64"
        # The mantissa's bits: adding and taking away `magic`, 1.5 times the
        # weight of its top bit, rounds every value up to `limit` exactly.
        comptime M = 52 if T == DType.float64 else 23
        comptime limit = Scalar[T](1 << (M - 1))
        comptime magic = Scalar[T](3 << (M - 1))
        var x = v * _ten[T](Int(fit >> 8)) * _tenth[T](Int(fit & 0xFF))
        # Not a number, or past the limit, as 0.
        var inside = x.le(limit) & x.ge(-limit)
        var rounded = (inside.select(x, 0) + magic) - magic
        return rounded.cast[Self.Encoded[T]]()

    @always_inline
    @staticmethod
    def decode[
        T: DType, W: Int
    ](v: SIMD[Self.Encoded[T], W], fit: UInt64) -> SIMD[T, W]:
        return v.cast[T]() * _ten[T](Int(fit & 0xFF)) * _tenth[T](Int(fit >> 8))

    @staticmethod
    def valid[T: DType](fit: UInt64) -> Bool:
        return _valid[T](fit)


comptime AlpPack = PatchPack[
    Alp, Encoder[Compose[OrderedEncode], ReferencePack[BitPack]]
]
"""Floats as the integers of their decimal digits, by frame of reference,
the values that do not come back kept apart."""

comptime AlpUnpack[T: DType] = PatchUnpack[
    Alp,
    Decoder[
        ReferenceUnpack[BitUnpack],
        Compose[OrderedDecode[Alp.Encoded[T]]],
        Unsigned[Alp.Encoded[T]],
    ],
]
"""`AlpPack`'s values, of the float dtype `T`."""


# --- AlpRd ---------------------------------------------------------------------


@fieldwise_init
struct AlpRdPacked[L: Copyable & Deinitable, R: Copyable & Deinitable](
    Copyable, Movable
):
    """Each float's bits cut in two: the left bits, packed by one packer, and
    the right, by another; and each segment's cut.

        [1.0, 1.5] as float64, cut 48  ->  left [0x3ff0, 0x3ff8], right [0, 0]
    """

    var cuts: PerSegment
    """Each segment's cut, the right part's bits, a `uint8`."""
    var left: Self.L
    """The left parts, as `uint16`."""
    var right: Self.R
    """The right parts, as the floats' unsigned twin."""


@fieldwise_init
struct AlpRdPack[Left: PackKernel, Right: PackKernel](PackKernel):
    """Floats as `AlpRdPacked`, their bits cut per segment of `size` values,
    at most 16 bits to the left: where, on 32 values spread evenly over the
    segment, the right bits and the bits of a code into the distinct left
    parts add up least, ties to the wider right. Every bit pattern comes
    back."""

    comptime name = "alp_rd_pack"
    comptime Packed = AlpRdPacked[Self.Left.Packed, Self.Right.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        Buffer[mut=True],
        Int,
        Self.Left.State[DType.uint16],
        Self.Right.State[Unsigned[T]],
    ]
    """The cuts, the values stored, and the two packers' states."""
    comptime STEP = 64
    comptime SAMPLE = 32
    """The values of a segment its cut is chosen on."""
    comptime LEFT = 16
    """The most bits a left part holds."""

    var size: Int
    var left: Self.Left
    var right: Self.Right

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "AlpRdPack takes float32 and float64"
        var f = self.size
        if f <= 0 or f % Self.STEP != 0 or BLOCK % f != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        return (
            Buffer.alloc_uninit[DType.uint8](ceildiv(count, f)),
            0,
            self.left.start[DType.uint16](count),
            self.right.start[Unsigned[T]](count),
        )

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime B = bit_width_of[T]()
        var m = len(block)
        if m > BLOCK or state[1] % BLOCK != 0:
            raise Self.block_error(m)
        var lefts = Array[UInt16, BLOCK](uninitialized=True)
        var rights = Array[UInt8, BLOCK * size_of[Scalar[T]]()](
            uninitialized=True
        )
        var left = BufferView(Span(lefts))[:m]
        var right = BufferView[U].from_bytes(Span(rights))[:m]
        var cuts = state[0].view[DType.uint8]()
        var f = self.size
        # The segment the block starts, as blocks hold whole segments.
        var s = state[1] // f
        var lo = 0
        while lo < m:
            var hi = min(lo + f, m)
            var n = hi - lo
            # The sample's bits sorted, so at any cut the values sharing a
            # left part lie together.
            var k = min(Self.SAMPLE, n)
            var sample = Array[UInt64, Self.SAMPLE](fill=0)
            for j in range(k):
                sample[j] = bitcast[U](block.unsafe_get(lo + j * n // k)).cast[
                    DType.uint64
                ]()
            var sampled = Span(sample)[:k]
            sort(sampled)
            var cut = B - 1
            var least = Int.MAX
            for p in range(B - 1, B - Self.LEFT - 1, -1):
                var distinct = 1
                for j in range(1, k):
                    if sample[j] >> UInt64(p) != sample[j - 1] >> UInt64(p):
                        distinct += 1
                var cost = p + Int(bit_width(distinct - 1))
                if cost < least:
                    least = cost
                    cut = p
            cuts.unsafe_set(s, UInt8(cut))
            s += 1
            var c = Scalar[U](cut)
            var low = (Scalar[U](1) << c) - 1

            @always_inline
            def split[W: Int](i: Int) {imm}:
                var bits = bitcast[U, W](block.load[W](lo + i))
                left.store[W](lo + i, (bits >> c).cast[DType.uint16]())
                right.store[W](lo + i, bits & low)

            vectorize[64 // size_of[Scalar[T]]()](n, split)
            lo = hi
        state[1] += m
        self.left.store[DType.uint16](state[2], left)
        self.right.store[U](state[3], right)

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        comptime U = Unsigned[T]
        var cuts = Optional[Buffer[mut=True]](None)
        var stored = 0
        var left = Optional[Self.Left.State[DType.uint16]](None)
        var right = Optional[Self.Right.State[U]](None)

        def take[
            k: Int
        ](var part: Self.State[T].Ts[k]) {
            mut cuts, mut stored, mut left, mut right
        }:
            comptime if k == 0:
                cuts = rebind_var[Buffer[mut=True]](part^)
            elif k == 1:
                stored = rebind_var[Int](part^)
            elif k == 2:
                left = rebind_var[Self.Left.State[DType.uint16]](part^)
            else:
                right = rebind_var[Self.Right.State[U]](part^)

        state^.consume_elements(take)
        return Self.Packed(
            PerSegment(cuts.take().to_immutable(), ceildiv(stored, self.size)),
            self.left.finish[DType.uint16](left.take()),
            self.right.finish[U](right.take()),
        )

    def nbytes[T: DType](self, packed: Self.Packed) -> Int:
        # A `uint8` cut a segment.
        return (
            len(packed.cuts)
            + self.left.nbytes[DType.uint16](packed.left)
            + self.right.nbytes[Unsigned[T]](packed.right)
        )


@fieldwise_init
struct AlpRdUnpack[Left: UnpackKernel, Right: UnpackKernel](UnpackKernel):
    """`AlpRdPacked`'s floats: each left part `Left` loads shifted past its
    segment's cut, joined with the right part `Right` loads, in segments of
    `size` values."""

    comptime name = "alp_rd_unpack"
    comptime Packed = AlpRdPacked[Self.Left.Packed, Self.Right.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        PerSegment,
        BySize,
        Self.Left.State[DType.uint16],
        Self.Right.State[Unsigned[T]],
    ]
    comptime STEP = 64
    comptime LEFT = 16
    """The most bits a left part holds."""

    var size: Int
    var left: Self.Left
    var right: Self.Right

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        comptime assert (
            T == DType.float64 or T == DType.float32
        ), "AlpRdUnpack yields float32 and float64"
        comptime B = bit_width_of[T]()
        var f = self.size
        if f <= 0 or f % Self.STEP != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        var segments = BySize(count, f)
        var s = segments.count()
        if len(packed.cuts) != s or not packed.cuts.holds[DType.uint8]():
            raise Self.error[CorruptError](
                t"{len(packed.cuts)} cuts in a buffer of"
                t" {len(packed.cuts.values)} bytes for {s} segments"
            )
        for g in range(s):
            var cut = Int(packed.cuts.get[DType.uint8](g))
            if cut < B - Self.LEFT or cut > B - 1:
                raise Self.error[CorruptError](t"cut {cut} for {T}")
        return (
            packed.cuts.copy(),
            segments,
            self.left.start[DType.uint16](packed.left.copy(), count),
            self.right.start[Unsigned[T]](packed.right.copy(), count),
        )

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        comptime U = Unsigned[T]
        var cut = state[0].get[DType.uint8](state[1].of(i)).cast[U]()
        var left = self.left.load[DType.uint16, W](state[2], i).cast[U]()
        var right = self.right.load[U, W](state[3], i)
        return bitcast[T, W]((left << SIMD[U, W](cut)) | right)

    def check[T: DType](self, state: Self.State[T]) raises:
        self.left.check[DType.uint16](state[2])
        self.right.check[Unsigned[T]](state[3])
