# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Kernels one after another: stream stages composed, and a codec built from
stages and a packer.

`Compose` runs stream stages in order on one register, as one stage: a
`UnaryKernel` through its `core`, a `ScanKernel` with its own slot of the
state. A `BinaryKernel` enters with its second operand named: `With[K, D]`
fixes it to a constant, `Scan` and `Lagged` (`scan.mojo`) take it from the
rows before.

`Encoder` runs stages before a packer and is a packer itself; `Decoder`
runs stages after an unpacker and is an unpacker itself, so either nests
inside another packer. `Codec` pairs an encoder with its decoder and drives a
whole column through them: a block of `BLOCK` values at a time through the
encoder, so the block stays in the L1 cache from stage to stage, and a
register at a time out of the decoder, written once.

    Codec(
        Encoder(Compose((DiffScan(),)), PlainPack()),
        Decoder[PlainUnpack, Compose[SumScan], DType.int64](
            PlainUnpack(), Compose((SumScan(),))
        ),
    )

`ChoicePack` packs each segment by whichever of its candidate packers packs
a sample of the segment smallest, and `ChoiceUnpack` reads each segment back
by the candidate that packed it. `Auto` lists candidates for integers and for
floats, a level nesting the one below for its inner streams.
"""

from std.builtin.rebind import downcast, rebind_var
from std.builtin.variadics import TypeList
from std.math import ceildiv
from std.sys import align_of, size_of
from std.utils import Variant

from ..buffers import Buffer
from ..errors import CorruptError, InvalidError
from ..utils import Fmix64
from ..views import BufferView
from .bitwise import OrderedDecode, OrderedEncode
from .core import (
    BinaryKernel,
    Kernel,
    PackKernel,
    ScanKernel,
    UnaryKernel,
    UnpackKernel,
)
from .dictionary import DictionaryPack, DictionaryUnpack
from .float import Alp, AlpRdPack, AlpRdUnpack
from .packing import (
    BLOCK,
    BitPack,
    BitUnpack,
    ConstantPack,
    ConstantUnpack,
    FrequencyPack,
    FrequencyUnpack,
    PatchPack,
    PatchUnpack,
    PforPack,
    PforUnpack,
    PlainPack,
    PlainUnpack,
    ReferencePack,
    ReferenceUnpack,
)
from .runs import RunsPack, RunsUnpack
from .scan import DiffScan, SumScan
from .segments import BySize


@fieldwise_init
struct NoState(Copyable, Defaultable, Deinitable, Movable):
    """The state of a stage that keeps none."""

    pass


@fieldwise_init
struct With[K: BinaryKernel, D: DType](Copyable, ScanKernel):
    """`K(v, constant)`: a binary kernel whose right operand is a constant
    of dtype `D`, the dtype of the values it takes.

        With[SubKernel, DType.int64](5) over [10, 12, 15]  ->  [5, 7, 10]
    """

    comptime name = "with"
    comptime Out[T: DType]: DType = Self.K.Out[T]
    comptime State[T: DType]: Movable & Deinitable = NoState

    var constant: Scalar[Self.D]

    def start[T: DType](self) -> NoState:
        return NoState()

    @always_inline
    def core[
        T: DType, W: Int
    ](self, v: SIMD[T, W], mut state: NoState) -> SIMD[Self.Out[T], W]:
        comptime assert T == Self.D, "With's constant has its values' dtype"
        return Self.K.core[T, W](
            v, SIMD[T, W](rebind[Scalar[T]](self.constant))
        )


@fieldwise_init
struct Compose[*Ks: Kernel & Movable & Deinitable](ScanKernel):
    """Stream stages in order, as one stage: each register goes through every
    stage before the next register is read.

    Each stage is a `UnaryKernel`, or a `ScanKernel` with its own slot of the
    state. Built from the stages, `Compose((With[SubKernel, DType.int64](5),
    DiffScan()))`: a stage with parameters is a value, so the stages are a
    tuple of values rather than type parameters alone.
    """

    comptime name = "compose"
    comptime N = len(Self.Ks)

    var stages: Tuple[*Self.Ks]

    @staticmethod
    def _in[k: Int, T: DType]() -> DType:
        """What stage `k` takes when the first takes `T`."""
        comptime if k == 0:
            return T
        else:
            comptime K = Self.Ks[k - 1]
            comptime I = Self._in[k - 1, T]()
            comptime if conforms_to(K, UnaryKernel):
                return downcast[K, UnaryKernel].Out[I]
            else:
                return downcast[K, ScanKernel].Out[I]

    comptime Out[T: DType]: DType = Self._in[Self.N, T]()

    comptime _Slot[T: DType, k: Int]: Movable & Deinitable & Defaultable = (
        Optional[downcast[Self.Ks[k], ScanKernel].State[Self._in[k, T]()]]
    ) if conforms_to(Self.Ks[k], ScanKernel) else NoState
    """Stage `k`'s slot of the state: optional, so the tuple of slots is
    built empty and each scan stage's filled in."""

    comptime State[T: DType]: Movable & Deinitable = Tuple[
        *TypeList.tabulate[
            Trait=Movable & Deinitable & Defaultable,
            Self.N,
            Self._Slot[T, ...],
        ]()
    ]

    def start[T: DType](self) -> Self.State[T]:
        var state = Self.State[T]()
        comptime for k in range(Self.N):
            comptime if conforms_to(Self.Ks[k], ScanKernel):
                comptime S = downcast[Self.Ks[k], ScanKernel]
                comptime I = Self._in[k, T]()
                ref slot = rebind[Optional[S.State[I]]](state[k])
                slot = Optional(rebind[S](self.stages[k]).start[I]())
        return state^

    @always_inline
    def core[
        T: DType, W: Int
    ](self, v: SIMD[T, W], mut state: Self.State[T]) -> SIMD[Self.Out[T], W]:
        return self._from[0, T, T, W](v, state)

    @always_inline
    def _from[
        k: Int, T: DType, I: DType, W: Int
    ](self, v: SIMD[I, W], mut state: Self.State[T]) -> SIMD[Self.Out[T], W]:
        """`v`, which stage `k` takes as `I`, through stages `k` on."""
        comptime if k == Self.N:
            return rebind[SIMD[Self.Out[T], W]](v)
        else:
            comptime K = Self.Ks[k]
            comptime if conforms_to(K, UnaryKernel):
                comptime U = downcast[K, UnaryKernel]
                return self._from[k + 1, T, U.Out[I], W](U.core[I, W](v), state)
            else:
                comptime S = downcast[K, ScanKernel]
                ref slot = rebind[Optional[S.State[I]]](state[k]).value()
                var out = rebind[S](self.stages[k]).core[I, W](v, slot)
                return self._from[k + 1, T, S.Out[I], W](out, state)


@fieldwise_init
struct Encoder[S: ScanKernel, P: PackKernel](PackKernel):
    """Stream stages, then a packer: each block goes through `S` a register
    at a time, into one block of scratch, which `P` stores."""

    comptime name = "encoder"
    comptime Packed = Self.P.Packed
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        Self.S.State[T], Self.P.State[Self.S.Out[T]]
    ]

    var stages: Self.S
    var pack: Self.P

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        return (
            self.stages.start[T](),
            self.pack.start[Self.S.Out[T]](count),
        )

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        comptime O = Self.S.Out[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var m = len(block)
        if m > BLOCK:
            raise Self.block_error(m)
        var scratch = Array[UInt8, BLOCK * size_of[Scalar[O]]()](
            uninitialized=True
        )
        var staged = BufferView[O].from_bytes(Span(scratch))[:m]
        var i = 0
        while i + W <= m:
            staged.store[W](
                i, self.stages.core[T, W](block.load[W](i), state[0])
            )
            i += W
        while i < m:
            staged.store[1](
                i, self.stages.core[T, 1](block.load[1](i), state[0])
            )
            i += 1
        self.pack.store[O](state[1], staged)

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        comptime O = Self.S.Out[T]
        var packing = Optional[Self.P.State[O]](None)

        # The stages' state is spent; the packer's finishes the column.
        def take[k: Int](var part: Self.State[T].Ts[k]) {mut packing}:
            comptime if k == 0:
                _ = rebind_var[Self.S.State[T]](part^)
            else:
                packing = rebind_var[Self.P.State[O]](part^)

        state^.consume_elements(take)
        return self.pack.finish[O](packing.take())

    def nbytes[T: DType](self, packed: Self.Packed) -> Int:
        return self.pack.nbytes[Self.S.Out[T]](packed)


@fieldwise_init
struct Decoder[U: UnpackKernel, S: ScanKernel, In: DType](UnpackKernel):
    """An unpacker, then stream stages: each register `U` loads as `In`, the
    dtype the paired encoder packed, goes through `S`, which must answer the
    column's dtype."""

    comptime name = "decoder"
    comptime Packed = Self.U.Packed
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        Self.U.State[Self.In], Self.S.State[Self.In]
    ]

    var unpack: Self.U
    var stages: Self.S

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        return (
            self.unpack.start[Self.In](packed^, count),
            self.stages.start[Self.In](),
        )

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        comptime assert (
            Self.S.Out[Self.In] == T
        ), "the decoder's stages must answer the column's dtype"
        var v = self.unpack.load[Self.In, W](state[0], i)
        return rebind[SIMD[T, W]](self.stages.core[Self.In, W](v, state[1]))

    def check[T: DType](self, state: Self.State[T]) raises:
        self.unpack.check[Self.In](state[0])


@fieldwise_init
struct Codec[E: PackKernel, D: UnpackKernel](Movable):
    """An encoder and the decoder of what it packs, driving whole columns.

    var codec = Codec(BitPack(), BitUnpack())
    var packed = codec.encode(BufferView(Span(values)))
    codec.decode(packed^, BufferView(Span(out)))
    """

    var encoder: Self.E
    var decoder: Self.D

    def encode[
        T: DType
    ](self, values: BufferView[T, _]) raises -> Self.E.Packed:
        """`values` packed, a block of `BLOCK` at a time."""
        comptime assert (
            Self.E.Packed == Self.D.Packed
        ), "the decoder must take what the encoder packs"
        var n = len(values)
        var state = self.encoder.start[T](n)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            self.encoder.store[T](state, values[lo : lo + m])
            lo += m
        return self.encoder.finish[T](state^)

    def decode[
        T: DType
    ](self, var packed: Self.E.Packed, dst: BufferView[mut=True, T, _]) raises:
        """The `len(dst)` values `packed` holds, into `dst`: one loop, a
        cache line of values at a time."""
        comptime assert (
            Self.E.Packed == Self.D.Packed
        ), "the decoder must take what the encoder packs"
        var state = self.decoder.start[T](
            rebind_var[Self.D.Packed](packed^), len(dst)
        )
        self._load[T](state, dst)
        self.decoder.check[T](state)

    def _load[
        T: DType
    ](self, mut state: Self.D.State[T], dst: BufferView[mut=True, T, _]):
        """Every value into `dst`, a cache line at a time."""
        # Its own function, so the loop is optimised with the state an
        # argument it has to itself: what `load` updates stays in registers,
        # where a local the caller goes on to use is stored back on every
        # register. Neither `@always_inline`, which inlines it before that,
        # nor `@no_inline`: once it is optimised, the inliner may take it.
        comptime W = 64 // size_of[Scalar[T]]()
        var count = len(dst)
        var i = 0
        while i + W <= count:
            dst.store[W](i, self.decoder.load[T, W](state, i))
            i += W
        while i < count:
            dst.store[1](i, self.decoder.load[T, 1](state, i))
            i += 1


# --- Choice --------------------------------------------------------------------


@fieldwise_init
struct _Candidate[k: Int, P: Copyable & Deinitable](Copyable, Movable):
    """A segment candidate `k` packed: in a `ChoicePacked`, the slot is the
    index of the candidate."""

    var packed: Self.P


@fieldwise_init
struct ChoicePacked[*Ts: Copyable & Deinitable](Copyable, Movable):
    """Each segment packed by the candidate its sample chose: that
    candidate's own `Packed`, in the slot of its index, and the bytes the
    sample predicted for the segment."""

    comptime N = len(Self.Ts)
    comptime Slot[k: Int]: Copyable & Deinitable = _Candidate[k, Self.Ts[k]]
    """A segment candidate `k` packed."""
    comptime Chosen = Variant[
        *TypeList.tabulate[
            Trait=Copyable & Deinitable, Self.N, Self.Slot[...]
        ]()
    ]
    """A segment, in the slot of the candidate that packed it."""

    var size: Int
    """Values a segment; the last may hold fewer."""
    var chosen: List[Self.Chosen]
    var predicted: List[Int]
    """The bytes each segment's sample predicted for it."""

    def chose(self, s: Int) -> Int:
        """The candidate segment `s` was packed by; -1 for a segment in no
        candidate's slot, which a `ChoicePack` never packs."""
        var k = -1
        comptime for j in range(Self.N):
            if self.chosen[s].isa[Self.Slot[j]]():
                k = j
        return k


struct _Choosing[P: Copyable & Deinitable](Movable):
    """The segment being filled, `at` values of it, and the segments packed
    so far."""

    var segment: Buffer[mut=True]
    var at: Int
    var packed: Self.P

    def __init__(out self, var segment: Buffer[mut=True], var packed: Self.P):
        self.segment = segment^
        self.at = 0
        self.packed = packed^

    def taken(deinit self) -> Self.P:
        """The segments packed."""
        return self.packed^


@fieldwise_init
struct ChoicePack[*Ps: PackKernel](PackKernel):
    """Values as `ChoicePacked`: each segment of `size` values packed by the
    candidate that packs a sample of it smallest -- ten runs of 64 values,
    spread over the segment. Of the candidates within 2% of the smallest,
    the one listed first is chosen, so a list in the order its candidates
    decode fastest picks the faster of two that pack alike. A segment its
    candidate packs more than 5% past what the sample predicted is packed by
    the next best by the same rule too, and the smaller of the two kept.

    `size` is a multiple of `BLOCK`; a segment of at most 640 values is its
    own sample. The runs are placed by the segment's index alone, so equal
    values pack to equal bytes.
    """

    comptime name = "choice_pack"
    comptime N = len(Self.Ps)
    comptime _PackedAt[k: Int]: Copyable & Deinitable = Self.Ps[k].Packed
    comptime Packed = ChoicePacked[
        *TypeList.tabulate[
            Trait=Copyable & Deinitable, Self.N, Self._PackedAt[...]
        ]()
    ]
    comptime State[T: DType]: Movable & Deinitable = _Choosing[Self.Packed]
    comptime SEGMENT = 65_536
    """The default values a segment."""
    comptime RUNS = 10
    comptime RUN = 64

    var size: Int
    var candidates: Tuple[*Self.Ps]

    def start[T: DType](self, count: Int) raises -> Self.State[T]:
        comptime assert Self.N <= 256, "a byte names a segment's candidate"
        # A list of variants grows safely when every slot is as aligned as
        # the most aligned one.
        comptime for k in range(Self.N):
            comptime assert (
                align_of[Self.Packed.Slot[k]]() == 8
            ), "ChoicePack keeps candidates whose Packed is 8-aligned"
        var f = self.size
        if f <= 0 or f % BLOCK != 0:
            raise Self.error[InvalidError](t"segments of {f} values")
        var segments = ceildiv(count, f)
        return _Choosing(
            Buffer.alloc_uninit[T](min(f, count)),
            Self.Packed(
                f,
                List[Self.Packed.Chosen](capacity=segments),
                List[Int](capacity=segments),
            ),
        )

    def store[
        T: DType
    ](self, mut state: Self.State[T], block: BufferView[T, _]) raises:
        var m = len(block)
        if m > BLOCK or state.at % BLOCK != 0:
            raise Self.block_error(m)
        state.segment.extend(block, state.at, m)
        state.at += m
        if state.at == self.size:
            self._close[T](state)

    def finish[T: DType](self, var state: Self.State[T]) raises -> Self.Packed:
        if state.at > 0:
            self._close[T](state)
        return state^.taken()

    def _close[T: DType](self, mut state: Self.State[T]) raises:
        """The segment filled, packed by the candidate its sample chose."""
        comptime S = Self.RUNS * Self.RUN
        var n = state.at
        var s = len(state.packed.chosen)
        var segment = state.segment.view[T](0, n)
        var scratch = Array[UInt8, S * size_of[Scalar[T]]()](uninitialized=True)
        var sample = BufferView[T].from_bytes(Span(scratch))[: min(n, S)]
        if n <= S:
            sample.copy_from(segment, n)
        else:
            # A run from each tenth, at an offset hashed from the segment
            # and the run.
            var part = n // Self.RUNS
            for r in range(Self.RUNS):
                var h = Fmix64.mix[1](UInt64(s << 8 | r))
                var at = r * part + Int(h % UInt64(part - Self.RUN + 1))
                sample[r * Self.RUN : (r + 1) * Self.RUN].copy_from(
                    segment[at : at + Self.RUN], Self.RUN
                )
        var sizes = Array[Int, Self.N](fill=0)
        comptime for k in range(Self.N):
            var trial = Self._packed(self.candidates[k], sample)
            sizes[k] = self.candidates[k].nbytes[T](trial)
        var best = Self._pick(sizes, -1)
        var predicted = ceildiv(sizes[best] * n, len(sample))
        var chosen = self._pack[T](best, segment)
        var nbytes = self._part_nbytes[T](chosen)
        # More than 5% past its prediction, the next best packs the segment
        # too, and the smaller of the two is kept.
        if Self.N > 1 and nbytes * 100 > predicted * 105:
            var next = Self._pick(sizes, best)
            var other = self._pack[T](next, segment)
            if self._part_nbytes[T](other) < nbytes:
                chosen = other^
                predicted = ceildiv(sizes[next] * n, len(sample))
        state.packed.chosen.append(chosen^)
        state.packed.predicted.append(predicted)
        state.at = 0

    @staticmethod
    def _pick(sizes: Array[Int, Self.N], skip: Int) -> Int:
        """The first candidate within 2% of the smallest, `skip` aside."""
        var least = Int.MAX
        for k in range(Self.N):
            if k != skip:
                least = min(least, sizes[k])
        var best = 0
        while best == skip or sizes[best] * 100 > least * 102:
            best += 1
        return best

    def _pack[
        T: DType
    ](self, k: Int, segment: BufferView[T, _]) raises -> Self.Packed.Chosen:
        """`segment` packed by candidate `k`, in its slot."""
        comptime for j in range(Self.N):
            if j == k:
                return _Candidate[j, Self.Ps[j].Packed](
                    Self._packed(self.candidates[j], segment)
                )
        raise Self.error[InvalidError](t"candidate {k} of {Self.N}")

    def nbytes[T: DType](self, packed: Self.Packed) -> Int:
        # A byte a segment names its candidate.
        var total = len(packed.chosen)
        for s in range(len(packed.chosen)):
            total += self.segment_nbytes[T](packed, s)
        return total

    def segment_nbytes[T: DType](self, packed: Self.Packed, s: Int) -> Int:
        """The bytes segment `s` is packed in, less the byte naming its
        candidate."""
        return self._part_nbytes[T](packed.chosen[s])

    def _part_nbytes[T: DType](self, chosen: Self.Packed.Chosen) -> Int:
        var total = 0
        comptime for k in range(Self.N):
            comptime S = Self.Packed.Slot[k]
            if chosen.isa[S]():
                total = self.candidates[k].nbytes[T](
                    rebind[Self.Ps[k].Packed](chosen[S].packed)
                )
        return total

    @staticmethod
    def _packed[
        P: PackKernel, T: DType
    ](packer: P, values: BufferView[T, _]) raises -> P.Packed:
        """`values` packed by `packer` as a column of their own, a block of
        `BLOCK` at a time."""
        var n = len(values)
        var state = packer.start[T](n)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            packer.store[T](state, values[lo : lo + m])
            lo += m
        return packer.finish[T](state^)


struct _Choice[P: Copyable & Deinitable](Movable):
    """What `ChoiceUnpack` reads: the segments, the one being read -- the
    candidate reading it, where it starts and where it ends -- and the first
    failure a segment's candidate raised."""

    var packed: Self.P
    var segments: BySize
    var segment: Int
    var lo: Int
    var hi: Int
    var active: Int
    """The candidate reading the segment, or -1 when it refused it."""
    var failure: String

    def __init__(out self, var packed: Self.P, segments: BySize):
        self.packed = packed^
        self.segments = segments
        self.segment = -1
        self.lo = 0
        self.hi = 0
        self.active = -1
        self.failure = String()

    def note(mut self, e: Error):
        """`e`, raised by the segment's candidate, unless one came first."""
        if not self.failure:
            self.failure = String(t"segment {self.segment}: {e}")


@fieldwise_init
struct ChoiceUnpack[*Us: UnpackKernel](UnpackKernel):
    """`ChoicePacked`'s values: each segment by the candidate that packed it.

    The candidate's state is started as its segment is entered, and the
    state before checked; what a candidate refuses is noted and raised by
    `check`, naming the segment.
    """

    comptime name = "choice_unpack"
    comptime N = len(Self.Us)
    comptime _PackedAt[k: Int]: Copyable & Deinitable = Self.Us[k].Packed
    comptime Packed = ChoicePacked[
        *TypeList.tabulate[
            Trait=Copyable & Deinitable, Self.N, Self._PackedAt[...]
        ]()
    ]
    comptime _Slot[T: DType, k: Int]: Movable & Deinitable & Defaultable = (
        Optional[Self.Us[k].State[T]]
    )
    """Candidate `k`'s state, filled while it reads a segment."""
    comptime _Slots[T: DType] = Tuple[
        *TypeList.tabulate[
            Trait=Movable & Deinitable & Defaultable,
            Self.N,
            Self._Slot[T, ...],
        ]()
    ]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        _Choice[Self.Packed], Self._Slots[T]
    ]
    comptime STEP = 64

    var candidates: Tuple[*Self.Us]

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        var f = packed.size
        var segments = ceildiv(count, f) if f > 0 else 0
        if f <= 0 or f % Self.STEP != 0 or len(packed.chosen) != segments:
            raise Self.error[CorruptError](
                t"{len(packed.chosen)} segments of {f} values for {count}"
                t" values"
            )
        return (
            _Choice[Self.Packed](packed^, BySize(count, f)),
            Self._Slots[T](),
        )

    # Once a segment, so out of the register loop: not `@always_inline`.
    def _enter[T: DType](self, mut state: Self.State[T], i: Int):
        """The segment holding value `i`, its candidate's state started; the
        segment read before checked and its state dropped."""
        comptime for k in range(Self.N):
            if state[0].active == k:
                ref slot = rebind[Optional[Self.Us[k].State[T]]](state[1][k])
                try:
                    self.candidates[k].check[T](slot.value())
                except e:
                    state[0].note(e)
                slot = None
        var s = state[0].segments.of(i)
        var lo = state[0].segments.start(s)
        var hi = state[0].segments.end(s)
        state[0].segment = s
        state[0].lo = lo
        state[0].hi = hi
        state[0].active = -1
        comptime for k in range(Self.N):
            comptime S = Self.Packed.Slot[k]
            if state[0].packed.chosen[s].isa[S]():
                var packed = state[0].packed.chosen[s][S].packed.copy()
                ref slot = rebind[Optional[Self.Us[k].State[T]]](state[1][k])
                try:
                    slot = Optional(
                        self.candidates[k].start[T](
                            rebind_var[Self.Us[k].Packed](packed^), hi - lo
                        )
                    )
                    state[0].active = k
                except e:
                    state[0].note(e)

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        # Segments hold whole registers, so a register lies in one.
        if i < state[0].lo or i >= state[0].hi:
            self._enter[T](state, i)
        var lo = state[0].lo
        comptime for k in range(Self.N):
            if state[0].active == k:
                ref slot = rebind[Optional[Self.Us[k].State[T]]](
                    state[1][k]
                ).value()
                return self.candidates[k].load[T, W](slot, i - lo)
        # A segment its candidate refused reads as zeros; `check` raises.
        return SIMD[T, W](0)

    def check[T: DType](self, state: Self.State[T]) raises:
        var failure = state[0].failure
        comptime for k in range(Self.N):
            if state[0].active == k:
                ref slot = rebind[Optional[Self.Us[k].State[T]]](state[1][k])
                try:
                    self.candidates[k].check[T](slot.value())
                except e:
                    if not failure:
                        failure = String(t"segment {state[0].segment}: {e}")
        if failure:
            raise Self.error[CorruptError](failure)


# --- Auto ----------------------------------------------------------------------


struct Auto:
    """Packers chosen per segment by sampling, a level deep for each name:
    each level chooses among packers whose inner streams the level below
    chooses for. `Int` levels take integers of any width, `Float` levels
    floats.

    Each list is in the order its candidates decode fastest -- the most
    common value with the rest patched, the values as they are, frame of
    reference, patched frame of reference, a dictionary, for integers the
    differences between the values and for floats their bits cut into a
    dictionary and the rest, and runs -- so of two that pack alike
    the faster is chosen. Runs come last: their values are packed by the
    level below, so runs of one value pack nearly as small as those values
    packed alone, and decode slowest. The differences are chosen for by the
    level below too, so a level of its own takes them again: differences of
    differences.

    An integer unpacker takes the column's dtype `T` and `U`, its unsigned
    twin; a float unpacker the dtypes its streams are read as: `I`, the
    integers of the floats' digits, and `U`, that integer's unsigned twin.

        Codec(Auto.int1(), Auto.int_unpack1[DType.int64, DType.uint64]())
    """

    comptime _For = Encoder[Compose[OrderedEncode], ReferencePack[BitPack]]
    """Frame of reference over the values' ordered bits."""
    comptime _ForUnpack[T: DType, U: DType] = Decoder[
        ReferenceUnpack[BitUnpack], Compose[OrderedDecode[T]], U
    ]
    comptime _Pfor = Encoder[Compose[OrderedEncode], ReferencePack[PforPack]]
    """Patched frame of reference over the values' ordered bits."""
    comptime _PforUnpack[T: DType, U: DType] = Decoder[
        ReferenceUnpack[PforUnpack], Compose[OrderedDecode[T]], U
    ]
    comptime _Delta[L: PackKernel] = Encoder[Compose[DiffScan], L]
    """The differences between the values; for run ends, the runs'
    lengths."""
    comptime _DeltaUnpack[L: UnpackKernel, T: DType] = Decoder[
        L, Compose[SumScan], T
    ]
    comptime _AlpRd = AlpRdPack[DictionaryPack[ReferencePack[BitPack]], BitPack]
    """The floats' bits cut in two, the left parts through a dictionary
    whose codes are framed a block at a time."""
    comptime _AlpRdUnpack = AlpRdUnpack[
        DictionaryUnpack[ReferenceUnpack[BitUnpack]], BitUnpack
    ]
    comptime _U64 = DType.uint64

    comptime Int0 = ChoicePack[PlainPack, Self._For, Self._Pfor]
    # The unpackers take `U` and `I` as parameters rather than deriving them
    # from the column's dtype: twins nesting derived dtypes this deep do not
    # finish compiling.
    comptime IntUnpack0[T: DType, U: DType] = ChoiceUnpack[
        PlainUnpack, Self._ForUnpack[T, U], Self._PforUnpack[T, U]
    ]
    # Delta before Runs: runs of one value pack nearly as small as their
    # values' own choice, which may be the differences, and decode slowest.
    comptime _IntOver[L: PackKernel] = ChoicePack[
        FrequencyPack,
        PlainPack,
        Self._For,
        Self._Pfor,
        DictionaryPack[BitPack],
        Self._Delta[L],
        RunsPack[L, Self._Delta[L]],
    ]
    """The integers' candidates, `L` the level below."""
    comptime _IntUnpackOver[
        L: UnpackKernel, E: UnpackKernel, T: DType, U: DType
    ] = ChoiceUnpack[
        FrequencyUnpack,
        PlainUnpack,
        Self._ForUnpack[T, U],
        Self._PforUnpack[T, U],
        DictionaryUnpack[BitUnpack],
        Self._DeltaUnpack[L, T],
        RunsUnpack[L, Self._DeltaUnpack[E, Self._U64]],
    ]
    comptime Int1 = Self._IntOver[Self.Int0]
    comptime IntUnpack1[T: DType, U: DType] = Self._IntUnpackOver[
        Self.IntUnpack0[T, U], Self.IntUnpack0[Self._U64, Self._U64], T, U
    ]
    comptime Int2 = Self._IntOver[Self.Int1]
    comptime IntUnpack2[T: DType, U: DType] = Self._IntUnpackOver[
        Self.IntUnpack1[T, U], Self.IntUnpack1[Self._U64, Self._U64], T, U
    ]
    comptime Float0 = ChoicePack[PlainPack, PatchPack[Alp, Self._For]]
    comptime FloatUnpack0[I: DType, U: DType] = ChoiceUnpack[
        PlainUnpack, PatchUnpack[Alp, Self._ForUnpack[I, U]]
    ]
    comptime _FloatOver[F: PackKernel, L: PackKernel] = ChoicePack[
        FrequencyPack,
        PlainPack,
        PatchPack[Alp, L],
        DictionaryPack[BitPack],
        Self._AlpRd,
        RunsPack[F, Self._Delta[L]],
    ]
    """The floats' candidates, `F` the float level below and `L` the
    integer one."""
    comptime _FloatUnpackOver[
        F: UnpackKernel, L: UnpackKernel, E: UnpackKernel
    ] = ChoiceUnpack[
        FrequencyUnpack,
        PlainUnpack,
        PatchUnpack[Alp, L],
        DictionaryUnpack[BitUnpack],
        Self._AlpRdUnpack,
        RunsUnpack[F, Self._DeltaUnpack[E, Self._U64]],
    ]
    comptime Float1 = Self._FloatOver[Self.Float0, Self.Int0]
    comptime FloatUnpack1[I: DType, U: DType] = Self._FloatUnpackOver[
        Self.FloatUnpack0[I, U],
        Self.IntUnpack0[I, U],
        Self.IntUnpack0[Self._U64, Self._U64],
    ]
    comptime Float2 = Self._FloatOver[Self.Float1, Self.Int1]
    comptime FloatUnpack2[I: DType, U: DType] = Self._FloatUnpackOver[
        Self.FloatUnpack1[I, U],
        Self.IntUnpack1[I, U],
        Self.IntUnpack1[Self._U64, Self._U64],
    ]

    @staticmethod
    def _for() -> Self._For:
        return Self._For(
            Compose((OrderedEncode(),)), ReferencePack(BLOCK, BitPack())
        )

    @staticmethod
    def _for_unpack[T: DType, U: DType]() -> Self._ForUnpack[T, U]:
        return Self._ForUnpack[T, U](
            ReferenceUnpack(BLOCK, BitUnpack()), Compose((OrderedDecode[T](),))
        )

    @staticmethod
    def _pfor() -> Self._Pfor:
        return Self._Pfor(
            Compose((OrderedEncode(),)),
            ReferencePack(BLOCK, PforPack(BLOCK, BitPack())),
        )

    @staticmethod
    def _pfor_unpack[T: DType, U: DType]() -> Self._PforUnpack[T, U]:
        return Self._PforUnpack[T, U](
            ReferenceUnpack(BLOCK, PforUnpack(BLOCK, BitUnpack())),
            Compose((OrderedDecode[T](),)),
        )

    @staticmethod
    def _alp_rd() -> Self._AlpRd:
        return Self._AlpRd(
            BLOCK, DictionaryPack(ReferencePack(BLOCK, BitPack())), BitPack()
        )

    @staticmethod
    def _alp_rd_unpack() -> Self._AlpRdUnpack:
        return Self._AlpRdUnpack(
            BLOCK,
            DictionaryUnpack(ReferenceUnpack(BLOCK, BitUnpack())),
            BitUnpack(),
        )

    @staticmethod
    def _delta[L: PackKernel](var inner: L) -> Self._Delta[L]:
        return Self._Delta[L](Compose((DiffScan(),)), inner^)

    @staticmethod
    def _delta_unpack[
        L: UnpackKernel, T: DType
    ](var inner: L) -> Self._DeltaUnpack[L, T]:
        return Self._DeltaUnpack[L, T](inner^, Compose((SumScan(),)))

    @staticmethod
    def _int_over[
        L: PackKernel
    ](size: Int, var deltas: L, var runs: L, var ends: L) -> Self._IntOver[L]:
        """A level's candidates, each of `deltas`, `runs` and `ends` the
        level below."""
        return Self._IntOver[L](
            size,
            (
                FrequencyPack(BLOCK, ConstantPack()),
                PlainPack(),
                Self._for(),
                Self._pfor(),
                DictionaryPack(BitPack()),
                Self._delta(deltas^),
                RunsPack(runs^, Self._delta(ends^)),
            ),
        )

    @staticmethod
    def _int_unpack_over[
        L: UnpackKernel, E: UnpackKernel, T: DType, U: DType
    ](var deltas: L, var runs: L, var ends: E) -> Self._IntUnpackOver[
        L, E, T, U
    ]:
        return Self._IntUnpackOver[L, E, T, U](
            (
                FrequencyUnpack(BLOCK, ConstantUnpack()),
                PlainUnpack(),
                Self._for_unpack[T, U](),
                Self._pfor_unpack[T, U](),
                DictionaryUnpack(BitUnpack()),
                Self._delta_unpack[T=T](deltas^),
                RunsUnpack(runs^, Self._delta_unpack[T=Self._U64](ends^)),
            )
        )

    @staticmethod
    def _float_over[
        F: PackKernel, L: PackKernel
    ](size: Int, var runs: F, var ints: L, var ends: L) -> Self._FloatOver[
        F, L
    ]:
        return Self._FloatOver[F, L](
            size,
            (
                FrequencyPack(BLOCK, ConstantPack()),
                PlainPack(),
                PatchPack[Alp, L](BLOCK, ints^),
                DictionaryPack(BitPack()),
                Self._alp_rd(),
                RunsPack(runs^, Self._delta(ends^)),
            ),
        )

    @staticmethod
    def _float_unpack_over[
        F: UnpackKernel, L: UnpackKernel, E: UnpackKernel
    ](var runs: F, var ints: L, var ends: E) -> Self._FloatUnpackOver[F, L, E]:
        return Self._FloatUnpackOver[F, L, E](
            (
                FrequencyUnpack(BLOCK, ConstantUnpack()),
                PlainUnpack(),
                PatchUnpack[Alp, L](BLOCK, ints^),
                DictionaryUnpack(BitUnpack()),
                Self._alp_rd_unpack(),
                RunsUnpack(runs^, Self._delta_unpack[T=Self._U64](ends^)),
            )
        )

    @staticmethod
    def int0(size: Int = Self.Int0.SEGMENT) -> Self.Int0:
        return Self.Int0(size, (PlainPack(), Self._for(), Self._pfor()))

    @staticmethod
    def int_unpack0[T: DType, U: DType]() -> Self.IntUnpack0[T, U]:
        return Self.IntUnpack0[T, U](
            (
                PlainUnpack(),
                Self._for_unpack[T, U](),
                Self._pfor_unpack[T, U](),
            )
        )

    @staticmethod
    def int1(size: Int = Self.Int1.SEGMENT) -> Self.Int1:
        return Self._int_over(size, Self.int0(), Self.int0(), Self.int0())

    @staticmethod
    def int_unpack1[T: DType, U: DType]() -> Self.IntUnpack1[T, U]:
        return Self._int_unpack_over[T=T, U=U](
            Self.int_unpack0[T, U](),
            Self.int_unpack0[T, U](),
            Self.int_unpack0[Self._U64, Self._U64](),
        )

    @staticmethod
    def int2(size: Int = Self.Int2.SEGMENT) -> Self.Int2:
        return Self._int_over(size, Self.int1(), Self.int1(), Self.int1())

    @staticmethod
    def int_unpack2[T: DType, U: DType]() -> Self.IntUnpack2[T, U]:
        return Self._int_unpack_over[T=T, U=U](
            Self.int_unpack1[T, U](),
            Self.int_unpack1[T, U](),
            Self.int_unpack1[Self._U64, Self._U64](),
        )

    @staticmethod
    def float0(size: Int = Self.Float0.SEGMENT) -> Self.Float0:
        return Self.Float0(
            size, (PlainPack(), PatchPack[Alp, Self._For](BLOCK, Self._for()))
        )

    @staticmethod
    def float_unpack0[I: DType, U: DType]() -> Self.FloatUnpack0[I, U]:
        return Self.FloatUnpack0[I, U](
            (
                PlainUnpack(),
                PatchUnpack[Alp, Self._ForUnpack[I, U]](
                    BLOCK, Self._for_unpack[I, U]()
                ),
            )
        )

    @staticmethod
    def float1(size: Int = Self.Float1.SEGMENT) -> Self.Float1:
        return Self._float_over(size, Self.float0(), Self.int0(), Self.int0())

    @staticmethod
    def float_unpack1[I: DType, U: DType]() -> Self.FloatUnpack1[I, U]:
        return Self._float_unpack_over(
            Self.float_unpack0[I, U](),
            Self.int_unpack0[I, U](),
            Self.int_unpack0[Self._U64, Self._U64](),
        )

    @staticmethod
    def float2(size: Int = Self.Float2.SEGMENT) -> Self.Float2:
        return Self._float_over(size, Self.float1(), Self.int1(), Self.int1())

    @staticmethod
    def float_unpack2[I: DType, U: DType]() -> Self.FloatUnpack2[I, U]:
        return Self._float_unpack_over(
            Self.float_unpack1[I, U](),
            Self.int_unpack1[I, U](),
            Self.int_unpack1[Self._U64, Self._U64](),
        )
