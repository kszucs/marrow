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
"""

from std.builtin.rebind import downcast, rebind_var
from std.builtin.variadics import TypeList
from std.sys import size_of

from ..views import BufferView
from .core import (
    BinaryKernel,
    Kernel,
    PackKernel,
    ScanKernel,
    UnaryKernel,
    UnpackKernel,
)
from .packing import BLOCK


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
