# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Runs: consecutive rows equal on a key.

A **run** is a maximal stretch of consecutive rows that hold the same key.
`RunsKernel` finds where each run starts and answers the positions a `ByEnds`
reads, so the runs are segments: a window's partitions and peer groups, and
the rows whose frames are equal.

"Equal" is `IS NOT DISTINCT FROM`, the key identity `GROUP BY` groups by:
nulls are equal to each other, every NaN is one value, and a nested key
compares structurally.

`RunsPack` stores a column's runs, each value once and where its run ends,
equal meaning the same bits; `RunsUnpack` reads them back a block at a time.
"""

from std.memory import bitcast
from std.sys import size_of

from ..arrays import BoolArray, DynArray, Int64Array
from ..buffers import Bitmap, Buffer
from ..builders import arange
from ..dtypes import Int32Type
from ..errors import CorruptError, InvalidError
from ..execution import ExecContext
from ..utils.integers import Unsigned
from ..views import BufferView
from .core import Kernel, PackKernel, UnpackKernel
from .filter import IndicesNonzeroKernel
from .hashing import KeyCompare
from .packing import BLOCK, Unpacked


struct RunsKernel(Kernel):
    """Where each run of rows equal on every key starts."""

    comptime name = "runs"

    @staticmethod
    def dispatch(
        keys: List[DynArray],
        length: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int64Array:
        """The ascending positions where a run starts, then `length`: the
        `count + 1` positions a `ByEnds` reads. `[0]` when `length` is 0, and
        `[0, length]` with no key, since every row is then one run.

        Raises:
            InvalidError: unless every key holds `length` rows.
        """
        for ref key in keys:
            if len(key) != length:
                raise Self.error[InvalidError](
                    t"a key holds {len(key)} rows, not {length}"
                )
        # One bit per row and one past the last: a run starts at row 0, and
        # the last run ends at `length`.
        var starts = Bitmap.alloc_zeroed(length + 1)
        starts.set(0)
        starts.set(length)
        if length > 1:
            # Bit `j` is whether row `j + 1` equals row `j` on every key;
            # each key compares only the pairs the earlier keys kept.
            var same = Bitmap.alloc_zeroed(length - 1)
            same.set_range(0, length - 1, True)
            var rows = arange[Int32Type](0, length - 1)
            for ref key in keys:
                KeyCompare.dispatch(
                    key.slice(1, length - 1),
                    rows,
                    key.slice(0, length - 1),
                    rows,
                    same,
                    ctx,
                )
            var changed = ~same.view(0, length - 1)
            starts.extend(changed.view(0, length - 1), 1, length - 1)
        var positions = IndicesNonzeroKernel.apply(
            BoolArray(
                length=length + 1,
                buffer=starts^.to_immutable(length=length + 1),
            )
        )
        # No position exceeds `length`, so its uint64 bits read as int64.
        return Int64Array(length=len(positions), buffer=positions.buffer)


# ---------------------------------------------------------------------------
# RunsPack, RunsUnpack
# ---------------------------------------------------------------------------


@fieldwise_init
struct RunsPacked[V: Copyable & Deinitable, E: Copyable & Deinitable](
    Copyable, Movable
):
    """Each run of equal values once: the runs' values, packed by one packer,
    and where each run ends, `uint64`s packed by another.

        [5, 5, 5, 1, 2, 2]  ->  values [5, 1, 2], ends [3, 4, 6]
    """

    var runs: Int
    var values: Self.V
    var ends: Self.E


struct _RunFinding(Movable):
    """The runs found so far, each one's value and where it ends, and the
    bits of the run still open, `at` values in."""

    var values: Buffer[mut=True]
    var ends: List[UInt64]
    var value: UInt64
    var at: Int

    def __init__(out self, var values: Buffer[mut=True]):
        self.values = values^
        self.ends = List[UInt64]()
        self.value = 0
        self.at = 0

    def close[T: DType](mut self):
        """The open run, found: it ends at `at`."""
        self.values.view[T]().unsafe_set(
            len(self.ends), bitcast[T](self.value.cast[Unsigned[T]]())
        )
        self.ends.append(UInt64(self.at))


@fieldwise_init
struct RunsPack[Values: PackKernel, Ends: PackKernel](PackKernel):
    """Values as `RunsPacked`: each run's value packed by `Values`, and its
    end by `Ends`. Values are equal when their bits are."""

    comptime name = "runs_pack"
    comptime Packed = RunsPacked[Self.Values.Packed, Self.Ends.Packed]
    comptime State[T: DType]: Movable & Deinitable = _RunFinding

    var values: Self.Values
    var ends: Self.Ends

    def start[T: DType](self, count: Int) raises -> _RunFinding:
        comptime assert (
            size_of[Scalar[T]]() <= 8
        ), "RunsPack takes values of at most 8 bytes"
        return _RunFinding(Buffer.alloc_uninit[T](count))

    def store[
        T: DType
    ](self, mut state: _RunFinding, block: BufferView[T, _]) raises:
        comptime U = Unsigned[T]
        comptime W = 64 // size_of[Scalar[T]]()
        var n = len(block)
        var i = 0
        while i < n:
            var bits = bitcast[U](block.load[1](i)).cast[DType.uint64]()
            if state.at == 0:
                state.value = bits
            elif bits != state.value:
                state.close[T]()
                state.value = bits
            state.at += 1
            i += 1
            # The run goes on while the values equal it, a register at a
            # time.
            var v = SIMD[U, W](state.value.cast[U]())
            while (
                i + W <= n
                and bitcast[U, W](block.load[W](i)).eq(v).reduce_and()
            ):
                state.at += W
                i += W

    def finish[T: DType](self, var state: _RunFinding) raises -> Self.Packed:
        if state.at > 0:
            state.close[T]()
        var runs = len(state.ends)
        return Self.Packed(
            runs,
            Self._packed(self.values, state.values.view[T](0, runs)),
            Self._packed(self.ends, BufferView(Span(state.ends))),
        )

    @staticmethod
    def _packed[
        P: PackKernel, T: DType
    ](packer: P, values: BufferView[T, _]) raises -> P.Packed:
        """`values` packed by `packer`, a block of `BLOCK` at a time: the runs
        are counted only once the column is, so they are packed at its end."""
        var n = len(values)
        var state = packer.start[T](n)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            packer.store[T](state, values[lo : lo + m])
            lo += m
        return packer.finish[T](state^)


struct _RunReading[T: DType](Movable):
    """Where `RunsUnpack` is: the run being read, its value, where it ends
    and how many of its values are left, the block decoded, and whether the
    runs did not add up."""

    var runs: Int
    var run: Int
    var value: Scalar[Self.T]
    var end: Int
    var left: Int
    var unpacked: Unpacked
    var bad: Bool

    def __init__(out self, runs: Int, count: Int):
        self.runs = runs
        self.run = -1
        self.value = 0
        self.end = 0
        self.left = 0
        self.unpacked = Unpacked(count)
        self.bad = False


@fieldwise_init
struct RunsUnpack[Values: UnpackKernel, Ends: UnpackKernel](UnpackKernel):
    """`RunsPacked`'s values, a block at a time: each run's value and end are
    read as the run starts."""

    comptime name = "runs_unpack"
    comptime Packed = RunsPacked[Self.Values.Packed, Self.Ends.Packed]
    comptime State[T: DType]: Movable & Deinitable = Tuple[
        _RunReading[T], Self.Values.State[T], Self.Ends.State[DType.uint64]
    ]

    var values: Self.Values
    var ends: Self.Ends

    def start[
        T: DType
    ](self, var packed: Self.Packed, count: Int) raises -> Self.State[T]:
        if packed.runs < 0:
            raise Self.error[CorruptError](t"{packed.runs} runs")
        return (
            _RunReading[T](packed.runs, count),
            self.values.start[T](packed.values.copy(), packed.runs),
            self.ends.start[DType.uint64](packed.ends.copy(), packed.runs),
        )

    @always_inline
    def _unpack[T: DType](self, mut state: Self.State[T]):
        # A run is written a cache line at a time, past its end into what
        # the runs after it overwrite.
        comptime W = 64 // size_of[Scalar[T]]()
        var out = state[0].unpacked.next[T]()
        var n = state[0].unpacked.size()
        var count = state[0].unpacked.count
        var i = 0
        while i < n:
            if state[0].left == 0:
                var r = state[0].run + 1
                var end = count
                if r < state[0].runs:
                    state[0].value = self.values.load[T, 1](state[1], r)
                    end = Int(self.ends.load[DType.uint64, 1](state[2], r))
                    state[0].run = r
                else:
                    state[0].value = 0
                # Ends that do not increase, or pass the count, are noted
                # and cut to the count, so the values left are filled.
                if r >= state[0].runs or end <= state[0].end or end > count:
                    state[0].bad = True
                    end = count
                state[0].left = end - state[0].end
                state[0].end = end
            var take = min(state[0].left, n - i)
            var v = SIMD[T, W](state[0].value)
            var j = 0
            while j < take:
                out.store[W](i + j, v)
                j += W
            state[0].left -= take
            i += take

    @always_inline
    def load[
        T: DType, W: Int
    ](self, mut state: Self.State[T], i: Int) -> SIMD[T, W]:
        if state[0].unpacked.due(i):
            self._unpack[T](state)
        return state[0].unpacked.load[T, W](i)

    def check[T: DType](self, state: Self.State[T]) raises:
        if state[0].bad or state[0].run + 1 != state[0].runs:
            raise Self.error[CorruptError](
                t"{state[0].runs} runs for {state[0].unpacked.count} values"
            )
        self.values.check[T](state[1])
        self.ends.check[DType.uint64](state[2])
