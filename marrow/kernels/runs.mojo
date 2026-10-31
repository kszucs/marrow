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

Three public kernels are built on them. `RunEndEncodeKernel` stores each run's
value once with the row its run ends at, a run-end encoded array, and
`RunEndDecodeKernel` repeats each value over its run again. `RunIndexKernel`
numbers each row by its run.

`RunsPack` stores a column's runs, each value once and where its run ends,
equal meaning the same bits; `RunsUnpack` reads them back a block at a time.
"""

from std.memory import bitcast
from std.sys import size_of

from ..arrays import (
    BoolArray,
    DynArray,
    Int64Array,
    PrimitiveArray,
    RunEndEncodedArray,
    UInt32Array,
)
from ..buffers import Bitmap, Buffer
from ..builders import arange
from ..dtypes import (
    DynType,
    Int16Type,
    Int32Type,
    Int64Type,
    PrimitiveType,
    UInt32Type,
    run_end_encoded,
    uint16,
    uint32,
    uint64,
)
from ..errors import CorruptError, InvalidError, TypeError
from ..execution import ExecContext
from ..utils.integers import Unsigned
from ..views import BitmapView, BufferView
from .cast import NumericCastKernel
from .core import Kernel, PackKernel, UnpackKernel
from .filter import IndicesNonzeroKernel, TakeKernel
from .hashing import KeyCompare
from .packing import BLOCK, Unpacked
from .segments import ByEnds, PerSegment, Segments


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
# RunEndEncodeKernel, RunEndDecodeKernel, RunIndexKernel
# ---------------------------------------------------------------------------


struct RunEndEncodeKernel(Kernel):
    """Each run of equal values once, with the row its run ends at.

    Equivalent to PyArrow's ``run_end_encode``. Rows are equal when their bits
    are: NaNs with different payloads are different values, and so are `-0.0`
    and `0.0`. Consecutive nulls are one run, of a null value.

        [5, 5, 5, 1, 2, 2]  ->  run ends [3, 4, 6], values [5, 1, 2]
    """

    comptime name = "run_end_encode"

    @staticmethod
    def dispatch(
        array: DynArray,
        run_end_type: DynType,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> RunEndEncodedArray:
        """`array`'s runs, their ends typed `run_end_type`.

        Raises:
            TypeError: when `run_end_type` is not int16, int32 or int64, or
                `array` holds nested, dictionary or run-end encoded values.
            InvalidError: when `array` has more rows than `run_end_type` can
                count.
        """
        var dtype = array.dtype()
        var n = len(array)
        var most: Int
        if run_end_type.is_int16():
            most = Int(Int16.MAX)
        elif run_end_type.is_int32():
            most = Int(Int32.MAX)
        elif run_end_type.is_int64():
            most = Int(Int64.MAX)
        else:
            raise Self.error[TypeError](
                t"run ends must be int16, int32 or int64, not {run_end_type}"
            )
        if (
            dtype.is_list_like()
            or dtype.is_fixed_size_list()
            or dtype.is_struct()
            or dtype.is_dictionary()
            or dtype.is_run_end_encoded()
        ):
            raise Self.error[TypeError](t"cannot encode {dtype} values")
        if n > most:
            raise Self.error[InvalidError](
                t"{n} rows are more than {run_end_type} run ends can hold:"
                t" {most}"
            )
        # A float compares as the unsigned integer of its width: by its bits.
        var keys = array.copy()
        if dtype.is_floating_point():
            if dtype.byte_width() == 2:
                keys = array.view(uint16)
            elif dtype.byte_width() == 4:
                keys = array.view(uint32)
            else:
                keys = array.view(uint64)
        var positions = RunsKernel.dispatch([keys^], n, ctx)
        var runs = len(positions) - 1
        var starts = NumericCastKernel.apply[Int64Type, Int32Type, False](
            positions.slice(0, runs), ctx
        )
        var ends = positions.slice(1, runs)
        var run_ends: DynArray
        if run_end_type.is_int16():
            run_ends = NumericCastKernel.apply[Int64Type, Int16Type, False](
                ends, ctx
            )
        elif run_end_type.is_int32():
            run_ends = NumericCastKernel.apply[Int64Type, Int32Type, False](
                ends, ctx
            )
        else:
            run_ends = NumericCastKernel.apply[Int64Type, Int64Type, False](
                ends, ctx
            )
        return RunEndEncodedArray(
            dtype=run_end_encoded(run_end_type.copy(), dtype^),
            length=n,
            offset=0,
            run_ends=run_ends^,
            values=TakeKernel.dispatch(array, starts, ctx),
        )


struct RunEndDecodeKernel(Kernel):
    """Each run's value repeated over its rows.

    Equivalent to PyArrow's ``run_end_decode``. The typed `apply` overloads
    decode one value per segment of any `Segments`, so a run may be empty.

        run ends [3, 4, 6], values [5, 1, 2]  ->  [5, 5, 5, 1, 2, 2]
    """

    comptime name = "run_end_decode"

    @staticmethod
    def apply[
        T: PrimitiveType, S: Segments
    ](values: PrimitiveArray[T], runs: S) raises -> PrimitiveArray[T]:
        """Row `i` of run `s` is `values[s]`, null where it is.

        The runs cut rows `[0, n)`, `n` where the last one ends: the first
        starts at row 0 and each starts where the one before ends.

        Raises:
            InvalidError: unless `values` holds one value per run.
        """
        comptime N = T.native
        var count = runs.count()
        if len(values) != count:
            raise Self.error[InvalidError](
                t"{len(values)} values for {count} runs"
            )
        var n = runs.end(count - 1) if count > 0 else 0
        # `PerSegment` reads its values from the start of their buffer.
        var src = values.buffer
        if values.offset > 0:
            var copied = Buffer.alloc_uninit[N](count)
            copied.view[N](0, count).copy_from(values.values(), count)
            src = copied^.to_immutable()
        var out = Buffer.alloc_uninit[N](n)
        PerSegment(src^, count).broadcast(runs, out.view[N](0, n))
        var validity: Optional[Bitmap[mut=False]] = None
        if values.null_count() > 0:
            validity = Self._valid_runs(values.validity().value(), runs, n)
        return PrimitiveArray[T](
            values.dtype,
            length=n,
            validity=validity,
            buffer=out^.to_immutable(),
        )

    @staticmethod
    def apply[S: Segments](values: BoolArray, runs: S) raises -> BoolArray:
        """Row `i` of run `s` is `values[s]`, null where it is.

        The runs cut rows `[0, n)`, as the primitive overload's do.

        Raises:
            InvalidError: unless `values` holds one value per run.
        """
        var count = runs.count()
        if len(values) != count:
            raise Self.error[InvalidError](
                t"{len(values)} values for {count} runs"
            )
        var n = runs.end(count - 1) if count > 0 else 0
        var bits = values.values()
        var out = Bitmap.alloc_zeroed(n)
        for s in range(count):
            if bits.test(s):
                out.set_range(runs.start(s), runs.end(s) - runs.start(s), True)
        var validity: Optional[Bitmap[mut=False]] = None
        if values.null_count() > 0:
            validity = Self._valid_runs(values.validity().value(), runs, n)
        return BoolArray(
            length=n, validity=validity^, buffer=out^.to_immutable()
        )

    @staticmethod
    def _valid_runs[
        S: Segments
    ](valid: BitmapView[_], runs: S, n: Int) -> Bitmap[mut=False]:
        """The `n` rows' validity: set over each run whose value is valid."""
        var out = Bitmap.alloc_zeroed(n)
        for s in range(runs.count()):
            if valid.test(s):
                out.set_range(runs.start(s), runs.end(s) - runs.start(s), True)
        return out^.to_immutable()

    @staticmethod
    def apply(
        array: RunEndEncodedArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> DynArray:
        """The rows of `array`'s window, as an array of its value type.

        Fixed-width and boolean values are repeated over their runs; every
        other layout is gathered at each row's run.
        """
        var ends = array.segment_ends()
        var runs = ByEnds(ends.values())
        var values = array.physical_values()
        if values.isa[BoolArray]():
            return Self.apply(values.as_bool(), runs).to_dyn()
        elif values.dtype().is_primitive():

            def primitive[T: PrimitiveType](d: T) raises {imm} -> DynArray:
                return Self.apply(values.as_primitive[T](), runs).to_dyn()

            return values.dtype().try_visit[DynArray, PrimitiveType](
                primitive, "dtype is not primitive"
            )
        else:
            var ids = Self.apply(arange[Int32Type](0, runs.count()), runs)
            return TakeKernel.dispatch(values, ids, ctx)

    @staticmethod
    def dispatch(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> DynArray:
        """`array`'s rows, decoded.

        Raises:
            TypeError: unless `array` is run-end encoded.
        """
        if array.isa[RunEndEncodedArray]():
            return Self.apply(array.as_run_end_encoded(), ctx)
        else:
            raise Self.error[TypeError](
                t"{array.dtype()} is not run-end encoded"
            )


struct RunIndexKernel(Kernel):
    """Each row's run, counting from 0.

    Rows are equal as `GROUP BY` keys are: nulls equal each other, every NaN
    is one value, and `-0.0` equals `0.0`.

        [1, 1, null, null, 2, 1]  ->  [0, 0, 1, 1, 2, 3]
    """

    comptime name = "run_index"

    @staticmethod
    def dispatch(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> UInt32Array:
        """One `uint32` per row of `array`: the number of its run.

        Raises:
            TypeError: for a dtype whose rows cannot be compared.
        """
        var positions = RunsKernel.dispatch([array.copy()], len(array), ctx)
        var runs = ByEnds(positions.values())
        return RunEndDecodeKernel.apply(
            arange[UInt32Type](0, runs.count()), runs
        )


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
