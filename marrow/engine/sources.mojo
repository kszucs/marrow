# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Where a phase's morsels come from.

A `Source` declares how many morsels it has and reads one when asked. It never
decides which worker reads which: that is the `Driver`'s strategy. Sources are
erased by `DynSource`, because the kind of a plan's sources is chosen at run
time from the plan, and a closed `Variant` would link every reader into every
binary.
"""

from std.memory import ArcPointer

from ..arrays import StructArray
from ..execution import ExecContext


trait Source(Deinitable, Movable):
    """The morsels of one phase: how many there are, and each one's batches."""

    def count(self) -> Int:
        """How many morsels there are, or -1 when only reading to the end
        tells. A source of unknown count is read in order, on one thread."""
        ...

    def rows(self) -> Int:
        """How many rows the morsels hold in all, or -1 when unknown."""
        ...

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        """Morsel `i`'s batches, read on `ctx`. A source of unknown count
        answers its next morsel, whatever `i` is, and nothing past the last."""
        ...

    def describe(self) -> String:
        """One line naming the source, for a plan report."""
        ...


def rows_of(batches: List[StructArray]) -> Int:
    """The rows of `batches`, together."""
    var n = 0
    for ref b in batches:
        n += len(b)
    return n


struct DynSource(Movable):
    """A `Source` of any kind, erased."""

    var _data: ArcPointer[NoneType]
    var _virt_count: def(ArcPointer[NoneType]) thin -> Int
    var _virt_rows: def(ArcPointer[NoneType]) thin -> Int
    var _virt_read: def(
        ArcPointer[NoneType], Int, ExecContext
    ) thin raises -> List[StructArray]
    var _virt_describe: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin
    """Releases the box's reference at the source's true type, so its
    destructor runs: an erased `ArcPointer` forgets it."""

    @staticmethod
    def _count_tramp[S: Source](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[S]](ptr)[].count()

    @staticmethod
    def _rows_tramp[S: Source](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[S]](ptr)[].rows()

    @staticmethod
    def _read_tramp[
        S: Source
    ](ptr: ArcPointer[NoneType], i: Int, ctx: ExecContext) raises -> List[
        StructArray
    ]:
        return rebind[ArcPointer[S]](ptr)[].read(i, ctx)

    @staticmethod
    def _describe_tramp[S: Source](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[S]](ptr)[].describe()

    @staticmethod
    def _drop_tramp[S: Source](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[S]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[S: Source](out self, var value: S):
        var ptr = ArcPointer[S](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_count = Self._count_tramp[S]
        self._virt_rows = Self._rows_tramp[S]
        self._virt_read = Self._read_tramp[S]
        self._virt_describe = Self._describe_tramp[S]
        self._virt_drop = Self._drop_tramp[S]

    def __deinit__(deinit self):
        self._virt_drop(self._data^)

    def count(self) -> Int:
        return self._virt_count(self._data)

    def rows(self) -> Int:
        return self._virt_rows(self._data)

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._virt_read(self._data, i, ctx)

    def describe(self) -> String:
        return self._virt_describe(self._data)


struct TableSource(Source):
    """Batches already in memory, a few at a time: each morsel is some of
    them, or a slice of one."""

    var _morsels: List[List[StructArray]]

    def __init__(out self, var morsels: List[List[StructArray]]):
        self._morsels = morsels^

    @staticmethod
    def of(var batches: List[StructArray]) -> Self:
        """One morsel per batch."""
        var morsels = List[List[StructArray]](capacity=len(batches))
        for ref b in batches:
            morsels.append([b.copy()])
        return Self(morsels^)

    @staticmethod
    def sliced(batch: StructArray, morsel_rows: Int) -> Self:
        """`batch` in morsels of `morsel_rows` rows, the last one shorter.
        A slice shares the batch's buffers. An empty batch is one empty
        morsel, so its schema still reaches the stages that read it."""
        var n = len(batch)
        var morsels = List[List[StructArray]]()
        if n <= morsel_rows:
            morsels.append([batch.copy()])
        else:
            var start = 0
            while start < n:
                var length = min(morsel_rows, n - start)
                morsels.append([batch.slice(start, length)])
                start += length
        return Self(morsels^)

    def count(self) -> Int:
        return len(self._morsels)

    def rows(self) -> Int:
        var n = 0
        for ref m in self._morsels:
            n += rows_of(m)
        return n

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._morsels[i].copy()

    def describe(self) -> String:
        return String("TableSource(", self.rows(), " rows)")


struct PlannedSource(Source):
    """Rows a dry run did not produce, where a phase would read them: no
    morsel holds anything; the phase reading it carries their estimate."""

    var _name: String
    var _morsels: Int

    def __init__(out self, var name: String, morsels: Int):
        self._name = name^
        self._morsels = morsels

    def count(self) -> Int:
        return self._morsels

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return []

    def describe(self) -> String:
        return self._name.copy()


trait Stream(Deinitable, Movable):
    """Batches read one after another until there are none: a source whose
    length only reading tells."""

    def next(mut self) raises -> List[StructArray]:
        """The next batches, never empty until the stream is spent; nothing
        once it is."""
        ...


struct StreamSource[S: Stream](Source):
    """A stream as a source: each morsel is the next batches it reads. Its
    count is unknown, so it is read in order, on one thread."""

    var _stream: ArcPointer[Self.S]
    var _name: String

    def __init__(out self, var stream: Self.S, var name: String):
        self._stream = ArcPointer(stream^)
        self._name = name^

    def count(self) -> Int:
        return -1

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._stream[].next()

    def describe(self) -> String:
        return self._name.copy()
