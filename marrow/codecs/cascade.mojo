# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Cascades: chains fixed at compile time, and their streaming writer and
reader."""

from std.builtin.rebind import downcast

from .byteorder import Leb128, LittleEndian
from .core import (
    Append,
    BLOCK,
    Codec,
    Decoder,
    Elementwise,
    Emitter,
    Mapped,
    Source,
    Unmapped,
    Values,
)
from .plain import Plain


struct Cascade[*Ts: Codec](Copyable, Defaultable, Movable):
    """A chain fixed at compile time: each codec runs over what the one
    before it handed on, as in
    `Cascade[Delta, Zigzag, BitPack].encode[DType.int64](values)`.

    Any codec may follow any other whose output it takes, and the chain may
    end anywhere: its last stream is stored as its values' bytes, as `Plain`
    writes them. An `Elementwise` codec runs inside the next one's loop --
    each value mapped as it is read, and back as it is decoded -- so a block
    is never stored in between; any other codec's bytes are, as the next
    codec's input. It writes the bytes `DynCascade(Delta(), Zigzag(),
    BitPack())` writes."""

    comptime n = len(Self.Ts)

    def __init__(out self):
        """The chain, as a value: what converts to a `DynCascade`."""
        pass

    @staticmethod
    def takes[T: DType]() -> Bool:
        """Whether every codec takes what the one before it hands on, for a
        column of `T`."""
        return Self._takes[0, T]()

    @staticmethod
    def _takes[k: Int, U: DType]() -> Bool:
        comptime if k == Self.n:
            return True
        else:
            comptime C = Self.Ts[k]
            comptime if C.takes[U]():
                return Self._takes[k + 1, C.Out[U]]()
            else:
                return False

    @staticmethod
    def check[T: DType]():
        comptime assert Self.takes[
            T
        ](), "a codec of the chain does not take what it is handed"

    @staticmethod
    def encode[
        T: DType
    ](values: Span[Scalar[T], _], block: Int = BLOCK) raises -> List[UInt8]:
        """`values` as one stream."""
        var writer = CascadeWriter[T, *Self.Ts](block)
        var out = List[UInt8]()
        writer.write(values, out)
        writer.finish(out)
        return out^

    @staticmethod
    def decode[T: DType](data: Span[UInt8, _]) raises -> List[Scalar[T]]:
        """Every value of a stream of this chain."""
        var reader = CascadeReader[T, *Self.Ts](data)
        var out = List[Scalar[T]](capacity=reader.count())
        while reader.read(out):
            pass
        return out^

    @staticmethod
    def write_block[
        T: DType
    ](values: Span[Scalar[T], _], mut out: List[UInt8]) raises:
        """One block: its count, its length, then what the chain wrote."""
        Leb128.write(out, UInt64(len(values)))
        var at = len(out)
        LittleEndian.append[DType.uint32](out, 0)
        Self._encode[0, T](Values(values), out)
        LittleEndian.write[DType.uint32](out, at, UInt32(len(out) - at - 4))

    @staticmethod
    def read_block[
        T: DType
    ](mut src: Decoder[_], count: Int, mut out: List[Scalar[T]]) raises:
        """Append one block's `count` values."""
        # Decoded into a list the emitter owns -- one it points at would be
        # appended to through memory -- then handed back.
        var read = Append[T](List[Scalar[T]]())
        swap(read.out, out)
        read = Self._decode[0, T](src, count, read^)
        out = read^.take()

    @staticmethod
    def _encode[
        k: Int, U: DType, S: Source
    ](var src: S, mut out: List[UInt8]) raises:
        """Stage `k` over `src`, a block of `U`, and every stage after it."""
        comptime if k == Self.n:
            Plain.encode[U](src^, out)
        else:
            comptime C = Self.Ts[k]
            comptime if conforms_to(C, Elementwise):
                comptime E = downcast[C, Elementwise]
                var mapped = Mapped[E, U](src^, out)
                Self._encode[k + 1, E.Out[U]](mapped^, out)
            elif k == Self.n - 1:
                C.encode[U](src^, out)
            else:
                var bytes = List[UInt8]()
                C.encode[U](src^, bytes)
                Leb128.write(out, UInt64(len(bytes)))
                Self._encode[k + 1, DType.uint8](Values(Span(bytes)), out)

    @staticmethod
    def _decode[
        k: Int, U: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        """Emit the `count` values of `U` stage `k` was given."""
        comptime if k == Self.n:
            return Plain.decode[U](src, count, out^)
        else:
            comptime C = Self.Ts[k]
            comptime if conforms_to(C, Elementwise):
                comptime W = downcast[C, Elementwise]
                var undo = Unmapped[W, U](out^, src)
                undo = Self._decode[k + 1, W.Out[U]](src, count, undo^)
                return undo^.take()
            elif k == Self.n - 1:
                return C.decode[U](src, count, out^)
            else:
                var m = src.length()
                var read = Self._decode[k + 1, DType.uint8](
                    src, m, Append[DType.uint8](List[UInt8]())
                )
                var bytes = read^.take()
                var inner = Decoder(Span(bytes))
                out = C.decode[U](inner, count, out^)
                inner.expect_end()
                return out^


struct CascadeWriter[T: DType, *Ts: Codec](Movable):
    """Writes a column through `Cascade[*Ts]` as it arrives: values in chunks of
    any size, a block written whenever one fills."""

    var block: Int
    var buf: List[Scalar[Self.T]]

    def __init__(out self, block: Int = BLOCK):
        Cascade[*Self.Ts].check[Self.T]()
        self.block = max(block, 1)
        self.buf = List[Scalar[Self.T]](capacity=self.block)

    def write(
        mut self, values: Span[Scalar[Self.T], _], mut out: List[UInt8]
    ) raises:
        """Take `values`, appending to `out` every block they fill."""
        var i = 0
        while i < len(values):
            var take = min(self.block - len(self.buf), len(values) - i)
            self.buf.extend(values[i : i + take])
            i += take
            if len(self.buf) == self.block:
                Cascade[*Self.Ts].write_block(Span(self.buf), out)
                self.buf.clear()

    def finish(mut self, mut out: List[UInt8]) raises:
        """Write the last, partial block and the end of the stream."""
        if len(self.buf) > 0:
            Cascade[*Self.Ts].write_block(Span(self.buf), out)
            self.buf.clear()
        out.append(0)


struct CascadeReader[o: ImmOrigin, //, T: DType, *Ts: Codec](Movable):
    """Reads a stream of `Cascade[*Ts]` a block at a time."""

    var src: Decoder[Self.o]

    def __init__(out self, data: Span[UInt8, Self.o]) raises:
        Cascade[*Self.Ts].check[Self.T]()
        self.src = Decoder(data)

    def count(self) raises -> Int:
        """The values left to read, counted without decoding them."""
        return self.src.count()

    def read(mut self, mut out: List[Scalar[Self.T]]) raises -> Bool:
        """Append the next block's values; False, having appended none, once
        the stream has ended."""
        var count, end = self.src.block()
        if count == 0:
            self.src.expect_end()
            return False
        Cascade[*Self.Ts].read_block(self.src, count, out)
        self.src.end_block(end)
        return True
