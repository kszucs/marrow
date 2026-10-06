# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The codec traits, the sources and emitters they read and write
through, and the read cursor.

A codec maps a block of values to another block -- of its own output type,
`Out[T]` -- and back. Most write bytes; an `Elementwise` codec maps one value
to one, given a running state, which lets a `Cascade` fuse it into whatever
follows. A stream is blocks of up to a few thousand values, each on its own
-- its count, its length in bytes, then what the chain wrote -- then an empty
block. It does not name its chain: the reader is given the chain that wrote
it. A reader can count a stream's values, or skip a block, without decoding
it, so a writer and a reader hold one block at a time, in cache, whatever the
column's length.
"""

comptime BLOCK = 4096
"""Values a block holds, unless the writer is given another size."""

from std.memory import bitcast
from std.reflection import reflect
from std.sys import size_of

from ..errors import CorruptError
from .bits import Bits
from .byteorder import Leb128, LittleEndian


trait Codec(Copyable, Defaultable, Deinitable):
    """A block of values to a block of `Out[T]`, and back: a field-less
    struct, one step of a `Cascade`."""

    comptime Out[T: DType]: DType = DType.uint8
    """What the codec hands on for a block of `T`: bytes, unless it says
    otherwise."""

    @staticmethod
    def name() -> StaticString:
        """The codec's type name, as a chain prints it."""
        comptime full = reflect[Self].name()
        return full[byte = full.rfind(".") + 1 :]

    @staticmethod
    def takes[T: DType]() -> Bool:
        """Whether the codec takes `T` values: every number, unless it says
        otherwise."""
        return T.is_numeric()

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        """Append the block `src` gives, reading it as many times as the
        codec needs."""
        ...

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        """Emit the `count` values `encode` wrote to `out`, in order, and
        hand it back. Raises `CorruptError` on input `encode` could not have
        written."""
        ...


trait Elementwise(Codec):
    """A codec that maps each value to one value of the same width, given a
    running state that starts at `init` of the block's first two values --
    which the stream stores. A `Cascade` maps each value as the next codec
    reads it, and back as the next codec decodes it, so a block is never
    stored in between."""

    @staticmethod
    def init[T: DType](first: Scalar[T], second: Scalar[T]) -> UInt64:
        return 0

    @staticmethod
    def forward[
        T: DType
    ](mut state: UInt64, v: Scalar[T]) -> Scalar[Self.Out[T]]:
        ...

    @staticmethod
    def inverse[
        T: DType
    ](mut state: UInt64, v: Scalar[Self.Out[T]]) -> Scalar[T]:
        ...


trait Source(Deinitable, Movable, Sized):
    """Where a codec's values come from as it encodes them: a block of
    values, or one mapped as it is read. A codec that reads its values more
    than once rewinds."""

    def rewind(mut self):
        ...

    def next[U: DType](mut self) -> Scalar[U]:
        ...

    def next8[U: DType](mut self) -> SIMD[U, 8]:
        var v = SIMD[U, 8](0)
        comptime for lane in range(8):
            v[lane] = self.next[U]()
        return v

    def collect[U: DType](mut self) -> List[Scalar[U]]:
        """Every value, for a codec that needs them at hand."""
        self.rewind()
        var out = List[Scalar[U]](capacity=len(self))
        for _ in range(len(self)):
            out.append(self.next[U]())
        return out^


struct Values[T: DType, o: ImmOrigin](Source):
    """A block of values, read as they are -- or as any type of their width,
    their bits unchanged."""

    var values: Span[Scalar[Self.T], Self.o]
    var at: Int

    def __init__(out self, values: Span[Scalar[Self.T], Self.o]):
        self.values = values
        self.at = 0

    def __len__(self) -> Int:
        return len(self.values)

    def rewind(mut self):
        self.at = 0

    @always_inline
    def next[U: DType](mut self) -> Scalar[U]:
        # A codec reads at most `len` values a pass; unchecked, the bound is
        # not re-proven through every source wrapped around this one.
        debug_assert(self.at < len(self.values), "Values: read past the block")
        var v = self.values.unsafe_get(self.at)
        self.at += 1
        return bitcast[U](v)


trait Emitter(Deinitable, Movable):
    """Where a codec's decoded values go, as it decodes them: appended to a
    list, or mapped back first."""

    def reserve(mut self, extra: Int):
        ...

    def emit[U: DType](mut self, v: Scalar[U]):
        ...

    def emit8[U: DType](mut self, v: SIMD[U, 8]):
        comptime for lane in range(8):
            self.emit(v[lane])


struct Append[T: DType](Emitter):
    """Appends each value -- of `T` or any type of its width, as its bits --
    to a list it owns until `take`."""

    var out: List[Scalar[Self.T]]

    def __init__(out self, var out: List[Scalar[Self.T]]):
        self.out = out^

    def take(deinit self) -> List[Scalar[Self.T]]:
        return self.out^

    def reserve(mut self, extra: Int):
        # Geometric: a column goes through block after block into one list,
        # and reserving exactly what each block needs would copy the whole
        # list every time.
        var need = len(self.out) + extra
        if need > self.out.capacity():
            self.out.reserve(max(need, 2 * self.out.capacity()))

    @always_inline
    def emit[U: DType](mut self, v: Scalar[U]):
        self.out.append(bitcast[Self.T](v))

    @always_inline
    def emit8[U: DType](mut self, v: SIMD[U, 8]):
        var w = bitcast[Self.T, 8](v)
        comptime for lane in range(8):
            self.out.append(w[lane])


struct Mapped[C: Elementwise, T: DType, S: Source](Source):
    """`src` through `C.forward` as it is read. It holds `src` itself, not a
    pointer to it, so a chain of them keeps its state in registers."""

    var src: Self.S
    var start: UInt64
    var state: UInt64

    def __init__(out self, var src: Self.S, mut out: List[UInt8]):
        """Start the map at the block's first two values, and write the
        state it starts from to `out`."""
        src.rewind()
        var first = src.next[Self.T]() if len(src) > 0 else Scalar[Self.T](0)
        var second = src.next[Self.T]() if len(src) > 1 else first
        src.rewind()
        self.start = Self.C.init(first, second)
        Leb128.write(out, self.start)
        self.state = self.start
        self.src = src^

    def __len__(self) -> Int:
        return len(self.src)

    def rewind(mut self):
        self.src.rewind()
        self.state = self.start

    @always_inline
    def next[U: DType](mut self) -> Scalar[U]:
        var v = self.src.next[Self.T]()
        return rebind[Scalar[U]](Self.C.forward[Self.T](self.state, v))


struct Unmapped[C: Elementwise, T: DType, E: Emitter](Emitter):
    """Each value emitted, through `C.inverse`, to `out` -- which it holds
    until `take`, rather than pointing at it."""

    var out: Self.E
    var state: UInt64

    def __init__(out self, var out: Self.E, mut src: Decoder[_]) raises:
        """Start from the state `Mapped` wrote, read from `src`."""
        self.state = src.varint()
        self.out = out^

    def take(deinit self) -> Self.E:
        return self.out^

    def reserve(mut self, extra: Int):
        self.out.reserve(extra)

    @always_inline
    def emit[U: DType](mut self, v: Scalar[U]):
        var x = rebind[Scalar[Self.C.Out[Self.T]]](v)
        self.out.emit(Self.C.inverse[Self.T](self.state, x))


struct Decoder[origin: ImmOrigin](Movable):
    """A read position over an encoded stream. Every read checks the range
    and raises `CorruptError` rather than read past the end."""

    var data: Span[UInt8, Self.origin]
    var pos: Int

    def __init__(out self, data: Span[UInt8, Self.origin]):
        self.data = data
        self.pos = 0

    def _remaining(self) -> Int:
        return len(self.data) - self.pos

    def need(self, n: Int) raises CorruptError:
        if n < 0 or n > self._remaining():
            raise CorruptError(
                t"codecs: need {n} bytes at offset {self.pos}, have"
                t" {self._remaining()}"
            )

    def expect_end(self) raises CorruptError:
        """The stream holds exactly one column."""
        if self._remaining() != 0:
            raise CorruptError(
                t"codecs: {self._remaining()} bytes left after the column"
            )

    def byte(mut self) raises CorruptError -> UInt8:
        self.need(1)
        var b = self.data[self.pos]
        self.pos += 1
        return b

    def varint(mut self) raises CorruptError -> UInt64:
        var v, p = Leb128.read(self.data, self.pos)
        self.pos = p
        return v

    def length(mut self) raises CorruptError -> Int:
        """A varint that counts something, so it must fit an `Int`."""
        var v = self.varint()
        if v > UInt64(Int.MAX):
            raise CorruptError(t"codecs: count {v} out of range")
        return Int(v)

    def block(mut self) raises CorruptError -> Tuple[Int, Int]:
        """A block's count and the position its bytes end at -- (0, here)
        for the empty block that ends a stream."""
        var count = self.length()
        if count == 0:
            return (0, self.pos)
        var size = Int(self.fixed[DType.uint32]())
        self.need(size)
        return (count, self.pos + size)

    def end_block(self, end: Int) raises CorruptError:
        if self.pos != end:
            raise CorruptError(
                t"codecs: a block ended at byte {self.pos}, its length says"
                t" {end}"
            )

    def count(self) raises CorruptError -> Int:
        """The values left in the stream, counted by skipping its blocks."""
        var scan = Decoder(self.data)
        scan.pos = self.pos
        var total = 0
        while True:
            var count, end = scan.block()
            if count == 0:
                return total
            total += count
            scan.pos = end

    def fixed[T: DType](mut self) raises CorruptError -> Scalar[T]:
        """A little-endian `T`."""
        comptime n = size_of[Scalar[T]]()
        self.need(n)
        var v = LittleEndian.fixed[Bits.unsigned[T]](self.data, self.pos)
        self.pos += n
        return bitcast[T](v)
