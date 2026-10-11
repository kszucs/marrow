# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The codec trait, the cascade that chains codecs, and the tree a cascade
stores a column as.

Every codec of a `Cascade` but the last is a *lane*: it turns each value
into one value, and may keep parameters per frame of values, such as the
smallest of them. The last codec is the *store*, which holds the values'
bytes. Encoding passes a block of `BLOCK` values from codec to codec, in
the L1 cache. Decoding is a single loop: the store loads a register of
values, each lane in turn decodes it, and it is written out -- nothing is
written down between the codecs.

What a codec makes besides its values -- the widths, the smallest values,
a dictionary -- is a *slot*, encoded by a cascade of its own, named by a
type parameter: `BitPack[Widths=Plain]`. A column is stored as an
`Encoded` tree: the cascade's node holds how many values there are and a
child per codec; a codec's node holds its bytes and a child per slot.

What codecs share is here too: `Frames` and `FrameValues`, a parameter a
frame; `Exceptions`, values set apart; and `Unpacked`, the block a store
decodes at a time.
"""

from std.bit import count_trailing_zeros
from std.sys import size_of

from ..arrays import PrimitiveArray
from ..buffers import Buffer
from ..dtypes import PrimitiveType, _FloatingType, _IntegerType
from ..errors import CorruptError, InvalidError
from ..utils.integers import Unsigned
from ..views import BufferView


comptime BLOCK = 1024
"""The values a cascade carries through its codecs at once: a block, in
each codec's type, stays in the L1 cache from codec to codec."""


comptime ArrowOf[D: DType] = _IntegerType[D] if D.is_integral() else (
    _FloatingType[D]
)
"""The Arrow type whose values are `D`s: what a slot of `D`s is typed as."""


struct Encoded(Copyable, Movable):
    """What a codec stored: its bytes, and the nodes of the codecs it holds
    -- a cascade's members, or a codec's slots. A column's node holds
    `count`, its values; a cascade's members take their cascade's. A
    container frames the bytes as it likes and builds the same tree to
    decode them."""

    var count: Int
    var buffer: Buffer[mut=False]
    var start: Int
    """The byte of `buffer` the node's bytes start at."""
    var size: Int
    """How many bytes are the node's."""
    var children: List[Encoded]

    def __init__(out self):
        """No values and no bytes: the node of a codec that stores
        nothing."""
        self = Self(0, List[Encoded]())

    def __init__(out self, count: Int, var children: List[Encoded]):
        """A node with no bytes: `count` values, and `children`."""
        self.count = count
        self.buffer = Buffer.alloc_uninit(0).to_immutable()
        self.start = 0
        self.size = 0
        self.children = children^

    def __init__(
        out self,
        var buffer: Buffer[mut=False],
        size: Int,
        var children: List[Encoded],
    ):
        """A codec's node: the first `size` bytes of `buffer`, and a child
        per slot."""
        self.count = 0
        self.buffer = buffer^
        self.start = 0
        self.size = size
        self.children = children^

    def __init__(out self, var buffer: Buffer[mut=False], size: Int):
        """A codec's node with no slots: the first `size` bytes of
        `buffer`."""
        self = Self(buffer^, size, List[Encoded]())

    def __init__(out self, *, bytes: Span[UInt8, _]):
        """A codec's node holding a copy of `bytes`."""
        var buffer = Buffer.alloc_uninit(len(bytes))
        buffer.extend(BufferView(bytes), 0, len(bytes))
        self = Self(buffer^.to_immutable(), len(bytes))

    def __init__[A: PrimitiveType](out self, array: PrimitiveArray[A]):
        """`Plain`'s node of `array`'s values, sharing its buffer."""
        comptime S = size_of[Scalar[A.native]]()
        self.count = len(array)
        self.buffer = array.buffer
        self.start = array.offset * S
        self.size = len(array) * S
        self.children = List[Encoded]()

    def __init__[T: DType](out self, values: Span[Scalar[T], _]):
        """`Plain`'s node of `values`, a copy."""
        var n = len(values)
        var buffer = Buffer.alloc_uninit[T](n)
        buffer.extend(BufferView(values), 0, n)
        self = Self(buffer^.to_immutable(), n * size_of[Scalar[T]]())
        self.count = n

    def __deinit__(deinit self):
        # Spelled out: the node holds a `List` of nodes.
        pass

    def __getitem__(self, i: Int) -> ref[self.children[i]] Encoded:
        return self.children[i]

    @always_inline
    def bytes(self) -> Span[UInt8, origin_of(self.buffer)]:
        return self.buffer.view[DType.uint8](self.start, self.size).as_span()

    @always_inline
    def values[T: DType](self) -> BufferView[T, origin_of(self.buffer)]:
        """The bytes as `T`s, read unaligned: `start` may be any byte."""
        return BufferView[T, origin_of(self.buffer)].from_bytes(self.bytes())

    def array[A: PrimitiveType](self, dtype: A) raises -> PrimitiveArray[A]:
        """`Plain`'s node as an array of `dtype`: over the buffer itself when
        `start` falls on a whole value, a copy otherwise."""
        comptime T = A.native
        comptime S = size_of[Scalar[T]]()
        var n = self.count
        if self.size != n * S:
            raise CorruptError(t"codecs: {self.size} bytes for {n} {T} values")
        if self.start % S == 0:
            return PrimitiveArray[A](
                dtype,
                length=n,
                nulls=0,
                offset=self.start // S,
                bitmap=None,
                buffer=self.buffer,
            )
        var copy = Buffer.alloc_uninit[T](n)
        copy.extend(BufferView(self.bytes()), 0, self.size)
        return PrimitiveArray[A](
            dtype,
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=copy^.to_immutable(),
        )


struct Frames(Copyable, Movable):
    """How many values share one set of a codec's parameters -- one width,
    one smallest value -- and which frame a value is in. Decoding takes any
    size of whole `step`s: a lane's `step` is 64, so a register of values
    never spans two frames. Encoding takes the sizes that divide `BLOCK`,
    so a block holds whole frames."""

    var size: Int
    """The values a frame holds: the size given, or `BLOCK` for 0."""
    var step: Int
    var shift: Int
    """`size` as a power of two, or -1 when it is not one."""

    def __init__(out self, size: Int, step: Int):
        self.size = size if size != 0 else BLOCK
        self.step = step
        var f = self.size
        self.shift = (
            count_trailing_zeros(f) if f > 0 and f & (f - 1) == 0 else -1
        )

    def over[encoding: Bool](self, count: Int) raises InvalidError -> Int:
        """The frames `count` values take; refuse a size these values
        cannot be cut into."""
        var f = self.size
        if f < 0 or f % self.step != 0 or (encoding and BLOCK % f != 0):
            raise InvalidError(t"codecs: frames of {f} values")
        return (count + f - 1) // f

    @always_inline
    def of(self, i: Int) -> Int:
        """The frame value `i` is in."""
        if self.shift >= 0:
            return i >> self.shift
        return i // self.size


struct FrameValues[S: Codec](Movable, Sized):
    """One value per frame of a codec's values -- a smallest value, a
    divisor, a width -- stored by the codec's slot: put a frame at a time
    while encoding; while decoding, looked up by the position of a value."""

    var frames: Frames
    var slot: Self.S
    var values: Buffer[mut=True]
    """The values put while encoding."""
    var table: Buffer[mut=False]
    """The values decoded, from `first` on."""
    var first: Int
    var length: Int
    """How many have been put, or decoded."""

    def __init__(out self, size: Int, step: Int, var slot: Self.S):
        self.frames = Frames(size, step)
        self.slot = slot^
        self.values = Buffer.alloc_uninit(0)
        self.table = Buffer.alloc_uninit(0).to_immutable()
        self.first = 0
        self.length = 0

    def start[T: DType](mut self, count: Int) raises:
        """Before `count` values are encoded, a frame's value put at a
        time."""
        self.values = Buffer.alloc_uninit[T](self.frames.over[True](count))
        self.length = 0

    @always_inline
    def put[T: DType](mut self, value: Scalar[T]):
        self.values.view[T]().unsafe_set(self.length, value)
        self.length += 1

    def finish[T: DType](mut self) raises -> Encoded:
        var values = self.values^.to_immutable()
        self.values = Buffer.alloc_uninit(0)
        return self.slot.encode_view(
            BufferView(values.view[T](0, self.length).as_span())
        )

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        """Before `count` values are decoded, their frames' values from
        `node`."""
        var frames = self.frames.over[False](count)
        if node.count != frames:
            raise CorruptError(
                t"codecs: {node.count} values for {frames} frames"
            )
        var table = self.slot.decode(node, ArrowOf[T]())
        self.table = table.buffer
        self.first = table.offset
        self.length = len(table)

    def __len__(self) -> Int:
        """The frames decoded."""
        return self.length

    @always_inline
    def get[T: DType](self, frame: Int) -> Scalar[T]:
        return self.table.unsafe_get[T](self.first + frame)

    @always_inline
    def at[T: DType](self, i: Int) -> Scalar[T]:
        """The value of the frame value `i` is in."""
        return self.get[T](self.frames.of(i))


struct Exceptions[Positions: Codec, Values: Codec](Movable):
    """Values set apart and where they were, two slots of the codec that
    sets them apart: each one's position in the column, ascending, and its
    value."""

    var positions: Self.Positions
    var values: Self.Values
    var found: List[UInt64]
    """Each exception's position."""
    var kept: Buffer[mut=True]
    """Each exception's value, while encoding."""
    var decoded: Encoded
    """Each exception's value, decoded."""
    var before: Int
    """Values encoded before the block being encoded."""
    var next: Int
    """The exception to put back next."""
    var ahead: Int
    """Its position; `Int.MAX` past the last."""

    def __init__(
        out self, var positions: Self.Positions, var values: Self.Values
    ):
        self.positions = positions^
        self.values = values^
        self.found = List[UInt64]()
        self.kept = Buffer.alloc_uninit(0)
        self.decoded = Encoded()
        self.before = 0
        self.next = 0
        self.ahead = Int.MAX

    # --- encoding ---

    def start[T: DType](mut self, count: Int):
        self.found = List[UInt64]()
        self.kept = Buffer.alloc_uninit[T](count)
        self.before = 0

    @always_inline
    def keep[T: DType](mut self, i: Int, value: Scalar[T]):
        """Set value `i` of the block being encoded apart."""
        self.kept.view[T]().unsafe_set(len(self.found), value)
        self.found.append(UInt64(self.before + i))

    def count(self) -> Int:
        """The values set apart so far."""
        return len(self.found)

    def fill[
        T: DType
    ](self, since: Int, dst: BufferView[mut=True, T, _], value: Scalar[T]):
        """Write `value` where each value set apart after the first `since`
        was, in `dst`, the block being encoded."""
        for k in range(since, len(self.found)):
            dst.unsafe_set(Int(self.found[k]) - self.before, value)

    def passed(mut self, n: Int):
        """The block being encoded, of `n` values, is done."""
        self.before += n

    def finish[T: DType](mut self, mut slots: List[Encoded]) raises:
        """Append the two slots' nodes to `slots`."""
        var kept = self.kept^.to_immutable()
        self.kept = Buffer.alloc_uninit(0)
        var n = len(self.found)
        slots.append(self.positions.encode_view(BufferView(Span(self.found))))
        slots.append(
            self.values.encode_view(BufferView(kept.view[T](0, n).as_span()))
        )

    # --- decoding ---

    def start[
        T: DType
    ](mut self, ref positions: Encoded, ref values: Encoded, count: Int) raises:
        var n = positions.count
        var stored = values.count
        if stored != n:
            raise CorruptError(t"codecs: {stored} exceptions at {n} positions")
        self.found = List[UInt64](length=n, fill=0)
        self.positions.decode_into[DType.uint64](
            positions, BufferView(Span(self.found))
        )
        var last = -1
        for p in self.found:
            if Int(p) <= last or Int(p) >= count:
                raise CorruptError(
                    t"codecs: an exception at {p}, after {last} of {count}"
                )
            last = Int(p)
        self.decoded = self.values.held[T](values)
        self.next = 0
        self.ahead = Int(self.found[0]) if n > 0 else Int.MAX

    @always_inline
    def put[T: DType, W: Int](mut self, at: Int, v: SIMD[T, W]) -> SIMD[T, W]:
        """`v`, the values from position `at`, with the exceptions among
        them put back."""
        if self.ahead >= at + W:
            return v
        var out = v
        var kept = self.decoded.values[T]()
        while self.ahead < at + W:
            out[self.ahead - at] = kept.unsafe_get(self.next)
            self.next += 1
            self.ahead = (
                Int(self.found[self.next]) if self.next
                < len(self.found) else Int.MAX
            )
        return out


struct Unpacked(Movable):
    """The block of values a store decodes a block at a time -- runs, or
    values of as many bits as each needs -- for the registers to load: the
    values from `first` on, with a cache line to spare past them."""

    var bytes: List[UInt8]
    var first: Int
    var count: Int
    """Values in the column."""

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.first = -BLOCK
        self.count = 0

    def start(mut self, count: Int):
        self.bytes.resize(8 * BLOCK + 64, 0)
        self.first = -BLOCK
        self.count = count

    @always_inline
    def due(self, i: Int) -> Bool:
        """Whether value `i` is past the block held."""
        return i >= self.first + BLOCK

    def next[T: DType](mut self) -> BufferView[T, origin_of(self.bytes)]:
        """The block after the one held, to write."""
        self.first += BLOCK
        return BufferView[T, origin_of(self.bytes)].from_bytes(Span(self.bytes))

    def size(self) -> Int:
        """The values of the block held."""
        return min(BLOCK, self.count - self.first)

    @always_inline
    def load[T: DType, W: Int](self, i: Int) -> SIMD[T, W]:
        return (
            BufferView[T, origin_of(self.bytes)]
            .from_bytes(Span(self.bytes))
            .load[W](i - self.first)
        )


trait Codec(Defaultable, Deinitable, Movable):
    """A codec: a lane, which turns each value into one value, or a store,
    which holds values' bytes -- a `Cascade` of lanes ending in a store is
    a store too. A codec writes the hooks of its role and leaves the rest to
    the defaults, which refuse, at compile time, a codec put in a role it
    has no hooks for. A store encodes and decodes a column on its own:
    `Plain().encode_view(BufferView(values))`.

    |                | encoding            | decoding             |
    |----------------|---------------------|----------------------|
    | the column     | `start`, `finish`   | `start`, `check`     |
    | a lane         | `forward`           | `backward`, a register |
    | the store      | `store`, a block    | `load`, a register   |
    | byte strings   | `forward`, `finish` | `start`, `decode`    |

    A lane's `forward` takes a block, or a register, which the block's
    default loops over. `backward` and `load` take the position of the
    register's first value in the column, and never raise: what they find
    wrong they note, and `check` raises it once the column is decoded. Byte
    strings are taken by the first codec of a cascade only.
    """

    comptime Out[T: DType]: DType = T
    """What a lane makes of values of `T`."""

    comptime BytesOut: DType = DType.uint8
    """What the first codec makes of byte strings."""

    comptime SLOTS: Int = 0
    """How many slots the codec's node holds."""

    comptime STORES_VALUES: Bool = False
    """Whether the store's bytes are its values as they are, so a slot it
    stores alone is read where it lies."""

    # --- encoding ---

    def start[T: DType](mut self, count: Int) raises:
        """Before `count` values of `T` are encoded."""
        pass

    def forward[
        T: DType
    ](
        mut self,
        src: BufferView[T, _],
        dst: BufferView[mut=True, Self.Out[T], _],
    ) raises:
        """A lane's block: `src` into `dst`, a register at a time."""
        comptime W = 64 // size_of[Scalar[T]]()
        var n = len(src)
        var i = 0
        while i + W <= n:
            dst.store[W](i, self.forward[T, W](src.load[W](i)))
            i += W
        while i < n:
            dst.store[1](i, self.forward[T, 1](src.load[1](i)))
            i += 1

    @always_inline
    def forward[
        T: DType, W: Int
    ](mut self, v: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        """A lane's register of values."""
        comptime assert W < 0, "the codec is no lane: put it last"
        return 0

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        """The store's block."""
        comptime assert size_of[Scalar[T]]() < 0, "end the cascade with a store"

    def finish[T: DType](mut self) raises -> Encoded:
        """The codec's node, once every value is encoded."""
        return Encoded()

    # --- decoding ---

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        """Before `count` values of `T` are decoded from `node`: decode the
        slots and refuse what does not add up."""
        pass

    @always_inline
    def backward[
        T: DType, W: Int
    ](mut self, i: Int, v: SIMD[Self.Out[T], W]) -> SIMD[T, W]:
        """A lane's register of values, from value `i`."""
        comptime assert W < 0, "the codec is no lane: put it last"
        return 0

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        """The store's register of values, from value `i`."""
        comptime assert W < 0, "end the cascade with a store"
        return 0

    def check(self) raises:
        """Raise what decoding noted wrong."""
        pass

    # --- a column, by a store ---

    @always_inline
    def encode[
        A: PrimitiveType
    ](mut self, array: PrimitiveArray[A]) raises -> Encoded:
        """`array`'s values, read where they lie; an array with nulls is
        refused -- a container sets them apart, or `Nullable` does."""
        if array.nulls != 0:
            raise InvalidError(
                t"codecs: {array.nulls} nulls; encode the values present"
            )
        return self.encode_view[A.native](array.values())

    @always_inline
    def decode[
        A: PrimitiveType
    ](mut self, ref node: Encoded, dtype: A) raises -> PrimitiveArray[A]:
        """The values `node` holds, as an array of `dtype`: over the node's
        own buffer when the store keeps them as they are."""
        comptime T = A.native
        comptime if Self.STORES_VALUES:
            self.start[T](node, node.count)
            return node.array(dtype)
        else:
            var n = node.count
            var buffer = Buffer.alloc_uninit[T](n)
            self.decode_into[T](node, buffer.view[T](0, n))
            return PrimitiveArray[A](
                dtype,
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buffer^.to_immutable(),
            )

    def encode_view[
        T: DType
    ](mut self, src: BufferView[T, _]) raises -> Encoded:
        """`src`'s values, a block at a time."""
        var n = len(src)
        self.start[T](n)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            self.store[T](src[lo : lo + m])
            lo += m
        var node = self.finish[T]()
        node.count = n
        return node^

    def decode_into[
        T: DType
    ](mut self, ref node: Encoded, dst: BufferView[mut=True, T, _]) raises:
        """The values `node` holds, into `dst`, which holds `node.count` --
        an array's buffer at an offset, say: one loop, a cache line of
        values at a time."""
        var n = node.count
        if len(dst) != n:
            raise InvalidError(t"codecs: {n} values decoded into {len(dst)}")
        if len(node.children) != Self.SLOTS:
            raise CorruptError(
                t"codecs: a node of {len(node.children)} slots, not"
                t" {Self.SLOTS}"
            )
        self.start[T](node, n)
        comptime W = 64 // size_of[Scalar[T]]()
        var i = 0
        while i + W <= n:
            dst.store[W](i, self.load[T, W](i))
            i += W
        while i < n:
            dst.store[1](i, self.load[T, 1](i))
            i += 1
        self.check()

    def held[T: DType](mut self, ref node: Encoded) raises -> Encoded:
        """The values `node` holds, as `Plain`'s node: `node` itself when
        the store keeps them as they are, or else decoded into one."""
        comptime if Self.STORES_VALUES:
            self.start[T](node, node.count)
            return node.copy()
        else:
            var n = node.count
            var buffer = Buffer.alloc_uninit[T](n)
            self.decode_into[T](node, buffer.view[T](0, n))
            var out = Encoded(buffer^.to_immutable(), n * size_of[Scalar[T]]())
            out.count = n
            return out^

    def encode(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        """Byte strings: string `i` is `data[offsets[i] : offsets[i + 1]]`
        -- by a cascade that starts with a codec taking them."""
        raise InvalidError("codecs: the codec stores no byte strings")

    def decode(
        mut self,
        ref node: Encoded,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """The byte strings `node` holds: their bytes appended to `data`,
        and where each ends to `offsets`."""
        raise InvalidError("codecs: the codec stores no byte strings")

    # --- byte strings, the first codec only ---

    def forward(
        mut self,
        offsets: Span[Int32, _],
        data: Span[UInt8, _],
        dst: BufferView[mut=True, Self.BytesOut, _],
    ) raises:
        """A block of byte strings into `dst`: string `i` is the bytes of
        `data` from `offsets[i]` to `offsets[i + 1]`."""
        raise InvalidError("codecs: the codec takes no byte strings")

    def finish(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        """The codec's node, once every byte string -- all of them in
        `offsets` and `data` -- is encoded; and a fresh start."""
        raise InvalidError("codecs: the codec takes no byte strings")

    def start(mut self, ref node: Encoded, mut data: List[UInt8]) raises:
        """Before byte strings are decoded from `node` onto the end of
        `data`."""
        raise InvalidError("codecs: the codec takes no byte strings")

    def decode(
        mut self,
        src: BufferView[Self.BytesOut, _],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """A block of byte strings from `src`: their bytes appended to
        `data`, and where each ends to `offsets`."""
        raise InvalidError("codecs: the codec takes no byte strings")


struct Cascade[*Ts: Codec](Codec):
    """Codecs one after another, every one but the last a lane and the last
    a store: itself a store. It holds its codecs and their state while it
    encodes or decodes a column.

        Cascade[Delta, Zigzag, BitPack[]]()
        Cascade((Delta(first), Ordered(), FrameOfReference(128), BitPack(32)))

    A codec with slots is written with brackets as a type -- `BitPack[]`,
    its slots stored plainly -- and without as a value.

    A cascade over byte strings starts with a codec that takes them --
    `Lengths`, `Prefixes`, `Dictionary` -- and goes on with what it makes
    of them.
    """

    comptime n = len(Self.Ts)
    comptime SLOTS = len(Self.Ts)

    var codecs: Tuple[*Self.Ts]

    def __init__(out self):
        self.codecs = Tuple[*Self.Ts]()

    def __init__(out self, var codecs: Tuple[*Self.Ts]):
        self.codecs = codecs^

    @staticmethod
    def _in[j: Int, k: Int, T: DType]() -> DType:
        """What codec `k` takes when codec `j` takes `T`."""
        comptime if k == j:
            return T
        else:
            return Self.Ts[k - 1].Out[Self._in[j, k - 1, T]()]

    @staticmethod
    def _widest[j: Int, T: DType]() -> Int:
        """The bytes of the widest value codecs `j` on take, codec `j`
        taking `T`: what a block of scratch holds."""
        var widest = 0
        comptime for k in range(j, Self.n):
            widest = max(widest, size_of[Scalar[Self._in[j, k, T]()]]())
        return widest

    # --- values, as a store ---

    def start[T: DType](mut self, count: Int) raises:
        comptime for k in range(Self.n):
            comptime I = Self._in[0, k, T]()
            self.codecs[k].start[I](count)

    def store[T: DType](mut self, src: BufferView[T, _]) raises:
        comptime if Self.n == 1:
            self.codecs[0].store[T](src)
        else:
            # A block between two codecs stays in two scratch blocks on the
            # stack, each lane writing the one the next codec reads.
            comptime O = Self.Ts[0].Out[T]
            comptime S = BLOCK * Self._widest[1, O]()
            var a = Array[UInt8, S](uninitialized=True)
            var b = Array[UInt8, S](uninitialized=True)
            var m = len(src)
            self.codecs[0].forward[T](
                src, BufferView[O].from_bytes(Span(a))[:m]
            )
            self._encode[1, O](m, Span(a), Span(b))

    def finish[T: DType](mut self) raises -> Encoded:
        return Encoded(0, self._finish[0, T]())

    def start[T: DType](mut self, ref node: Encoded, count: Int) raises:
        self._start[0, T](node, count)

    @always_inline
    def load[T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        return self._up[0, T, W](i)

    def check(self) raises:
        comptime for k in range(Self.n):
            self.codecs[k].check()

    # --- byte strings ---

    def encode(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        comptime assert Self.n > 1, "byte strings need a codec after the first"
        comptime B = Self.Ts[0].BytesOut
        var n = max(len(offsets) - 1, 0)
        comptime for k in range(1, Self.n):
            comptime I = Self._in[1, k, B]()
            self.codecs[k].start[I](n)
        comptime S = BLOCK * Self._widest[1, B]()
        var a = Array[UInt8, S](uninitialized=True)
        var b = Array[UInt8, S](uninitialized=True)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            self.codecs[0].forward(
                offsets[lo : lo + m + 1],
                data,
                BufferView[B].from_bytes(Span(a))[:m],
            )
            self._encode[1, B](m, Span(a), Span(b))
            lo += m
        var children = List[Encoded](capacity=Self.n)
        children.append(self.codecs[0].finish(offsets, data))
        children.extend(self._finish[1, B]())
        return Encoded(n, children^)

    def decode(
        mut self,
        ref node: Encoded,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        comptime assert Self.n > 1, "byte strings need a codec after the first"
        comptime B = Self.Ts[0].BytesOut
        var n = node.count
        self._start[1, B](node, n)
        Self._slots[0](node[0])
        self.codecs[0].start(node[0], data)
        offsets.reserve(len(offsets) + n)
        var scratch = Buffer.alloc_uninit[B](BLOCK)
        var lo = 0
        while lo < n:
            var m = min(BLOCK, n - lo)
            var values = scratch.view[B](0, m)
            self._fill[1](values, lo)
            self.codecs[0].decode(values, offsets, data)
            lo += m
        self.check()

    # --- the drivers ---

    def _encode[
        k: Int, T: DType
    ](
        mut self,
        m: Int,
        a: Span[mut=True, UInt8, _],
        b: Span[mut=True, UInt8, _],
    ) raises:
        """A block of `m` values of `T`, in `a`, through codecs `k` on: a
        lane writes `b`, which the codec after it reads."""
        comptime if k == Self.n - 1:
            self.codecs[k].store[T](BufferView[T].from_bytes(a)[:m])
        else:
            comptime O = Self.Ts[k].Out[T]
            self.codecs[k].forward[T](
                BufferView[T].from_bytes(a)[:m],
                BufferView[O].from_bytes(b)[:m],
            )
            self._encode[k + 1, O](m, b, a)

    def _finish[j: Int, T: DType](mut self) raises -> List[Encoded]:
        """The nodes of codecs `j` on, codec `j` having taken `T`."""
        var out = List[Encoded](capacity=Self.n - j)
        comptime for k in range(j, Self.n):
            comptime I = Self._in[j, k, T]()
            out.append(self.codecs[k].finish[I]())
        return out^

    def _start[
        j: Int, T: DType
    ](mut self, ref node: Encoded, count: Int) raises:
        """Codecs `j` on, codec `j` taking `T`, started on their nodes for
        `count` values."""
        if len(node.children) != Self.n:
            raise CorruptError(
                t"codecs: {len(node.children)} nodes for {Self.n} codecs"
            )
        comptime for k in range(j, Self.n):
            comptime I = Self._in[j, k, T]()
            Self._slots[k](node[k])
            self.codecs[k].start[I](node[k], count)

    @staticmethod
    def _slots[k: Int](ref node: Encoded) raises CorruptError:
        """Refuse codec `k`'s node without a child per slot."""
        comptime slots = Self.Ts[k].SLOTS
        if len(node.children) != slots:
            raise CorruptError(
                t"codecs: a node of {len(node.children)} slots, not {slots}"
            )

    def _fill[
        j: Int, T: DType
    ](mut self, dst: BufferView[mut=True, T, _], lo: Int):
        """The values from value `lo` through codecs `j` on into `dst`: one
        loop, a cache line of values at a time."""
        comptime W = 64 // size_of[Scalar[T]]()
        var m = len(dst)
        var i = 0
        while i + W <= m:
            dst.store[W](i, self._up[j, T, W](lo + i))
            i += W
        while i < m:
            dst.store[1](i, self._up[j, T, 1](lo + i))
            i += 1

    @always_inline
    def _up[k: Int, T: DType, W: Int](mut self, i: Int) -> SIMD[T, W]:
        """The `W` values from value `i`, through codecs `k` on, codec `k`
        taking `T`."""
        comptime if k == Self.n - 1:
            return self.codecs[k].load[T, W](i)
        else:
            return self.codecs[k].backward[T, W](
                i, self._up[k + 1, Self.Ts[k].Out[T], W](i)
            )
