# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Codecs that take byte strings apart: `Lengths` and `Prefixes`."""

from ..errors import CorruptError
from ..views import BufferView
from .core import Cascade, Codec, Encoded
from ..compress.lz77 import match_length
from .stores import Plain


struct Lengths[Bytes: Codec = Plain](Codec):
    """Byte strings as their lengths, `int32`s; their bytes end to end are
    the `Bytes` slot. Parquet's DELTA_LENGTH_BYTE_ARRAY is `Lengths`, then
    DELTA_BINARY_PACKED's codecs.

        ["ab", "abc"]  ->  [2, 3], bytes 61 62 61 62 63
    """

    comptime BytesOut: DType = DType.int32
    comptime SLOTS = 1
    comptime BYTES = 0
    """The slot of the bytes."""

    var _bytes: Self.Bytes
    var end: Int
    """Where the next string decoded starts in the data."""
    var last: Int
    """Where the decoded bytes end in the data."""
    var bad: Bool
    """Whether a length was negative or ran past them."""

    @staticmethod
    def node(var bytes: Encoded) -> Encoded:
        """`Lengths`' node in a cascade's tree: `bytes`, the node of its
        `Bytes` slot."""
        return Encoded(0, [bytes^])

    @staticmethod
    def bytes(
        node: Encoded,
    ) -> Span[UInt8, origin_of(node.children[Self.BYTES].buffer)]:
        """The byte strings' bytes `node` holds, end to end, read where they
        lie: the `Bytes` slot keeps them as they are."""
        comptime assert (
            Self.Bytes.STORES_VALUES
        ), "the Bytes slot does not keep its values as they are"
        return node[Self.BYTES].bytes()

    def __init__(out self):
        self = Self(Self.Bytes())

    def __init__(out self, var bytes: Self.Bytes):
        self._bytes = bytes^
        self.end = 0
        self.last = 0
        self.bad = False

    def forward(
        mut self,
        offsets: Span[Int32, _],
        data: Span[UInt8, _],
        dst: BufferView[mut=True, DType.int32, _],
    ) raises:
        comptime W = 16
        var ends = BufferView(offsets)
        var i = 0
        while i + W <= len(dst):
            dst.store[W](i, ends.load[W](i + 1) - ends.load[W](i))
            i += W
        while i < len(dst):
            dst.store[1](i, ends.load[1](i + 1) - ends.load[1](i))
            i += 1

    def finish(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        var n = len(offsets) - 1
        var lo = Int(offsets[0]) if n > 0 else 0
        var hi = Int(offsets[n]) if n > 0 else 0
        return Self.node(self._bytes.encode_view(BufferView(data[lo:hi])))

    def start(mut self, ref node: Encoded, mut data: List[UInt8]) raises:
        # The bytes go where they belong at once; each block then only
        # says where its strings end.
        self.end = len(data)
        data.resize(unsafe_uninit_length=self.end + node[Self.BYTES].count)
        self._bytes.decode_into[DType.uint8](
            node[Self.BYTES], BufferView(Span(data)[self.end :])
        )
        self.last = len(data)
        self.bad = False

    def decode(
        mut self,
        src: BufferView[DType.int32, _],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        var base = len(offsets)
        offsets.resize(unsafe_uninit_length=base + len(src))
        var out = BufferView(Span(offsets))
        # Checked once the block is done: with no length negative, the
        # ends only grow, so the last end within the bytes means all are.
        var end = self.end
        var signs = Int32(0)
        for i in range(len(src)):
            var length = src.unsafe_get(i)
            signs |= length
            end += Int(length)
            out.unsafe_set(base + i, Int32(end))
        self.end = end
        self.bad |= signs < 0 or end > self.last

    def check(self) raises:
        if self.bad or self.end != self.last:
            raise CorruptError("codecs: lengths do not cover the bytes")


struct Prefixes[Suffixes: Codec = Cascade[Lengths[], Plain]](Codec):
    """Byte strings as the length of the prefix each shares with the one
    before, `int32`s; what follows the prefix of each, its suffix, is the
    `Suffixes` slot, byte strings again.

        ["ab", "abc"]  ->  [0, 2], suffixes ["ab", "c"]
    """

    comptime BytesOut: DType = DType.int32
    comptime SLOTS = 1
    comptime SUFFIXES = 0
    """The slot of the suffixes."""

    var _suffixes: Self.Suffixes
    var ends: List[Int32]
    """Where each suffix ends in `bytes`, after a 0."""
    var bytes: List[UInt8]
    """The suffixes' bytes."""
    var lo: Int
    """Where the string before starts in the data."""
    var hi: Int
    """Where it ends."""
    var taken: Int
    """Suffixes decoded."""
    var bad: Bool
    """Whether a prefix was longer than the string before, or a string had
    no suffix."""

    @staticmethod
    def node(var suffixes: Encoded) -> Encoded:
        """`Prefixes`' node in a cascade's tree: `suffixes`, the node of its
        `Suffixes` slot."""
        return Encoded(0, [suffixes^])

    @staticmethod
    def suffixes(node: Encoded) -> ref[node.children[Self.SUFFIXES]] Encoded:
        """The node of the suffixes `node` holds."""
        return node[Self.SUFFIXES]

    def __init__(out self):
        self = Self(Self.Suffixes())

    def __init__(out self, var suffixes: Self.Suffixes):
        self._suffixes = suffixes^
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self.lo = 0
        self.hi = 0
        self.taken = 0
        self.bad = False

    def forward(
        mut self,
        offsets: Span[Int32, _],
        data: Span[UInt8, _],
        dst: BufferView[mut=True, DType.int32, _],
    ) raises:
        if len(self.ends) == 1:
            # The suffixes take at most the strings' bytes.
            self.bytes.resize(unsafe_uninit_length=len(data))
        var ends = BufferView(offsets)
        var out = Span(self.bytes)
        var at = Int(self.ends[len(self.ends) - 1])
        for i in range(len(dst)):
            var lo = Int(ends.unsafe_get(i))
            var hi = Int(ends.unsafe_get(i + 1))
            var most = min(self.hi - self.lo, hi - lo)
            var p = match_length(data, lo, self.lo, lo + most)
            dst.store[1](i, Int32(p))
            out[at : at + hi - lo - p].copy_from(data[lo + p : hi])
            at += hi - lo - p
            self.ends.append(Int32(at))
            self.lo = lo
            self.hi = hi

    def finish(
        mut self, offsets: Span[Int32, _], data: Span[UInt8, _]
    ) raises -> Encoded:
        self.bytes.resize(
            unsafe_uninit_length=Int(self.ends[len(self.ends) - 1])
        )
        var node = Self.node(
            self._suffixes.encode(Span(self.ends), Span(self.bytes))
        )
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self.lo = 0
        self.hi = 0
        return node^

    def start(mut self, ref node: Encoded, mut data: List[UInt8]) raises:
        self.ends = [Int32(0)]
        self.bytes = List[UInt8]()
        self._suffixes.decode(node[Self.SUFFIXES], self.ends, self.bytes)
        self.lo = len(data)
        self.hi = self.lo
        self.taken = 0
        self.bad = False

    def decode(
        mut self,
        src: BufferView[DType.int32, _],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        var ends = BufferView(Span(self.ends))
        var bytes = BufferView(Span(self.bytes))
        var suffixes = len(self.ends) - 1
        for i in range(len(src)):
            var p = Int(src.unsafe_get(i))
            if (
                self.bad
                or self.taken >= suffixes
                or p < 0
                or p > self.hi - self.lo
            ):
                self.bad = True
                offsets.append(Int32(len(data)))
                continue
            var lo = Int(ends.unsafe_get(self.taken))
            var hi = Int(ends.unsafe_get(self.taken + 1))
            var start = len(data)
            data.resize(unsafe_uninit_length=start + p + hi - lo)
            # The prefix is the start of the string before, which ends where
            # this one starts, so the two never overlap.
            var out = BufferView(Span(data))
            for k in range(p):
                out.unsafe_set(start + k, out.unsafe_get(self.lo + k))
            out.as_span()[start + p : start + p + hi - lo].copy_from(
                bytes.as_span()[lo:hi]
            )
            self.taken += 1
            offsets.append(Int32(len(data)))
            self.lo = start
            self.hi = len(data)

    def check(self) raises:
        if self.bad or self.taken != len(self.ends) - 1:
            raise CorruptError(
                "codecs: prefixes do not match the strings before, or the"
                " suffixes"
            )
