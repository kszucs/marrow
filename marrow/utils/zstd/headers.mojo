# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The headers of a zstd frame, of its blocks, and of a compressed block's
two sections (`zstd_compression_format.md`, RFC 8878).

| Type | What it is |
|---|---|
| `FrameHeader` | a frame's descriptor, window, dictionary id and content size |
| `BlockHeader` | a block's last flag, type and size |
| `LiteralsHeader` | a literals section's type, streams, literal count and size |
| `SequencesHeader` | a sequences section's count and each field's table mode |

Each type both reads its header and writes it, so the decoder and the
encoder share one definition of every format.
"""

from std.bit import log2_floor

from ...errors import CorruptError, DynError, NotImplementedError
from .fse import Alphabet
from ...codecs.byteorder import LittleEndian


comptime BLOCK_MAX = 1 << 17
"""The most a block holds, compressed or not: 128 KiB."""


@fieldwise_init
struct FrameHeader(ImplicitlyCopyable, Movable):
    """What a frame's header says, after its magic number: whether the
    frame is one segment -- its window the whole content -- whether a
    checksum follows it, its window, and its content size, -1 when the
    header leaves it out."""

    var single: Bool
    var checksum: Bool
    var window: Int
    var content: Int

    @staticmethod
    def describing(content: Int, window: Int) -> Self:
        """The header of a frame of `content` bytes compressed within
        `window`: one segment when the window covers the content, and no
        checksum."""
        return Self(content <= window, False, window, content)

    @staticmethod
    def read(src: Span[UInt8, _], mut ip: Int) raises DynError -> Self:
        """The header at `ip`, which is moved past it. A frame that needs a
        dictionary raises `NotImplementedError`."""
        var n_in = len(src)
        if ip >= n_in:
            raise CorruptError("zstd: truncated frame header")
        var fhd = Int(src[ip])
        ip += 1
        var single = fhd & 0x20 != 0
        var checksum = fhd & 0x04 != 0
        if fhd & 0x08 != 0:
            raise CorruptError("zstd: reserved frame header bit set")
        var dict_size = (1 << (fhd & 3)) >> 1  # 0, 1, 2 or 4
        var fcs_flag = fhd >> 6
        var fcs_size = 1 << fcs_flag if fcs_flag > 0 else Int(single)
        var need = (0 if single else 1) + dict_size + fcs_size
        if n_in - ip < need:
            raise CorruptError("zstd: truncated frame header")
        var window = 0
        if not single:
            var wd = Int(src[ip])
            ip += 1
            var base = 1 << (10 + (wd >> 3))
            window = base + (base >> 3) * (wd & 7)
        if dict_size > 0:
            var id = Int(
                LittleEndian.partial[DType.uint32](src[ip : ip + dict_size], 0)
            )
            ip += dict_size
            if id != 0:
                raise NotImplementedError(
                    t"zstd: frames that need dictionary {id}"
                )
        var content = -1
        if fcs_size > 0:
            content = Int(
                LittleEndian.partial[DType.uint64](src[ip : ip + fcs_size], 0)
            )
            if fcs_size == 2:
                content += 256
            ip += fcs_size
        if single:
            window = content
        return Self(single, checksum, window, content)

    def write(self, mut dst: List[UInt8]):
        """Append the header: no dictionary id, the window as a power of two
        from 1 KiB unless the frame is one segment, and the content size in
        1 (one segment only), 2 (less 256), 4 or 8 bytes."""
        var n = self.content
        var code = Int(n >= 256) + Int(n >= 65536 + 256) + Int(n >= 0xFFFFFFFF)
        dst.append(
            UInt8(code << 6 | Int(self.single) << 5 | Int(self.checksum) << 2)
        )
        if not self.single:
            dst.append(UInt8((log2_floor(self.window) - 10) << 3))
        if code > 0 or self.single:
            LittleEndian.put_le(
                dst, UInt64(n - 256 if code == 1 else n), 1 << code
            )

    def block_max(self) -> Int:
        """The most a block of this frame may hold."""
        return min(self.window, BLOCK_MAX)


@fieldwise_init
struct BlockHeader(ImplicitlyCopyable, Movable):
    """A block's 3-byte header: whether it is the frame's last, its type,
    and its size -- the bytes it holds, or for an RLE block the bytes it
    repeats its one byte to."""

    var last: Bool
    var kind: Int
    var size: Int

    comptime RAW = 0
    comptime RLE = 1
    comptime COMPRESSED = 2

    comptime SIZE = 3
    """The header's own bytes."""

    @staticmethod
    def read(src: Span[UInt8, _], ip: Int) raises CorruptError -> Self:
        """The header at `ip`."""
        if len(src) - ip < Self.SIZE:
            raise CorruptError("zstd: truncated block header")
        var v = Int(LittleEndian.partial[DType.uint32](src[ip : ip + 3], 0))
        var kind = (v >> 1) & 3
        if kind == 3:
            raise CorruptError("zstd: reserved block type")
        return Self(v & 1 != 0, kind, v >> 3)

    def payload(self) -> Int:
        """The bytes after the header that belong to the block."""
        return 1 if self.kind == Self.RLE else self.size

    def _value(self) -> Int:
        return Int(self.last) | self.kind << 1 | self.size << 3

    def write(self, mut out: List[UInt8]):
        """Append the header."""
        LittleEndian.put_le(out, UInt64(self._value()), Self.SIZE)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the `SIZE` bytes reserved for it at `at`."""
        LittleEndian.put_le_at(out, at, UInt64(self._value()), Self.SIZE)


@fieldwise_init
struct LiteralsHeader(ImplicitlyCopyable, Movable):
    """A literals section's header: its type, how many Huffman streams, how
    many literals, the header's own bytes, and the bytes after it -- the
    literals themselves, their one repeated byte, or their streams."""

    var kind: Int
    var streams: Int
    var n: Int
    var header: Int
    var size: Int

    comptime RAW = 0
    comptime RLE = 1
    comptime COMPRESSED = 2
    """Huffman-coded, with the tree."""
    comptime TREELESS = 3
    """Huffman-coded with the previous block's tree."""

    @staticmethod
    def raw(n: Int) -> Self:
        """The header of `n` literals stored as they are."""
        return Self(Self.RAW, 1, n, Self._plain_length(n), n)

    @staticmethod
    def rle(n: Int) -> Self:
        """The header of `n` copies of one byte."""
        return Self(Self.RLE, 1, n, Self._plain_length(n), 1)

    @staticmethod
    def coded(n: Int, streams: Int) -> Self:
        """The header of `n` Huffman-coded literals in `streams` streams,
        with a tree; its type and size are set once the code is."""
        var header = 3 + Int(n >= 1024) + Int(n >= 16384)
        return Self(Self.COMPRESSED, streams, n, header, 0)

    @staticmethod
    def _plain_length(n: Int) -> Int:
        return 1 if n < 32 else (2 if n < 4096 else 3)

    @staticmethod
    def read(block: Span[UInt8, _]) raises CorruptError -> Self:
        """The header the block starts with; the section must fit in it."""
        var n_in = len(block)
        if n_in == 0:
            raise CorruptError("zstd: an empty compressed block")
        var b0 = Int(block[0])
        var kind = b0 & 3
        var size_format = (b0 >> 2) & 3
        if kind <= Self.RLE:
            var n: Int
            var header: Int
            if size_format & 1 == 0:
                n = b0 >> 3
                header = 1
            elif size_format == 1:
                if n_in < 2:
                    raise CorruptError("zstd: truncated literals header")
                n = (b0 >> 4) + (Int(block[1]) << 4)
                header = 2
            else:
                if n_in < 3:
                    raise CorruptError("zstd: truncated literals header")
                n = (b0 >> 4) + (Int(block[1]) << 4) + (Int(block[2]) << 12)
                header = 3
            var size = n if kind == Self.RAW else 1
            if size > n_in - header:
                raise CorruptError(
                    "zstd: truncated raw literals" if kind
                    == Self.RAW else "zstd: truncated RLE literals"
                )
            return Self(kind, 1, n, header, size)
        var header = 3 if size_format <= 1 else size_format + 2
        if n_in < header:
            raise CorruptError("zstd: truncated literals header")
        var v = Int(LittleEndian.partial[DType.uint64](block[:header], 0))
        var width = Self._width(header)
        var n = (v >> 4) & ((1 << width) - 1)
        var size = (v >> (4 + width)) & ((1 << width) - 1)
        if size > n_in - header:
            raise CorruptError("zstd: truncated Huffman literals")
        var streams = 1 if size_format == 0 else 4
        if streams == 4 and n < 6:
            raise CorruptError(t"zstd: {n} literals in 4 streams")
        return Self(kind, streams, n, header, size)

    def end(self) -> Int:
        """Where the section ends."""
        return self.header + self.size

    @staticmethod
    def _width(header: Int) -> Int:
        """The bits a Huffman-coded section's header gives its literal count
        and its size each: 10, 14 or 18."""
        return 4 * header - 2

    def _value(self) -> Int:
        """The header as `read` parses it: the type, the size format -- for
        stored or repeated literals 0, 1 or 3 by the header's length, for
        coded ones the streams or the length -- and the counts after it."""
        if self.kind <= Self.RLE:
            var format = self.header - 1 + Int(self.header == 3)
            return (
                self.kind | format << 2 | self.n << (3 + Int(self.header > 1))
            )
        else:
            var format = (
                Int(self.streams == 4) if self.header == 3 else self.header - 2
            )
            return (
                self.kind
                | format << 2
                | self.n << 4
                | self.size << (4 + Self._width(self.header))
            )

    def write(self, mut out: List[UInt8]):
        """Append the header."""
        LittleEndian.put_le(out, UInt64(self._value()), self.header)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the bytes reserved for it at `at`."""
        LittleEndian.put_le_at(out, at, UInt64(self._value()), self.header)


@fieldwise_init
struct SequencesHeader(ImplicitlyCopyable, Movable):
    """A sequences section's header: how many sequences, and -- when there
    are any -- the table mode of each field, two bits apiece; and the
    header's own bytes, which a reader takes as written: a count may be
    longer than it needs to be."""

    var count: Int
    var modes: Int
    var length: Int

    comptime PREDEFINED = 0
    comptime RLE = 1
    comptime COMPRESSED = 2
    comptime REPEAT = 3

    @staticmethod
    def of(count: Int) -> Self:
        """The header of `count` sequences, in as few bytes as hold it; the
        modes are set as the tables are chosen."""
        var count_bytes = 1 if count < 128 else (2 if count < 0x7F00 else 3)
        return Self(count, 0, count_bytes + Int(count > 0))

    @staticmethod
    def read(src: Span[UInt8, _]) raises CorruptError -> Self:
        """The header the section starts with."""
        var n_in = len(src)
        if n_in == 0:
            raise CorruptError("zstd: missing sequences section")
        var b0 = Int(src[0])
        var count: Int
        var ip: Int
        if b0 < 128:
            count = b0
            ip = 1
        elif b0 < 255:
            if n_in < 2:
                raise CorruptError("zstd: truncated sequence count")
            count = ((b0 - 128) << 8) + Int(src[1])
            ip = 2
        else:
            if n_in < 3:
                raise CorruptError("zstd: truncated sequence count")
            count = Int(src[1]) + (Int(src[2]) << 8) + 0x7F00
            ip = 3
        var modes = 0
        if count > 0:
            if ip >= n_in:
                raise CorruptError("zstd: truncated sequence modes")
            modes = Int(src[ip])
            if modes & 3 != 0:
                raise CorruptError("zstd: reserved sequence mode bits set")
            ip += 1
        return Self(count, modes, ip)

    def mode(self, field: Alphabet) -> Int:
        """`field`'s table mode."""
        return (self.modes >> (6 - 2 * field.index)) & 3

    def set_mode(mut self, field: Alphabet, mode: Int):
        self.modes |= mode << (6 - 2 * field.index)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the `length` bytes reserved for it at
        `at`."""
        var n = self.count
        var pos = at
        if n < 128:
            out[pos] = UInt8(n)
            pos += 1
        elif n < 0x7F00:
            out[pos] = UInt8((n >> 8) + 0x80)
            out[pos + 1] = UInt8(n & 0xFF)
            pos += 2
        else:
            out[pos] = 0xFF
            out[pos + 1] = UInt8((n - 0x7F00) & 0xFF)
            out[pos + 2] = UInt8((n - 0x7F00) >> 8)
            pos += 3
        if n > 0:
            out[pos] = UInt8(self.modes)
