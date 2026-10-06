# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro's binary encoding of single values.

`int` and `long` are zigzag varints, `float` and `double` little-endian IEEE
754, `bytes` and `string` a `long` length then the bytes, and `fixed` its bytes
alone. Arrays and maps are written as blocks: a `long` count, that many items,
and a zero count to end; a negative count is followed by the block's byte size,
which a reader may use to skip it.

The logical types whose bytes differ from the Arrow value -- a decimal's
big-endian two's complement, a uuid's RFC 4122 text, a duration's three
little-endian counts -- are read by `AvroCursor` and written by `AvroBytes`
side by side, so the two directions cannot drift apart.
"""

from std.memory import bitcast
from std.sys import size_of

from ..errors import CorruptError, InvalidError
from ..utils import hex_digit
from ..codecs import BigEndian, LittleEndian, Leb128, Zigzag


struct AvroCursor[origin: ImmOrigin](Movable):
    """A read position over Avro-encoded bytes. Every read checks the range and
    raises `CorruptError` rather than read past the end."""

    var data: Span[UInt8, Self.origin]
    var pos: Int

    def __init__(out self, data: Span[UInt8, Self.origin], pos: Int = 0):
        self.data = data
        self.pos = pos

    def remaining(self) -> Int:
        return len(self.data) - self.pos

    def _need(self, n: Int) raises CorruptError:
        if n < 0 or n > self.remaining():
            raise CorruptError(
                t"avro: need {n} bytes at offset {self.pos}, have"
                t" {self.remaining()}"
            )

    def long(mut self) raises CorruptError -> Int64:
        var v, p = Leb128.read(self.data, self.pos)
        self.pos = p
        return Zigzag.decode_value[DType.int64](v)

    def int(mut self) raises CorruptError -> Int32:
        var v = self.long()
        if v < Int64(Int32.MIN) or v > Int64(Int32.MAX):
            raise CorruptError(t"avro: int out of range: {v}")
        return Int32(v)

    def length(mut self) raises CorruptError -> Int:
        """A `long` that counts something -- a byte length or a block size --
        so a negative one is corrupt."""
        var n = self.long()
        if n < 0:
            raise CorruptError(t"avro: negative length {n}")
        return Int(n)

    def boolean(mut self) raises CorruptError -> Bool:
        self._need(1)
        var b = self.data[self.pos]
        self.pos += 1
        if b > 1:
            raise CorruptError(t"avro: invalid boolean byte {b}")
        return b == 1

    def float(mut self) raises CorruptError -> Float32:
        self._need(4)
        var v = LittleEndian.fixed[DType.float32](self.data, self.pos)
        self.pos += 4
        return v

    def double(mut self) raises CorruptError -> Float64:
        self._need(8)
        var v = LittleEndian.fixed[DType.float64](self.data, self.pos)
        self.pos += 8
        return v

    def fixed(mut self, n: Int) raises CorruptError -> Span[UInt8, Self.origin]:
        self._need(n)
        var s = self.data[self.pos : self.pos + n]
        self.pos += n
        return s

    def bytes(mut self) raises CorruptError -> Span[UInt8, Self.origin]:
        """`bytes` or `string`: a length, then that many bytes, borrowed."""
        var n = self.length()
        return self.fixed(n)

    def skip(mut self, n: Int) raises CorruptError:
        self._need(n)
        self.pos += n

    def block(mut self) raises CorruptError -> Tuple[Int, Int]:
        """The header of the next array or map block: `(count, byte_size)`,
        `byte_size` -1 when the writer did not record it. A count of 0 ends
        the sequence."""
        var count = self.long()
        if count >= 0:
            return (Int(count), -1)
        if count == Int64.MIN:
            raise CorruptError("avro: block count out of range")
        return (Int(-count), self.length())

    # --- logical types ----------------------------------------------------

    def decimal[T: DType](mut self, size: Int) raises CorruptError -> Scalar[T]:
        """A decimal's unscaled value: a `fixed` of `size` bytes, or `bytes`
        when `size` is -1 -- big-endian two's complement, narrowed to `T` once
        its redundant sign bytes are dropped."""
        var raw = self.bytes() if size < 0 else self.fixed(size)
        var start = 0
        while len(raw) - start > size_of[Scalar[T]]():
            var b = raw[start]
            var next_negative = raw[start + 1] >= 0x80
            if (b == 0x00 and not next_negative) or (
                b == 0xFF and next_negative
            ):
                start += 1
            else:
                raise CorruptError(
                    t"avro: decimal of {len(raw)} bytes overflows {T}"
                )
        return BigEndian.signed[T](raw[start:])

    def uuid_text(mut self) raises CorruptError -> Array[UInt8, 16]:
        """The 16 bytes an RFC 4122 `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
        `string` spells."""
        var text = self.bytes()
        var out = Array[UInt8, 16](fill=0)
        var ok = len(text) == 36
        var j = 0
        var i = 0
        while ok and i < 36:
            if i == 8 or i == 13 or i == 18 or i == 23:
                ok = text[i] == UInt8(ord("-"))
                i += 1
                continue
            var hi = hex_digit(text[i])
            var lo = hex_digit(text[i + 1])
            ok = hi >= 0 and lo >= 0
            out[j] = UInt8(hi * 16 + lo)
            j += 1
            i += 2
        if not ok:
            raise CorruptError(
                t"avro: invalid uuid '{StringSlice(unsafe_from_utf8=text)}'"
            )
        return out^

    def duration(mut self) raises CorruptError -> Scalar[DType.int128]:
        """A duration -- three little-endian `uint32`s: months, days,
        milliseconds -- as Arrow's month-day-nano interval: `int32` months,
        `int32` days and `int64` nanoseconds packed little-endian."""
        var raw = self.fixed(12)
        var months = UInt128(LittleEndian.fixed[DType.uint32](raw, 0))
        var days = UInt128(LittleEndian.fixed[DType.uint32](raw, 4))
        var millis = LittleEndian.fixed[DType.uint32](raw, 8)
        var nanos = UInt128(UInt64(millis) * 1_000_000)
        return (months | (days << 32) | (nanos << 64)).cast[DType.int128]()


struct AvroBytes(Movable, Sized):
    """Avro-encoded bytes being appended to -- the writing counterpart of
    `AvroCursor`."""

    var _data: List[UInt8]

    def __init__(out self):
        self._data = List[UInt8]()

    def __len__(self) -> Int:
        return len(self._data)

    def written(ref self) -> Span[UInt8, origin_of(self._data)]:
        """What has been written so far."""
        return Span(self._data)

    def clear(mut self):
        self._data.clear()

    def long(mut self, v: Int64):
        Leb128.write(self._data, Zigzag.encode_value(v))

    def int(mut self, v: Int32):
        self.long(Int64(v))

    def boolean(mut self, v: Bool):
        self._data.append(UInt8(1) if v else UInt8(0))

    def float(mut self, v: Float32):
        LittleEndian.append[DType.uint32](self._data, bitcast[DType.uint32](v))

    def double(mut self, v: Float64):
        LittleEndian.append[DType.uint64](self._data, bitcast[DType.uint64](v))

    def fixed(mut self, v: Span[UInt8, _]):
        self._data.extend(v)

    def bytes(mut self, v: Span[UInt8, _]):
        self.long(Int64(len(v)))
        self._data.extend(v)

    # --- logical types ----------------------------------------------------

    def decimal[T: DType](mut self, v: Scalar[T], size: Int) raises:
        """A decimal's unscaled value as big-endian two's complement: a
        `fixed` of `size` bytes, or `bytes` of the fewest that hold it when
        `size` is -1."""
        var width = size
        if size < 0:
            width = 1
            while width < size_of[Scalar[T]]() and not _fits(v, width):
                width += 1
            self.long(Int64(width))
        elif not _fits(v, width):
            raise InvalidError(
                t"avro: decimal {v} does not fit in fixed({size})"
            )
        BigEndian.put_signed[T](self._data, v, width)

    def uuid_text(mut self, uuid: Span[UInt8, _]):
        """16 bytes as a `string` of RFC 4122 text, lower-case."""
        comptime digits = "0123456789abcdef"
        self.long(36)
        for j in range(16):
            if j == 4 or j == 6 or j == 8 or j == 10:
                self._data.append(UInt8(ord("-")))
            self._data.append(digits.as_bytes()[Int(uuid[j] >> 4)])
            self._data.append(digits.as_bytes()[Int(uuid[j] & 0xF)])

    def duration(mut self, v: Scalar[DType.int128]) raises:
        """Arrow's month-day-nano interval as a duration: the months and days
        must not be negative, and the nanoseconds must be whole, non-negative
        milliseconds that fit 32 bits."""
        var bits = v.cast[DType.uint128]()
        var months = (
            (bits & 0xFFFFFFFF).cast[DType.uint32]().cast[DType.int32]()
        )
        var days = (
            ((bits >> 32) & 0xFFFFFFFF).cast[DType.uint32]().cast[DType.int32]()
        )
        var nanos = (bits >> 64).cast[DType.uint64]().cast[DType.int64]()
        if months < 0 or days < 0 or nanos < 0 or nanos % 1_000_000 != 0:
            raise InvalidError(
                t"avro: interval ({months} months, {days} days, {nanos} ns)"
                t" has no duration form"
            )
        var millis = nanos // 1_000_000
        if millis > Int64(UInt32.MAX):
            raise InvalidError(t"avro: {millis} ms overflows a duration")
        for part in [UInt32(months), UInt32(days), UInt32(millis)]:
            LittleEndian.append[DType.uint32](self._data, part)


def _fits[T: DType](v: Scalar[T], width: Int) -> Bool:
    """Whether `v` survives truncation to `width` bytes of two's
    complement."""
    if width >= size_of[Scalar[T]]():
        return True
    var top = v >> Scalar[T](width * 8 - 1)
    return top == 0 or top == -1
