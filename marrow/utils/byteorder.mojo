# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Byte order: values as little- and big-endian bytes, and as varints.

The reads and writes every format in marrow goes through -- Arrow IPC,
Parquet, Avro, the compressors and the hashes -- whatever the host's byte
order.
"""

from std.bit import byte_swap
from std.sys import size_of
from std.sys.info import Endian
from ..errors import CorruptError
from ..views import BufferView
from .integers import unzigzag, zigzag


struct LittleEndian:
    """Little-endian reads and writes: of a `T`, or of a value's low bytes."""

    @staticmethod
    def fixed[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """Read a `T`-width little-endian scalar at byte `pos`. The bound is a
        `debug_assert` -- checked in a test build (`-D ASSERT=all`), free in a
        release one -- so callers validate `pos`, as the hot decode paths do."""
        # One wide load, and unaligned: a byte span promises no alignment,
        # and an aligned load of an `int128` faults on x86-64.
        Self._check[T](len(data), pos)
        var v = (
            BufferView[T, data.origin]
            .from_bytes(data[pos:])
            .load_unaligned[1](0)
        )
        comptime if Endian.native() == Endian.big:
            return byte_swap(v)
        else:
            return v

    @staticmethod
    def store[
        T: DType
    ](data: Span[mut=True, UInt8, _], pos: Int, value: Scalar[T]):
        """Write `value` as `T`-width little-endian bytes at byte `pos` --
        `fixed`'s inverse, one unaligned wide store, its bound asserted the
        same way."""
        Self._check[T](len(data), pos)
        var v: Scalar[T]
        comptime if Endian.native() == Endian.big:
            v = byte_swap(value)
        else:
            v = value
        BufferView[T, data.origin].from_bytes(data[pos:]).store_unaligned[1](
            0, v
        )

    @staticmethod
    @always_inline
    def _check[T: DType](length: Int, pos: Int):
        debug_assert(
            0 <= pos and pos + size_of[Scalar[T]]() <= length,
            "LittleEndian: ",
            size_of[Scalar[T]](),
            "-byte access at ",
            pos,
            " is out of bounds for ",
            length,
            " bytes",
        )

    @staticmethod
    def partial[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """`fixed`, but the bytes past the end of `data` read as zero -- for a
        wide load at the tail of a span that carries no padding."""
        comptime assert T.is_integral(), "LittleEndian.partial needs an integer"
        var v = Scalar[T](0)
        for b in range(min(size_of[Scalar[T]](), len(data) - pos)):
            v |= Scalar[T](data[pos + b]) << Scalar[T](8 * b)
        return v

    @staticmethod
    def checked[
        T: DType
    ](data: Span[UInt8, _], pos: Int) raises CorruptError -> Scalar[T]:
        """`fixed`, but raising when the read would run past the end -- for
        offsets a file gave."""
        if pos < 0 or pos + size_of[Scalar[T]]() > len(data):
            raise CorruptError(
                t"LittleEndian.checked: {size_of[Scalar[T]]()}-byte read at "
                t"{pos} is out of bounds for {len(data)} bytes"
            )
        return Self.fixed[T](data, pos)

    @staticmethod
    def append[T: DType](mut buf: List[UInt8], val: Scalar[T]):
        """Append `val` as `T`-width bytes."""
        var at = len(buf)
        buf.resize(unsafe_uninit_length=at + size_of[Scalar[T]]())
        Self.store[T](Span(buf), at, val)

    @staticmethod
    def append(mut buf: List[UInt8], bits: UInt64, width: Int):
        """Append the low `width` bytes of `bits`."""
        for i in range(width):
            buf.append(UInt8((bits >> UInt64(i * 8)) & 0xFF))

    @staticmethod
    def write[T: DType](mut buf: List[UInt8], pos: Int, val: Scalar[T]):
        """Write `val` as `T`-width bytes at `pos`, over bytes `buf` holds."""
        Self.store[T](Span(buf), pos, val)

    @staticmethod
    def write(mut buf: List[UInt8], pos: Int, bits: UInt64, width: Int):
        """Write the low `width` bytes of `bits` at `pos`, over bytes `buf`
        holds."""
        for i in range(width):
            buf[pos + i] = UInt8((bits >> UInt64(i * 8)) & 0xFF)


struct Leb128:
    """Unsigned integers as ULEB128 -- seven bits a byte, least significant
    first, the high bit set on every byte but the last -- and signed ones
    zigzagged first: the varints of Parquet's Thrift, Parquet's RLE and
    delta headers, Avro and Snappy's preamble."""

    @staticmethod
    @always_inline
    def scan(data: Span[UInt8, _], pos: Int) -> Tuple[UInt64, Int]:
        """The value at `pos` and the position after it -- or a position of
        -1 when the value runs past `data` or past 64 bits."""
        var result: UInt64 = 0
        var shift: UInt64 = 0
        var p = pos
        while p < len(data) and shift < 64:
            var b = data[p]
            p += 1
            result |= UInt64(b & 0x7F) << shift
            if b < 0x80:
                return (result, p)
            shift += 7
        return (result, -1)

    @staticmethod
    def read(
        data: Span[UInt8, _], pos: Int
    ) raises CorruptError -> Tuple[UInt64, Int]:
        """The value at `pos` and the position after it."""
        var v, p = Self.scan(data, pos)
        if p < 0:
            raise CorruptError(t"a varint at byte {pos} runs past its end")
        return (v, p)

    @staticmethod
    def read_signed(
        data: Span[UInt8, _], pos: Int
    ) raises CorruptError -> Tuple[Int64, Int]:
        """The zigzagged value at `pos` and the position after it."""
        var v, p = Self.read(data, pos)
        return (unzigzag[DType.int64](v), p)

    @staticmethod
    def write(mut out: List[UInt8], var v: UInt64):
        """Append `v`."""
        while True:
            var b = UInt8(v & 0x7F)
            v >>= 7
            if v != 0:
                out.append(b | 0x80)
            else:
                out.append(b)
                break

    @staticmethod
    def write_signed(mut out: List[UInt8], v: Int64):
        """Append `v`, zigzagged: small magnitudes stay small."""
        Self.write(out, zigzag(v).cast[DType.uint64]())


struct BigEndian:
    """The few big-endian reads and writes marrow's formats need: Parquet's and
    Avro's two's-complement decimals, Avro's snappy checksum trailer, and
    Hadoop-framed LZ4's sizes."""

    @staticmethod
    def fixed[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """A big-endian `T` at byte `pos`; the caller has checked the
        range."""
        return byte_swap(LittleEndian.fixed[T](data, pos))

    @staticmethod
    def append[T: DType](mut buf: List[UInt8], val: Scalar[T]):
        """Append `val` as `T`-width big-endian bytes."""
        LittleEndian.append[T](buf, byte_swap(val))

    @staticmethod
    def write[T: DType](mut buf: List[UInt8], pos: Int, val: Scalar[T]):
        """Write `val` as `T`-width big-endian bytes at `pos`, over bytes that
        already exist."""
        LittleEndian.write[T](buf, pos, byte_swap(val))

    @staticmethod
    def signed[T: DType](data: Span[UInt8, _]) -> Scalar[T]:
        """A big-endian two's-complement integer of `len(data)` bytes,
        sign-extended to `T`. `len(data)` must not exceed `T`'s width."""
        comptime FULL = size_of[Scalar[T]]()
        debug_assert(len(data) <= FULL, "BigEndian.signed: too wide")
        var arr = Array[UInt8, FULL](fill=0)
        if len(data) > 0 and (data[0] & 0x80) != 0:
            for i in range(FULL):
                arr[i] = 0xFF
        for i in range(len(data)):
            arr[FULL - len(data) + i] = data[i]
        # Not `SIMD.from_bytes`: it dereferences the byte array at the native
        # type's alignment, which for an `int128`/`int256` decimal is an aligned
        # SSE load that faults on x86-64. `fixed` reads little-endian, so a
        # swap turns the big-endian bytes into the value on either host.
        return byte_swap(LittleEndian.fixed[T](arr, 0))

    @staticmethod
    def put_signed[T: DType](mut out: List[UInt8], v: Scalar[T], width: Int):
        """Append the low `width` bytes of `v`, most significant first -- the
        inverse of `signed` when `v` fits in `width` bytes."""
        var bytes = v.as_bytes[endian=Endian.big]()
        out.extend(Span(bytes)[size_of[Scalar[T]]() - width :])
