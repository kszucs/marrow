# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Byte, bit and varint primitives.

The low-level serialization helpers shared by the Arrow IPC (FlatBuffers),
Parquet (Thrift / page) and Avro codecs. Fixed-width scalars are read and written as
little-endian bytes independent of the host byte order: the
`from_bytes[big_endian=False]` read and the shift/mask write both assemble LE
bytes numerically, so no host byteswap is needed.
"""

from std.bit import byte_swap
from std.sys import size_of
from std.sys.info import is_big_endian
from ..errors import CorruptError


struct LittleEndian:
    """Little-endian byte, bit, and LEB128-varint reads/writes over a byte span.
    """

    @staticmethod
    def fixed[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """Read a `T`-width little-endian scalar at byte `pos`. The bound is a
        `debug_assert` -- checked in a test build (`-D ASSERT=all`), free in a
        release one -- so callers validate `pos`, as the hot decode paths do.

        **One unaligned wide load, not a byte loop.** This used to copy `W`
        bytes into an `Array` and call `SIMD.from_bytes`, which cost ~8 loads
        and a stack temporary per 64-bit read — measured at 14-38x on the hash
        kernel's string path against `std.hashlib`, which loads wide. Every
        Parquet and IPC decode paid it too. The bitcast is why `unsafe_ptr` is
        used here: this module is the byte-order abstraction, and confining the
        raw load to it is what keeps it out of the decoders.

        **The load is unaligned.** A byte span promises no alignment, and a
        typed dereference assumes `align_of[Scalar[T]]` -- 16 for `int128` and
        32 for `int256` -- which x86-64 lowers to an aligned SSE move that
        faults on an odd address. That is exactly what `SIMD.from_bytes` does,
        and it crashed `PrimitiveScalar.value()` on Linux x86-64 while ARM, which
        does not trap on alignment, passed."""
        Self._check[T](len(data), pos)
        var v = (
            data.unsafe_ptr()
            .unsafe_offset(pos)
            .unsafe_bitcast[Scalar[T]]()
            .unsafe_load[alignment=1](0)
        )
        comptime if is_big_endian():
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
        comptime if is_big_endian():
            v = byte_swap(value)
        else:
            v = value
        data.unsafe_ptr().unsafe_offset(pos).unsafe_bitcast[
            Scalar[T]
        ]().unsafe_store[alignment=1](v)

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
        """`fixed`, but raising when the read would run past the end.

        The bounds-checked form belongs here rather than being re-derived by
        each caller: a format parser reads *untrusted* offsets, so "raise rather
        than read past the end" is a property of the read, not of any one
        parser. `ipc.mojo` had its own `_read_le` doing exactly this over a
        `List`, which also pinned its buffers to `List` and made a memory-mapped
        source impossible.
        """
        if pos < 0 or pos + size_of[Scalar[T]]() > len(data):
            raise CorruptError(
                t"LittleEndian.checked: {size_of[Scalar[T]]()}-byte read at "
                t"{pos} is out of bounds for {len(data)} bytes"
            )
        return Self.fixed[T](data, pos)

    @staticmethod
    def write[T: DType](mut buf: List[UInt8], pos: Int, val: Scalar[T]):
        """Write `val` as `T`-width little-endian bytes into `buf` at `pos` (the
        destination slots must already exist)."""
        comptime for i in range(size_of[Scalar[T]]()):
            buf[pos + i] = (val >> Scalar[T](i * 8)).cast[DType.uint8]()

    @staticmethod
    def append[T: DType](mut buf: List[UInt8], val: Scalar[T]):
        """Append `val` as `T`-width little-endian bytes to `buf`."""
        comptime for i in range(size_of[Scalar[T]]()):
            buf.append((val >> Scalar[T](i * 8)).cast[DType.uint8]())

    @staticmethod
    def u32(body: Span[UInt8, _], off: Int) -> Int:
        return Int(Self.fixed[DType.uint32](body, off))

    @staticmethod
    def put_u32(mut out: List[UInt8], v: Int):
        Self.append[DType.uint32](out, UInt32(v))

    @staticmethod
    def put_le(mut out: List[UInt8], bits: UInt64, width: Int):
        """Append the low `width` bytes of `bits`, least-significant first."""
        for i in range(width):
            out.append(UInt8((bits >> UInt64(i * 8)) & 0xFF))

    @staticmethod
    def varint(
        data: Span[UInt8, _], pos: Int
    ) raises CorruptError -> Tuple[UInt64, Int]:
        """Read an unsigned LEB128 varint at `pos`; return `(value, next_pos)`.
        """
        var result: UInt64 = 0
        var shift: Int = 0
        var p = pos
        while True:
            if p >= len(data):
                raise CorruptError("varint out of bounds")
            var b = data[p]
            p += 1
            result |= UInt64(b & 0x7F) << UInt64(shift)
            if b & 0x80 == 0:
                break
            shift += 7
            if shift >= 64:
                raise CorruptError("varint too long")
        return (result, p)

    @staticmethod
    def put_varint(mut out: List[UInt8], var v: UInt64):
        """Append `v` as an unsigned LEB128 varint."""
        while True:
            var b = UInt8(v & 0x7F)
            v >>= 7
            if v != 0:
                out.append(b | 0x80)
            else:
                out.append(b)
                break

    @staticmethod
    def bits(data: Span[UInt8, _], bit_offset: Int, nbits: Int) -> UInt64:
        """Read `nbits` starting at absolute `bit_offset`, least-significant
        first."""
        var result: UInt64 = 0
        for i in range(nbits):
            var abs_bit = bit_offset + i
            var byte_idx = abs_bit >> 3
            var bit_idx = abs_bit & 7
            var bit = (UInt64(data[byte_idx]) >> UInt64(bit_idx)) & 1
            result |= bit << UInt64(i)
        return result

    @staticmethod
    def bytes_less(a: Span[UInt8, _], b: Span[UInt8, _]) -> Bool:
        """Unsigned byte-wise lexicographic `a < b` (BYTE_ARRAY ordering)."""
        var n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return a[i] < b[i]
        return len(a) < len(b)


struct Zigzag:
    """Signed <-> unsigned mapping so small-magnitude signed integers stay small
    as varints — shared by Parquet's delta codecs and Thrift Compact Protocol,
    and by Avro's `int` and `long`. Stateless; a namespace of static methods."""

    @staticmethod
    @always_inline
    def encode(v: Int64) -> UInt64:
        return UInt64((v << 1) ^ (v >> 63))

    @staticmethod
    @always_inline
    def decode(u: UInt64) -> Int64:
        return Int64(u >> 1) ^ -Int64(u & 1)


struct BigEndian:
    """The few big-endian reads and writes marrow's formats need: Parquet's and
    Avro's two's-complement decimals, and Avro's snappy checksum trailer."""

    @staticmethod
    def u32(data: Span[UInt8, _], pos: Int) -> UInt32:
        """The 4 bytes at `pos` as a big-endian `UInt32`; the caller has
        checked the range."""
        return byte_swap(LittleEndian.fixed[DType.uint32](data, pos))

    @staticmethod
    def put_u32(mut out: List[UInt8], v: UInt32):
        """Append `v` as 4 big-endian bytes."""
        LittleEndian.append[DType.uint32](out, byte_swap(v))

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
        var bytes = v.as_bytes[big_endian=True]()
        out.extend(Span(bytes)[size_of[Scalar[T]]() - width :])
