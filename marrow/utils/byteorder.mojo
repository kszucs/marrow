"""Little-endian byte, bit and varint primitives.

The low-level serialization helpers shared by the Arrow IPC (FlatBuffers) and
Parquet (Thrift / page) codecs. Fixed-width scalars are read and written as
little-endian bytes independent of the host byte order: the
`from_bytes[big_endian=False]` read and the shift/mask write both assemble LE
bytes numerically, so no host byteswap is needed.
"""

from std.sys import size_of


struct LittleEndian:
    """Little-endian byte, bit, and LEB128-varint reads/writes over a byte span.
    """

    @staticmethod
    def fixed[T: DType](data: Span[UInt8, _], pos: Int) -> Scalar[T]:
        """Read a `T`-width little-endian scalar at byte `pos`. Not bounds-checked
        — callers validate `pos` (matches the raw span reads in the hot decode
        paths)."""
        comptime W = size_of[Scalar[T]]()
        var arr = Array[UInt8, W](fill=0)
        for i in range(W):
            arr[i] = data[pos + i]
        return SIMD[T, 1].from_bytes[big_endian=False](arr)

    @staticmethod
    def checked[T: DType](data: Span[UInt8, _], pos: Int) raises -> Scalar[T]:
        """`fixed`, but raising when the read would run past the end.

        The bounds-checked form belongs here rather than being re-derived by
        each caller: a format parser reads *untrusted* offsets, so "raise rather
        than read past the end" is a property of the read, not of any one
        parser. `ipc.mojo` had its own `_read_le` doing exactly this over a
        `List`, which also pinned its buffers to `List` and made a memory-mapped
        source impossible.
        """
        if pos < 0 or pos + size_of[Scalar[T]]() > len(data):
            raise Error(
                "LittleEndian.checked: ",
                size_of[Scalar[T]](),
                "-byte read at ",
                pos,
                " is out of bounds for ",
                len(data),
                " bytes",
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
    def varint(data: Span[UInt8, _], pos: Int) raises -> Tuple[UInt64, Int]:
        """Read an unsigned LEB128 varint at `pos`; return `(value, next_pos)`.
        """
        var result: UInt64 = 0
        var shift: Int = 0
        var p = pos
        while True:
            if p >= len(data):
                raise Error("varint out of bounds")
            var b = data[p]
            p += 1
            result |= UInt64(b & 0x7F) << UInt64(shift)
            if b & 0x80 == 0:
                break
            shift += 7
            if shift >= 64:
                raise Error("varint too long")
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
