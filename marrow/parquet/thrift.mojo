"""Thrift Compact Protocol codec — the minimal subset Parquet needs.

Parquet stores its file footer and page headers as Thrift Compact Protocol
structures. Mojo has no Thrift runtime, so this is a hand-written reader/writer
for exactly the wire format the `parquet.thrift` IDL uses — varint (ULEB128),
zigzag signed integers, nibble-packed field/list headers, and a recursive
`skip` for forward-compatible unknown fields.

Modelled on arrow-rs `parquet/src/parquet_thrift.rs` (which likewise avoids any
Thrift code generator). The per-struct read/write logic lives in `format.mojo`;
this file is only the protocol primitives.

Compact protocol field/element type nibbles:
    0 STOP  1 BOOL_TRUE  2 BOOL_FALSE  3 BYTE  4 I16  5 I32  6 I64
    7 DOUBLE  8 BINARY(/string)  9 LIST  10 SET  11 MAP  12 STRUCT
"""

from std.memory import bitcast

comptime TC_STOP: UInt8 = 0
comptime TC_BOOL_TRUE: UInt8 = 1
comptime TC_BOOL_FALSE: UInt8 = 2
comptime TC_BYTE: UInt8 = 3
comptime TC_I16: UInt8 = 4
comptime TC_I32: UInt8 = 5
comptime TC_I64: UInt8 = 6
comptime TC_DOUBLE: UInt8 = 7
comptime TC_BINARY: UInt8 = 8
comptime TC_LIST: UInt8 = 9
comptime TC_SET: UInt8 = 10
comptime TC_MAP: UInt8 = 11
comptime TC_STRUCT: UInt8 = 12


def zigzag_encode(v: Int64) -> UInt64:
    """Map a signed integer to an unsigned one so small magnitudes stay small.
    """
    return UInt64((v << 1) ^ (v >> 63))


def zigzag_decode(v: UInt64) -> Int64:
    """Inverse of `zigzag_encode`."""
    return Int64(v >> 1) ^ (-(Int64(v & 1)))


# ---------------------------------------------------------------------------
# Reader — slice-backed, zero-copy for byte strings
# ---------------------------------------------------------------------------


struct CompactReader[o: Origin[mut=False]](Movable):
    """Reads Thrift Compact Protocol values from an immutable byte span.

    `pos` advances as values are consumed. Byte strings are returned as
    sub-spans into the same backing buffer (zero-copy); callers copy when they
    need ownership.
    """

    var data: Span[UInt8, Self.o]
    var pos: Int

    def __init__(out self, data: Span[UInt8, Self.o]):
        self.data = data
        self.pos = 0

    def __init__(out self, data: Span[UInt8, Self.o], pos: Int):
        self.data = data
        self.pos = pos

    @always_inline
    def _u8(mut self) raises -> UInt8:
        if self.pos >= len(self.data):
            raise Error("thrift: unexpected end of input")
        var b = self.data[self.pos]
        self.pos += 1
        return b

    def read_varint(mut self) raises -> UInt64:
        """Read an unsigned LEB128 varint."""
        var result: UInt64 = 0
        var shift: Int = 0
        while True:
            var b = self._u8()
            result |= UInt64(b & 0x7F) << UInt64(shift)
            if b & 0x80 == 0:
                break
            shift += 7
            if shift >= 64:
                raise Error("thrift: varint too long")
        return result

    def read_i16(mut self) raises -> Int16:
        return Int16(zigzag_decode(self.read_varint()))

    def read_i32(mut self) raises -> Int32:
        return Int32(zigzag_decode(self.read_varint()))

    def read_i64(mut self) raises -> Int64:
        return zigzag_decode(self.read_varint())

    def read_byte(mut self) raises -> Int8:
        return Int8(self._u8())

    def read_double(mut self) raises -> Float64:
        var bits: UInt64 = 0
        for i in range(8):
            var sh = UInt64(i * 8)
            bits |= UInt64(self._u8()) << sh
        return bitcast[DType.float64](bits)

    def read_bytes(mut self) raises -> Span[UInt8, Self.o]:
        """Read a length-prefixed byte string as a zero-copy sub-span."""
        var n = Int(self.read_varint())
        if self.pos + n > len(self.data):
            raise Error("thrift: byte string exceeds input")
        var start = self.pos
        self.pos += n
        return self.data[start : start + n]

    def read_string(mut self) raises -> String:
        var raw = self.read_bytes()
        return String(from_utf8=raw)

    def read_bool(mut self, field_type: UInt8) -> Bool:
        """Booleans carry their value in the field-type nibble (1=true)."""
        return field_type == TC_BOOL_TRUE

    def read_field_header(
        mut self, last_field_id: Int
    ) raises -> Tuple[UInt8, Int]:
        """Read a field header, returning `(field_type, field_id)`.

        A zero type nibble is STOP. Otherwise the high nibble is a delta from
        `last_field_id`; a zero delta means a full zigzag field id follows.
        """
        var b = self._u8()
        if b == 0:
            return (TC_STOP, 0)
        var field_type = b & 0x0F
        var delta = Int(b >> 4)
        var field_id: Int
        if delta == 0:
            field_id = Int(self.read_i16())
        else:
            field_id = last_field_id + delta
        return (field_type, field_id)

    def read_list_header(mut self) raises -> Tuple[UInt8, Int]:
        """Read a list/set header, returning `(element_type, size)`."""
        var b = self._u8()
        var elem_type = b & 0x0F
        var size = Int(b >> 4)
        if size == 15:
            size = Int(self.read_varint())
        return (elem_type, size)

    def skip(mut self, field_type: UInt8) raises:
        """Recursively skip a value of the given type (forward compat)."""
        if field_type == TC_BOOL_TRUE or field_type == TC_BOOL_FALSE:
            pass
        elif field_type == TC_BYTE:
            _ = self._u8()
        elif (
            field_type == TC_I16
            or field_type == TC_I32
            or (field_type == TC_I64)
        ):
            _ = self.read_varint()
        elif field_type == TC_DOUBLE:
            _ = self.read_double()
        elif field_type == TC_BINARY:
            _ = self.read_bytes()
        elif field_type == TC_LIST or field_type == TC_SET:
            var elem_type, size = self.read_list_header()
            for _ in range(size):
                self.skip(elem_type)
        elif field_type == TC_MAP:
            var size = Int(self.read_varint())
            if size > 0:
                var kv = self._u8()
                var ktype = kv >> 4
                var vtype = kv & 0x0F
                for _ in range(size):
                    self.skip(ktype)
                    self.skip(vtype)
        elif field_type == TC_STRUCT:
            var last = 0
            while True:
                var ftype, fid = self.read_field_header(last)
                if ftype == TC_STOP:
                    break
                last = fid
                self.skip(ftype)
        else:
            raise Error("thrift: unknown field type " + String(field_type))


# ---------------------------------------------------------------------------
# Writer — appends to an owned byte buffer
# ---------------------------------------------------------------------------


struct CompactWriter(Movable):
    """Builds a Thrift Compact Protocol byte stream."""

    var buf: List[UInt8]

    def __init__(out self):
        self.buf = List[UInt8]()

    def write_varint(mut self, var v: UInt64):
        while True:
            var b = UInt8(v & 0x7F)
            v >>= 7
            if v != 0:
                self.buf.append(b | 0x80)
            else:
                self.buf.append(b)
                break

    def write_i16(mut self, v: Int16):
        self.write_varint(zigzag_encode(Int64(v)))

    def write_i32(mut self, v: Int32):
        self.write_varint(zigzag_encode(Int64(v)))

    def write_i64(mut self, v: Int64):
        self.write_varint(zigzag_encode(v))

    def write_byte(mut self, v: Int8):
        self.buf.append(UInt8(v))

    def write_double(mut self, v: Float64):
        var bits = UInt64(v.to_bits())
        for i in range(8):
            var sh = UInt64(i * 8)
            self.buf.append(UInt8((bits >> sh) & 0xFF))

    def write_bytes(mut self, data: Span[UInt8, _]):
        self.write_varint(UInt64(len(data)))
        self.buf.extend(data)

    def write_string(mut self, s: String):
        self.write_bytes(s.as_bytes())

    def write_field_begin(
        mut self, field_type: UInt8, field_id: Int, last_field_id: Int
    ) -> Int:
        """Write a field header; return the new `last_field_id`."""
        var delta = field_id - last_field_id
        if 0 < delta <= 15:
            self.buf.append((UInt8(delta) << 4) | field_type)
        else:
            self.buf.append(field_type)
            self.write_i16(Int16(field_id))
        return field_id

    def write_bool_field(
        mut self, value: Bool, field_id: Int, last_field_id: Int
    ) -> Int:
        """Bool fields carry their value in the type nibble."""
        var t = TC_BOOL_TRUE if value else TC_BOOL_FALSE
        return self.write_field_begin(t, field_id, last_field_id)

    def write_field_stop(mut self):
        self.buf.append(TC_STOP)

    def write_list_begin(mut self, elem_type: UInt8, size: Int):
        if size < 15:
            self.buf.append((UInt8(size) << 4) | elem_type)
        else:
            self.buf.append(UInt8(0xF0) | elem_type)
            self.write_varint(UInt64(size))
