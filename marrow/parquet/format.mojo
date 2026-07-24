"""The Parquet file-format layer: the Thrift Compact Protocol codec plus the
metadata structures it serializes.

`ThriftCompactReader` / `ThriftCompactWriter` are a hand-written subset of the Thrift Compact
Protocol (varint / zigzag / nibble-packed field & list headers + a recursive
`skip` for forward compatibility), modelled on arrow-rs `parquet_thrift.rs` — no
Thrift runtime or codegen. On top of them sit the metadata structs for exactly
the subset of `parquet.thrift` the reader/writer touch: the file footer
(`FileMetaData` → `RowGroup` → `ColumnChunk` → `ColumnMetaData`), the schema
element list, and the page headers. Field IDs and enum values come straight from
the `parquet.thrift` IDL; the small enum discriminants are namespaced value
types (`PhysicalType`, `Encoding`, …) rather than bare integer constants.
"""

from std.memory import bitcast

from .codecs import Encoding, Zigzag
from ..utils import LittleEndian


# ---------------------------------------------------------------------------
# Thrift Compact Protocol codec
#
# Field/element type nibbles:
#     0 STOP  1 BOOL_TRUE  2 BOOL_FALSE  3 BYTE  4 I16  5 I32  6 I64
#     7 DOUBLE  8 BINARY(/string)  9 LIST  10 SET  11 MAP  12 STRUCT
# ---------------------------------------------------------------------------

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


struct FieldHeader(Copyable, Movable):
    """Iteration state for one Thrift struct's fields: the current field's `id`
    and wire `type`, plus the running `last` id the compact header-delta decoding
    needs. One instance per struct frame — nested structs each keep their own.
    """

    var id: Int
    var type: UInt8
    var last: Int

    def __init__(out self):
        self.id = 0
        self.type = TC_STOP
        self.last = 0


struct ThriftCompactReader[o: Origin[mut=False]](Movable):
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
        """Read an unsigned LEB128 varint, advancing `pos`."""
        var value: UInt64
        value, self.pos = LittleEndian.varint(self.data, self.pos)
        return value

    def read_i16(mut self) raises -> Int16:
        return Int16(Zigzag.decode(self.read_varint()))

    def read_i32(mut self) raises -> Int32:
        return Int32(Zigzag.decode(self.read_varint()))

    def read_i64(mut self) raises -> Int64:
        return Zigzag.decode(self.read_varint())

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

    def next_field(mut self, mut field: FieldHeader) raises -> Bool:
        """Advance to the next field of the current struct, driving the
        `while r.next_field(f):` loop that every `read` body shares.

        Returns `False` at the STOP marker (ending the loop). On `True`,
        `field.id` / `field.type` describe the field the body must consume;
        the running `field.last` is tracked internally. Fields the body does
        not recognise still need an explicit `r.skip(field.type)`.
        """
        var ftype, fid = self.read_field_header(field.last)
        if ftype == TC_STOP:
            return False
        field.last = fid
        field.id = fid
        field.type = ftype
        return True

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
            var f = FieldHeader()
            while self.next_field(f):
                self.skip(f.type)
        else:
            raise Error("thrift: unknown field type " + String(field_type))


struct ThriftCompactWriter(Movable):
    """Builds a Thrift Compact Protocol byte stream."""

    var buf: List[UInt8]

    def __init__(out self):
        self.buf = List[UInt8]()

    def write_varint(mut self, var v: UInt64):
        LittleEndian.put_varint(self.buf, v)

    def write_i16(mut self, v: Int16):
        self.write_varint(Zigzag.encode(Int64(v)))

    def write_i32(mut self, v: Int32):
        self.write_varint(Zigzag.encode(Int64(v)))

    def write_i64(mut self, v: Int64):
        self.write_varint(Zigzag.encode(v))

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

    def write_struct_list[
        T: ThriftWritable
    ](mut self, field_id: Int, last: Int, items: List[T]) -> Int:
        """Write a `list<struct>` field (each element via its own `write`);
        return `field_id` as the new `last_field_id`."""
        var this = self.write_field_begin(TC_LIST, field_id, last)
        self.write_list_begin(TC_STRUCT, len(items))
        for i in range(len(items)):
            items[i].write(self)
        return this


trait ThriftWritable(Copyable, Movable):
    """A metadata struct that serializes itself into a `ThriftCompactWriter`. `write`
    is the only requirement; `append_to` and `ThriftCompactWriter.write_struct_list`
    build on it so every footer / header struct shares one serialization path.
    """

    def write(self, mut w: ThriftCompactWriter):
        ...

    def append_to(self, mut out: List[UInt8]) -> Int:
        """Serialize `self` into `out`; return its byte length."""
        var w = ThriftCompactWriter()
        self.write(w)
        out.extend(Span(w.buf))
        return len(w.buf)


comptime PARQUET_MAGIC: List[UInt8] = [0x50, 0x41, 0x52, 0x31]  # "PAR1"


# ---------------------------------------------------------------------------
# Enum discriminants — one small value type per Parquet IDL enum, so callers
# compare against namespaced members (`PhysicalType.INT32`) instead of importing
# a pile of bare integer constants. Field IDs and enum values come straight from
# the `parquet.thrift` IDL. `NONE` (-1) is the "absent"/"group node" sentinel.
# ---------------------------------------------------------------------------


@fieldwise_init
struct PhysicalType(Equatable, ImplicitlyCopyable, Movable):
    """Parquet `Type` — `SchemaElement.type` / `ColumnMetaData.type`."""

    var code: Int

    comptime NONE = Self(-1)  # group node, no physical type
    comptime BOOLEAN = Self(0)
    comptime INT32 = Self(1)
    comptime INT64 = Self(2)
    comptime INT96 = Self(3)
    comptime FLOAT = Self(4)
    comptime DOUBLE = Self(5)
    comptime BYTE_ARRAY = Self(6)
    comptime FIXED_LEN_BYTE_ARRAY = Self(7)


@fieldwise_init
struct Repetition(Equatable, ImplicitlyCopyable, Movable):
    """Parquet `FieldRepetitionType` — `NONE` (-1) marks the root group."""

    var code: Int

    comptime NONE = Self(-1)
    comptime REQUIRED = Self(0)
    comptime OPTIONAL = Self(1)
    comptime REPEATED = Self(2)


@fieldwise_init
struct ConvertedType(Equatable, ImplicitlyCopyable, Movable):
    """Parquet `ConvertedType` (legacy logical annotation); `NONE` = absent."""

    var code: Int

    comptime NONE = Self(-1)
    comptime UTF8 = Self(0)
    comptime MAP = Self(1)
    comptime MAP_KEY_VALUE = Self(2)  # legacy map annotation (read tolerance)
    comptime LIST = Self(3)
    comptime DECIMAL = Self(5)
    comptime DATE = Self(6)
    comptime TIME_MILLIS = Self(7)
    comptime TIME_MICROS = Self(8)
    comptime TIMESTAMP_MILLIS = Self(9)
    comptime TIMESTAMP_MICROS = Self(10)
    comptime UINT_8 = Self(11)
    comptime UINT_16 = Self(12)
    comptime UINT_32 = Self(13)
    comptime UINT_64 = Self(14)
    comptime INT_8 = Self(15)
    comptime INT_16 = Self(16)
    comptime INT_32 = Self(17)
    comptime INT_64 = Self(18)


@fieldwise_init
struct LogicalType(Equatable, ImplicitlyCopyable, Movable):
    """Parquet `LogicalType` union member id; `NONE` = absent."""

    var code: Int

    comptime NONE = Self(-1)
    comptime STRING = Self(1)
    comptime MAP = Self(2)
    comptime LIST = Self(3)
    comptime DECIMAL = Self(5)
    comptime DATE = Self(6)
    comptime TIME = Self(7)
    comptime TIMESTAMP = Self(8)
    comptime INTEGER = Self(10)
    comptime FLOAT16 = Self(15)


@fieldwise_init
struct PageType(Equatable, ImplicitlyCopyable, Movable):
    """Parquet `PageType`."""

    var code: Int

    comptime NONE = Self(-1)
    comptime DATA = Self(0)
    comptime INDEX = Self(1)
    comptime DICTIONARY = Self(2)
    comptime DATA_V2 = Self(3)


# ---------------------------------------------------------------------------
# SchemaElement — one node of the flattened schema tree
# ---------------------------------------------------------------------------


struct SchemaElement(Copyable, Movable, ThriftWritable):
    var type: PhysicalType  # NONE if a group node
    var type_length: Int  # for FIXED_LEN_BYTE_ARRAY
    var repetition_type: Repetition  # NONE at root
    var name: String
    var num_children: Int  # >0 for group nodes
    var converted_type: ConvertedType  # NONE if absent
    var scale: Int
    var precision: Int
    var logical_type: LogicalType  # union member id, NONE if absent
    var logical_unit: Int  # TimeUnit for TIMESTAMP/TIME: 1=ms 2=us 3=ns, else -1
    var logical_utc: Bool  # isAdjustedToUTC for TIMESTAMP/TIME

    def __init__(out self):
        self.type = PhysicalType.NONE
        self.type_length = 0
        self.repetition_type = Repetition.NONE
        self.name = String()
        self.num_children = 0
        self.converted_type = ConvertedType.NONE
        self.scale = 0
        self.precision = 0
        self.logical_type = LogicalType.NONE
        self.logical_unit = -1
        self.logical_utc = False

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.type = PhysicalType(Int(r.read_i32()))
            elif f.id == 2:
                out.type_length = Int(r.read_i32())
            elif f.id == 3:
                out.repetition_type = Repetition(Int(r.read_i32()))
            elif f.id == 4:
                out.name = r.read_string()
            elif f.id == 5:
                out.num_children = Int(r.read_i32())
            elif f.id == 6:
                out.converted_type = ConvertedType(Int(r.read_i32()))
            elif f.id == 7:
                out.scale = Int(r.read_i32())
            elif f.id == 8:
                out.precision = Int(r.read_i32())
            elif f.id == 10:
                out._read_logical_type(r)
            else:
                r.skip(f.type)
        return out^

    def _read_logical_type[
        o: Origin[mut=False]
    ](mut self, mut r: ThriftCompactReader[o]) raises:
        """Parse the `LogicalType` union into `logical_type`, and for TIMESTAMP /
        TIME also the nested `TimeUnit` (`logical_unit`) and `isAdjustedToUTC`.
        """
        var f = FieldHeader()
        while r.next_field(f):
            self.logical_type = LogicalType(f.id)
            if (
                self.logical_type == LogicalType.TIMESTAMP
                or self.logical_type == LogicalType.TIME
            ):
                # TimestampType/TimeType = {1: isAdjustedToUTC, 2: TimeUnit unit}
                var f2 = FieldHeader()
                while r.next_field(f2):
                    if f2.id == 1:
                        self.logical_utc = r.read_bool(f2.type)
                    elif f2.id == 2:
                        # TimeUnit union: the single set field id is the unit
                        var f3 = FieldHeader()
                        while r.next_field(f3):
                            self.logical_unit = f3.id
                            r.skip(f3.type)
                    else:
                        r.skip(f2.type)
            else:
                r.skip(f.type)

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        if self.type != PhysicalType.NONE:
            last = w.write_field_begin(TC_I32, 1, last)
            w.write_i32(Int32(self.type.code))
        if self.type == PhysicalType.FIXED_LEN_BYTE_ARRAY:
            last = w.write_field_begin(TC_I32, 2, last)
            w.write_i32(Int32(self.type_length))
        if self.repetition_type != Repetition.NONE:
            last = w.write_field_begin(TC_I32, 3, last)
            w.write_i32(Int32(self.repetition_type.code))
        last = w.write_field_begin(TC_BINARY, 4, last)
        w.write_string(self.name)
        if self.num_children > 0:
            last = w.write_field_begin(TC_I32, 5, last)
            w.write_i32(Int32(self.num_children))
        if self.converted_type != ConvertedType.NONE:
            last = w.write_field_begin(TC_I32, 6, last)
            w.write_i32(Int32(self.converted_type.code))
        if self.logical_type == LogicalType.DECIMAL:
            last = w.write_field_begin(TC_I32, 7, last)
            w.write_i32(Int32(self.scale))
            last = w.write_field_begin(TC_I32, 8, last)
            w.write_i32(Int32(self.precision))
        if self.logical_type != LogicalType.NONE:
            _ = w.write_field_begin(TC_STRUCT, 10, last)
            self._write_logical_type(w)
        w.write_field_stop()

    def _write_logical_type(self, mut w: ThriftCompactWriter):
        """Serialize the `LogicalType` union (field 10): one member field whose
        id is the logical type, holding that member's struct. TIME/TIMESTAMP carry
        `{isAdjustedToUTC, unit}` (unit a `TimeUnit` union), DECIMAL carries
        `{scale, precision}`, and the rest (DATE, STRING, …) are empty structs.
        """
        _ = w.write_field_begin(TC_STRUCT, self.logical_type.code, 0)
        if (
            self.logical_type == LogicalType.TIME
            or self.logical_type == LogicalType.TIMESTAMP
        ):
            var last = w.write_bool_field(self.logical_utc, 1, 0)
            _ = w.write_field_begin(TC_STRUCT, 2, last)  # unit: TimeUnit union
            _ = w.write_field_begin(TC_STRUCT, self.logical_unit, 0)
            w.write_field_stop()  # close the TimeUnit member (empty struct)
            w.write_field_stop()  # close the TimeUnit union
            w.write_field_stop()  # close the TimeType / TimestampType struct
        elif self.logical_type == LogicalType.DECIMAL:
            var last = w.write_field_begin(TC_I32, 1, 0)
            w.write_i32(Int32(self.scale))
            _ = w.write_field_begin(TC_I32, 2, last)
            w.write_i32(Int32(self.precision))
            w.write_field_stop()  # close the DecimalType struct
        else:
            w.write_field_stop()  # close the (empty) member struct
        w.write_field_stop()  # close the LogicalType union struct


# ---------------------------------------------------------------------------
# Statistics, page headers
# ---------------------------------------------------------------------------


struct DataPageHeader(Copyable, Movable):
    var num_values: Int
    var encoding: Encoding
    var definition_level_encoding: Encoding
    var repetition_level_encoding: Encoding

    def __init__(out self):
        self.num_values = 0
        self.encoding = Encoding.PLAIN
        self.definition_level_encoding = Encoding.RLE
        self.repetition_level_encoding = Encoding.RLE

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.num_values = Int(r.read_i32())
            elif f.id == 2:
                out.encoding = Encoding(Int(r.read_i32()))
            elif f.id == 3:
                out.definition_level_encoding = Encoding(Int(r.read_i32()))
            elif f.id == 4:
                out.repetition_level_encoding = Encoding(Int(r.read_i32()))
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_values))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.encoding.code))
        last = w.write_field_begin(TC_I32, 3, last)
        w.write_i32(Int32(self.definition_level_encoding.code))
        _ = w.write_field_begin(TC_I32, 4, last)
        w.write_i32(Int32(self.repetition_level_encoding.code))
        w.write_field_stop()


struct DataPageHeaderV2(Copyable, Movable):
    var num_values: Int
    var num_nulls: Int
    var num_rows: Int
    var encoding: Encoding
    var definition_levels_byte_length: Int
    var repetition_levels_byte_length: Int
    var is_compressed: Bool

    def __init__(out self):
        self.num_values = 0
        self.num_nulls = 0
        self.num_rows = 0
        self.encoding = Encoding.PLAIN
        self.definition_levels_byte_length = 0
        self.repetition_levels_byte_length = 0
        self.is_compressed = True

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.num_values = Int(r.read_i32())
            elif f.id == 2:
                out.num_nulls = Int(r.read_i32())
            elif f.id == 3:
                out.num_rows = Int(r.read_i32())
            elif f.id == 4:
                out.encoding = Encoding(Int(r.read_i32()))
            elif f.id == 5:
                out.definition_levels_byte_length = Int(r.read_i32())
            elif f.id == 6:
                out.repetition_levels_byte_length = Int(r.read_i32())
            elif f.id == 7:
                out.is_compressed = r.read_bool(f.type)
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_values))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.num_nulls))
        last = w.write_field_begin(TC_I32, 3, last)
        w.write_i32(Int32(self.num_rows))
        last = w.write_field_begin(TC_I32, 4, last)
        w.write_i32(Int32(self.encoding.code))
        last = w.write_field_begin(TC_I32, 5, last)
        w.write_i32(Int32(self.definition_levels_byte_length))
        last = w.write_field_begin(TC_I32, 6, last)
        w.write_i32(Int32(self.repetition_levels_byte_length))
        _ = w.write_bool_field(self.is_compressed, 7, last)
        w.write_field_stop()


struct DictionaryPageHeader(Copyable, Movable):
    var num_values: Int
    var encoding: Encoding

    def __init__(out self):
        self.num_values = 0
        self.encoding = Encoding.PLAIN

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.num_values = Int(r.read_i32())
            elif f.id == 2:
                out.encoding = Encoding(Int(r.read_i32()))
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_values))
        _ = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.encoding.code))
        w.write_field_stop()


struct PageHeader(Copyable, Movable, ThriftWritable):
    var type: PageType
    var uncompressed_page_size: Int
    var compressed_page_size: Int
    var crc: Int  # CRC-32 of the page body, or -1 when absent
    var data_page_header: Optional[DataPageHeader]
    var data_page_header_v2: Optional[DataPageHeaderV2]
    var dictionary_page_header: Optional[DictionaryPageHeader]

    def __init__(out self):
        self.type = PageType.NONE
        self.uncompressed_page_size = 0
        self.compressed_page_size = 0
        self.crc = -1
        self.data_page_header = None
        self.data_page_header_v2 = None
        self.dictionary_page_header = None

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.type = PageType(Int(r.read_i32()))
            elif f.id == 2:
                out.uncompressed_page_size = Int(r.read_i32())
            elif f.id == 3:
                out.compressed_page_size = Int(r.read_i32())
            elif f.id == 4:
                # stored signed; keep the unsigned 32-bit value so -1 = absent
                out.crc = Int(r.read_i32()) & 0xFFFFFFFF
            elif f.id == 5:
                out.data_page_header = DataPageHeader.read(r)
            elif f.id == 7:
                out.dictionary_page_header = DictionaryPageHeader.read(r)
            elif f.id == 8:
                out.data_page_header_v2 = DataPageHeaderV2.read(r)
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.type.code))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.uncompressed_page_size))
        last = w.write_field_begin(TC_I32, 3, last)
        w.write_i32(Int32(self.compressed_page_size))
        if self.crc >= 0:
            last = w.write_field_begin(TC_I32, 4, last)
            w.write_i32(Int32(self.crc))
        if self.data_page_header:
            _ = w.write_field_begin(TC_STRUCT, 5, last)
            self.data_page_header.value().write(w)
        elif self.dictionary_page_header:
            _ = w.write_field_begin(TC_STRUCT, 7, last)
            self.dictionary_page_header.value().write(w)
        elif self.data_page_header_v2:
            _ = w.write_field_begin(TC_STRUCT, 8, last)
            self.data_page_header_v2.value().write(w)
        w.write_field_stop()

    @staticmethod
    def read_at[
        o: Origin[mut=False]
    ](data: Span[UInt8, o], mut pos: Int) raises -> Self:
        """Read the page header at `pos`, advancing `pos` to the page body."""
        var r = ThriftCompactReader(data, pos)
        var ph = Self.read(r)
        pos = r.pos
        return ph^

    @staticmethod
    def data_page(
        uncompressed_size: Int,
        compressed_size: Int,
        num_values: Int,
        encoding: Encoding = Encoding.PLAIN,
    ) -> Self:
        """Build a v1 data-page header. `encoding` is the value encoding (PLAIN,
        RLE_DICTIONARY, or a DELTA_* variant); levels are always RLE."""
        var ph = Self()
        ph.type = PageType.DATA
        ph.uncompressed_page_size = uncompressed_size
        ph.compressed_page_size = compressed_size
        var dph = DataPageHeader()
        dph.num_values = num_values
        dph.encoding = encoding
        dph.definition_level_encoding = Encoding.RLE
        dph.repetition_level_encoding = Encoding.RLE
        ph.data_page_header = dph^
        return ph^

    @staticmethod
    def data_page_v2(
        uncompressed_size: Int,
        compressed_size: Int,
        num_values: Int,
        num_nulls: Int,
        num_rows: Int,
        def_levels_byte_length: Int,
        is_compressed: Bool,
        rep_levels_byte_length: Int = 0,
        encoding: Encoding = Encoding.PLAIN,
    ) -> Self:
        """Build a v2 data-page header (levels stored uncompressed ahead of the —
        optionally compressed — values). `encoding` is the value encoding."""
        var ph = Self()
        ph.type = PageType.DATA_V2
        ph.uncompressed_page_size = uncompressed_size
        ph.compressed_page_size = compressed_size
        var dph = DataPageHeaderV2()
        dph.num_values = num_values
        dph.num_nulls = num_nulls
        dph.num_rows = num_rows
        dph.encoding = encoding
        dph.definition_levels_byte_length = def_levels_byte_length
        dph.repetition_levels_byte_length = rep_levels_byte_length
        dph.is_compressed = is_compressed
        ph.data_page_header_v2 = dph^
        return ph^

    @staticmethod
    def dictionary_page(
        uncompressed_size: Int, compressed_size: Int, num_values: Int
    ) -> Self:
        """Build a dictionary-page header (PLAIN-encoded distinct values)."""
        var ph = Self()
        ph.type = PageType.DICTIONARY
        ph.uncompressed_page_size = uncompressed_size
        ph.compressed_page_size = compressed_size
        var dph = DictionaryPageHeader()
        dph.num_values = num_values
        dph.encoding = Encoding.PLAIN
        ph.dictionary_page_header = dph^
        return ph^


# ---------------------------------------------------------------------------
# ColumnMetaData / ColumnChunk / RowGroup / FileMetaData
# ---------------------------------------------------------------------------


struct ColumnMetaData(Copyable, Movable):
    var type: Int
    var path_in_schema: List[String]
    var codec: Int
    var num_values: Int
    var total_uncompressed_size: Int
    var total_compressed_size: Int
    var data_page_offset: Int
    var dictionary_page_offset: Int  # -1 if absent
    var encodings: List[Int]  # Encoding codes actually used in the chunk
    var null_count: Int  # -1 if unknown; written as Statistics.null_count
    var distinct_count: Int  # -1 if unknown; Statistics.distinct_count
    var has_min_max: Bool  # Statistics.min_value/max_value present
    var min_value: List[
        UInt8
    ]  # PLAIN-encoded min (no length prefix for BYTE_ARRAY)
    var max_value: List[UInt8]  # PLAIN-encoded max
    var bloom_filter_offset: Int  # -1 if absent
    var bloom_filter_length: Int

    def __init__(out self):
        self.type = -1
        self.path_in_schema = List[String]()
        self.codec = 0
        self.num_values = 0
        self.total_uncompressed_size = 0
        self.total_compressed_size = 0
        self.data_page_offset = 0
        self.dictionary_page_offset = -1
        self.encodings = [Encoding.RLE.code, Encoding.PLAIN.code]
        self.null_count = -1
        self.distinct_count = -1
        self.has_min_max = False
        self.min_value = List[UInt8]()
        self.max_value = List[UInt8]()
        self.bloom_filter_offset = -1
        self.bloom_filter_length = 0

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 3:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.path_in_schema.append(r.read_string())
            elif f.id == 1:
                out.type = Int(r.read_i32())
            elif f.id == 4:
                out.codec = Int(r.read_i32())
            elif f.id == 5:
                out.num_values = Int(r.read_i64())
            elif f.id == 6:
                out.total_uncompressed_size = Int(r.read_i64())
            elif f.id == 7:
                out.total_compressed_size = Int(r.read_i64())
            elif f.id == 9:
                out.data_page_offset = Int(r.read_i64())
            elif f.id == 11:
                out.dictionary_page_offset = Int(r.read_i64())
            elif f.id == 12:
                out._read_statistics(r)
            elif f.id == 14:
                out.bloom_filter_offset = Int(r.read_i64())
            elif f.id == 15:
                out.bloom_filter_length = Int(r.read_i32())
            else:
                r.skip(f.type)
        return out^

    def _read_statistics[
        o: Origin[mut=False]
    ](mut self, mut r: ThriftCompactReader[o]) raises:
        """Parse the nested Statistics struct, keeping null_count and the modern
        min_value/max_value (fields 6/5). The deprecated min/max (fields 2/1) are
        skipped — modern writers populate min_value/max_value."""
        var f = FieldHeader()
        var seen_min = False
        var seen_max = False
        while r.next_field(f):
            if f.id == 3:
                self.null_count = Int(r.read_i64())
            elif f.id == 4:
                self.distinct_count = Int(r.read_i64())
            elif f.id == 5:
                var bytes = r.read_bytes()
                self.max_value = List[UInt8](Span(bytes))
                seen_max = True
            elif f.id == 6:
                var bytes = r.read_bytes()
                self.min_value = List[UInt8](Span(bytes))
                seen_min = True
            else:
                r.skip(f.type)
        # min/max are only usable when both bounds are present.
        self.has_min_max = seen_min and seen_max

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.type))
        # encodings: list<Encoding> actually used — RLE levels plus the value
        # encoding(s) (PLAIN, or RLE_DICTIONARY + PLAIN for a dictionary chunk).
        last = w.write_field_begin(TC_LIST, 2, last)
        w.write_list_begin(TC_I32, len(self.encodings))
        for e in self.encodings:
            w.write_i32(Int32(e))
        last = w.write_field_begin(TC_LIST, 3, last)
        w.write_list_begin(TC_BINARY, len(self.path_in_schema))
        for p in self.path_in_schema:
            w.write_string(p)
        last = w.write_field_begin(TC_I32, 4, last)
        w.write_i32(Int32(self.codec))
        last = w.write_field_begin(TC_I64, 5, last)
        w.write_i64(Int64(self.num_values))
        last = w.write_field_begin(TC_I64, 6, last)
        w.write_i64(Int64(self.total_uncompressed_size))
        last = w.write_field_begin(TC_I64, 7, last)
        w.write_i64(Int64(self.total_compressed_size))
        last = w.write_field_begin(TC_I64, 9, last)
        w.write_i64(Int64(self.data_page_offset))
        if self.dictionary_page_offset >= 0:
            last = w.write_field_begin(TC_I64, 11, last)
            w.write_i64(Int64(self.dictionary_page_offset))
        if self.null_count >= 0 or self.distinct_count >= 0 or self.has_min_max:
            # Statistics (field 12): null_count (3), distinct_count (4), and the
            # modern max_value (5) / min_value (6) with their exactness flags
            # (7/8). Fields are written in ascending id order per the compact
            # protocol.
            last = w.write_field_begin(TC_STRUCT, 12, last)
            var slast = 0
            if self.null_count >= 0:
                slast = w.write_field_begin(TC_I64, 3, slast)
                w.write_i64(Int64(self.null_count))
            if self.distinct_count >= 0:
                slast = w.write_field_begin(TC_I64, 4, slast)
                w.write_i64(Int64(self.distinct_count))
            if self.has_min_max:
                slast = w.write_field_begin(TC_BINARY, 5, slast)
                w.write_bytes(Span(self.max_value))
                slast = w.write_field_begin(TC_BINARY, 6, slast)
                w.write_bytes(Span(self.min_value))
                slast = w.write_bool_field(True, 7, slast)  # is_max_value_exact
                _ = w.write_bool_field(True, 8, slast)  # is_min_value_exact
            w.write_field_stop()
        if self.bloom_filter_offset >= 0:
            last = w.write_field_begin(TC_I64, 14, last)
            w.write_i64(Int64(self.bloom_filter_offset))
            _ = w.write_field_begin(TC_I32, 15, last)
            w.write_i32(Int32(self.bloom_filter_length))
        w.write_field_stop()


# ---------------------------------------------------------------------------
# Page index — OffsetIndex / ColumnIndex, stored near the footer and pointed to
# by ColumnChunk.{offset,column}_index_offset. Read on demand (selective scans
# only); a full-row-group scan never touches them.
# ---------------------------------------------------------------------------


struct PageLocation(Copyable, Movable, ThriftWritable):
    """Locates one data page: byte `offset` in the file, `compressed_page_size`
    (header included), and `first_row_index` (row-group-relative, on a row
    boundary)."""

    var offset: Int
    var compressed_page_size: Int
    var first_row_index: Int

    def __init__(out self):
        self.offset = 0
        self.compressed_page_size = 0
        self.first_row_index = 0

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.offset = Int(r.read_i64())
            elif f.id == 2:
                out.compressed_page_size = Int(r.read_i32())
            elif f.id == 3:
                out.first_row_index = Int(r.read_i64())
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I64, 1, last)
        w.write_i64(Int64(self.offset))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.compressed_page_size))
        _ = w.write_field_begin(TC_I64, 3, last)
        w.write_i64(Int64(self.first_row_index))
        w.write_field_stop()


struct OffsetIndex(Copyable, Movable, ThriftWritable):
    """Per-page locations for one column chunk (field 1 = list<PageLocation>).
    """

    var page_locations: List[PageLocation]

    def __init__(out self):
        self.page_locations = List[PageLocation]()

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.page_locations.append(PageLocation.read(r))
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        _ = w.write_struct_list(1, 0, self.page_locations)
        w.write_field_stop()


struct ColumnIndex(Copyable, Movable, ThriftWritable):
    """Per-page statistics for one column chunk: `null_pages[i]` (page holds only
    nulls → its min/max are empty), the PLAIN-encoded `min_values`/`max_values`
    bounds, `boundary_order` (0 UNORDERED, 1 ASCENDING, 2 DESCENDING), and the
    optional per-page `null_counts`. Bounds follow the column's ColumnOrder."""

    var null_pages: List[Bool]
    var min_values: List[List[UInt8]]
    var max_values: List[List[UInt8]]
    var boundary_order: Int
    var null_counts: List[Int]

    def __init__(out self):
        self.null_pages = List[Bool]()
        self.min_values = List[List[UInt8]]()
        self.max_values = List[List[UInt8]]()
        self.boundary_order = 0
        self.null_counts = List[Int]()

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                var _, n = r.read_list_header()
                for _ in range(n):
                    # list bool elements: 1 = true, 2 = false (compact protocol)
                    out.null_pages.append(Int(r.read_byte()) == 1)
            elif f.id == 2:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.min_values.append(List[UInt8](Span(r.read_bytes())))
            elif f.id == 3:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.max_values.append(List[UInt8](Span(r.read_bytes())))
            elif f.id == 4:
                out.boundary_order = Int(r.read_i32())
            elif f.id == 5:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.null_counts.append(Int(r.read_i64()))
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        # null_pages: list<bool> — compact lists encode each bool as one byte
        # (1 = true, 2 = false) under a BOOL_TRUE element-type nibble.
        last = w.write_field_begin(TC_LIST, 1, last)
        w.write_list_begin(TC_BOOL_TRUE, len(self.null_pages))
        for p in self.null_pages:
            w.buf.append(UInt8(1) if p else UInt8(2))
        last = w.write_field_begin(TC_LIST, 2, last)
        w.write_list_begin(TC_BINARY, len(self.min_values))
        for i in range(len(self.min_values)):
            w.write_bytes(Span(self.min_values[i]))
        last = w.write_field_begin(TC_LIST, 3, last)
        w.write_list_begin(TC_BINARY, len(self.max_values))
        for i in range(len(self.max_values)):
            w.write_bytes(Span(self.max_values[i]))
        last = w.write_field_begin(TC_I32, 4, last)
        w.write_i32(Int32(self.boundary_order))
        if len(self.null_counts) > 0:
            _ = w.write_field_begin(TC_LIST, 5, last)
            w.write_list_begin(TC_I64, len(self.null_counts))
            for i in range(len(self.null_counts)):
                w.write_i64(Int64(self.null_counts[i]))
        w.write_field_stop()


struct ColumnChunk(Copyable, Movable, ThriftWritable):
    var file_offset: Int
    var meta_data: ColumnMetaData
    var offset_index_offset: Int  # -1 if absent
    var offset_index_length: Int
    var column_index_offset: Int  # -1 if absent
    var column_index_length: Int
    var offset_index_out: OffsetIndex  # writer-only: per-data-page locations
    var column_index_out: ColumnIndex  # writer-only: per-data-page stats
    var write_column_index: Bool  # writer-only: emit the ColumnIndex
    var bloom_bytes: List[UInt8]  # writer-only: the built bloom filter bitset

    def __init__(out self):
        self.file_offset = 0
        self.meta_data = ColumnMetaData()
        self.offset_index_offset = -1
        self.offset_index_length = 0
        self.column_index_offset = -1
        self.column_index_length = 0
        self.offset_index_out = OffsetIndex()
        self.column_index_out = ColumnIndex()
        self.write_column_index = False
        self.bloom_bytes = List[UInt8]()

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 2:
                out.file_offset = Int(r.read_i64())
            elif f.id == 3:
                out.meta_data = ColumnMetaData.read(r)
            elif f.id == 4:
                out.offset_index_offset = Int(r.read_i64())
            elif f.id == 5:
                out.offset_index_length = Int(r.read_i32())
            elif f.id == 6:
                out.column_index_offset = Int(r.read_i64())
            elif f.id == 7:
                out.column_index_length = Int(r.read_i32())
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I64, 2, last)
        w.write_i64(Int64(self.file_offset))
        last = w.write_field_begin(TC_STRUCT, 3, last)
        self.meta_data.write(w)
        # page-index pointers (written after the pages, before the footer)
        if self.offset_index_offset >= 0:
            last = w.write_field_begin(TC_I64, 4, last)
            w.write_i64(Int64(self.offset_index_offset))
            last = w.write_field_begin(TC_I32, 5, last)
            w.write_i32(Int32(self.offset_index_length))
        if self.column_index_offset >= 0:
            last = w.write_field_begin(TC_I64, 6, last)
            w.write_i64(Int64(self.column_index_offset))
            _ = w.write_field_begin(TC_I32, 7, last)
            w.write_i32(Int32(self.column_index_length))
        w.write_field_stop()


struct RowGroup(Copyable, Movable, ThriftWritable):
    var columns: List[ColumnChunk]
    var total_byte_size: Int
    var num_rows: Int

    def __init__(out self):
        self.columns = List[ColumnChunk]()
        self.total_byte_size = 0
        self.num_rows = 0

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.columns.append(ColumnChunk.read(r))
            elif f.id == 2:
                out.total_byte_size = Int(r.read_i64())
            elif f.id == 3:
                out.num_rows = Int(r.read_i64())
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = w.write_struct_list(1, 0, self.columns)
        last = w.write_field_begin(TC_I64, 2, last)
        w.write_i64(Int64(self.total_byte_size))
        _ = w.write_field_begin(TC_I64, 3, last)
        w.write_i64(Int64(self.num_rows))
        w.write_field_stop()


struct KeyValue(Copyable, Movable, ThriftWritable):
    """A file-level metadata entry: a `key` and its optional string `value`."""

    var key: String
    var value: String

    def __init__(out self, key: String = String(), value: String = String()):
        self.key = key
        self.value = value

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.key = r.read_string()
            elif f.id == 2:
                out.value = r.read_string()
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_BINARY, 1, last)
        w.write_string(self.key)
        _ = w.write_field_begin(TC_BINARY, 2, last)
        w.write_string(self.value)
        w.write_field_stop()


struct FileMetaData(Copyable, Movable):
    var version: Int
    var schema: List[SchemaElement]
    var num_rows: Int
    var row_groups: List[RowGroup]
    var key_value_metadata: List[KeyValue]
    var created_by: String

    def __init__(out self):
        self.version = 1
        self.schema = List[SchemaElement]()
        self.num_rows = 0
        self.row_groups = List[RowGroup]()
        self.key_value_metadata = List[KeyValue]()
        self.created_by = String()

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.version = Int(r.read_i32())
            elif f.id == 2:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.schema.append(SchemaElement.read(r))
            elif f.id == 3:
                out.num_rows = Int(r.read_i64())
            elif f.id == 4:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.row_groups.append(RowGroup.read(r))
            elif f.id == 5:
                var _, n = r.read_list_header()
                for _ in range(n):
                    out.key_value_metadata.append(KeyValue.read(r))
            elif f.id == 6:
                out.created_by = r.read_string()
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.version))
        last = w.write_struct_list(2, last, self.schema)
        last = w.write_field_begin(TC_I64, 3, last)
        w.write_i64(Int64(self.num_rows))
        last = w.write_struct_list(4, last, self.row_groups)
        if len(self.key_value_metadata) > 0:
            last = w.write_struct_list(5, last, self.key_value_metadata)
        last = w.write_field_begin(TC_BINARY, 6, last)
        w.write_string(self.created_by)
        # column_orders (7): one ColumnOrder per leaf column, each the
        # TYPE_ORDER (TypeDefinedOrder) union member — declares that min_value/
        # max_value follow the column's logical (type-defined) ordering, so
        # readers trust unsigned-int and byte-array bounds.
        var num_leaves = 0
        for s in self.schema:
            if s.num_children == 0:
                num_leaves += 1
        _ = w.write_field_begin(TC_LIST, 7, last)
        w.write_list_begin(TC_STRUCT, num_leaves)
        for _ in range(num_leaves):
            # ColumnOrder union: field 1 = TypeDefinedOrder (empty struct)
            _ = w.write_field_begin(TC_STRUCT, 1, 0)
            w.write_field_stop()  # empty TypeDefinedOrder
            w.write_field_stop()  # end ColumnOrder union
        w.write_field_stop()

    @staticmethod
    def write_magic(mut out: List[UInt8]):
        """Append the 4-byte `PAR1` magic (file header and footer)."""
        out.append(0x50)
        out.append(0x41)
        out.append(0x52)
        out.append(0x31)

    @staticmethod
    def read_footer[o: Origin[mut=False]](data: Span[UInt8, o]) raises -> Self:
        """Parse the file footer: the trailing 8 bytes are a 4-byte LE metadata
        length then the `PAR1` magic; the thrift blob precedes them."""
        var n = len(data)
        if n < 12:
            raise Error("parquet: file too small")
        if not (
            data[n - 4] == 0x50
            and data[n - 3] == 0x41
            and data[n - 2] == 0x52
            and data[n - 1] == 0x31
        ):
            raise Error("parquet: missing PAR1 footer magic")
        var meta_len = (
            Int(data[n - 8])
            | (Int(data[n - 7]) << 8)
            | (Int(data[n - 6]) << 16)
            | (Int(data[n - 5]) << 24)
        )
        var start = n - 8 - meta_len
        if start < 4:
            raise Error("parquet: corrupt footer length")
        var r = ThriftCompactReader(data, start)
        return Self.read(r)

    def write_footer(self, mut out: List[UInt8]) raises:
        """Serialize the thrift blob, then the 4-byte LE length and `PAR1` magic
        that close the file."""
        var w = ThriftCompactWriter()
        self.write(w)
        var meta_len = len(w.buf)
        out.extend(Span(w.buf))
        for i in range(4):
            out.append(UInt8((meta_len >> (i * 8)) & 0xFF))
        Self.write_magic(out)
