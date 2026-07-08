"""Parquet metadata structures and their Thrift Compact Protocol serde.

Hand-written per-struct read/write against `thrift.CompactReader`/`CompactWriter`,
covering exactly the subset of `parquet.thrift` the reader/writer touches:
the file footer (`FileMetaData` → `RowGroup` → `ColumnChunk` → `ColumnMetaData`),
the schema element list, and the page headers. Field IDs and enum values come
straight from the `parquet.thrift` IDL.

Unknown/optional fields are skipped via the protocol's recursive `skip`, so
forward-compatible files still parse.
"""

from .thrift import (
    CompactReader,
    CompactWriter,
    TC_STOP,
    TC_I32,
    TC_I64,
    TC_BINARY,
    TC_LIST,
    TC_STRUCT,
)

# --- Physical types (SchemaElement.type / ColumnMetaData.type) ---
comptime PT_BOOLEAN: Int = 0
comptime PT_INT32: Int = 1
comptime PT_INT64: Int = 2
comptime PT_INT96: Int = 3
comptime PT_FLOAT: Int = 4
comptime PT_DOUBLE: Int = 5
comptime PT_BYTE_ARRAY: Int = 6
comptime PT_FIXED_LEN_BYTE_ARRAY: Int = 7

# --- Field repetition ---
comptime REP_REQUIRED: Int = 0
comptime REP_OPTIONAL: Int = 1
comptime REP_REPEATED: Int = 2

# --- ConvertedType (legacy logical annotation) ---
comptime CT_UTF8: Int = 0
comptime CT_MAP: Int = 1
comptime CT_LIST: Int = 3
comptime CT_DECIMAL: Int = 5
comptime CT_DATE: Int = 6
comptime CT_TIME_MILLIS: Int = 7
comptime CT_TIME_MICROS: Int = 8
comptime CT_TIMESTAMP_MILLIS: Int = 9
comptime CT_TIMESTAMP_MICROS: Int = 10
comptime CT_UINT_8: Int = 11
comptime CT_UINT_16: Int = 12
comptime CT_UINT_32: Int = 13
comptime CT_UINT_64: Int = 14
comptime CT_INT_8: Int = 15
comptime CT_INT_16: Int = 16
comptime CT_INT_32: Int = 17
comptime CT_INT_64: Int = 18

# --- LogicalType union member ids ---
comptime LT_STRING: Int = 1
comptime LT_MAP: Int = 2
comptime LT_LIST: Int = 3
comptime LT_DECIMAL: Int = 5
comptime LT_DATE: Int = 6
comptime LT_TIME: Int = 7
comptime LT_TIMESTAMP: Int = 8
comptime LT_INTEGER: Int = 10

# --- Encoding ---
comptime ENC_PLAIN: Int = 0
comptime ENC_PLAIN_DICTIONARY: Int = 2
comptime ENC_RLE: Int = 3
comptime ENC_BIT_PACKED: Int = 4
comptime ENC_DELTA_BINARY_PACKED: Int = 5
comptime ENC_DELTA_LENGTH_BYTE_ARRAY: Int = 6
comptime ENC_DELTA_BYTE_ARRAY: Int = 7
comptime ENC_RLE_DICTIONARY: Int = 8
comptime ENC_BYTE_STREAM_SPLIT: Int = 9

# --- PageType ---
comptime PAGE_DATA: Int = 0
comptime PAGE_INDEX: Int = 1
comptime PAGE_DICTIONARY: Int = 2
comptime PAGE_DATA_V2: Int = 3

comptime PARQUET_MAGIC: List[UInt8] = [0x50, 0x41, 0x52, 0x31]  # "PAR1"


def read_footer[
    o: Origin[mut=False]
](data: Span[UInt8, o]) raises -> FileMetaData:
    """Parse the file footer: trailing 8 bytes are a 4-byte LE metadata length
    followed by the `PAR1` magic; the thrift `FileMetaData` blob precedes it."""
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
    var r = CompactReader(data, start)
    return FileMetaData.read(r)


def read_page_header[
    o: Origin[mut=False]
](data: Span[UInt8, o], mut pos: Int) raises -> PageHeader:
    """Read the page header at `pos`, advancing `pos` to the page body."""
    var r = CompactReader(data, pos)
    var ph = PageHeader.read(r)
    pos = r.pos
    return ph^


def write_magic(mut out: List[UInt8]):
    """Append the 4-byte `PAR1` magic (file header and footer)."""
    out.append(0x50)
    out.append(0x41)
    out.append(0x52)
    out.append(0x31)


def append_page_header(mut out: List[UInt8], ph: PageHeader) raises -> Int:
    """Serialize a page header into `out`; return its byte length."""
    var w = CompactWriter()
    ph.write(w)
    out.extend(Span(w.buf))
    return len(w.buf)


def write_footer(
    mut out: List[UInt8],
    meta: FileMetaData,
    row_group_encodings: List[List[Int]],
) raises:
    """Serialize the `FileMetaData` thrift blob, then the 4-byte LE length and
    the `PAR1` magic that close the file."""
    var w = CompactWriter()
    meta.write(w, row_group_encodings)
    var meta_len = len(w.buf)
    out.extend(Span(w.buf))
    for i in range(4):
        out.append(UInt8((meta_len >> (i * 8)) & 0xFF))
    write_magic(out)


# ---------------------------------------------------------------------------
# SchemaElement — one node of the flattened schema tree
# ---------------------------------------------------------------------------


struct SchemaElement(Copyable, Movable):
    var type: Int  # physical type, -1 if a group node
    var type_length: Int  # for FIXED_LEN_BYTE_ARRAY
    var repetition_type: Int  # REP_*, -1 at root
    var name: String
    var num_children: Int  # >0 for group nodes
    var converted_type: Int  # CT_*, -1 if absent
    var scale: Int
    var precision: Int
    var field_id: Int
    var logical_type: Int  # LT_* union member id, -1 if absent
    var logical_unit: Int  # TimeUnit for TIMESTAMP/TIME: 1=ms 2=us 3=ns, else -1
    var logical_utc: Bool  # isAdjustedToUTC for TIMESTAMP/TIME

    def __init__(out self):
        self.type = -1
        self.type_length = 0
        self.repetition_type = -1
        self.name = String()
        self.num_children = 0
        self.converted_type = -1
        self.scale = 0
        self.precision = 0
        self.field_id = -1
        self.logical_type = -1
        self.logical_unit = -1
        self.logical_utc = False

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.type = Int(r.read_i32())
            elif fid == 2:
                out.type_length = Int(r.read_i32())
            elif fid == 3:
                out.repetition_type = Int(r.read_i32())
            elif fid == 4:
                out.name = r.read_string()
            elif fid == 5:
                out.num_children = Int(r.read_i32())
            elif fid == 6:
                out.converted_type = Int(r.read_i32())
            elif fid == 7:
                out.scale = Int(r.read_i32())
            elif fid == 8:
                out.precision = Int(r.read_i32())
            elif fid == 9:
                out.field_id = Int(r.read_i32())
            elif fid == 10:
                _read_logical_type(r, out)
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter):
        var last = 0
        if self.type >= 0:
            last = w.write_field_begin(TC_I32, 1, last)
            w.write_i32(Int32(self.type))
        if self.type == PT_FIXED_LEN_BYTE_ARRAY:
            last = w.write_field_begin(TC_I32, 2, last)
            w.write_i32(Int32(self.type_length))
        if self.repetition_type >= 0:
            last = w.write_field_begin(TC_I32, 3, last)
            w.write_i32(Int32(self.repetition_type))
        last = w.write_field_begin(TC_BINARY, 4, last)
        w.write_string(self.name)
        if self.num_children > 0:
            last = w.write_field_begin(TC_I32, 5, last)
            w.write_i32(Int32(self.num_children))
        if self.converted_type >= 0:
            last = w.write_field_begin(TC_I32, 6, last)
            w.write_i32(Int32(self.converted_type))
        w.write_field_stop()


def _read_logical_type[
    o: Origin[mut=False]
](mut r: CompactReader[o], mut out: SchemaElement) raises:
    """Parse the `LogicalType` union into `out.logical_type`, and for TIMESTAMP /
    TIME also the nested `TimeUnit` (`logical_unit`) and `isAdjustedToUTC`."""
    var last = 0
    while True:
        var ftype, fid = r.read_field_header(last)
        if ftype == TC_STOP:
            break
        last = fid
        out.logical_type = fid
        if fid == LT_TIMESTAMP or fid == LT_TIME:
            # TimestampType/TimeType = {1: bool isAdjustedToUTC, 2: TimeUnit unit}
            var l2 = 0
            while True:
                var ft2, fid2 = r.read_field_header(l2)
                if ft2 == TC_STOP:
                    break
                l2 = fid2
                if fid2 == 1:
                    out.logical_utc = r.read_bool(ft2)
                elif fid2 == 2:
                    # TimeUnit union: the single set field id is the unit
                    var l3 = 0
                    while True:
                        var ft3, fid3 = r.read_field_header(l3)
                        if ft3 == TC_STOP:
                            break
                        l3 = fid3
                        out.logical_unit = fid3
                        r.skip(ft3)
                else:
                    r.skip(ft2)
        else:
            r.skip(ftype)


# ---------------------------------------------------------------------------
# Statistics, page headers
# ---------------------------------------------------------------------------


struct DataPageHeader(Copyable, Movable):
    var num_values: Int
    var encoding: Int
    var definition_level_encoding: Int
    var repetition_level_encoding: Int

    def __init__(out self):
        self.num_values = 0
        self.encoding = ENC_PLAIN
        self.definition_level_encoding = ENC_RLE
        self.repetition_level_encoding = ENC_RLE

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.num_values = Int(r.read_i32())
            elif fid == 2:
                out.encoding = Int(r.read_i32())
            elif fid == 3:
                out.definition_level_encoding = Int(r.read_i32())
            elif fid == 4:
                out.repetition_level_encoding = Int(r.read_i32())
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_values))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.encoding))
        last = w.write_field_begin(TC_I32, 3, last)
        w.write_i32(Int32(self.definition_level_encoding))
        last = w.write_field_begin(TC_I32, 4, last)
        w.write_i32(Int32(self.repetition_level_encoding))
        w.write_field_stop()


struct DataPageHeaderV2(Copyable, Movable):
    var num_values: Int
    var num_nulls: Int
    var num_rows: Int
    var encoding: Int
    var definition_levels_byte_length: Int
    var repetition_levels_byte_length: Int
    var is_compressed: Bool

    def __init__(out self):
        self.num_values = 0
        self.num_nulls = 0
        self.num_rows = 0
        self.encoding = ENC_PLAIN
        self.definition_levels_byte_length = 0
        self.repetition_levels_byte_length = 0
        self.is_compressed = True

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.num_values = Int(r.read_i32())
            elif fid == 2:
                out.num_nulls = Int(r.read_i32())
            elif fid == 3:
                out.num_rows = Int(r.read_i32())
            elif fid == 4:
                out.encoding = Int(r.read_i32())
            elif fid == 5:
                out.definition_levels_byte_length = Int(r.read_i32())
            elif fid == 6:
                out.repetition_levels_byte_length = Int(r.read_i32())
            elif fid == 7:
                out.is_compressed = r.read_bool(ftype)
            else:
                r.skip(ftype)
        return out^


struct DictionaryPageHeader(Copyable, Movable):
    var num_values: Int
    var encoding: Int

    def __init__(out self):
        self.num_values = 0
        self.encoding = ENC_PLAIN

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.num_values = Int(r.read_i32())
            elif fid == 2:
                out.encoding = Int(r.read_i32())
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_values))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.encoding))
        w.write_field_stop()


struct PageHeader(Copyable, Movable):
    var type: Int
    var uncompressed_page_size: Int
    var compressed_page_size: Int
    var data_page_header: Optional[DataPageHeader]
    var data_page_header_v2: Optional[DataPageHeaderV2]
    var dictionary_page_header: Optional[DictionaryPageHeader]

    def __init__(out self):
        self.type = -1
        self.uncompressed_page_size = 0
        self.compressed_page_size = 0
        self.data_page_header = None
        self.data_page_header_v2 = None
        self.dictionary_page_header = None

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.type = Int(r.read_i32())
            elif fid == 2:
                out.uncompressed_page_size = Int(r.read_i32())
            elif fid == 3:
                out.compressed_page_size = Int(r.read_i32())
            elif fid == 5:
                out.data_page_header = DataPageHeader.read(r)
            elif fid == 7:
                out.dictionary_page_header = DictionaryPageHeader.read(r)
            elif fid == 8:
                out.data_page_header_v2 = DataPageHeaderV2.read(r)
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.type))
        last = w.write_field_begin(TC_I32, 2, last)
        w.write_i32(Int32(self.uncompressed_page_size))
        last = w.write_field_begin(TC_I32, 3, last)
        w.write_i32(Int32(self.compressed_page_size))
        if self.data_page_header:
            _ = w.write_field_begin(TC_STRUCT, 5, last)
            self.data_page_header.value().write(w)
        elif self.dictionary_page_header:
            _ = w.write_field_begin(TC_STRUCT, 7, last)
            self.dictionary_page_header.value().write(w)
        w.write_field_stop()

    @staticmethod
    def data_page(
        uncompressed_size: Int, compressed_size: Int, num_values: Int
    ) -> Self:
        """Build a v1 data-page header (PLAIN values, RLE levels)."""
        var ph = Self()
        ph.type = PAGE_DATA
        ph.uncompressed_page_size = uncompressed_size
        ph.compressed_page_size = compressed_size
        var dph = DataPageHeader()
        dph.num_values = num_values
        dph.encoding = ENC_PLAIN
        dph.definition_level_encoding = ENC_RLE
        dph.repetition_level_encoding = ENC_RLE
        ph.data_page_header = dph^
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
    var null_count: Int  # -1 if unknown; written as Statistics.null_count

    def __init__(out self):
        self.type = -1
        self.path_in_schema = List[String]()
        self.codec = 0
        self.num_values = 0
        self.total_uncompressed_size = 0
        self.total_compressed_size = 0
        self.data_page_offset = 0
        self.dictionary_page_offset = -1
        self.null_count = -1

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 3:
                var et, n = r.read_list_header()
                for _ in range(n):
                    out.path_in_schema.append(r.read_string())
            elif fid == 1:
                out.type = Int(r.read_i32())
            elif fid == 4:
                out.codec = Int(r.read_i32())
            elif fid == 5:
                out.num_values = Int(r.read_i64())
            elif fid == 6:
                out.total_uncompressed_size = Int(r.read_i64())
            elif fid == 7:
                out.total_compressed_size = Int(r.read_i64())
            elif fid == 9:
                out.data_page_offset = Int(r.read_i64())
            elif fid == 11:
                out.dictionary_page_offset = Int(r.read_i64())
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter, encoding: Int):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.type))
        # encodings: list<Encoding> = [RLE, encoding]
        last = w.write_field_begin(TC_LIST, 2, last)
        w.write_list_begin(TC_I32, 2)
        w.write_i32(Int32(ENC_RLE))
        w.write_i32(Int32(encoding))
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
        if self.null_count >= 0:
            # Statistics (field 12) with only null_count (field 3) populated
            last = w.write_field_begin(TC_STRUCT, 12, last)
            _ = w.write_field_begin(TC_I64, 3, 0)
            w.write_i64(Int64(self.null_count))
            w.write_field_stop()
        w.write_field_stop()


struct ColumnChunk(Copyable, Movable):
    var file_offset: Int
    var meta_data: ColumnMetaData

    def __init__(out self):
        self.file_offset = 0
        self.meta_data = ColumnMetaData()

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 2:
                out.file_offset = Int(r.read_i64())
            elif fid == 3:
                out.meta_data = ColumnMetaData.read(r)
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter, encoding: Int):
        var last = 0
        last = w.write_field_begin(TC_I64, 2, last)
        w.write_i64(Int64(self.file_offset))
        last = w.write_field_begin(TC_STRUCT, 3, last)
        self.meta_data.write(w, encoding)
        w.write_field_stop()


struct RowGroup(Copyable, Movable):
    var columns: List[ColumnChunk]
    var total_byte_size: Int
    var num_rows: Int

    def __init__(out self):
        self.columns = List[ColumnChunk]()
        self.total_byte_size = 0
        self.num_rows = 0

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                var et, n = r.read_list_header()
                for _ in range(n):
                    out.columns.append(ColumnChunk.read(r))
            elif fid == 2:
                out.total_byte_size = Int(r.read_i64())
            elif fid == 3:
                out.num_rows = Int(r.read_i64())
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter, encodings: List[Int]):
        var last = 0
        last = w.write_field_begin(TC_LIST, 1, last)
        w.write_list_begin(TC_STRUCT, len(self.columns))
        for i in range(len(self.columns)):
            self.columns[i].write(w, encodings[i])
        last = w.write_field_begin(TC_I64, 2, last)
        w.write_i64(Int64(self.total_byte_size))
        last = w.write_field_begin(TC_I64, 3, last)
        w.write_i64(Int64(self.num_rows))
        w.write_field_stop()


struct FileMetaData(Copyable, Movable):
    var version: Int
    var schema: List[SchemaElement]
    var num_rows: Int
    var row_groups: List[RowGroup]
    var created_by: String

    def __init__(out self):
        self.version = 1
        self.schema = List[SchemaElement]()
        self.num_rows = 0
        self.row_groups = List[RowGroup]()
        self.created_by = String()

    @staticmethod
    def read[o: Origin[mut=False]](mut r: CompactReader[o]) raises -> Self:
        var out = Self()
        var last = 0
        while True:
            var ftype, fid = r.read_field_header(last)
            if ftype == TC_STOP:
                break
            last = fid
            if fid == 1:
                out.version = Int(r.read_i32())
            elif fid == 2:
                var et, n = r.read_list_header()
                for _ in range(n):
                    out.schema.append(SchemaElement.read(r))
            elif fid == 3:
                out.num_rows = Int(r.read_i64())
            elif fid == 4:
                var et, n = r.read_list_header()
                for _ in range(n):
                    out.row_groups.append(RowGroup.read(r))
            elif fid == 6:
                out.created_by = r.read_string()
            else:
                r.skip(ftype)
        return out^

    def write(self, mut w: CompactWriter, row_group_encodings: List[List[Int]]):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.version))
        last = w.write_field_begin(TC_LIST, 2, last)
        w.write_list_begin(TC_STRUCT, len(self.schema))
        for s in self.schema:
            s.write(w)
        last = w.write_field_begin(TC_I64, 3, last)
        w.write_i64(Int64(self.num_rows))
        last = w.write_field_begin(TC_LIST, 4, last)
        w.write_list_begin(TC_STRUCT, len(self.row_groups))
        for i in range(len(self.row_groups)):
            self.row_groups[i].write(w, row_group_encodings[i])
        _ = w.write_field_begin(TC_BINARY, 6, last)
        w.write_string(self.created_by)
        w.write_field_stop()
