"""Arrow IPC file and stream reader/writer.

Public API
----------
Top-level functions:
  read_ipc_file(path)              → List[RecordBatch]
  write_ipc_file(path, batches)
  read_ipc_stream(path)            → List[RecordBatch]
  write_ipc_stream(path, batches)
  read_ipc_file_schema(path)       → RecordBatch (0-row, schema only)
  read_ipc_stream_schema(path)     → RecordBatch (0-row, schema only)

Reader/writer classes for incremental I/O:
  RecordBatchFileWriter(path, schema)   — write IPC file incrementally
  RecordBatchStreamWriter(path, schema) — write IPC stream incrementally
  RecordBatchFileReader(path)           — read IPC file with random access
  RecordBatchStreamReader(path)         — read IPC stream sequentially

Supported types: bool, int8-64, uint8-64, float16/32/64, binary, utf8,
list, fixed_size_list, struct, dictionary.
"""

from std.math import ceildiv
from std.pathlib import Path
from std.sys import size_of
from .arrays import AnyArray, ArrayData, DictionaryArray, NullArray
from .buffers import Buffer, Bitmap
from .schema import Schema
from .tabular import RecordBatch
from .utils import LittleEndian
from . import dtypes as dt


# ---------------------------------------------------------------------------
# Arrow IPC wire protocol constants
# ---------------------------------------------------------------------------

comptime _HEADER_SCHEMA: UInt8 = 1
comptime _HEADER_DICTIONARY_BATCH: UInt8 = 2
comptime _HEADER_RECORD_BATCH: UInt8 = 3
comptime _METADATA_VERSION_V5: Int16 = 4
comptime _ENDIANNESS_LITTLE: Int16 = 0
comptime _TYPE_NULL: UInt8 = 1
comptime _TYPE_INT: UInt8 = 2
comptime _TYPE_FLOATING_POINT: UInt8 = 3
comptime _TYPE_BINARY: UInt8 = 4
comptime _TYPE_UTF8: UInt8 = 5
comptime _TYPE_LARGE_BINARY: UInt8 = 19
comptime _TYPE_LARGE_UTF8: UInt8 = 20
comptime _TYPE_LARGE_LIST: UInt8 = 21
comptime _TYPE_INTERVAL: UInt8 = 11
comptime _INTERVAL_UNIT_YEAR_MONTH: UInt16 = 0
comptime _INTERVAL_UNIT_DAY_TIME: UInt16 = 1
comptime _INTERVAL_UNIT_MONTH_DAY_NANO: UInt16 = 2

comptime _TYPE_BOOL: UInt8 = 6
comptime _TYPE_DECIMAL: UInt8 = 7
comptime _TYPE_DATE: UInt8 = 8
comptime _TYPE_TIME: UInt8 = 9
comptime _TYPE_TIMESTAMP: UInt8 = 10
comptime _TYPE_LIST: UInt8 = 12
comptime _TYPE_STRUCT: UInt8 = 13
comptime _TYPE_FIXED_SIZE_BINARY: UInt8 = 15
comptime _TYPE_FIXED_SIZE_LIST: UInt8 = 16
comptime _TYPE_DURATION: UInt8 = 18
comptime _PRECISION_HALF: UInt16 = 0
comptime _PRECISION_SINGLE: UInt16 = 1
comptime _PRECISION_DOUBLE: UInt16 = 2
comptime _DATE_UNIT_DAY: UInt16 = 0
comptime _DATE_UNIT_MILLISECOND: UInt16 = 1
comptime _TIME_UNIT_SECOND: UInt16 = 0
comptime _TIME_UNIT_MILLISECOND: UInt16 = 1
comptime _TIME_UNIT_MICROSECOND: UInt16 = 2
comptime _TIME_UNIT_NANOSECOND: UInt16 = 3


def _magic() -> List[UInt8]:
    var m = List[UInt8](capacity=8)
    m.append(65)  # 'A'
    m.append(82)  # 'R'
    m.append(82)  # 'R'
    m.append(79)  # 'O'
    m.append(87)  # 'W'
    m.append(49)  # '1'
    m.append(0)
    m.append(0)
    return m^


# ---------------------------------------------------------------------------
# Internal wire format structs
# ---------------------------------------------------------------------------


@fieldwise_init
struct _FieldNode(ImplicitlyCopyable, Movable):
    var length: Int64
    var null_count: Int64


@fieldwise_init
struct _BodyBuffer(ImplicitlyCopyable, Movable):
    var offset: Int64
    var length: Int64


@fieldwise_init
struct _Block(ImplicitlyCopyable, Movable):
    var offset: Int64
    var metadata_length: Int32
    var body_length: Int64


struct _DictPair(Copyable, Movable):
    """A (dict_id, values) pair collected from an ArrayData tree."""

    var dict_id: Int
    var values: AnyArray

    def __init__(out self, dict_id: Int, var values: AnyArray):
        self.dict_id = dict_id
        self.values = values^

    def __init__(out self, *, copy: Self):
        self.dict_id = copy.dict_id
        self.values = copy.values.copy()


struct _DictLookup(Movable):
    """Result of searching a (_dtype, _FieldIpcInfo) tree for a dict_id."""

    var value_type: dt.AnyDataType
    var value_ipc_info: _FieldIpcInfo

    def __init__(
        out self,
        var value_type: dt.AnyDataType,
        var value_ipc_info: _FieldIpcInfo,
    ):
        self.value_type = value_type^
        self.value_ipc_info = value_ipc_info^


struct _FieldIpcInfo(Copyable, Movable):
    """IPC-only metadata shadow for a field: dict_id and value-type children.

    Mirrors the Arrow type tree but carries only IPC serialization metadata
    (dict_ids assigned during schema write/read), decoupled from the logical
    type system.  For dictionary fields, `children` holds the IPC infos of the
    VALUE TYPE's children; for non-dict fields it holds the direct type children
    (list child, struct children).  `dict_id == -1` means no DictionaryEncoding.
    """

    var dict_id: Int
    var children: List[_FieldIpcInfo]

    def __init__(out self, dict_id: Int = -1):
        self.dict_id = dict_id
        self.children = List[_FieldIpcInfo]()

    def __init__(out self, dict_id: Int, var children: List[_FieldIpcInfo]):
        self.dict_id = dict_id
        self.children = children^

    def __init__(out self, *, copy: Self):
        self.dict_id = copy.dict_id
        self.children = copy.children.copy()

    # Explicit (empty) destructor so this self-referential struct
    # (`children: List[_FieldIpcInfo]`) is ImplicitlyDeletable; fields are still
    # destroyed automatically after the body runs.
    def __del__(deinit self):
        pass

    @staticmethod
    def find(
        dtype: dt.AnyDataType, ipc_info: _FieldIpcInfo, target_id: Int
    ) raises -> Optional[_DictLookup]:
        """Search the (dtype, ipc_info) shadow tree for dict_id == target_id.

        Returns a `_DictLookup` whose `value_ipc_info` has `dict_id=-1` with
        the matched node's children, so that nested dicts inside the value type
        can still be resolved during decoding.
        """
        if ipc_info.dict_id == target_id:
            ref d = dtype.as_dictionary()
            var vt_ipc = _FieldIpcInfo(-1, ipc_info.children.copy())
            return _DictLookup(d.value_type().copy(), vt_ipc^)
        if dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var vt_ipc = _FieldIpcInfo(-1, ipc_info.children.copy())
            return _FieldIpcInfo.find(d.value_type().copy(), vt_ipc^, target_id)
        elif dtype.is_list():
            if len(ipc_info.children) > 0:
                return _FieldIpcInfo.find(
                    dtype.as_list().value_type().copy(),
                    ipc_info.children[0].copy(),
                    target_id,
                )
        elif dtype.is_struct():
            ref st = dtype.as_struct()
            for i in range(len(st.fields)):
                if i < len(ipc_info.children):
                    var found = _FieldIpcInfo.find(
                        st.fields[i].dtype.copy(),
                        ipc_info.children[i].copy(),
                        target_id,
                    )
                    if found:
                        return found^
        return None

    @staticmethod
    def find_in_schema(
        fields: List[dt.Field],
        ipc_infos: List[_FieldIpcInfo],
        target_id: Int,
    ) raises -> Optional[_DictLookup]:
        """Search schema fields and their IPC shadow tree for the given dict_id.
        """
        for i in range(len(fields)):
            if i < len(ipc_infos):
                var found = _FieldIpcInfo.find(
                    fields[i].dtype.copy(), ipc_infos[i].copy(), target_id
                )
                if found:
                    return found^
        return None


# ---------------------------------------------------------------------------
# Little-endian integer read/write helpers
# ---------------------------------------------------------------------------


def _read_le[T: DType](buf: List[UInt8], pos: Int) raises -> Scalar[T]:
    """Bounds-checked little-endian read from a `List` (IPC parses untrusted
    metadata, so an out-of-range offset raises rather than reading past the end).
    """
    if pos < 0 or pos + size_of[T]() - 1 >= len(buf):
        raise Error("ipc: _read_le out of bounds at " + String(pos))
    return LittleEndian.fixed[T](Span(buf), pos)


def _padding_to(pos: Int, alignment: Int) -> Int:
    return (alignment - (pos % alignment)) % alignment


def _pad_to(mut buf: List[UInt8], alignment: Int):
    var r = len(buf) % alignment
    if r != 0:
        for _ in range(alignment - r):
            buf.append(UInt8(0))


# ---------------------------------------------------------------------------
# Generic FlatBuffers codec
# ---------------------------------------------------------------------------


@fieldwise_init
struct _FieldOffset(Copyable, Movable):
    """Slot index and tail-distance recorded after prepending a field."""

    var slot: Int
    var at: UInt32


struct _FlatbufWriter(Movable):
    """Prepend-model FlatBuffers builder. `_buf[_head:]` is valid content."""

    var _buf: List[UInt8]
    var _head: Int
    var _min_align: Int

    def __init__(out self, initial_capacity: Int = 256):
        self._buf = List[UInt8](capacity=initial_capacity)
        for _ in range(initial_capacity):
            self._buf.append(UInt8(0))
        self._head = initial_capacity
        self._min_align = 1

    def _grow(mut self) raises:
        var old_size = len(self._buf)
        if old_size > 0x3FFF_FFFF_FFFF_FFFF:
            raise Error("flatbuffers: buffer too large to grow")
        var new_size = old_size * 2
        var written = old_size - self._head
        var new_buf = List[UInt8](capacity=new_size)
        var new_head = new_size - written
        for _ in range(new_head):
            new_buf.append(UInt8(0))
        for i in range(written):
            new_buf.append(self._buf[self._head + i])
        self._buf = new_buf^
        self._head = new_head

    def _prep(mut self, align: Int, needed: Int = 0) raises:
        if align > self._min_align:
            self._min_align = align
        while self._head < needed + align:
            self._grow()
        var written = len(self._buf) - self._head
        var pad = _padding_to(written + needed, align)
        for _ in range(pad):
            self._head -= 1
            self._buf[self._head] = UInt8(0)

    def offset(self) -> UInt32:
        return UInt32(len(self._buf) - self._head)

    def prepend_u8(mut self, val: UInt8) raises -> UInt32:
        self._prep(1, 1)
        self._head -= 1
        self._buf[self._head] = val
        return self.offset()

    def prepend_bool(mut self, val: Bool) raises -> UInt32:
        return self.prepend_u8(UInt8(1) if val else UInt8(0))

    def prepend_u16(mut self, val: UInt16) raises -> UInt32:
        self._prep(2, 2)
        self._head -= 2
        LittleEndian.write[DType.uint16](self._buf, self._head, val)
        return self.offset()

    def prepend_i16(mut self, val: Int16) raises -> UInt32:
        self._prep(2, 2)
        self._head -= 2
        LittleEndian.write[DType.int16](self._buf, self._head, val)
        return self.offset()

    def prepend_i32(mut self, val: Int32) raises -> UInt32:
        self._prep(4, 4)
        self._head -= 4
        LittleEndian.write[DType.int32](self._buf, self._head, val)
        return self.offset()

    def prepend_i64(mut self, val: Int64) raises -> UInt32:
        self._prep(8, 8)
        self._head -= 8
        LittleEndian.write[DType.int64](self._buf, self._head, val)
        return self.offset()

    def prepend_uoffset(mut self, val: UInt32) raises -> UInt32:
        self._prep(4, 4)
        self._head -= 4
        var stored_abs = len(self._buf) - self._head
        LittleEndian.write[DType.uint32](
            self._buf, self._head, UInt32(stored_abs - Int(val))
        )
        return self.offset()

    def create_string(mut self, s: String) raises -> UInt32:
        var bytes = s.as_bytes()
        var n = len(bytes)
        self._prep(4, n + 1)
        self._head -= 1
        self._buf[self._head] = UInt8(0)
        for i in range(n - 1, -1, -1):
            self._head -= 1
            self._buf[self._head] = bytes[i]
        self._head -= 4
        LittleEndian.write[DType.uint32](self._buf, self._head, UInt32(n))
        return self.offset()

    def create_vector_u8(mut self, data: List[UInt8]) raises -> UInt32:
        var n = len(data)
        self._prep(4, n)
        for i in range(n - 1, -1, -1):
            self._head -= 1
            self._buf[self._head] = data[i]
        self._head -= 4
        LittleEndian.write[DType.uint32](self._buf, self._head, UInt32(n))
        return self.offset()

    def create_vector_offsets(mut self, offsets: List[UInt32]) raises -> UInt32:
        var n = len(offsets)
        self._prep(4, n * 4)
        for i in range(n - 1, -1, -1):
            self._head -= 4
            var stored_abs = len(self._buf) - self._head
            var rel = stored_abs - Int(offsets[i])
            LittleEndian.write[DType.uint32](self._buf, self._head, UInt32(rel))
        self._head -= 4
        LittleEndian.write[DType.uint32](self._buf, self._head, UInt32(n))
        return self.offset()

    def create_vector_structs(
        mut self,
        data: List[UInt8],
        count: Int,
        struct_size: Int,
        struct_align: Int,
    ) raises -> UInt32:
        if count < 0:
            raise Error("flatbuffers: create_vector_structs: negative count")
        if len(data) != count * struct_size:
            raise Error(
                "flatbuffers: create_vector_structs: data length "
                + String(len(data))
                + " != count("
                + String(count)
                + ") * struct_size("
                + String(struct_size)
                + ")"
            )
        var n_bytes = count * struct_size
        self._prep(struct_align, n_bytes)
        for i in range(n_bytes - 1, -1, -1):
            self._head -= 1
            self._buf[self._head] = data[i]
        self._head -= 4
        LittleEndian.write[DType.uint32](self._buf, self._head, UInt32(count))
        return self.offset()

    def finish(mut self, root: UInt32) raises -> List[UInt8]:
        self._prep(self._min_align, 4)
        self._head -= 4
        var table_pos_in_result = (len(self._buf) - Int(root)) - self._head
        LittleEndian.write[DType.uint32](
            self._buf, self._head, UInt32(table_pos_in_result)
        )
        var result = List[UInt8](capacity=len(self._buf) - self._head)
        for i in range(self._head, len(self._buf)):
            result.append(self._buf[i])
        return result^

    def write_table(
        mut self,
        fields: List[_FieldOffset],
        table_start: UInt32,
    ) raises -> UInt32:
        var num_slots = 0
        for i in range(len(fields)):
            var s = fields[i].slot + 1
            if s > num_slots:
                num_slots = s

        self._prep(4, 4)
        self._head -= 4
        LittleEndian.write[DType.int32](self._buf, self._head, Int32(0))
        var table_pos = self.offset()

        var object_size = UInt16(Int(table_pos) - Int(table_start))
        var vtable_size = UInt16(4 + num_slots * 2)

        var vtable_slots = List[UInt16](capacity=num_slots)
        for s in range(num_slots):
            var voff = UInt16(0)
            for i in range(len(fields)):
                if fields[i].slot == s:
                    voff = UInt16(Int(table_pos) - Int(fields[i].at))
                    break
            vtable_slots.append(voff)

        self._prep(1, 4 + num_slots * 2)
        for s in range(num_slots - 1, -1, -1):
            self._head -= 2
            LittleEndian.write[DType.uint16](
                self._buf, self._head, vtable_slots[s]
            )
        self._head -= 2
        LittleEndian.write[DType.uint16](self._buf, self._head, object_size)
        self._head -= 2
        LittleEndian.write[DType.uint16](self._buf, self._head, vtable_size)
        var new_vt_offset = self.offset()

        var soffset = Int32(Int(new_vt_offset) - Int(table_pos))
        LittleEndian.write[DType.int32](
            self._buf, len(self._buf) - Int(table_pos), soffset
        )

        return table_pos


struct _FlatbufReader(Movable):
    """Generic FlatBuffers reader. Table positions are absolute byte offsets."""

    var _buf: List[UInt8]

    def __init__(out self, var buf: List[UInt8]):
        self._buf = buf^

    def root(self) raises -> UInt32:
        return _read_le[DType.uint32](self._buf, 0)

    def _field_voffset(self, table_pos: UInt32, slot: Int) raises -> UInt16:
        var tp = Int(table_pos)
        var soffset_raw = _read_le[DType.uint32](self._buf, tp)
        var vt = Int(table_pos - soffset_raw)
        if vt < 0 or vt >= len(self._buf):
            raise Error(
                "flatbuffers: vtable position out of bounds: " + String(vt)
            )
        var vt_size = Int(_read_le[DType.uint16](self._buf, vt))
        var slot_byte = 4 + slot * 2
        if slot_byte + 1 >= vt_size:
            return UInt16(0)
        if vt + slot_byte + 1 >= len(self._buf):
            return UInt16(0)
        return _read_le[DType.uint16](self._buf, vt + slot_byte)

    def read_u8(
        self, tp: UInt32, slot: Int, default: UInt8 = 0
    ) raises -> UInt8:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return _read_le[DType.uint8](self._buf, Int(tp) + Int(voff))

    def read_u16(
        self, tp: UInt32, slot: Int, default: UInt16 = 0
    ) raises -> UInt16:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return _read_le[DType.uint16](self._buf, Int(tp) + Int(voff))

    def read_i32(
        self, tp: UInt32, slot: Int, default: Int32 = 0
    ) raises -> Int32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return _read_le[DType.int32](self._buf, Int(tp) + Int(voff))

    def read_i64(
        self, tp: UInt32, slot: Int, default: Int64 = 0
    ) raises -> Int64:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return _read_le[DType.int64](self._buf, Int(tp) + Int(voff))

    def read_bool(
        self, tp: UInt32, slot: Int, default: Bool = False
    ) raises -> Bool:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return _read_le[DType.uint8](self._buf, Int(tp) + Int(voff)) != UInt8(0)

    def read_string(
        self, tp: UInt32, slot: Int, default: String = ""
    ) raises -> String:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        var ref_pos = Int(tp) + Int(voff)
        var str_pos = ref_pos + Int(_read_le[DType.uint32](self._buf, ref_pos))
        var length = Int(_read_le[DType.uint32](self._buf, str_pos))
        if str_pos + 4 + length > len(self._buf):
            raise Error("flatbuffers: string extends beyond buffer")
        var bytes = List[UInt8](capacity=length)
        for i in range(length):
            bytes.append(self._buf[str_pos + 4 + i])
        return String(unsafe_from_utf8=bytes^)

    def read_vector(self, tp: UInt32, slot: Int) raises -> UInt32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            raise Error(
                "flatbuffers: absent offset field at slot " + String(slot)
            )
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + _read_le[DType.uint32](self._buf, ref_pos)

    def read_table(self, tp: UInt32, slot: Int) raises -> UInt32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            raise Error(
                "flatbuffers: absent offset field at slot " + String(slot)
            )
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + _read_le[DType.uint32](self._buf, ref_pos)

    def vector_len(self, vec_pos: UInt32) raises -> UInt32:
        return _read_le[DType.uint32](self._buf, Int(vec_pos))

    def vec_offset(self, vec_pos: UInt32, i: UInt32) raises -> UInt32:
        var vlen = self.vector_len(vec_pos)
        if i >= vlen:
            raise Error(
                "flatbuffers: vec index "
                + String(i)
                + " >= len "
                + String(vlen)
            )
        var elem_pos = Int(vec_pos) + 4 + Int(i) * 4
        return UInt32(elem_pos) + _read_le[DType.uint32](self._buf, elem_pos)

    def vec_struct_bytes(
        self, vec_pos: UInt32, i: UInt32, struct_size: Int
    ) raises -> List[UInt8]:
        var vlen = self.vector_len(vec_pos)
        if i >= vlen:
            raise Error(
                "flatbuffers: vec_struct_bytes index "
                + String(i)
                + " >= len "
                + String(vlen)
            )
        var start = Int(vec_pos) + 4 + Int(i) * struct_size
        var end = start + struct_size
        if end > len(self._buf):
            raise Error("flatbuffers: vec_struct_bytes extends beyond buffer")
        var result = List[UInt8](capacity=struct_size)
        for j in range(struct_size):
            result.append(self._buf[start + j])
        return result^


# ---------------------------------------------------------------------------
# Arrow IPC metadata encoder
# ---------------------------------------------------------------------------


struct _IpcEncoder(Movable):
    """Encodes Arrow IPC metadata (schema, record batch, footer) into FlatBuffers.
    """

    var _fb: _FlatbufWriter

    def __init__(out self, capacity: Int = 512):
        self._fb = _FlatbufWriter(capacity)

    @staticmethod
    def _time_unit_to_wire(unit: dt.TimeUnit) -> UInt16:
        if unit == dt.second:
            return _TIME_UNIT_SECOND
        elif unit == dt.millisecond:
            return _TIME_UNIT_MILLISECOND
        elif unit == dt.microsecond:
            return _TIME_UNIT_MICROSECOND
        else:
            return _TIME_UNIT_NANOSECOND

    @staticmethod
    def encode_schema(schema: Schema) raises -> List[UInt8]:
        var enc = _IpcEncoder(256)
        var schema_pos = enc._write_schema_table(schema)
        return enc._finish(schema_pos)

    @staticmethod
    def encode_schema_message(schema: Schema) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var schema_pos = enc._write_schema_table(schema)
        var msg_pos = enc._write_message_table(
            _HEADER_SCHEMA, schema_pos, Int64(0)
        )
        return enc._finish(msg_pos)

    @staticmethod
    def encode_record_batch(
        length: Int64,
        nodes: List[_FieldNode],
        buffers: List[_BodyBuffer],
    ) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var nodes_vec = enc._write_field_nodes_vec(nodes)
        var bufs_vec = enc._write_body_buffers_vec(buffers)
        var rb_pos = enc._write_record_batch_table(length, nodes_vec, bufs_vec)

        var max_end = Int64(0)
        for b in buffers:
            max_end = max(max_end, b.offset + b.length)
        var r = max_end % Int64(8)
        var body_len = max_end + (Int64(8) - r) % Int64(8)

        var msg_pos = enc._write_message_table(
            _HEADER_RECORD_BATCH, rb_pos, body_len
        )
        return enc._finish(msg_pos)

    @staticmethod
    def encode_footer(
        schema: Schema,
        dict_blocks: List[_Block],
        blocks: List[_Block],
    ) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var schema_pos = enc._write_schema_table(schema)
        var dicts_vec = enc._write_blocks_vec(dict_blocks)
        var blocks_vec = enc._write_blocks_vec(blocks)
        var footer_pos = enc._write_footer_table(
            schema_pos, dicts_vec, blocks_vec
        )
        return enc._finish(footer_pos)

    @staticmethod
    def frame_message(metadata: List[UInt8], body: List[UInt8]) -> List[UInt8]:
        var out = List[UInt8]()
        LittleEndian.append[DType.uint32](out, UInt32(0xFFFFFFFF))
        var meta_len = len(metadata)
        var padded_len = meta_len + (8 - meta_len % 8) % 8
        LittleEndian.append[DType.int32](out, Int32(padded_len))
        out.extend(Span(metadata))
        _pad_to(out, 8)
        out.extend(Span(body))
        return out^

    def _finish(mut self, root: UInt32) raises -> List[UInt8]:
        return self._fb.finish(root)

    def _write_record_batch_table(
        mut self,
        length: Int64,
        nodes_vec: UInt32,
        bufs_vec: UInt32,
    ) raises -> UInt32:
        var ts = self._fb.offset()
        var bv_at = self._fb.prepend_uoffset(bufs_vec)
        var nv_at = self._fb.prepend_uoffset(nodes_vec)
        var ln_at = self._fb.prepend_i64(length)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, ln_at))
        flds.append(_FieldOffset(1, nv_at))
        flds.append(_FieldOffset(2, bv_at))
        return self._fb.write_table(flds, ts)

    def _write_footer_table(
        mut self,
        schema_pos: UInt32,
        dicts_vec: UInt32,
        blocks_vec: UInt32,
    ) raises -> UInt32:
        var ts = self._fb.offset()
        var bv_at = self._fb.prepend_uoffset(blocks_vec)
        var dv_at = self._fb.prepend_uoffset(dicts_vec)
        var sc_at = self._fb.prepend_uoffset(schema_pos)
        var ver_at = self._fb.prepend_i16(_METADATA_VERSION_V5)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, ver_at))
        flds.append(_FieldOffset(1, sc_at))
        flds.append(_FieldOffset(2, dv_at))
        flds.append(_FieldOffset(3, bv_at))
        return self._fb.write_table(flds, ts)

    def _write_field_nodes_vec(
        mut self, nodes: List[_FieldNode]
    ) raises -> UInt32:
        var n = len(nodes)
        var data = List[UInt8](capacity=n * 16)
        for _ in range(n * 16):
            data.append(UInt8(0))
        for i in range(n):
            LittleEndian.write[DType.int64](data, i * 16, nodes[i].length)
            LittleEndian.write[DType.int64](
                data, i * 16 + 8, nodes[i].null_count
            )
        return self._fb.create_vector_structs(data, n, 16, 8)

    def _write_body_buffers_vec(
        mut self, bufs: List[_BodyBuffer]
    ) raises -> UInt32:
        var n = len(bufs)
        var data = List[UInt8](capacity=n * 16)
        for _ in range(n * 16):
            data.append(UInt8(0))
        for i in range(n):
            LittleEndian.write[DType.int64](data, i * 16, bufs[i].offset)
            LittleEndian.write[DType.int64](data, i * 16 + 8, bufs[i].length)
        return self._fb.create_vector_structs(data, n, 16, 8)

    def _write_blocks_vec(mut self, blocks: List[_Block]) raises -> UInt32:
        var n = len(blocks)
        var data = List[UInt8](capacity=n * 24)
        for _ in range(n * 24):
            data.append(UInt8(0))
        for i in range(n):
            LittleEndian.write[DType.int64](data, i * 24, blocks[i].offset)
            LittleEndian.write[DType.int32](
                data, i * 24 + 8, blocks[i].metadata_length
            )
            # 4 bytes padding at offset 12 (Arrow Block struct alignment)
            LittleEndian.write[DType.int64](
                data, i * 24 + 16, blocks[i].body_length
            )
        return self._fb.create_vector_structs(data, n, 24, 8)

    def _type_code(self, dtype: dt.AnyDataType) raises -> UInt8:
        if dtype.is_null():
            return _TYPE_NULL
        elif dtype.is_bool():
            return _TYPE_BOOL
        elif dtype.is_integer():
            return _TYPE_INT
        elif dtype.is_floating_point():
            return _TYPE_FLOATING_POINT
        elif dtype.is_binary():
            return _TYPE_BINARY
        elif dtype.is_large_binary():
            return _TYPE_LARGE_BINARY
        elif dtype.is_string():
            return _TYPE_UTF8
        elif dtype.is_large_string():
            return _TYPE_LARGE_UTF8
        elif dtype.is_list():
            return _TYPE_LIST
        elif dtype.is_large_list():
            return _TYPE_LARGE_LIST
        elif dtype.is_fixed_size_list():
            return _TYPE_FIXED_SIZE_LIST
        elif dtype.is_fixed_size_binary():
            return _TYPE_FIXED_SIZE_BINARY
        elif dtype.is_date32() or dtype.is_date64():
            return _TYPE_DATE
        elif dtype.is_time32() or dtype.is_time64():
            return _TYPE_TIME
        elif dtype.is_timestamp():
            return _TYPE_TIMESTAMP
        elif dtype.is_duration():
            return _TYPE_DURATION
        elif dtype.is_interval():
            return _TYPE_INTERVAL
        elif dtype.is_decimal():
            return _TYPE_DECIMAL
        elif dtype.is_struct():
            return _TYPE_STRUCT
        elif dtype.is_dictionary():
            # Schema encodes the value type; DictionaryEncoding carries index type.
            return self._type_code(dtype.as_dictionary().value_type())
        else:
            raise Error("_IpcEncoder: unsupported dtype: " + String(dtype))

    def _write_type_table(mut self, dtype: dt.AnyDataType) raises -> UInt32:
        if (
            dtype.is_null()
            or dtype.is_bool()
            or dtype.is_binary()
            or dtype.is_large_binary()
            or dtype.is_string()
            or dtype.is_large_string()
            or dtype.is_list()
            or dtype.is_large_list()
            or dtype.is_struct()
        ):
            var ts = self._fb.offset()
            return self._fb.write_table(List[_FieldOffset](), ts)
        elif dtype.is_integer():
            var bw: Int32
            var signed: Bool
            if dtype == dt.int8:
                bw = 8
                signed = True
            elif dtype == dt.int16:
                bw = 16
                signed = True
            elif dtype == dt.int32:
                bw = 32
                signed = True
            elif dtype == dt.int64:
                bw = 64
                signed = True
            elif dtype == dt.uint8:
                bw = 8
                signed = False
            elif dtype == dt.uint16:
                bw = 16
                signed = False
            elif dtype == dt.uint32:
                bw = 32
                signed = False
            else:
                bw = 64
                signed = False
            var ts = self._fb.offset()
            var signed_at = self._fb.prepend_bool(signed)
            var bw_at = self._fb.prepend_i32(bw)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, bw_at))
            flds.append(_FieldOffset(1, signed_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_floating_point():
            var prec: UInt16
            if dtype == dt.float16:
                prec = _PRECISION_HALF
            elif dtype == dt.float32:
                prec = _PRECISION_SINGLE
            else:
                prec = _PRECISION_DOUBLE
            var ts = self._fb.offset()
            var prec_at = self._fb.prepend_u16(prec)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, prec_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_fixed_size_list():
            ref fsl = dtype.as_fixed_size_list()
            var ts = self._fb.offset()
            var sz_at = self._fb.prepend_i32(Int32(fsl.size))
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, sz_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_fixed_size_binary():
            ref fsb = dtype.as_fixed_size_binary()
            var ts = self._fb.offset()
            var bw_at = self._fb.prepend_i32(Int32(fsb.byte_width))
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, bw_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_date32():
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(_DATE_UNIT_DAY)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_date64():
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(_DATE_UNIT_MILLISECOND)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_time32() or dtype.is_time64():
            var unit: dt.TimeUnit
            var bw: Int32
            if dtype.is_time32():
                unit = dtype.as_time32().unit
                bw = 32
            else:
                unit = dtype.as_time64().unit
                bw = 64
            var ipc_unit = _IpcEncoder._time_unit_to_wire(unit)
            var ts = self._fb.offset()
            var bw_at = self._fb.prepend_i32(bw)
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            flds.append(_FieldOffset(1, bw_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_timestamp():
            ref tstype = dtype.as_timestamp()
            var ipc_unit = _IpcEncoder._time_unit_to_wire(tstype.unit)
            var ts = self._fb.offset()
            var tz_at: Optional[UInt32] = None
            if tstype.timezone:
                var tz_str_pos = self._fb.create_string(tstype.timezone)
                tz_at = self._fb.prepend_uoffset(tz_str_pos)
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            if tz_at:
                flds.append(_FieldOffset(1, tz_at.value()))
            return self._fb.write_table(flds, ts)
        elif dtype.is_duration():
            var unit = dtype.as_duration().unit
            var ipc_unit = _IpcEncoder._time_unit_to_wire(unit)
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_interval():
            var unit: UInt16
            if dtype.is_year_month_interval():
                unit = _INTERVAL_UNIT_YEAR_MONTH
            elif dtype.is_day_time_interval():
                unit = _INTERVAL_UNIT_DAY_TIME
            else:
                unit = _INTERVAL_UNIT_MONTH_DAY_NANO
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(unit)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_decimal():
            var precision: Int32
            var scale: Int32
            var bit_width: Int32
            if dtype.is_decimal32():
                ref d = dtype.as_decimal32()
                precision = Int32(d.precision)
                scale = Int32(d.scale)
                bit_width = 32
            elif dtype.is_decimal64():
                ref d = dtype.as_decimal64()
                precision = Int32(d.precision)
                scale = Int32(d.scale)
                bit_width = 64
            elif dtype.is_decimal128():
                ref d = dtype.as_decimal128()
                precision = Int32(d.precision)
                scale = Int32(d.scale)
                bit_width = 128
            else:
                ref d = dtype.as_decimal256()
                precision = Int32(d.precision)
                scale = Int32(d.scale)
                bit_width = 256
            var ts = self._fb.offset()
            var bw_at = self._fb.prepend_i32(bit_width)
            var scale_at = self._fb.prepend_i32(scale)
            var prec_at = self._fb.prepend_i32(precision)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, prec_at))
            flds.append(_FieldOffset(1, scale_at))
            flds.append(_FieldOffset(2, bw_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_dictionary():
            return self._write_type_table(dtype.as_dictionary().value_type())
        else:
            raise Error(
                "_IpcEncoder: unsupported dtype for type table: "
                + String(dtype)
            )

    def _write_dictionary_encoding_table(
        mut self, dict_id: Int64, index_dtype: dt.AnyDataType, ordered: Bool
    ) raises -> UInt32:
        var idx_type_pos = self._write_type_table(index_dtype)
        var ts = self._fb.offset()
        var ord_at = self._fb.prepend_bool(ordered)
        var idx_at = self._fb.prepend_uoffset(idx_type_pos)
        var id_at = self._fb.prepend_i64(dict_id)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, id_at))
        flds.append(_FieldOffset(1, idx_at))
        flds.append(_FieldOffset(2, ord_at))
        return self._fb.write_table(flds, ts)

    def _write_dictionary_batch_table(
        mut self, dict_id: Int64, is_delta: Bool, rb_pos: UInt32
    ) raises -> UInt32:
        var ts = self._fb.offset()
        var delta_at = self._fb.prepend_bool(is_delta)
        var data_at = self._fb.prepend_uoffset(rb_pos)
        var id_at = self._fb.prepend_i64(dict_id)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, id_at))
        flds.append(_FieldOffset(1, data_at))
        flds.append(_FieldOffset(2, delta_at))
        return self._fb.write_table(flds, ts)

    def _write_kv_vec(
        mut self, metadata: Dict[String, String]
    ) raises -> UInt32:
        var kv_positions = List[UInt32]()
        for entry in metadata.items():
            var key_pos = self._fb.create_string(entry.key)
            var val_pos = self._fb.create_string(entry.value)
            var ts = self._fb.offset()
            var val_at = self._fb.prepend_uoffset(val_pos)
            var key_at = self._fb.prepend_uoffset(key_pos)
            var kv_flds = List[_FieldOffset]()
            kv_flds.append(_FieldOffset(0, key_at))
            kv_flds.append(_FieldOffset(1, val_at))
            kv_positions.append(self._fb.write_table(kv_flds, ts))
        return self._fb.create_vector_offsets(kv_positions)

    def _write_field(
        mut self, f: dt.Field, mut next_dict_id: Int
    ) raises -> UInt32:
        """Write a Field FlatBuffer table. Returns the table offset.

        Dict_ids are assigned in DFS inner-first order: `next_dict_id` is
        mutated in-place so nested (child) dictionaries receive lower ids
        than their enclosing parent dictionaries.
        """
        var child_positions = List[UInt32]()
        var dtype = f.dtype.copy()
        var own_dict_id = -1

        if dtype.is_list():
            child_positions.append(
                self._write_field(
                    dtype.as_list().value_field().copy(), next_dict_id
                )
            )
        elif dtype.is_large_list():
            child_positions.append(
                self._write_field(
                    dtype.as_large_list().value_field().copy(), next_dict_id
                )
            )
        elif dtype.is_fixed_size_list():
            child_positions.append(
                self._write_field(
                    dtype.as_fixed_size_list().value_field().copy(),
                    next_dict_id,
                )
            )
        elif dtype.is_struct():
            ref st = dtype.as_struct()
            for i in range(len(st.fields)):
                child_positions.append(
                    self._write_field(st.fields[i], next_dict_id)
                )
        elif dtype.is_dictionary():
            # Write children of the VALUE TYPE first (inner dicts before outer).
            var val_type = dtype.as_dictionary().value_type().copy()
            if val_type.is_list():
                child_positions.append(
                    self._write_field(
                        val_type.as_list().value_field().copy(), next_dict_id
                    )
                )
            elif val_type.is_struct():
                ref st = val_type.as_struct()
                for i in range(len(st.fields)):
                    child_positions.append(
                        self._write_field(st.fields[i], next_dict_id)
                    )
            # Assign this dictionary's own id AFTER all nested children.
            own_dict_id = next_dict_id
            next_dict_id += 1

        var type_code = self._type_code(dtype)
        var type_pos = self._write_type_table(dtype)
        var name_pos = self._fb.create_string(f.name)

        var children_vec_pos: Optional[UInt32] = None
        if len(child_positions) > 0:
            children_vec_pos = self._fb.create_vector_offsets(child_positions)

        var meta_vec_pos: Optional[UInt32] = None
        if len(f.metadata) > 0:
            meta_vec_pos = self._write_kv_vec(f.metadata)

        var dict_enc_pos: Optional[UInt32] = None
        if own_dict_id >= 0:
            ref d = dtype.as_dictionary()
            dict_enc_pos = self._write_dictionary_encoding_table(
                Int64(own_dict_id), d.index_type().copy(), d.ordered
            )

        var ts = self._fb.offset()
        var ch_at = UInt32(0)
        if children_vec_pos:
            ch_at = self._fb.prepend_uoffset(children_vec_pos.value())
        var meta_at = UInt32(0)
        if meta_vec_pos:
            meta_at = self._fb.prepend_uoffset(meta_vec_pos.value())
        var de_at = UInt32(0)
        if dict_enc_pos:
            de_at = self._fb.prepend_uoffset(dict_enc_pos.value())
        var tp_at = self._fb.prepend_uoffset(type_pos)
        var tc_at = self._fb.prepend_u8(type_code)
        var nb_at = self._fb.prepend_bool(f.nullable)
        var nm_at = self._fb.prepend_uoffset(name_pos)

        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, nm_at))
        flds.append(_FieldOffset(1, nb_at))
        flds.append(_FieldOffset(2, tc_at))
        flds.append(_FieldOffset(3, tp_at))
        if dict_enc_pos:
            flds.append(_FieldOffset(4, de_at))
        if children_vec_pos:
            flds.append(_FieldOffset(5, ch_at))
        if meta_vec_pos:
            flds.append(_FieldOffset(6, meta_at))
        return self._fb.write_table(flds, ts)

    def _write_schema_table(mut self, schema: Schema) raises -> UInt32:
        var field_positions = List[UInt32]()
        var next_dict_id = 0
        for f in schema.fields:
            field_positions.append(self._write_field(f, next_dict_id))
        var fields_vec = self._fb.create_vector_offsets(field_positions)

        var meta_vec_pos: Optional[UInt32] = None
        if len(schema.metadata) > 0:
            meta_vec_pos = self._write_kv_vec(schema.metadata)

        var ts = self._fb.offset()
        var meta_at = UInt32(0)
        if meta_vec_pos:
            meta_at = self._fb.prepend_uoffset(meta_vec_pos.value())
        var fv_at = self._fb.prepend_uoffset(fields_vec)
        var en_at = self._fb.prepend_i16(_ENDIANNESS_LITTLE)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, en_at))
        flds.append(_FieldOffset(1, fv_at))
        if meta_vec_pos:
            flds.append(_FieldOffset(2, meta_at))
        return self._fb.write_table(flds, ts)

    def _write_message_table(
        mut self,
        header_type: UInt8,
        header_pos: UInt32,
        body_len: Int64,
    ) raises -> UInt32:
        var ts = self._fb.offset()
        var bl_at = self._fb.prepend_i64(body_len)
        var hdr_at = self._fb.prepend_uoffset(header_pos)
        var ht_at = self._fb.prepend_u8(header_type)
        var ver_at = self._fb.prepend_i16(_METADATA_VERSION_V5)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, ver_at))
        flds.append(_FieldOffset(1, ht_at))
        flds.append(_FieldOffset(2, hdr_at))
        flds.append(_FieldOffset(3, bl_at))
        return self._fb.write_table(flds, ts)


# ---------------------------------------------------------------------------
# Arrow IPC metadata decoder
# ---------------------------------------------------------------------------


struct _IpcDecoder(Movable):
    """Decodes Arrow IPC metadata (schema, record batch, footer) from FlatBuffers.
    """

    var _r: _FlatbufReader

    def __init__(out self, var buf: List[UInt8]):
        self._r = _FlatbufReader(buf^)

    def peek_header(self) raises -> UInt8:
        return self._r.read_u8(self._r.root(), 1, 0)

    def body_length(self) raises -> Int64:
        """Body length in bytes from the Message table (slot 3)."""
        return self._r.read_i64(self._r.root(), 3, 0)

    def peek_dict_id(self) raises -> Int:
        """Dict id from a DictionaryBatch message header."""
        var msg_tp = self._r.root()
        var db_pos = self._r.read_table(msg_tp, 2)
        return Int(self._r.read_i64(db_pos, 0, 0))

    @staticmethod
    def _wire_to_time_unit(v: UInt16) -> dt.TimeUnit:
        if v == _TIME_UNIT_SECOND:
            return dt.second
        elif v == _TIME_UNIT_MILLISECOND:
            return dt.millisecond
        elif v == _TIME_UNIT_MICROSECOND:
            return dt.microsecond
        else:
            return dt.nanosecond

    def _read_kv_vec(
        self, table_pos: UInt32, slot: Int
    ) raises -> Dict[String, String]:
        var result = Dict[String, String]()
        var meta_vec = self._r.read_vector(table_pos, slot)
        var n = Int(self._r.vector_len(meta_vec))
        for i in range(n):
            var kv_pos = self._r.vec_offset(meta_vec, UInt32(i))
            var key = self._r.read_string(kv_pos, 0)
            var val = self._r.read_string(kv_pos, 1)
            result[key] = val
        return result^

    def decode_schema(self, mut out_ipc: List[_FieldIpcInfo]) raises -> Schema:
        var msg_pos = self._r.root()
        var schema_pos = self._r.read_table(msg_pos, 2)
        var fields = self._decode_schema_fields(schema_pos, out_ipc)
        var metadata = Dict[String, String]()
        try:
            metadata = self._read_kv_vec(schema_pos, 2)
        except:
            pass
        return Schema(fields=fields^, metadata=metadata^)

    def decode_dict_batch(
        mut self,
        value_dtype: dt.AnyDataType,
        values_ipc_info: _FieldIpcInfo,
        var body: List[UInt8],
        dict_values: List[AnyArray] = List[AnyArray](),
    ) raises -> AnyArray:
        var msg_tp = self._r.root()
        var db_pos = self._r.read_table(msg_tp, 2)
        var rb_pos = self._r.read_table(db_pos, 1)
        var nodes = List[_FieldNode]()
        var bufs = List[_BodyBuffer]()
        var _l = self._read_record_batch_meta(rb_pos, nodes, bufs)
        var batch_dec = _BatchDecoder(body^, 0, nodes^, bufs^, dict_values)
        return batch_dec.read_array(value_dtype, values_ipc_info)

    def decode_record_batch(
        mut self,
        schema: Schema,
        ipc_infos: List[_FieldIpcInfo],
        var body: List[UInt8],
        dict_values: List[AnyArray] = List[AnyArray](),
    ) raises -> RecordBatch:
        var msg_tp = self._r.root()
        var hdr_type = self._r.read_u8(msg_tp, 1, 0)
        if Int(hdr_type) != Int(_HEADER_RECORD_BATCH):
            raise Error(
                "_IpcDecoder: expected record-batch header, got "
                + String(Int(hdr_type))
            )
        var rb_pos = self._r.read_table(msg_tp, 2)
        var nodes = List[_FieldNode]()
        var bufs = List[_BodyBuffer]()
        var _l = self._read_record_batch_meta(rb_pos, nodes, bufs)
        var batch_dec = _BatchDecoder(body^, 0, nodes^, bufs^, dict_values)
        var columns = List[AnyArray]()
        for i in range(len(schema.fields)):
            var ipc = (
                ipc_infos[i].copy() if i < len(ipc_infos) else _FieldIpcInfo()
            )
            columns.append(batch_dec.read_array(schema.fields[i].dtype, ipc))
        return RecordBatch(schema=schema, columns=columns^)

    def read_footer(
        self,
        mut dict_blocks: List[_Block],
        mut blocks: List[_Block],
        mut out_ipc: List[_FieldIpcInfo],
    ) raises -> Schema:
        var footer_pos = self._r.root()
        var schema_pos = self._r.read_table(footer_pos, 1)
        var fields = self._decode_schema_fields(schema_pos, out_ipc)
        var metadata = Dict[String, String]()
        try:
            metadata = self._read_kv_vec(schema_pos, 2)
        except:
            pass
        var dv = self._r.read_vector(footer_pos, 2)
        var nd = Int(self._r.vector_len(dv))
        for i in range(nd):
            var sb = self._r.vec_struct_bytes(dv, UInt32(i), 24)
            dict_blocks.append(
                _Block(
                    _read_le[DType.int64](sb, 0),
                    _read_le[DType.int32](sb, 8),
                    _read_le[DType.int64](sb, 16),
                )
            )
        var rb_vec = self._r.read_vector(footer_pos, 3)
        var n = Int(self._r.vector_len(rb_vec))
        for i in range(n):
            var sb = self._r.vec_struct_bytes(rb_vec, UInt32(i), 24)
            blocks.append(
                _Block(
                    _read_le[DType.int64](sb, 0),
                    _read_le[DType.int32](sb, 8),
                    _read_le[DType.int64](sb, 16),
                )
            )
        return Schema(fields=fields^, metadata=metadata^)

    def _decode_schema_fields(
        self, schema_pos: UInt32, mut out_ipc: List[_FieldIpcInfo]
    ) raises -> List[dt.Field]:
        var fields = List[dt.Field]()
        var fields_vec = self._r.read_vector(schema_pos, 1)
        var n = Int(self._r.vector_len(fields_vec))
        for i in range(n):
            var fp = self._r.vec_offset(fields_vec, UInt32(i))
            fields.append(self._read_field(fp, out_ipc))
        return fields^

    def _read_record_batch_meta(
        self,
        rb_pos: UInt32,
        mut nodes: List[_FieldNode],
        mut bufs: List[_BodyBuffer],
    ) raises -> Int64:
        var length = self._r.read_i64(rb_pos, 0, 0)

        var nodes_vec = self._r.read_vector(rb_pos, 1)
        var nn = Int(self._r.vector_len(nodes_vec))
        for i in range(nn):
            var sb = self._r.vec_struct_bytes(nodes_vec, UInt32(i), 16)
            nodes.append(
                _FieldNode(
                    _read_le[DType.int64](sb, 0), _read_le[DType.int64](sb, 8)
                )
            )

        var bufs_vec = self._r.read_vector(rb_pos, 2)
        var nb = Int(self._r.vector_len(bufs_vec))
        for i in range(nb):
            var sb = self._r.vec_struct_bytes(bufs_vec, UInt32(i), 16)
            bufs.append(
                _BodyBuffer(
                    _read_le[DType.int64](sb, 0), _read_le[DType.int64](sb, 8)
                )
            )

        return length

    def _read_field(
        self, fp: UInt32, mut out_ipc: List[_FieldIpcInfo]
    ) raises -> dt.Field:
        var name = self._r.read_string(fp, 0)
        var nullable = self._r.read_bool(fp, 1, False)
        var type_type = self._r.read_u8(fp, 2, 0)

        var children = List[dt.Field]()
        var child_ipc = List[_FieldIpcInfo]()
        try:
            var children_vec = self._r.read_vector(fp, 5)
            var n = Int(self._r.vector_len(children_vec))
            for i in range(n):
                var child_pos = self._r.vec_offset(children_vec, UInt32(i))
                children.append(self._read_field(child_pos, child_ipc))
        except:
            pass  # absent children vector is normal for leaf types

        var metadata = Dict[String, String]()
        try:
            metadata = self._read_kv_vec(fp, 6)
        except:
            pass

        var dtype: dt.AnyDataType
        if type_type == _TYPE_NULL:
            dtype = dt.null
        elif type_type == _TYPE_BOOL:
            dtype = dt.bool_
        elif type_type == _TYPE_DECIMAL:
            var tp = self._r.read_table(fp, 3)
            var precision = Int(self._r.read_i32(tp, 0, 0))
            var scale = Int(self._r.read_i32(tp, 1, 0))
            var bit_width = Int(self._r.read_i32(tp, 2, 128))
            if bit_width == 32:
                dtype = dt.decimal32(precision, scale)
            elif bit_width == 64:
                dtype = dt.decimal64(precision, scale)
            elif bit_width == 256:
                dtype = dt.decimal256(precision, scale)
            else:
                dtype = dt.decimal128(precision, scale)
        elif type_type == _TYPE_INT:
            var tp = self._r.read_table(fp, 3)
            var bw = Int(self._r.read_i32(tp, 0, 32))
            var signed = self._r.read_bool(tp, 1, False)
            if signed:
                if bw == 8:
                    dtype = dt.int8
                elif bw == 16:
                    dtype = dt.int16
                elif bw == 32:
                    dtype = dt.int32
                else:
                    dtype = dt.int64
            else:
                if bw == 8:
                    dtype = dt.uint8
                elif bw == 16:
                    dtype = dt.uint16
                elif bw == 32:
                    dtype = dt.uint32
                else:
                    dtype = dt.uint64
        elif type_type == _TYPE_FLOATING_POINT:
            var tp = self._r.read_table(fp, 3)
            var prec = self._r.read_u16(tp, 0, _PRECISION_DOUBLE)
            if prec == _PRECISION_HALF:
                dtype = dt.float16
            elif prec == _PRECISION_SINGLE:
                dtype = dt.float32
            else:
                dtype = dt.float64
        elif type_type == _TYPE_BINARY:
            dtype = dt.binary
        elif type_type == _TYPE_LARGE_BINARY:
            dtype = dt.large_binary
        elif type_type == _TYPE_UTF8:
            dtype = dt.string
        elif type_type == _TYPE_LARGE_UTF8:
            dtype = dt.large_string
        elif type_type == _TYPE_LIST:
            if len(children) == 0:
                raise Error("list Field must have 1 child, got 0")
            # Preserve the child Field as-is (its name may not be the default
            # "item" — e.g. arrow-rs uses "inner_list" for nested lists).
            dtype = dt.ListType(children[0].copy()).to_any()
        elif type_type == _TYPE_LARGE_LIST:
            if len(children) == 0:
                raise Error("large_list Field must have 1 child, got 0")
            dtype = dt.LargeListType(children[0].copy()).to_any()
        elif type_type == _TYPE_FIXED_SIZE_LIST:
            var tp = self._r.read_table(fp, 3)
            var list_size = Int(self._r.read_i32(tp, 0, 0))
            if len(children) == 0:
                raise Error("fixed_size_list Field must have 1 child, got 0")
            dtype = dt.FixedSizeListType(children[0].copy(), list_size).to_any()
        elif type_type == _TYPE_FIXED_SIZE_BINARY:
            var tp = self._r.read_table(fp, 3)
            var byte_width = Int(self._r.read_i32(tp, 0, 0))
            dtype = dt.FixedSizeBinaryType(byte_width).to_any()
        elif type_type == _TYPE_DATE:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _DATE_UNIT_MILLISECOND)
            if unit_v == _DATE_UNIT_DAY:
                dtype = dt.date32()
            else:
                dtype = dt.date64()
        elif type_type == _TYPE_TIME:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _TIME_UNIT_MILLISECOND)
            var bw = Int(self._r.read_i32(tp, 1, 32))
            var unit = _IpcDecoder._wire_to_time_unit(unit_v)
            if bw == 32:
                dtype = dt.time32(unit)
            else:
                dtype = dt.time64(unit)
        elif type_type == _TYPE_TIMESTAMP:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _TIME_UNIT_SECOND)
            var tz = self._r.read_string(tp, 1)
            dtype = dt.timestamp(
                _IpcDecoder._wire_to_time_unit(unit_v), tz
            ).to_any()
        elif type_type == _TYPE_DURATION:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _TIME_UNIT_MILLISECOND)
            dtype = dt.duration(_IpcDecoder._wire_to_time_unit(unit_v)).to_any()
        elif type_type == _TYPE_INTERVAL:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _INTERVAL_UNIT_YEAR_MONTH)
            if unit_v == _INTERVAL_UNIT_DAY_TIME:
                dtype = dt.day_time_interval().to_any()
            elif unit_v == _INTERVAL_UNIT_MONTH_DAY_NANO:
                dtype = dt.month_day_nano_interval().to_any()
            else:
                dtype = dt.year_month_interval().to_any()
        elif type_type == _TYPE_STRUCT:
            dtype = dt.struct_(children^)
        else:
            raise Error(
                "_IpcDecoder: unsupported type_type: " + String(Int(type_type))
            )

        # Check for DictionaryEncoding at slot 4 — wraps the value type in DictionaryType.
        # The dict_id is stored in _FieldIpcInfo rather than on the type itself.
        var own_dict_id = -1
        try:
            var de_pos = self._r.read_table(fp, 4)
            own_dict_id = Int(self._r.read_i64(de_pos, 0, 0))
            var idx_tp = self._r.read_table(de_pos, 1)
            var idx_bw = Int(self._r.read_i32(idx_tp, 0, 32))
            var idx_signed = self._r.read_bool(idx_tp, 1, False)
            var ordered = self._r.read_bool(de_pos, 2, False)
            var index_dtype: dt.AnyDataType
            if idx_signed:
                if idx_bw == 8:
                    index_dtype = dt.int8
                elif idx_bw == 16:
                    index_dtype = dt.int16
                elif idx_bw == 32:
                    index_dtype = dt.int32
                else:
                    index_dtype = dt.int64
            else:
                if idx_bw == 8:
                    index_dtype = dt.uint8
                elif idx_bw == 16:
                    index_dtype = dt.uint16
                elif idx_bw == 32:
                    index_dtype = dt.uint32
                else:
                    index_dtype = dt.uint64
            dtype = dt.dictionary(index_dtype^, dtype.copy(), ordered).to_any()
        except:
            pass  # no DictionaryEncoding at slot 4

        out_ipc.append(_FieldIpcInfo(own_dict_id, child_ipc^))
        return dt.Field(name, dtype^, nullable, metadata^)


# ---------------------------------------------------------------------------
# IPC framing helpers and framed message reader
# ---------------------------------------------------------------------------


struct _MessageReader(Movable):
    """Reads framed IPC messages (continuation + length + metadata + body)."""

    var _bytes: List[UInt8]
    var _pos: Int

    def __init__(out self, var bytes: List[UInt8], start_pos: Int = 0):
        self._bytes = bytes^
        self._pos = start_pos

    def pos(self) -> Int:
        return self._pos

    def seek(mut self, pos: Int):
        self._pos = pos

    def __len__(self) -> Int:
        return len(self._bytes)

    def read_next(
        mut self,
        mut meta: List[UInt8],
        mut body: List[UInt8],
    ) raises -> Bool:
        """Parse one message at the current position. Returns False at end-of-stream.
        """
        var n = len(self._bytes)
        if self._pos + 4 > n:
            return False

        var marker = _read_le[DType.int32](self._bytes, self._pos)
        var metadata_len: Int
        var meta_start: Int
        if UInt32(marker) == UInt32(0xFFFFFFFF):
            if self._pos + 8 > n:
                return False
            metadata_len = Int(
                _read_le[DType.int32](self._bytes, self._pos + 4)
            )
            meta_start = self._pos + 8
        else:
            metadata_len = Int(marker)
            meta_start = self._pos + 4

        if metadata_len == 0:
            self._pos = meta_start
            return False

        for i in range(metadata_len):
            meta.append(self._bytes[meta_start + i])

        var raw_end = meta_start + metadata_len
        var meta_end = raw_end + (8 - raw_end % 8) % 8

        var dec = _IpcDecoder(meta.copy())
        var body_len = Int(dec.body_length())

        for i in range(body_len):
            body.append(self._bytes[meta_end + i])

        self._pos = meta_end + body_len
        return True


# ---------------------------------------------------------------------------
# Batch body encoder: traverses ArrayData trees to collect raw bytes
# ---------------------------------------------------------------------------


@fieldwise_init
struct _EncodedBatch(Movable):
    var msg: List[UInt8]
    var metadata_length: Int32
    var body_length: Int64


struct _BatchEncoder(Movable):
    """Traverses an ArrayData tree collecting _FieldNode metadata and raw buffer bytes.
    """

    var nodes: List[_FieldNode]
    var raw_bufs: List[List[UInt8]]

    def __init__(out self):
        self.nodes = List[_FieldNode]()
        self.raw_bufs = List[List[UInt8]]()

    def write_array(mut self, root: ArrayData) raises:
        var stack = List[ArrayData]()
        stack.append(root.copy())
        while len(stack) > 0:
            var data = stack.pop()

            self.nodes.append(_FieldNode(Int64(data.length), Int64(data.nulls)))

            # Null type: emit FieldNode but NO body buffers (not even validity).
            if data.dtype.is_null():
                continue

            var validity_bytes = List[UInt8]()
            if data.nulls > 0 and data.bitmap:
                var bv = data.bitmap.value()
                var n_bits = data.offset + data.length
                var n_bytes = ceildiv(n_bits, 8)
                for byte_idx in range(n_bytes):
                    var byte_val = UInt8(0)
                    for bit_idx in range(8):
                        var bit_pos = byte_idx * 8 + bit_idx
                        if bit_pos < n_bits and bv.unsafe_test(bit_pos):
                            byte_val |= UInt8(1 << bit_idx)
                    validity_bytes.append(byte_val)
            self.raw_bufs.append(validity_bytes^)

            for buf in data.buffers:
                var n = buf.length[DType.uint8]()
                var bytes = List[UInt8](capacity=n)
                for i in range(n):
                    bytes.append(buf.unsafe_get[DType.uint8](i))
                self.raw_bufs.append(bytes^)

            # For dictionary arrays, children hold the dictionary values which
            # go in a separate DictionaryBatch message — not in the RecordBatch body.
            if not data.dtype.is_dictionary():
                for i in range(len(data.children) - 1, -1, -1):
                    stack.append(data.children[i].copy())

    @staticmethod
    def collect_dict_pairs(
        data: ArrayData, mut pairs: List[_DictPair], mut next_id: Int
    ) raises:
        """DFS inner-first: append (dict_id, values) for every dict array in the tree.
        """
        if data.dtype.is_dictionary():
            _BatchEncoder.collect_dict_pairs(data.children[0], pairs, next_id)
            pairs.append(
                _DictPair(next_id, AnyArray.from_data(data.children[0]))
            )
            next_id += 1
        elif data.dtype.is_list():
            if len(data.children) > 0:
                _BatchEncoder.collect_dict_pairs(
                    data.children[0], pairs, next_id
                )
        elif data.dtype.is_struct():
            for i in range(len(data.children)):
                _BatchEncoder.collect_dict_pairs(
                    data.children[i], pairs, next_id
                )

    def _build_body(
        mut self, mut buf_meta: List[_BodyBuffer], mut body: List[UInt8]
    ):
        """Assemble raw buffers into a padded body, populating buf_meta offsets.
        """
        for buf in self.raw_bufs:
            _pad_to(body, 8)
            buf_meta.append(_BodyBuffer(Int64(len(body)), Int64(len(buf))))
            body.extend(Span(buf))
        _pad_to(body, 8)

    def encode(mut self, batch: RecordBatch) raises -> _EncodedBatch:
        for col in batch.columns:
            self.write_array(col.to_data())
        var buf_meta = List[_BodyBuffer]()
        var body = List[UInt8]()
        self._build_body(buf_meta, body)
        var rb_meta = _IpcEncoder.encode_record_batch(
            Int64(batch.num_rows()), self.nodes, buf_meta
        )
        var meta_len = len(rb_meta)
        var padded_meta = meta_len + (8 - meta_len % 8) % 8
        var metadata_length = Int32(8 + padded_meta)
        var body_length = Int64(len(body))
        var msg = _IpcEncoder.frame_message(rb_meta, body)
        self.nodes = List[_FieldNode]()
        self.raw_bufs = List[List[UInt8]]()
        return _EncodedBatch(msg^, metadata_length, body_length)

    @staticmethod
    def encode_dict_message(
        dict_id: Int64, values: AnyArray
    ) raises -> _EncodedBatch:
        """Encode a dictionary values array as a DictionaryBatch IPC message."""
        var benc = _BatchEncoder()
        benc.write_array(values.to_data())
        var buf_meta = List[_BodyBuffer]()
        var body = List[UInt8]()
        benc._build_body(buf_meta, body)

        var enc = _IpcEncoder(512)
        var nodes_vec = enc._write_field_nodes_vec(benc.nodes)
        var bufs_vec = enc._write_body_buffers_vec(buf_meta)
        var rb_pos = enc._write_record_batch_table(
            Int64(values.length()), nodes_vec, bufs_vec
        )
        var db_pos = enc._write_dictionary_batch_table(dict_id, False, rb_pos)

        var max_end = Int64(0)
        for b in buf_meta:
            max_end = max(max_end, b.offset + b.length)
        var r = max_end % Int64(8)
        var body_len = max_end + (Int64(8) - r) % Int64(8)

        var msg_pos = enc._write_message_table(
            _HEADER_DICTIONARY_BATCH, db_pos, body_len
        )
        var meta = enc._finish(msg_pos)
        var meta_len = len(meta)
        var padded_meta = meta_len + (8 - meta_len % 8) % 8
        var metadata_length = Int32(8 + padded_meta)
        var body_length = Int64(len(body))
        var msg = _IpcEncoder.frame_message(meta, body)
        return _EncodedBatch(msg^, metadata_length, body_length)


# ---------------------------------------------------------------------------
# Batch body decoder: reconstructs AnyArray from raw bytes + cursor state
# ---------------------------------------------------------------------------


struct _BatchDecoder(Movable):
    """Reconstructs AnyArray values from a record batch body using node/buffer cursors.
    """

    var body: List[UInt8]
    var body_offset: Int
    var nodes: List[_FieldNode]
    var bufs: List[_BodyBuffer]
    var node_idx: Int
    var buf_idx: Int
    var dict_values: List[AnyArray]

    def __init__(
        out self,
        var body: List[UInt8],
        body_offset: Int,
        var nodes: List[_FieldNode],
        var bufs: List[_BodyBuffer],
        dict_values: List[AnyArray] = List[AnyArray](),
    ):
        self.body = body^
        self.body_offset = body_offset
        self.nodes = nodes^
        self.bufs = bufs^
        self.node_idx = 0
        self.buf_idx = 0
        self.dict_values = dict_values.copy()

    def read_array(
        mut self, dtype: dt.AnyDataType, ipc_info: _FieldIpcInfo
    ) raises -> AnyArray:
        var node = self.nodes[self.node_idx]
        self.node_idx += 1

        var length = Int(node.length)
        var null_count = Int(node.null_count)

        # Null type: FieldNode is consumed but no body buffers — neither validity
        # nor data — per Arrow spec.
        if dtype.is_null():
            return NullArray(length)

        var validity_buf = self.bufs[self.buf_idx]
        self.buf_idx += 1

        var bitmap: Optional[Bitmap[mut=False]] = None
        if null_count > 0 and validity_buf.length > 0:
            var off = Int(validity_buf.offset) + self.body_offset
            var n_bytes = Int(validity_buf.length)
            bitmap = Bitmap[mut=False](
                self._slice_body(off, n_bytes), length=length
            )

        var data_buffers = List[Buffer[mut=False]]()
        var children = List[ArrayData]()

        # Dictionary: consume index buffer then reconstruct from dict_values lookup.
        # The dict_id comes from ipc_info (not the logical type) so that the type
        # system remains free of IPC metadata.
        if dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var dict_id = ipc_info.dict_id
            if dict_id < 0 or dict_id >= len(self.dict_values):
                raise Error(
                    "_BatchDecoder: no values for dict_id " + String(dict_id)
                )
            var indices = self._consume_primitive_array(
                d.index_type().copy(), length, null_count, bitmap^
            )
            var values = self.dict_values[dict_id].copy()
            return DictionaryArray.from_arrays(
                indices^, values^, d.ordered
            ).to_any()

        if (
            dtype.is_string()
            or dtype.is_binary()
            or dtype.is_large_string()
            or dtype.is_large_binary()
        ):
            self._consume_buffer(data_buffers)
            self._consume_buffer(data_buffers)
        elif (
            dtype.is_bool()
            or dtype.is_primitive()
            or dtype.is_list()
            or dtype.is_large_list()
            or dtype.is_fixed_size_binary()
        ):
            self._consume_buffer(data_buffers)

        if dtype.is_list():
            var child_ipc = (
                ipc_info.children[0].copy() if len(ipc_info.children)
                > 0 else _FieldIpcInfo()
            )
            children.append(
                self.read_array(
                    dtype.as_list().value_type(), child_ipc
                ).to_data()
            )
        elif dtype.is_large_list():
            var child_ipc = (
                ipc_info.children[0].copy() if len(ipc_info.children)
                > 0 else _FieldIpcInfo()
            )
            children.append(
                self.read_array(
                    dtype.as_large_list().value_type(), child_ipc
                ).to_data()
            )
        elif dtype.is_fixed_size_list():
            var child_ipc = (
                ipc_info.children[0].copy() if len(ipc_info.children)
                > 0 else _FieldIpcInfo()
            )
            children.append(
                self.read_array(
                    dtype.as_fixed_size_list().value_type(), child_ipc
                ).to_data()
            )
        elif dtype.is_struct():
            ref st = dtype.as_struct()
            for i in range(len(st.fields)):
                var child_ipc = (
                    ipc_info.children[i].copy() if i
                    < len(ipc_info.children) else _FieldIpcInfo()
                )
                children.append(
                    self.read_array(st.fields[i].dtype, child_ipc).to_data()
                )

        var ad = ArrayData(
            dtype=dtype.copy(),
            length=length,
            nulls=null_count,
            offset=0,
            bitmap=bitmap,
            buffers=data_buffers^,
            children=children^,
        )
        return AnyArray.from_data(ad)

    def _slice_body(self, off: Int, n_bytes: Int) -> Buffer[mut=False]:
        var buf = Buffer.alloc_uninit[DType.uint8](n_bytes)
        for i in range(n_bytes):
            buf.unsafe_set[DType.uint8](i, self.body[off + i])
        return buf.to_immutable()

    def _consume_buffer(mut self, mut out: List[Buffer[mut=False]]) raises:
        var bb = self.bufs[self.buf_idx]
        self.buf_idx += 1
        var n_bytes = Int(bb.length)
        if n_bytes > 0:
            out.append(
                self._slice_body(Int(bb.offset) + self.body_offset, n_bytes)
            )
        else:
            out.append(Buffer.alloc_zeroed[DType.uint8](0).to_immutable())

    def _consume_primitive_array(
        mut self,
        dtype: dt.AnyDataType,
        length: Int,
        nulls: Int,
        var bitmap: Optional[Bitmap[mut=False]],
    ) raises -> AnyArray:
        """Read the next body buffer and build a primitive AnyArray of the given dtype.
        """
        var bufs = List[Buffer[mut=False]]()
        self._consume_buffer(bufs)
        return AnyArray.from_data(
            ArrayData(
                dtype=dtype.copy(),
                length=length,
                nulls=nulls,
                offset=0,
                bitmap=bitmap,
                buffers=bufs^,
                children=List[ArrayData](),
            )
        )


# ---------------------------------------------------------------------------
# Public: incremental file and stream writers
# ---------------------------------------------------------------------------


struct RecordBatchFileWriter(Movable):
    """Incremental writer for the Arrow IPC file format.

    Write batches with `write_batch`, then call `close()` to flush the
    footer and write the file to disk.
    """

    var _out: List[UInt8]
    var _path: String
    var _schema: Schema
    var _dict_blocks: List[_Block]
    var _blocks: List[_Block]
    var _enc: _BatchEncoder
    var _dicts_written: List[Bool]
    var _closed: Bool

    def __init__(out self, path: String, schema: Schema) raises:
        self._out = List[UInt8]()
        self._path = path
        self._schema = Schema(copy=schema)
        self._dict_blocks = List[_Block]()
        self._blocks = List[_Block]()
        self._enc = _BatchEncoder()
        self._dicts_written = List[Bool]()
        self._closed = False

        for b in _magic():
            self._out.append(b)
        var schema_msg = _IpcEncoder.frame_message(
            _IpcEncoder.encode_schema_message(self._schema), List[UInt8]()
        )
        self._out.extend(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise Error("RecordBatchFileWriter: writer is closed")
        # Collect all (dict_id, values) pairs in DFS inner-first order.
        # In FILE format each dict_id is written exactly once.
        var pairs = List[_DictPair]()
        var next_id = 0
        for col in batch.columns:
            _BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for j in range(len(pairs)):
            var did = pairs[j].dict_id
            while len(self._dicts_written) <= did:
                self._dicts_written.append(False)
            if self._dicts_written[did]:
                continue
            var dict_blk_start = Int64(len(self._out))
            var eb = _BatchEncoder.encode_dict_message(
                Int64(did), pairs[j].values
            )
            self._out.extend(Span(eb.msg))
            self._dict_blocks.append(
                _Block(dict_blk_start, eb.metadata_length, eb.body_length)
            )
            self._dicts_written[did] = True
        var blk_start = Int64(len(self._out))
        var eb = self._enc.encode(batch)
        self._out.extend(Span(eb.msg))
        self._blocks.append(
            _Block(blk_start, eb.metadata_length, eb.body_length)
        )

    def close(mut self) raises:
        if self._closed:
            return
        _pad_to(self._out, 8)
        var footer_bytes = _IpcEncoder.encode_footer(
            self._schema, self._dict_blocks, self._blocks
        )
        self._out.extend(Span(footer_bytes))
        LittleEndian.append[DType.int32](self._out, Int32(len(footer_bytes)))
        var magic = _magic()
        for i in range(6):
            self._out.append(magic[i])
        Path(self._path).write_bytes(self._out^)
        self._closed = True


struct RecordBatchStreamWriter(Movable):
    """Incremental writer for the Arrow IPC stream format.

    Write batches with `write_batch`, then call `close()` to write the
    EOS marker and flush the stream to disk.
    """

    var _out: List[UInt8]
    var _path: String
    var _enc: _BatchEncoder
    var _closed: Bool

    def __init__(out self, path: String, schema: Schema) raises:
        self._out = List[UInt8]()
        self._path = path
        self._enc = _BatchEncoder()
        self._closed = False

        var schema_msg = _IpcEncoder.frame_message(
            _IpcEncoder.encode_schema_message(schema),
            List[UInt8](),
        )
        self._out.extend(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise Error("RecordBatchStreamWriter: writer is closed")
        # Stream format sends all dicts before each record batch.
        var pairs = List[_DictPair]()
        var next_id = 0
        for col in batch.columns:
            _BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for j in range(len(pairs)):
            var eb = _BatchEncoder.encode_dict_message(
                Int64(pairs[j].dict_id), pairs[j].values
            )
            self._out.extend(Span(eb.msg))
        self._out.extend(Span(self._enc.encode(batch).msg))

    def close(mut self) raises:
        if self._closed:
            return
        LittleEndian.append[DType.uint32](self._out, UInt32(0xFFFFFFFF))
        LittleEndian.append[DType.int32](self._out, Int32(0))
        Path(self._path).write_bytes(self._out^)
        self._closed = True


# ---------------------------------------------------------------------------
# Public: file and stream readers
# ---------------------------------------------------------------------------


struct RecordBatchFileReader(Movable):
    """Reader for the Arrow IPC file format with random-access batch reads."""

    var schema: Schema
    var _ipc_infos: List[_FieldIpcInfo]
    var _blocks: List[_Block]
    var _msg_reader: _MessageReader
    var _dict_values: List[AnyArray]

    def __init__(out self, path: String) raises:
        var file_bytes = Path(path).read_bytes()
        var n = len(file_bytes)
        if n < 14:
            raise Error("IPC file too short")
        var magic = _magic()
        for i in range(8):
            if file_bytes[i] != magic[i]:
                raise Error("IPC file: bad magic bytes")
        for i in range(6):
            if file_bytes[n - 6 + i] != magic[i]:
                raise Error("IPC file: bad trailing magic")

        var footer_size = Int(_read_le[DType.int32](file_bytes, n - 10))
        var footer_start = n - 10 - footer_size
        var footer_bytes = List[UInt8](capacity=footer_size)
        for i in range(footer_size):
            footer_bytes.append(file_bytes[footer_start + i])

        var dec = _IpcDecoder(footer_bytes^)
        var dict_blocks = List[_Block]()
        var blocks = List[_Block]()
        var ipc_infos = List[_FieldIpcInfo]()
        self.schema = dec.read_footer(dict_blocks, blocks, ipc_infos)
        self._ipc_infos = ipc_infos^
        self._blocks = blocks^
        self._msg_reader = _MessageReader(file_bytes^)
        self._dict_values = List[AnyArray]()

        # Load dictionary values from their footer-registered blocks.
        # dict_values is indexed by dict_id; pass partial list to decode_dict_batch
        # so that nested dicts (already loaded at lower ids) can be resolved.
        for di in range(len(dict_blocks)):
            self._msg_reader.seek(Int(dict_blocks[di].offset))
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not self._msg_reader.read_next(meta, body):
                break
            var dict_id = _IpcDecoder(meta.copy()).peek_dict_id()
            var lkup = _FieldIpcInfo.find_in_schema(
                self.schema.fields, self._ipc_infos, dict_id
            )
            if lkup:
                var dec = _IpcDecoder(meta^)
                var values = dec.decode_dict_batch(
                    lkup.value().value_type,
                    lkup.value().value_ipc_info,
                    body^,
                    self._dict_values,
                )
                while len(self._dict_values) <= dict_id:
                    self._dict_values.append(NullArray(0))
                self._dict_values[dict_id] = values^

    def num_record_batches(self) -> Int:
        return len(self._blocks)

    def read_batch(mut self, i: Int) raises -> RecordBatch:
        if i < 0 or i >= len(self._blocks):
            raise Error("RecordBatchFileReader: batch index out of range")
        self._msg_reader.seek(Int(self._blocks[i].offset))
        var meta = List[UInt8]()
        var body = List[UInt8]()
        var _ok = self._msg_reader.read_next(meta, body)
        var dec = _IpcDecoder(meta^)
        return dec.decode_record_batch(
            self.schema, self._ipc_infos, body^, self._dict_values
        )

    def read_all(mut self) raises -> List[RecordBatch]:
        # Footer only lists record-batch blocks, so no header check needed.
        var batches = List[RecordBatch]()
        for i in range(len(self._blocks)):
            batches.append(self.read_batch(i))
        return batches^


struct RecordBatchStreamReader(Movable):
    """Reader for the Arrow IPC stream format."""

    var schema: Schema
    var _ipc_infos: List[_FieldIpcInfo]
    var _msg_reader: _MessageReader

    def __init__(out self, path: String) raises:
        var file_bytes = Path(path).read_bytes()
        var msg_reader = _MessageReader(file_bytes^)
        var meta = List[UInt8]()
        var body = List[UInt8]()
        if not msg_reader.read_next(meta, body):
            raise Error("RecordBatchStreamReader: missing schema message")
        var ipc_infos = List[_FieldIpcInfo]()
        var dec = _IpcDecoder(meta^)
        self.schema = dec.decode_schema(ipc_infos)
        self._ipc_infos = ipc_infos^
        self._msg_reader = msg_reader^

    def read_all(mut self) raises -> List[RecordBatch]:
        var dict_values = List[AnyArray]()
        var batches = List[RecordBatch]()
        while True:
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not self._msg_reader.read_next(meta, body):
                break
            var header_type: UInt8
            var peek = _IpcDecoder(meta.copy())
            header_type = peek.peek_header()
            if Int(header_type) == Int(_HEADER_DICTIONARY_BATCH):
                var dict_id = _IpcDecoder(meta.copy()).peek_dict_id()
                var lkup = _FieldIpcInfo.find_in_schema(
                    self.schema.fields, self._ipc_infos, dict_id
                )
                if lkup:
                    var dec = _IpcDecoder(meta^)
                    var values = dec.decode_dict_batch(
                        lkup.value().value_type,
                        lkup.value().value_ipc_info,
                        body^,
                        dict_values,
                    )
                    while len(dict_values) <= dict_id:
                        dict_values.append(NullArray(0))
                    dict_values[dict_id] = values^
            elif Int(header_type) == Int(_HEADER_RECORD_BATCH):
                var dec = _IpcDecoder(meta^)
                batches.append(
                    dec.decode_record_batch(
                        self.schema, self._ipc_infos, body^, dict_values
                    )
                )
        return batches^


# ---------------------------------------------------------------------------
# Public top-level functions
# ---------------------------------------------------------------------------


def write_ipc_file(
    path: String, schema: Schema, batches: List[RecordBatch]
) raises:
    """Write RecordBatches to an Arrow IPC file with an explicit schema."""
    var w = RecordBatchFileWriter(path, schema)
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_file(path: String, batches: List[RecordBatch]) raises:
    """Write RecordBatches to an Arrow IPC file."""
    if len(batches) == 0:
        raise Error(
            "write_ipc_file: no batches; use write_ipc_file(path, schema,"
            " batches) for schema-only files"
        )
    write_ipc_file(path, batches[0].schema, batches)


def write_ipc_stream(
    path: String, schema: Schema, batches: List[RecordBatch]
) raises:
    """Write RecordBatches to an Arrow IPC stream with an explicit schema."""
    var w = RecordBatchStreamWriter(path, schema)
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_stream(path: String, batches: List[RecordBatch]) raises:
    """Write RecordBatches to an Arrow IPC stream."""
    if len(batches) == 0:
        raise Error(
            "write_ipc_stream: no batches; use write_ipc_stream(path, schema,"
            " batches) for schema-only streams"
        )
    write_ipc_stream(path, batches[0].schema, batches)


def read_ipc_file(path: String) raises -> List[RecordBatch]:
    """Read an Arrow IPC file and return all RecordBatches."""
    var r = RecordBatchFileReader(path)
    return r.read_all()


def read_ipc_stream(path: String) raises -> List[RecordBatch]:
    """Read an Arrow IPC stream and return all RecordBatches."""
    var r = RecordBatchStreamReader(path)
    return r.read_all()


def read_ipc_file_schema(path: String) raises -> RecordBatch:
    """Read the schema from an Arrow IPC file; return a 0-row RecordBatch."""
    var r = RecordBatchFileReader(path)
    return RecordBatch.empty(r.schema)


def read_ipc_stream_schema(path: String) raises -> RecordBatch:
    """Read the schema from an Arrow IPC stream; return a 0-row RecordBatch."""
    var r = RecordBatchStreamReader(path)
    return RecordBatch.empty(r.schema)
