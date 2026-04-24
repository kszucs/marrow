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
list, fixed_size_list, struct.
"""

from std.bit import byte_swap
from std.math import ceildiv
from std.memory import Span
from std.pathlib import Path
from std.sys import size_of
from std.sys.info import is_big_endian
from .arrays import AnyArray, ArrayData
from .buffers import Buffer, Bitmap
from .dtypes import (
    AnyDataType,
    Field,
    field as mk_field,
    list_ as mk_list,
    fixed_size_list_ as mk_fixed_size_list,
    struct_ as mk_struct,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    binary,
    string,
)
from .schema import Schema
from .tabular import RecordBatch


# ---------------------------------------------------------------------------
# Arrow IPC wire protocol constants
# ---------------------------------------------------------------------------

comptime _HEADER_SCHEMA: UInt8 = 1
comptime _HEADER_RECORD_BATCH: UInt8 = 3
comptime _METADATA_VERSION_V5: Int16 = 4
comptime _ENDIANNESS_LITTLE: Int16 = 0
comptime _TYPE_INT: UInt8 = 2
comptime _TYPE_FLOATING_POINT: UInt8 = 3
comptime _TYPE_BINARY: UInt8 = 4
comptime _TYPE_UTF8: UInt8 = 5
comptime _TYPE_BOOL: UInt8 = 6
comptime _TYPE_LIST: UInt8 = 12
comptime _TYPE_STRUCT: UInt8 = 13
comptime _TYPE_FIXED_SIZE_LIST: UInt8 = 16
comptime _PRECISION_HALF: UInt16 = 0
comptime _PRECISION_SINGLE: UInt16 = 1
comptime _PRECISION_DOUBLE: UInt16 = 2


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


# ---------------------------------------------------------------------------
# Little-endian integer read/write helpers
# ---------------------------------------------------------------------------


def _write_le[T: DType](mut buf: List[UInt8], pos: Int, val: Scalar[T]):
    var v = val
    comptime if is_big_endian():
        v = byte_swap(v)
    comptime for i in range(size_of[T]()):
        buf[pos + i] = (v >> Scalar[T](i * 8)).cast[DType.uint8]()


def _read_le[T: DType](buf: List[UInt8], pos: Int) raises -> Scalar[T]:
    if pos < 0 or pos + size_of[T]() - 1 >= len(buf):
        raise Error("ipc: _read_le out of bounds at " + String(pos))
    var result = Scalar[T](0)
    comptime for i in range(size_of[T]()):
        result |= Scalar[T](buf[pos + i]) << Scalar[T](i * 8)
    return result


def _padding_to(pos: Int, alignment: Int) -> Int:
    return (alignment - (pos % alignment)) % alignment


def _append_le[T: DType](mut buf: List[UInt8], val: Scalar[T]):
    var v = val
    comptime if is_big_endian():
        v = byte_swap(v)
    comptime for i in range(size_of[T]()):
        buf.append((v >> Scalar[T](i * 8)).cast[DType.uint8]())


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
        _write_le[DType.uint16](self._buf, self._head, val)
        return self.offset()

    def prepend_i16(mut self, val: Int16) raises -> UInt32:
        self._prep(2, 2)
        self._head -= 2
        _write_le[DType.int16](self._buf, self._head, val)
        return self.offset()

    def prepend_i32(mut self, val: Int32) raises -> UInt32:
        self._prep(4, 4)
        self._head -= 4
        _write_le[DType.int32](self._buf, self._head, val)
        return self.offset()

    def prepend_i64(mut self, val: Int64) raises -> UInt32:
        self._prep(8, 8)
        self._head -= 8
        _write_le[DType.int64](self._buf, self._head, val)
        return self.offset()

    def prepend_uoffset(mut self, val: UInt32) raises -> UInt32:
        self._prep(4, 4)
        self._head -= 4
        var stored_abs = len(self._buf) - self._head
        _write_le[DType.uint32](self._buf, self._head, UInt32(stored_abs - Int(val)))
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
        _write_le[DType.uint32](self._buf, self._head, UInt32(n))
        return self.offset()

    def create_vector_u8(mut self, data: List[UInt8]) raises -> UInt32:
        var n = len(data)
        self._prep(4, n)
        for i in range(n - 1, -1, -1):
            self._head -= 1
            self._buf[self._head] = data[i]
        self._head -= 4
        _write_le[DType.uint32](self._buf, self._head, UInt32(n))
        return self.offset()

    def create_vector_offsets(mut self, offsets: List[UInt32]) raises -> UInt32:
        var n = len(offsets)
        self._prep(4, n * 4)
        for i in range(n - 1, -1, -1):
            self._head -= 4
            var stored_abs = len(self._buf) - self._head
            var rel = stored_abs - Int(offsets[i])
            _write_le[DType.uint32](self._buf, self._head, UInt32(rel))
        self._head -= 4
        _write_le[DType.uint32](self._buf, self._head, UInt32(n))
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
        _write_le[DType.uint32](self._buf, self._head, UInt32(count))
        return self.offset()

    def finish(mut self, root: UInt32) raises -> List[UInt8]:
        self._prep(self._min_align, 4)
        self._head -= 4
        var table_pos_in_result = (len(self._buf) - Int(root)) - self._head
        _write_le[DType.uint32](self._buf, self._head, UInt32(table_pos_in_result))
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
        _write_le[DType.int32](self._buf, self._head, Int32(0))
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
            _write_le[DType.uint16](self._buf, self._head, vtable_slots[s])
        self._head -= 2
        _write_le[DType.uint16](self._buf, self._head, object_size)
        self._head -= 2
        _write_le[DType.uint16](self._buf, self._head, vtable_size)
        var new_vt_offset = self.offset()

        var soffset = Int32(Int(new_vt_offset) - Int(table_pos))
        _write_le[DType.int32](self._buf, len(self._buf) - Int(table_pos), soffset)

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
            raise Error("flatbuffers: vtable position out of bounds: " + String(vt))
        var vt_size = Int(_read_le[DType.uint16](self._buf, vt))
        var slot_byte = 4 + slot * 2
        if slot_byte + 1 >= vt_size:
            return UInt16(0)
        if vt + slot_byte + 1 >= len(self._buf):
            return UInt16(0)
        return _read_le[DType.uint16](self._buf, vt + slot_byte)

    def read_u8(self, tp: UInt32, slot: Int, default: UInt8 = 0) raises -> UInt8:
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
            raise Error("flatbuffers: absent offset field at slot " + String(slot))
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + _read_le[DType.uint32](self._buf, ref_pos)

    def read_table(self, tp: UInt32, slot: Int) raises -> UInt32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            raise Error("flatbuffers: absent offset field at slot " + String(slot))
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + _read_le[DType.uint32](self._buf, ref_pos)

    def vector_len(self, vec_pos: UInt32) raises -> UInt32:
        return _read_le[DType.uint32](self._buf, Int(vec_pos))

    def vec_offset(self, vec_pos: UInt32, i: UInt32) raises -> UInt32:
        var vlen = self.vector_len(vec_pos)
        if i >= vlen:
            raise Error(
                "flatbuffers: vec index " + String(i) + " >= len " + String(vlen)
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
    """Encodes Arrow IPC metadata (schema, record batch, footer) into FlatBuffers."""

    var _fb: _FlatbufWriter

    def __init__(out self, capacity: Int = 512):
        self._fb = _FlatbufWriter(capacity)

    @staticmethod
    def encode_schema(fields: List[Field]) raises -> List[UInt8]:
        var enc = _IpcEncoder(256)
        var schema_pos = enc._write_schema_table(fields)
        return enc._finish(schema_pos)

    @staticmethod
    def encode_schema_message(fields: List[Field]) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var schema_pos = enc._write_schema_table(fields)
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
        fields: List[Field],
        blocks: List[_Block],
    ) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var schema_pos = enc._write_schema_table(fields)
        var blocks_vec = enc._write_blocks_vec(blocks)
        var footer_pos = enc._write_footer_table(schema_pos, blocks_vec)
        return enc._finish(footer_pos)

    @staticmethod
    def frame_message(metadata: List[UInt8], body: List[UInt8]) -> List[UInt8]:
        var out = List[UInt8]()
        _append_le[DType.uint32](out, UInt32(0xFFFFFFFF))
        var meta_len = len(metadata)
        var padded_len = meta_len + (8 - meta_len % 8) % 8
        _append_le[DType.int32](out, Int32(padded_len))
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
        blocks_vec: UInt32,
    ) raises -> UInt32:
        var empty = List[UInt32]()
        var dicts_vec = self._fb.create_vector_offsets(empty)
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
            _write_le[DType.int64](data, i * 16, nodes[i].length)
            _write_le[DType.int64](data, i * 16 + 8, nodes[i].null_count)
        return self._fb.create_vector_structs(data, n, 16, 8)

    def _write_body_buffers_vec(
        mut self, bufs: List[_BodyBuffer]
    ) raises -> UInt32:
        var n = len(bufs)
        var data = List[UInt8](capacity=n * 16)
        for _ in range(n * 16):
            data.append(UInt8(0))
        for i in range(n):
            _write_le[DType.int64](data, i * 16, bufs[i].offset)
            _write_le[DType.int64](data, i * 16 + 8, bufs[i].length)
        return self._fb.create_vector_structs(data, n, 16, 8)

    def _write_blocks_vec(mut self, blocks: List[_Block]) raises -> UInt32:
        var n = len(blocks)
        var data = List[UInt8](capacity=n * 24)
        for _ in range(n * 24):
            data.append(UInt8(0))
        for i in range(n):
            _write_le[DType.int64](data, i * 24, blocks[i].offset)
            _write_le[DType.int32](data, i * 24 + 8, blocks[i].metadata_length)
            # 4 bytes padding at offset 12 (Arrow Block struct alignment)
            _write_le[DType.int64](data, i * 24 + 16, blocks[i].body_length)
        return self._fb.create_vector_structs(data, n, 24, 8)

    def _type_code(self, dtype: AnyDataType) raises -> UInt8:
        if dtype.is_bool():
            return _TYPE_BOOL
        elif dtype.is_integer():
            return _TYPE_INT
        elif dtype.is_floating_point():
            return _TYPE_FLOATING_POINT
        elif dtype.is_binary():
            return _TYPE_BINARY
        elif dtype.is_string():
            return _TYPE_UTF8
        elif dtype.is_list():
            return _TYPE_LIST
        elif dtype.is_fixed_size_list():
            return _TYPE_FIXED_SIZE_LIST
        elif dtype.is_struct():
            return _TYPE_STRUCT
        else:
            raise Error("_IpcEncoder: unsupported dtype: " + String(dtype))

    def _write_type_table(mut self, dtype: AnyDataType) raises -> UInt32:
        if (
            dtype.is_bool()
            or dtype.is_binary()
            or dtype.is_string()
            or dtype.is_list()
            or dtype.is_struct()
        ):
            var ts = self._fb.offset()
            return self._fb.write_table(List[_FieldOffset](), ts)
        elif dtype.is_integer():
            var bw: Int32
            var signed: Bool
            if dtype == int8:
                bw = 8
                signed = True
            elif dtype == int16:
                bw = 16
                signed = True
            elif dtype == int32:
                bw = 32
                signed = True
            elif dtype == int64:
                bw = 64
                signed = True
            elif dtype == uint8:
                bw = 8
                signed = False
            elif dtype == uint16:
                bw = 16
                signed = False
            elif dtype == uint32:
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
            if dtype == float16:
                prec = _PRECISION_HALF
            elif dtype == float32:
                prec = _PRECISION_SINGLE
            else:
                prec = _PRECISION_DOUBLE
            var ts = self._fb.offset()
            var prec_at = self._fb.prepend_u16(prec)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, prec_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_fixed_size_list():
            var fsl = dtype.as_fixed_size_list_type()
            var ts = self._fb.offset()
            var sz_at = self._fb.prepend_i32(Int32(fsl.size))
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, sz_at))
            return self._fb.write_table(flds, ts)
        else:
            raise Error("_IpcEncoder: unsupported dtype for type table: " + String(dtype))

    def _write_field(mut self, f: Field) raises -> UInt32:
        var child_positions = List[UInt32]()
        var dtype = f.dtype.copy()
        if dtype.is_list():
            child_positions.append(
                self._write_field(dtype.as_list_type().value_field().copy())
            )
        elif dtype.is_fixed_size_list():
            child_positions.append(
                self._write_field(
                    dtype.as_fixed_size_list_type().value_field().copy()
                )
            )
        elif dtype.is_struct():
            var st = dtype.as_struct_type()
            for i in range(len(st.fields)):
                child_positions.append(self._write_field(st.fields[i]))

        var type_code = self._type_code(dtype)
        var type_pos = self._write_type_table(dtype)
        var name_pos = self._fb.create_string(f.name)

        var children_vec_pos: Optional[UInt32] = None
        if len(child_positions) > 0:
            children_vec_pos = self._fb.create_vector_offsets(child_positions)

        # Slot 4 (dictionary) is intentionally absent.
        var ts = self._fb.offset()
        var ch_at = UInt32(0)
        if children_vec_pos:
            ch_at = self._fb.prepend_uoffset(children_vec_pos.value())
        var tp_at = self._fb.prepend_uoffset(type_pos)
        var tc_at = self._fb.prepend_u8(type_code)
        var nb_at = self._fb.prepend_bool(f.nullable)
        var nm_at = self._fb.prepend_uoffset(name_pos)

        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, nm_at))
        flds.append(_FieldOffset(1, nb_at))
        flds.append(_FieldOffset(2, tc_at))
        flds.append(_FieldOffset(3, tp_at))
        if children_vec_pos:
            flds.append(_FieldOffset(5, ch_at))
        return self._fb.write_table(flds, ts)

    def _write_schema_table(mut self, fields: List[Field]) raises -> UInt32:
        var field_positions = List[UInt32]()
        for f in fields:
            field_positions.append(self._write_field(f))
        var fields_vec = self._fb.create_vector_offsets(field_positions)
        var ts = self._fb.offset()
        var fv_at = self._fb.prepend_uoffset(fields_vec)
        var en_at = self._fb.prepend_i16(_ENDIANNESS_LITTLE)
        var flds = List[_FieldOffset]()
        flds.append(_FieldOffset(0, en_at))
        flds.append(_FieldOffset(1, fv_at))
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
    """Decodes Arrow IPC metadata (schema, record batch, footer) from FlatBuffers."""

    var _r: _FlatbufReader

    def __init__(out self, var buf: List[UInt8]):
        self._r = _FlatbufReader(buf^)

    def peek_header(self) raises -> UInt8:
        return self._r.read_u8(self._r.root(), 1, 0)

    def decode_schema(self) raises -> Schema:
        var msg_pos = self._r.root()
        var schema_pos = self._r.read_table(msg_pos, 2)
        return Schema(fields=self._decode_schema_fields(schema_pos))

    def decode_record_batch(
        self,
        schema: Schema,
        var body: List[UInt8],
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
        var batch_dec = _BatchDecoder(body^, 0, nodes^, bufs^)
        var columns = List[AnyArray]()
        for f in schema.fields:
            columns.append(batch_dec.read_array(f.dtype))
        return RecordBatch(schema=schema, columns=columns^)

    def read_footer(
        self,
        mut fields: List[Field],
        mut blocks: List[_Block],
    ) raises:
        var footer_pos = self._r.root()
        var schema_pos = self._r.read_table(footer_pos, 1)
        fields = self._decode_schema_fields(schema_pos)
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

    def _decode_schema_fields(self, schema_pos: UInt32) raises -> List[Field]:
        var fields = List[Field]()
        var fields_vec = self._r.read_vector(schema_pos, 1)
        var n = Int(self._r.vector_len(fields_vec))
        for i in range(n):
            var fp = self._r.vec_offset(fields_vec, UInt32(i))
            fields.append(self._read_field(fp))
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
                _FieldNode(_read_le[DType.int64](sb, 0), _read_le[DType.int64](sb, 8))
            )

        var bufs_vec = self._r.read_vector(rb_pos, 2)
        var nb = Int(self._r.vector_len(bufs_vec))
        for i in range(nb):
            var sb = self._r.vec_struct_bytes(bufs_vec, UInt32(i), 16)
            bufs.append(
                _BodyBuffer(_read_le[DType.int64](sb, 0), _read_le[DType.int64](sb, 8))
            )

        return length

    def _read_field(self, fp: UInt32) raises -> Field:
        var name = self._r.read_string(fp, 0)
        var nullable = self._r.read_bool(fp, 1, True)
        var type_type = self._r.read_u8(fp, 2, 0)

        var children = List[Field]()
        try:
            var children_vec = self._r.read_vector(fp, 5)
            var n = Int(self._r.vector_len(children_vec))
            for i in range(n):
                var child_pos = self._r.vec_offset(children_vec, UInt32(i))
                children.append(self._read_field(child_pos))
        except:
            pass  # absent children vector is normal for leaf types

        var dtype: AnyDataType
        if type_type == _TYPE_BOOL:
            dtype = bool_
        elif type_type == _TYPE_INT:
            var tp = self._r.read_table(fp, 3)
            var bw = Int(self._r.read_i32(tp, 0, 32))
            var signed = self._r.read_bool(tp, 1, True)
            if signed:
                if bw == 8:
                    dtype = int8
                elif bw == 16:
                    dtype = int16
                elif bw == 32:
                    dtype = int32
                else:
                    dtype = int64
            else:
                if bw == 8:
                    dtype = uint8
                elif bw == 16:
                    dtype = uint16
                elif bw == 32:
                    dtype = uint32
                else:
                    dtype = uint64
        elif type_type == _TYPE_FLOATING_POINT:
            var tp = self._r.read_table(fp, 3)
            var prec = self._r.read_u16(tp, 0, _PRECISION_DOUBLE)
            if prec == _PRECISION_HALF:
                dtype = float16
            elif prec == _PRECISION_SINGLE:
                dtype = float32
            else:
                dtype = float64
        elif type_type == _TYPE_BINARY:
            dtype = binary
        elif type_type == _TYPE_UTF8:
            dtype = string
        elif type_type == _TYPE_LIST:
            if len(children) == 0:
                raise Error("list Field must have 1 child, got 0")
            dtype = mk_list(children[0].dtype.copy())
        elif type_type == _TYPE_FIXED_SIZE_LIST:
            var tp = self._r.read_table(fp, 3)
            var list_size = Int(self._r.read_i32(tp, 0, 0))
            if len(children) == 0:
                raise Error("fixed_size_list Field must have 1 child, got 0")
            dtype = mk_fixed_size_list(children[0].dtype.copy(), list_size)
        elif type_type == _TYPE_STRUCT:
            dtype = mk_struct(children^)
            return mk_field(name, dtype^, nullable)
        else:
            raise Error(
                "_IpcDecoder: unsupported type_type: " + String(Int(type_type))
            )

        return mk_field(name, dtype^, nullable)


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
        """Parse one message at the current position. Returns False at end-of-stream."""
        var n = len(self._bytes)
        if self._pos + 4 > n:
            return False

        var marker = _read_le[DType.int32](self._bytes, self._pos)
        var metadata_len: Int
        var meta_start: Int
        if UInt32(marker) == UInt32(0xFFFFFFFF):
            if self._pos + 8 > n:
                return False
            metadata_len = Int(_read_le[DType.int32](self._bytes, self._pos + 4))
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
        var body_len = Int(
            dec._r.read_i64(dec._r.root(), 3, 0)
        )

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
    """Traverses an ArrayData tree collecting _FieldNode metadata and raw buffer bytes."""

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

            for i in range(len(data.children) - 1, -1, -1):
                stack.append(data.children[i].copy())

    def encode(mut self, batch: RecordBatch) raises -> _EncodedBatch:
        for col in batch.columns:
            self.write_array(col.to_data())
        var buf_meta = List[_BodyBuffer]()
        var body = List[UInt8]()
        for buf in self.raw_bufs:
            _pad_to(body, 8)
            buf_meta.append(_BodyBuffer(Int64(len(body)), Int64(len(buf))))
            body.extend(Span(buf))
        _pad_to(body, 8)
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


# ---------------------------------------------------------------------------
# Batch body decoder: reconstructs AnyArray from raw bytes + cursor state
# ---------------------------------------------------------------------------


struct _BatchDecoder(Movable):
    """Reconstructs AnyArray values from a record batch body using node/buffer cursors."""

    var body: List[UInt8]
    var body_offset: Int
    var nodes: List[_FieldNode]
    var bufs: List[_BodyBuffer]
    var node_idx: Int
    var buf_idx: Int

    def __init__(
        out self,
        var body: List[UInt8],
        body_offset: Int,
        var nodes: List[_FieldNode],
        var bufs: List[_BodyBuffer],
    ):
        self.body = body^
        self.body_offset = body_offset
        self.nodes = nodes^
        self.bufs = bufs^
        self.node_idx = 0
        self.buf_idx = 0

    def read_array(mut self, dtype: AnyDataType) raises -> AnyArray:
        var node = self.nodes[self.node_idx]
        self.node_idx += 1

        var length = Int(node.length)
        var null_count = Int(node.null_count)

        var validity_buf = self.bufs[self.buf_idx]
        self.buf_idx += 1

        var bitmap: Optional[Bitmap[mut=False]] = None
        if null_count > 0 and validity_buf.length > 0:
            var off = Int(validity_buf.offset) + self.body_offset
            var n_bytes = Int(validity_buf.length)
            bitmap = Bitmap[mut=False](self._slice_body(off, n_bytes), length=length)

        var data_buffers = List[Buffer[mut=False]]()
        var children = List[ArrayData]()

        for _ in range(dtype.n_data_buffers()):
            self._consume_buffer(data_buffers)
        for child_dtype in dtype.child_dtypes():
            children.append(self.read_array(child_dtype).to_data())

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
            out.append(self._slice_body(Int(bb.offset) + self.body_offset, n_bytes))
        else:
            out.append(Buffer.alloc_zeroed[DType.uint8](0).to_immutable())


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
    var _fields: List[Field]
    var _blocks: List[_Block]
    var _enc: _BatchEncoder
    var _closed: Bool

    def __init__(out self, path: String, schema: Schema) raises:
        self._out = List[UInt8]()
        self._path = path
        self._fields = schema.fields.copy()
        self._blocks = List[_Block]()
        self._enc = _BatchEncoder()
        self._closed = False

        for b in _magic():
            self._out.append(b)
        var schema_msg = _IpcEncoder.frame_message(
            _IpcEncoder.encode_schema_message(self._fields), List[UInt8]()
        )
        self._out.extend(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise Error("RecordBatchFileWriter: writer is closed")
        var blk_start = Int64(len(self._out))
        var eb = self._enc.encode(batch)
        self._out.extend(Span(eb.msg))
        self._blocks.append(_Block(blk_start, eb.metadata_length, eb.body_length))

    def close(mut self) raises:
        if self._closed:
            return
        _pad_to(self._out, 8)
        var footer_bytes = _IpcEncoder.encode_footer(self._fields, self._blocks)
        self._out.extend(Span(footer_bytes))
        _append_le[DType.int32](self._out, Int32(len(footer_bytes)))
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
            _IpcEncoder.encode_schema_message(schema.fields.copy()),
            List[UInt8](),
        )
        self._out.extend(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise Error("RecordBatchStreamWriter: writer is closed")
        self._out.extend(Span(self._enc.encode(batch).msg))

    def close(mut self) raises:
        if self._closed:
            return
        _append_le[DType.uint32](self._out, UInt32(0xFFFFFFFF))
        _append_le[DType.int32](self._out, Int32(0))
        Path(self._path).write_bytes(self._out^)
        self._closed = True


# ---------------------------------------------------------------------------
# Public: file and stream readers
# ---------------------------------------------------------------------------


struct RecordBatchFileReader(Movable):
    """Reader for the Arrow IPC file format with random-access batch reads."""

    var schema: Schema
    var _blocks: List[_Block]
    var _msg_reader: _MessageReader

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
        var fields = List[Field]()
        var blocks = List[_Block]()
        dec.read_footer(fields, blocks)

        self.schema = Schema(fields=fields^)
        self._blocks = blocks^
        self._msg_reader = _MessageReader(file_bytes^)

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
        return dec.decode_record_batch(self.schema, body^)

    def read_all(mut self) raises -> List[RecordBatch]:
        # Footer only lists record-batch blocks, so no header check needed.
        var batches = List[RecordBatch]()
        for i in range(len(self._blocks)):
            batches.append(self.read_batch(i))
        return batches^


struct RecordBatchStreamReader(Movable):
    """Reader for the Arrow IPC stream format."""

    var schema: Schema
    var _msg_reader: _MessageReader

    def __init__(out self, path: String) raises:
        var file_bytes = Path(path).read_bytes()
        var msg_reader = _MessageReader(file_bytes^)
        var meta = List[UInt8]()
        var body = List[UInt8]()
        if not msg_reader.read_next(meta, body):
            raise Error("RecordBatchStreamReader: missing schema message")
        var dec = _IpcDecoder(meta^)
        self.schema = dec.decode_schema()
        self._msg_reader = msg_reader^

    def read_all(mut self) raises -> List[RecordBatch]:
        var batches = List[RecordBatch]()
        while True:
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not self._msg_reader.read_next(meta, body):
                break
            # Reuse one decoder for both the header peek and the decode.
            var dec = _IpcDecoder(meta^)
            if Int(dec.peek_header()) != Int(_HEADER_RECORD_BATCH):
                continue
            batches.append(dec.decode_record_batch(self.schema, body^))
        return batches^


# ---------------------------------------------------------------------------
# Public top-level functions
# ---------------------------------------------------------------------------


def write_ipc_file(
    path: String, fields: List[Field], batches: List[RecordBatch]
) raises:
    """Write RecordBatches to an Arrow IPC file with an explicit field list."""
    var w = RecordBatchFileWriter(path, Schema(fields=fields.copy()))
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_file(path: String, batches: List[RecordBatch]) raises:
    """Write RecordBatches to an Arrow IPC file."""
    if len(batches) == 0:
        raise Error(
            "write_ipc_file: no batches; use write_ipc_file(path, fields, batches)"
            " for schema-only files"
        )
    write_ipc_file(path, batches[0].schema.fields.copy(), batches)


def write_ipc_stream(
    path: String, fields: List[Field], batches: List[RecordBatch]
) raises:
    """Write RecordBatches to an Arrow IPC stream with an explicit field list."""
    var w = RecordBatchStreamWriter(path, Schema(fields=fields.copy()))
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_stream(path: String, batches: List[RecordBatch]) raises:
    """Write RecordBatches to an Arrow IPC stream."""
    if len(batches) == 0:
        raise Error(
            "write_ipc_stream: no batches; use write_ipc_stream(path, fields,"
            " batches) for schema-only streams"
        )
    write_ipc_stream(path, batches[0].schema.fields.copy(), batches)


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
