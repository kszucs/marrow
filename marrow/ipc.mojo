# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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

Bodies compressed with LZ4_FRAME or ZSTD (`BodyCompression`) are read, and
written when a writer is given one, with the native codecs in
`marrow.utils`.
"""

from std.math import ceildiv

from .errors import (
    CorruptError,
    DynError,
    IndexError,
    InternalError,
    InvalidError,
    NotImplementedError,
)
from .arrays import (
    ArrayData,
    BinaryViewArray,
    DynArray,
    Int32Array,
    NullArray,
    StringViewArray,
)
from .buffers import Buffer, Bitmap
from .views import BufferView
from .execution import ExecContext
from .io import (
    FOOTER_READ_SIZE,
    BufferSource,
    BufferedSink,
    ByteSink,
    ByteSource,
    DynSink,
    DynSource,
    FileSink,
    StorageOptions,
)
from .schema import Schema
from .tabular import RecordBatch
from .builders import Int32Builder
from .kernels.concat import concat as _concat
from .kernels.hashing import KeyCompare
from .kernels.filter import take as _take
from .utils import CompressionLibs, Lz4, Zstd
from . import dtypes as dt
from .codecs import LittleEndian


# ---------------------------------------------------------------------------
# Arrow IPC wire protocol constants
# ---------------------------------------------------------------------------

comptime _HEADER_SCHEMA: UInt8 = 1
comptime _HEADER_DICTIONARY_BATCH: UInt8 = 2
comptime _HEADER_RECORD_BATCH: UInt8 = 3
comptime _METADATA_VERSION_V5: Int16 = 4
comptime _ENDIANNESS_LITTLE: Int16 = 0
# How deeply schema fields may nest.
comptime _MAX_NESTING_DEPTH = 64
comptime _TYPE_NULL: UInt8 = 1
comptime _TYPE_INT: UInt8 = 2
comptime _TYPE_FLOATING_POINT: UInt8 = 3
comptime _TYPE_BINARY: UInt8 = 4
comptime _TYPE_UTF8: UInt8 = 5
comptime _TYPE_LARGE_BINARY: UInt8 = 19
comptime _TYPE_LARGE_UTF8: UInt8 = 20
comptime _TYPE_LARGE_LIST: UInt8 = 21
comptime _TYPE_BINARY_VIEW: UInt8 = 23
comptime _TYPE_UTF8_VIEW: UInt8 = 24
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
comptime _TYPE_MAP: UInt8 = 17
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
struct BodyCompression(Equatable, ImplicitlyCopyable, Movable):
    """How a record batch's body buffers are compressed (`Message.fbs`'s
    `BodyCompression`, method `BUFFER`): each on its own, with one of two
    codecs, prefixed by its uncompressed length as a little-endian i64 -- `-1`
    for a buffer stored as is. An empty buffer carries no prefix."""

    var codec: Int

    comptime LZ4_FRAME = Self(0)
    comptime ZSTD = Self(1)

    @staticmethod
    def from_code(code: Int) raises CorruptError -> Self:
        """The codec a `BodyCompression` table names."""
        if code != 0 and code != 1:
            raise CorruptError(t"ipc: unknown body compression codec {code}")
        return Self(code)

    @staticmethod
    def from_name(name: String) raises InvalidError -> Self:
        """`lz4` (or `lz4_frame`) or `zstd`, in any case, as pyarrow's
        `IpcWriteOptions` names them."""
        var lower = name.lower()
        if lower == "lz4" or lower == "lz4_frame":
            return Self.LZ4_FRAME
        elif lower == "zstd":
            return Self.ZSTD
        raise InvalidError(t"ipc: unknown body compression '{name}'")

    def compress(
        self,
        src: Span[mut=False, UInt8, _],
        mut out: List[UInt8],
        native: Bool = True,
    ) raises:
        """Append `src`, compressed behind its length, to `out` -- always
        compressed, as Arrow C++ writes it unless asked for a minimum saving
        -- or nothing for an empty `src`. Without `native`, liblz4 or
        libzstd compresses it."""
        if len(src) > 0:
            LittleEndian.append[DType.int64](out, Int64(len(src)))
            if self == Self.LZ4_FRAME:
                if native:
                    Lz4.compress_frame(src, out)
                else:
                    CompressionLibs.lz4_compress_frame(src, out)
            elif native:
                Zstd.compress(src, out)
            else:
                CompressionLibs.zstd_compress(src, out)

    def decompress(
        self, src: Span[UInt8, _], native: Bool = True
    ) raises DynError -> Buffer[mut=False]:
        """The buffer whose compressed form is `src` -- empty for an empty
        `src`, as `compress` writes an empty buffer. Without `native`,
        liblz4 or libzstd decompresses it."""
        var n = 0
        var payload = src
        var raw = False
        if len(src) > 0:
            n = Int(LittleEndian.checked[DType.int64](src, 0))
            payload = src[8:]
            raw = n == -1
        if raw:
            n = len(payload)
        else:
            # Checked before allocating: the length comes from the file.
            var most: Int
            if self == Self.LZ4_FRAME:
                most = Lz4.max_decompressed_length(len(payload))
            else:
                most = Zstd.max_decompressed_length(len(payload))
            if n < 0 or n > most:
                raise CorruptError(
                    t"ipc: a compressed buffer of {n} bytes in {len(payload)}"
                )
        var buf = Buffer.alloc_uninit[DType.uint8](n)
        var dst = buf.view(0, n)
        if raw:
            dst.copy_from(BufferView(payload), n)
        elif n > 0:
            # A length of 0 is an empty buffer, whatever follows it: Arrow
            # Java writes one with nothing after it.
            if native and self == Self.LZ4_FRAME:
                Lz4.decompress_frame_into(payload, dst.as_span())
            elif native:
                Zstd.decompress_into(payload, dst.as_span())
            else:
                try:
                    if self == Self.LZ4_FRAME:
                        CompressionLibs.lz4_decompress_frame(
                            payload, dst.as_span()
                        )
                    else:
                        CompressionLibs.zstd_decompress(payload, dst.as_span())
                except e:
                    raise DynError(e)
        return buf^.to_immutable()


@fieldwise_init
struct _Block(ImplicitlyCopyable, Movable):
    var offset: Int64
    var metadata_length: Int32
    var body_length: Int64


struct _DictPair(Copyable, Movable):
    """A (dict_id, values) pair collected from an ArrayData tree."""

    var dict_id: Int
    var values: DynArray

    def __init__(out self, dict_id: Int, var values: DynArray):
        self.dict_id = dict_id
        self.values = values^

    def __init__(out self, *, copy: Self):
        self.dict_id = copy.dict_id
        self.values = copy.values.copy()


struct _DictLookup(Movable):
    """Result of searching a (_dtype, _FieldIpcInfo) tree for a dict_id."""

    var value_type: dt.DynType
    var value_ipc_info: _FieldIpcInfo

    def __init__(
        out self,
        var value_type: dt.DynType,
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
    # (`children: List[_FieldIpcInfo]`) is Deinitable; fields are still
    # destroyed automatically after the body runs.
    def __deinit__(deinit self):
        pass

    @staticmethod
    def find(
        dtype: dt.DynType, ipc_info: _FieldIpcInfo, target_id: Int
    ) raises -> Optional[_DictLookup]:
        """Search the (dtype, ipc_info) shadow tree for dict_id == target_id.

        Returns a `_DictLookup` whose `value_ipc_info` has `dict_id=-1` with
        the matched node's children, so that nested dicts inside the value type
        can still be resolved during decoding.
        """
        # Every node that is not a dictionary carries -1, which a dictionary
        # batch may name too.
        if ipc_info.dict_id == target_id and dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var vt_ipc = _FieldIpcInfo(-1, ipc_info.children.copy())
            return _DictLookup(d.value_type().copy(), vt_ipc^)
        if dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var vt_ipc = _FieldIpcInfo(-1, ipc_info.children.copy())
            return _FieldIpcInfo.find(d.value_type().copy(), vt_ipc^, target_id)
        var n = min(dtype.layout().num_children, len(ipc_info.children))
        for i in range(n):
            var found = _FieldIpcInfo.find(
                dtype.child_type(i), ipc_info.children[i].copy(), target_id
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
            raise InvalidError("flatbuffers: buffer too large to grow")
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
            raise InternalError(
                "flatbuffers: create_vector_structs: negative count"
            )
        if len(data) != count * struct_size:
            raise InternalError(
                t"flatbuffers: create_vector_structs: data length "
                t"{len(data)} != count({count}) * struct_size("
                t"{struct_size})"
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

    def size(self) -> Int:
        """The buffer's length in bytes."""
        return len(self._buf)

    def root(self) raises -> UInt32:
        return LittleEndian.checked[DType.uint32](self._buf, 0)

    def _field_voffset(self, table_pos: UInt32, slot: Int) raises -> UInt16:
        var tp = Int(table_pos)
        var soffset_raw = LittleEndian.checked[DType.uint32](self._buf, tp)
        var vt = Int(table_pos - soffset_raw)
        if vt < 0 or vt >= len(self._buf):
            raise CorruptError(
                t"flatbuffers: vtable position out of bounds: {vt}"
            )
        var vt_size = Int(LittleEndian.checked[DType.uint16](self._buf, vt))
        var slot_byte = 4 + slot * 2
        if slot_byte + 1 >= vt_size:
            return UInt16(0)
        if vt + slot_byte + 1 >= len(self._buf):
            return UInt16(0)
        return LittleEndian.checked[DType.uint16](self._buf, vt + slot_byte)

    def read_u8(
        self, tp: UInt32, slot: Int, default: UInt8 = 0
    ) raises -> UInt8:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return LittleEndian.checked[DType.uint8](self._buf, Int(tp) + Int(voff))

    def read_u16(
        self, tp: UInt32, slot: Int, default: UInt16 = 0
    ) raises -> UInt16:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return LittleEndian.checked[DType.uint16](
            self._buf, Int(tp) + Int(voff)
        )

    def read_i32(
        self, tp: UInt32, slot: Int, default: Int32 = 0
    ) raises -> Int32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return LittleEndian.checked[DType.int32](self._buf, Int(tp) + Int(voff))

    def read_i64(
        self, tp: UInt32, slot: Int, default: Int64 = 0
    ) raises -> Int64:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return LittleEndian.checked[DType.int64](self._buf, Int(tp) + Int(voff))

    def read_bool(
        self, tp: UInt32, slot: Int, default: Bool = False
    ) raises -> Bool:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        return LittleEndian.checked[DType.uint8](
            self._buf, Int(tp) + Int(voff)
        ) != UInt8(0)

    def read_string(
        self, tp: UInt32, slot: Int, default: String = ""
    ) raises -> String:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            return default
        var ref_pos = Int(tp) + Int(voff)
        var str_pos = ref_pos + Int(
            LittleEndian.checked[DType.uint32](self._buf, ref_pos)
        )
        var length = Int(LittleEndian.checked[DType.uint32](self._buf, str_pos))
        if str_pos + 4 + length > len(self._buf):
            raise CorruptError("flatbuffers: string extends beyond buffer")
        try:
            return String(
                from_utf8=Span(self._buf)[str_pos + 4 : str_pos + 4 + length]
            )
        except:
            raise CorruptError(
                t"flatbuffers: the string at {str_pos} is not valid UTF-8"
            )

    def read_vector(self, tp: UInt32, slot: Int) raises -> UInt32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            raise CorruptError(
                t"flatbuffers: absent offset field at slot {slot}"
            )
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + LittleEndian.checked[DType.uint32](
            self._buf, ref_pos
        )

    def has_field(self, tp: UInt32, slot: Int) raises -> Bool:
        """Whether an optional table field is present.

        FlatBuffers tables omit absent fields from the vtable, so a reader must
        ask before reading — the typed readers raise on absence.
        """
        return self._field_voffset(tp, slot) != 0

    def read_table(self, tp: UInt32, slot: Int) raises -> UInt32:
        var voff = self._field_voffset(tp, slot)
        if voff == 0:
            raise CorruptError(
                t"flatbuffers: absent offset field at slot {slot}"
            )
        var ref_pos = Int(tp) + Int(voff)
        return UInt32(ref_pos) + LittleEndian.checked[DType.uint32](
            self._buf, ref_pos
        )

    def vector_len(self, vec_pos: UInt32) raises -> UInt32:
        return LittleEndian.checked[DType.uint32](self._buf, Int(vec_pos))

    def vec_offset(self, vec_pos: UInt32, i: UInt32) raises -> UInt32:
        var vlen = self.vector_len(vec_pos)
        if i >= vlen:
            raise CorruptError(t"flatbuffers: vec index {i} >= len {vlen}")
        var elem_pos = Int(vec_pos) + 4 + Int(i) * 4
        return UInt32(elem_pos) + LittleEndian.checked[DType.uint32](
            self._buf, elem_pos
        )

    def vec_struct_bytes(
        self, vec_pos: UInt32, i: UInt32, struct_size: Int
    ) raises -> List[UInt8]:
        var vlen = self.vector_len(vec_pos)
        if i >= vlen:
            raise CorruptError(
                t"flatbuffers: vec_struct_bytes index {i} >= len {vlen}"
            )
        var start = Int(vec_pos) + 4 + Int(i) * struct_size
        var end = start + struct_size
        if end > len(self._buf):
            raise CorruptError(
                "flatbuffers: vec_struct_bytes extends beyond buffer"
            )
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
        variadic_counts: List[Int64],
        compression: Optional[BodyCompression],
    ) raises -> List[UInt8]:
        var enc = _IpcEncoder(512)
        var nodes_vec = enc._write_field_nodes_vec(nodes)
        var bufs_vec = enc._write_body_buffers_vec(buffers)
        var rb_pos = enc._write_record_batch_table(
            length, nodes_vec, bufs_vec, variadic_counts, compression
        )

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
        variadic_counts: List[Int64],
        compression: Optional[BodyCompression],
    ) raises -> UInt32:
        # Slot 4, `variadicBufferCounts`: one entry per view-layout node, in
        # node order, giving how many data buffers follow its views buffer.
        # Written only when a view column is present, as the spec allows.
        var vc_vec = UInt32(0)
        if len(variadic_counts) > 0:
            var data = List[UInt8](capacity=len(variadic_counts) * 8)
            for c in variadic_counts:
                LittleEndian.append[DType.int64](data, c)
            vc_vec = self._fb.create_vector_structs(
                data, len(variadic_counts), 8, 8
            )
        # The BodyCompression table, slot 3, goes in ahead of its parent too.
        var bc_pos = UInt32(0)
        if compression:
            var bts = self._fb.offset()
            var method_at = self._fb.prepend_u8(0)  # BUFFER
            var codec_at = self._fb.prepend_u8(UInt8(compression.value().codec))
            var bflds = List[_FieldOffset]()
            bflds.append(_FieldOffset(0, codec_at))
            bflds.append(_FieldOffset(1, method_at))
            bc_pos = self._fb.write_table(bflds, bts)
        var ts = self._fb.offset()
        var flds = List[_FieldOffset]()
        var vc_at = UInt32(0)
        if len(variadic_counts) > 0:
            vc_at = self._fb.prepend_uoffset(vc_vec)
        if compression:
            flds.append(_FieldOffset(3, self._fb.prepend_uoffset(bc_pos)))
        var bv_at = self._fb.prepend_uoffset(bufs_vec)
        var nv_at = self._fb.prepend_uoffset(nodes_vec)
        var ln_at = self._fb.prepend_i64(length)
        flds.append(_FieldOffset(0, ln_at))
        flds.append(_FieldOffset(1, nv_at))
        flds.append(_FieldOffset(2, bv_at))
        if len(variadic_counts) > 0:
            flds.append(_FieldOffset(4, vc_at))
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

    def _type_code(self, dtype: dt.DynType) raises -> UInt8:
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
        elif dtype.is_binary_view():
            return _TYPE_BINARY_VIEW
        elif dtype.is_string_view():
            return _TYPE_UTF8_VIEW
        elif dtype.is_map():
            # Before `is_list()`: a map is a list of entry structs, and if
            # `is_list()` answered first a map would be written as a plain list
            # and read back as one.
            return _TYPE_MAP
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
            raise NotImplementedError(
                t"_IpcEncoder: unsupported dtype: {dtype}"
            )

    def _write_type_table(mut self, dtype: dt.DynType) raises -> UInt32:
        if (
            dtype.is_null()
            or dtype.is_bool()
            or dtype.is_binary()
            or dtype.is_large_binary()
            or dtype.is_string()
            or dtype.is_large_string()
            or dtype.is_string_view()
            or dtype.is_binary_view()
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
        elif dtype.is_map():
            ref m = dtype.as_map()
            var ts = self._fb.offset()
            var ks_at = self._fb.prepend_bool(m.keys_sorted)
            var flds = List[_FieldOffset]()
            flds.append(_FieldOffset(0, ks_at))
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
                precision = Int32(d.precision())
                scale = Int32(d.scale())
                bit_width = 32
            elif dtype.is_decimal64():
                ref d = dtype.as_decimal64()
                precision = Int32(d.precision())
                scale = Int32(d.scale())
                bit_width = 64
            elif dtype.is_decimal128():
                ref d = dtype.as_decimal128()
                precision = Int32(d.precision())
                scale = Int32(d.scale())
                bit_width = 128
            else:
                ref d = dtype.as_decimal256()
                precision = Int32(d.precision())
                scale = Int32(d.scale())
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
            raise NotImplementedError(
                t"_IpcEncoder: unsupported dtype for type table: {dtype}"
            )

    def _write_dictionary_encoding_table(
        mut self, dict_id: Int64, index_dtype: dt.DynType, ordered: Bool
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

        # Children first -- a dictionary's are its value type's -- so inner
        # dictionaries get lower ids than outer ones. A leaf has none.
        var children = dtype.children()
        if len(children) > 0:
            for child in children:
                child_positions.append(self._write_field(child, next_dict_id))
        if dtype.is_dictionary():
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

    def peek_is_delta(self) raises -> Bool:
        """`isDelta` from a DictionaryBatch header (slot 2).

        A delta batch carries only the values appended since the last one. The
        field is optional and defaults to false, so absence means "replace".
        """
        var msg_tp = self._r.root()
        var db_pos = self._r.read_table(msg_tp, 2)
        if not self._r.has_field(db_pos, 2):
            return False
        return self._r.read_bool(db_pos, 2, False)

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
        if self._r.has_field(schema_pos, 2):
            metadata = self._read_kv_vec(schema_pos, 2)
        return Schema(fields=fields^, metadata=metadata^)

    def decode_dict_batch(
        mut self,
        value_dtype: dt.DynType,
        values_ipc_info: _FieldIpcInfo,
        var body: List[UInt8],
        dict_values: List[DynArray] = List[DynArray](),
        native_codecs: Bool = True,
    ) raises -> DynArray:
        var msg_tp = self._r.root()
        var db_pos = self._r.read_table(msg_tp, 2)
        var rb_pos = self._r.read_table(db_pos, 1)
        var nodes = List[_FieldNode]()
        var bufs = List[_BodyBuffer]()
        var variadic = List[Int64]()
        var compression = self._read_record_batch_meta(
            rb_pos, nodes, bufs, variadic
        )
        var batch_dec = _BatchDecoder(
            body^,
            0,
            nodes^,
            bufs^,
            variadic^,
            compression,
            native_codecs,
            dict_values,
        )
        return batch_dec.read_array(value_dtype, values_ipc_info)

    def decode_record_batch(
        mut self,
        schema: Schema,
        ipc_infos: List[_FieldIpcInfo],
        var body: List[UInt8],
        dict_values: List[DynArray] = List[DynArray](),
        native_codecs: Bool = True,
    ) raises -> RecordBatch:
        var msg_tp = self._r.root()
        var hdr_type = self._r.read_u8(msg_tp, 1, 0)
        if Int(hdr_type) != Int(_HEADER_RECORD_BATCH):
            raise CorruptError(
                t"_IpcDecoder: expected record-batch header, got "
                t"{Int(hdr_type)}"
            )
        var rb_pos = self._r.read_table(msg_tp, 2)
        var nodes = List[_FieldNode]()
        var bufs = List[_BodyBuffer]()
        var variadic = List[Int64]()
        var compression = self._read_record_batch_meta(
            rb_pos, nodes, bufs, variadic
        )
        var batch_dec = _BatchDecoder(
            body^,
            0,
            nodes^,
            bufs^,
            variadic^,
            compression,
            native_codecs,
            dict_values,
        )
        var length = Int(self._r.read_i64(rb_pos, 0, 0))
        var columns = List[DynArray]()
        for i in range(len(schema.fields)):
            var ipc = (
                ipc_infos[i].copy() if i < len(ipc_infos) else _FieldIpcInfo()
            )
            columns.append(batch_dec.read_array(schema.fields[i].dtype, ipc))
            if columns[i].length() != length:
                raise CorruptError(
                    t"ipc: column {i} holds {columns[i].length()} values in a"
                    t" batch of {length} rows"
                )
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
        if self._r.has_field(schema_pos, 2):
            metadata = self._read_kv_vec(schema_pos, 2)
        var dv = self._r.read_vector(footer_pos, 2)
        var nd = Int(self._r.vector_len(dv))
        for i in range(nd):
            var sb = self._r.vec_struct_bytes(dv, UInt32(i), 24)
            dict_blocks.append(
                _Block(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int32](sb, 8),
                    LittleEndian.checked[DType.int64](sb, 16),
                )
            )
        var rb_vec = self._r.read_vector(footer_pos, 3)
        var n = Int(self._r.vector_len(rb_vec))
        for i in range(n):
            var sb = self._r.vec_struct_bytes(rb_vec, UInt32(i), 24)
            blocks.append(
                _Block(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int32](sb, 8),
                    LittleEndian.checked[DType.int64](sb, 16),
                )
            )
        return Schema(fields=fields^, metadata=metadata^)

    def _decode_schema_fields(
        self, schema_pos: UInt32, mut out_ipc: List[_FieldIpcInfo]
    ) raises -> List[dt.Field]:
        # Field 0 is the endianness of every buffer that follows; reading
        # big-endian data as little-endian would answer wrong values silently.
        if self._r.read_u16(schema_pos, 0) != UInt16(_ENDIANNESS_LITTLE):
            raise NotImplementedError("ipc: big-endian data is not supported")
        var fields = List[dt.Field]()
        var fields_vec = self._r.read_vector(schema_pos, 1)
        var n = Int(self._r.vector_len(fields_vec))
        var budget = self._r.size() // 4
        for i in range(n):
            var fp = self._r.vec_offset(fields_vec, UInt32(i))
            fields.append(self._read_field(fp, out_ipc, budget, 0))
        return fields^

    def _read_record_batch_meta(
        self,
        rb_pos: UInt32,
        mut nodes: List[_FieldNode],
        mut bufs: List[_BodyBuffer],
        mut variadic_counts: List[Int64],
    ) raises -> Optional[BodyCompression]:
        """Read the batch's field nodes, buffers and view buffer counts into
        `nodes`, `bufs` and `variadic_counts`; return how its body is
        compressed, if it is."""
        # Slot 3 is `BodyCompression`; absent, the buffers are raw.
        var compression = Optional[BodyCompression]()
        if self._r.has_field(rb_pos, 3):
            var bc = self._r.read_table(rb_pos, 3)
            var method = Int(self._r.read_u8(bc, 1, 0))
            if method != 0:
                raise NotImplementedError(
                    t"ipc: body compression method {method}"
                )
            compression = BodyCompression.from_code(
                Int(self._r.read_u8(bc, 0, 0))
            )

        var nodes_vec = self._r.read_vector(rb_pos, 1)
        var nn = Int(self._r.vector_len(nodes_vec))
        for i in range(nn):
            var sb = self._r.vec_struct_bytes(nodes_vec, UInt32(i), 16)
            nodes.append(
                _FieldNode(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int64](sb, 8),
                )
            )

        var bufs_vec = self._r.read_vector(rb_pos, 2)
        var nb = Int(self._r.vector_len(bufs_vec))
        for i in range(nb):
            var sb = self._r.vec_struct_bytes(bufs_vec, UInt32(i), 16)
            bufs.append(
                _BodyBuffer(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int64](sb, 8),
                )
            )

        if self._r.has_field(rb_pos, 4):
            var vc_vec = self._r.read_vector(rb_pos, 4)
            var nv = Int(self._r.vector_len(vc_vec))
            for i in range(nv):
                var sb = self._r.vec_struct_bytes(vc_vec, UInt32(i), 8)
                variadic_counts.append(LittleEndian.checked[DType.int64](sb, 0))

        return compression

    def _read_field(
        self,
        fp: UInt32,
        mut out_ipc: List[_FieldIpcInfo],
        mut budget: Int,
        depth: Int,
    ) raises -> dt.Field:
        # Child offsets may point back at an enclosing table, or several at
        # one table, so both the nesting and the number of fields read are
        # bounded rather than trusted. A tree of fields spends at least one
        # 4-byte offset per field, which is what `budget` counts down.
        if depth >= _MAX_NESTING_DEPTH:
            raise CorruptError(
                t"ipc: fields nested deeper than {_MAX_NESTING_DEPTH} levels"
            )
        budget -= 1
        if budget < 0:
            raise CorruptError(
                "ipc: the schema names more fields than its metadata can hold"
            )
        var name = self._r.read_string(fp, 0)
        var nullable = self._r.read_bool(fp, 1, False)
        var type_type = self._r.read_u8(fp, 2, 0)

        var children = List[dt.Field]()
        var child_ipc = List[_FieldIpcInfo]()
        # Absent for a leaf. A present but malformed one, or a child that does
        # not read, is corrupt rather than empty.
        if self._r.has_field(fp, 5):
            var children_vec = self._r.read_vector(fp, 5)
            var n = Int(self._r.vector_len(children_vec))
            for i in range(n):
                var child_pos = self._r.vec_offset(children_vec, UInt32(i))
                children.append(
                    self._read_field(child_pos, child_ipc, budget, depth + 1)
                )

        var metadata = Dict[String, String]()
        if self._r.has_field(fp, 6):
            metadata = self._read_kv_vec(fp, 6)

        var dtype: dt.DynType
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
            var prec = self._r.read_u16(tp, 0, _PRECISION_HALF)
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
        elif type_type == _TYPE_BINARY_VIEW:
            dtype = dt.binary_view
        elif type_type == _TYPE_UTF8_VIEW:
            dtype = dt.string_view
        elif type_type == _TYPE_MAP:
            var tp = self._r.read_table(fp, 3)
            var keys_sorted = self._r.read_bool(tp, 0, False)
            if len(children) == 0:
                raise CorruptError("map Field must have 1 child, got 0")
            # `MapType` reads its key and item out of a two-field struct.
            ref entries = children[0].dtype
            if not entries.is_struct() or len(entries.as_struct().fields) != 2:
                raise CorruptError(
                    t"map Field must hold a struct of key and value, got"
                    t" {entries}"
                )
            dtype = dt.MapType(children[0].copy(), keys_sorted).to_dyn()
        elif type_type == _TYPE_LIST:
            if len(children) == 0:
                raise CorruptError("list Field must have 1 child, got 0")
            # Preserve the child Field as-is (its name may not be the default
            # "item" — e.g. arrow-rs uses "inner_list" for nested lists).
            dtype = dt.ListType(children[0].copy()).to_dyn()
        elif type_type == _TYPE_LARGE_LIST:
            if len(children) == 0:
                raise CorruptError("large_list Field must have 1 child, got 0")
            dtype = dt.LargeListType(children[0].copy()).to_dyn()
        elif type_type == _TYPE_FIXED_SIZE_LIST:
            var tp = self._r.read_table(fp, 3)
            var list_size = Int(self._r.read_i32(tp, 0, 0))
            if len(children) == 0:
                raise CorruptError(
                    "fixed_size_list Field must have 1 child, got 0"
                )
            dtype = dt.FixedSizeListType(children[0].copy(), list_size).to_dyn()
        elif type_type == _TYPE_FIXED_SIZE_BINARY:
            var tp = self._r.read_table(fp, 3)
            var byte_width = Int(self._r.read_i32(tp, 0, 0))
            dtype = dt.FixedSizeBinaryType(byte_width).to_dyn()
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
            ).to_dyn()
        elif type_type == _TYPE_DURATION:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _TIME_UNIT_MILLISECOND)
            dtype = dt.duration(_IpcDecoder._wire_to_time_unit(unit_v)).to_dyn()
        elif type_type == _TYPE_INTERVAL:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, _INTERVAL_UNIT_YEAR_MONTH)
            if unit_v == _INTERVAL_UNIT_DAY_TIME:
                dtype = dt.day_time_interval().to_dyn()
            elif unit_v == _INTERVAL_UNIT_MONTH_DAY_NANO:
                dtype = dt.month_day_nano_interval().to_dyn()
            else:
                dtype = dt.year_month_interval().to_dyn()
        elif type_type == _TYPE_STRUCT:
            dtype = dt.struct_(children^)
        else:
            raise NotImplementedError(
                t"_IpcDecoder: unsupported type_type: {Int(type_type)}"
            )

        # DictionaryEncoding, slot 4, wraps the value type in DictionaryType.
        # The dict_id is stored in _FieldIpcInfo rather than on the type itself.
        var own_dict_id = -1
        if self._r.has_field(fp, 4):
            var de_pos = self._r.read_table(fp, 4)
            own_dict_id = Int(self._r.read_i64(de_pos, 0, 0))
            if own_dict_id < 0:
                # The readers index their dictionaries by id.
                raise NotImplementedError(
                    t"ipc: negative dictionary id {own_dict_id}"
                )
            # Without an index type the indices are signed 32-bit.
            var idx_bw = 32
            var idx_signed = True
            if self._r.has_field(de_pos, 1):
                var idx_tp = self._r.read_table(de_pos, 1)
                idx_bw = Int(self._r.read_i32(idx_tp, 0, 32))
                idx_signed = self._r.read_bool(idx_tp, 1, False)
            var ordered = self._r.read_bool(de_pos, 2, False)
            var index_dtype: dt.DynType
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
            dtype = dt.dictionary(index_dtype^, dtype.copy(), ordered).to_dyn()

        out_ipc.append(_FieldIpcInfo(own_dict_id, child_ipc^))
        return dt.Field(name, dtype^, nullable, metadata^)


# ---------------------------------------------------------------------------
# IPC framing helpers and framed message reader
# ---------------------------------------------------------------------------


def _read_message[
    S: ByteSource
](
    ref src: S,
    mut pos: Int,
    mut meta: List[UInt8],
    mut body: List[UInt8],
) raises -> Bool:
    """Parse one framed IPC message (continuation + length + metadata + body)
    at `pos`, advancing it past the message. Returns False at end-of-stream.

    A free generic function rather than a method on a `_MessageReader[S]`
    struct: the readers already hold the source and the cursor, and a struct
    whose only job was to pair them added a nested generic instantiation for
    nothing.

    Three ranged reads per message — the frame prefix, the metadata, the body.
    On a memory map those are three `Buffer.view` calls; on a remote source
    they are three round trips, and a caller that knows a message's extent from
    a footer `_Block` could ask for it in one. That optimisation is not here.

    The metadata and the body go through `read_ranges`, not `read_at`, and that
    is not a style choice: `read_at` hands back a span borrowed from storage
    the source keeps alive for *its* whole lifetime, so a fetching source
    retains every byte it ever served. A message body is the bulk of the file,
    so reading a 5 GB stream that way would end holding 5 GB. `read_ranges`
    returns a batch this function owns and drops. The 8-byte frame prefix goes
    the same way: it is tiny, but one arena entry *per message* still grows
    with the stream, and "bounded per call" is not the same as bounded.
    """
    var n = src.size()
    if pos + 4 > n:
        return False
    # Every fetch below is one range, so there is nothing to fan out.
    var one = ExecContext.serial()

    # The frame prefix is 4 bytes, or 8 when the continuation marker is
    # present. Ask for as much of it as the source has left.
    var pre_fetched = src.read_ranges([(pos, min(8, n - pos))], one)
    var pre = pre_fetched.span(0)
    var marker = LittleEndian.checked[DType.int32](pre, 0)
    var metadata_len: Int
    var meta_start: Int
    if UInt32(marker) == UInt32(0xFFFFFFFF):
        if pos + 8 > n:
            return False
        metadata_len = Int(LittleEndian.checked[DType.int32](pre, 4))
        meta_start = pos + 8
    else:
        metadata_len = Int(marker)
        meta_start = pos + 4

    if metadata_len == 0:
        pos = meta_start
        return False
    if metadata_len < 0 or meta_start + metadata_len > n:
        raise CorruptError(
            t"IPC: message metadata at {meta_start} runs past the end of a "
            t"{n}-byte source"
        )
    var meta_fetched = src.read_ranges([(meta_start, metadata_len)], one)
    meta.extend(meta_fetched.span(0))

    var raw_end = meta_start + metadata_len
    var meta_end = raw_end + (8 - raw_end % 8) % 8

    var dec = _IpcDecoder(meta.copy())
    var body_len = Int(dec.body_length())
    if body_len < 0 or meta_end + body_len > n:
        raise CorruptError(
            t"IPC: message body at {meta_end} runs past the end of a {n}-byte "
            t"source"
        )
    var body_fetched = src.read_ranges([(meta_end, body_len)], one)
    body.extend(body_fetched.span(0))

    pos = meta_end + body_len
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
    var variadic_counts: List[Int64]
    var compression: Optional[BodyCompression]
    var native_codecs: Bool

    def __init__(
        out self, compression: Optional[BodyCompression], native_codecs: Bool
    ):
        self.nodes = List[_FieldNode]()
        self.raw_bufs = List[List[UInt8]]()
        self.variadic_counts = List[Int64]()
        self.compression = compression
        self.native_codecs = native_codecs

    @staticmethod
    def dense(arr: DynArray) raises -> ArrayData:
        """Rebase a sliced array to offset 0.

        Arrow IPC bodies are written dense from index 0 — there is no offset
        field on the wire. The encoder writes whole value buffers under the
        slice's declared length and the decoder hardcodes `offset=0`, so a
        sliced column would round-trip as the parent's *first* `length`
        elements. Gathering `0..n` produces a dense copy for every layout,
        nested children included, and is paid only when `offset != 0`.
        """
        var data = arr.to_data()
        if data.offset == 0:
            return data^
        var idx = Int32Builder(capacity=len(arr))
        for i in range(len(arr)):
            idx.append(Int32(i))
        return _take(arr, idx.finish()).to_data()

    def write_array(mut self, root: ArrayData) raises:
        var stack = List[ArrayData]()
        stack.append(root.copy())
        while len(stack) > 0:
            var data = stack.pop()

            self.nodes.append(_FieldNode(Int64(data.length), Int64(data.nulls)))

            # Null type: emit FieldNode but NO body buffers (not even validity).
            if data.dtype.is_null():
                continue

            # A sparse view node -- a slice, or a filter or take result still
            # sharing its source's buffers -- is compacted first, or the body
            # would carry every byte those buffers hold. The same ladder as
            # `Pipeline.collect`'s.
            if data.dtype.is_string_view():
                data = (
                    StringViewArray(unsafe_from_data=data).compact().to_data()
                )
            elif data.dtype.is_binary_view():
                data = (
                    BinaryViewArray(unsafe_from_data=data).compact().to_data()
                )

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

            if data.dtype.is_string_view() or data.dtype.is_binary_view():
                self.variadic_counts.append(Int64(len(data.buffers) - 1))
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
                _DictPair(next_id, DynArray.from_data(data.children[0]))
            )
            next_id += 1
        else:
            for child in data.children:
                _BatchEncoder.collect_dict_pairs(child, pairs, next_id)

    def _build_body(
        mut self, mut buf_meta: List[_BodyBuffer], mut body: List[UInt8]
    ) raises:
        """Assemble the buffers into a padded body, populating buf_meta
        offsets -- each non-empty one compressed if the batch is."""
        for buf in self.raw_bufs:
            _pad_to(body, 8)
            var at = len(body)
            if self.compression:
                self.compression.value().compress(
                    Span(buf), body, self.native_codecs
                )
            else:
                body.extend(Span(buf))
            buf_meta.append(_BodyBuffer(Int64(at), Int64(len(body) - at)))
        _pad_to(body, 8)

    def encode(mut self, batch: RecordBatch) raises -> _EncodedBatch:
        for col in batch.columns:
            self.write_array(_BatchEncoder.dense(col))
        var buf_meta = List[_BodyBuffer]()
        var body = List[UInt8]()
        self._build_body(buf_meta, body)
        var rb_meta = _IpcEncoder.encode_record_batch(
            Int64(batch.num_rows()),
            self.nodes,
            buf_meta,
            self.variadic_counts,
            self.compression,
        )
        var meta_len = len(rb_meta)
        var padded_meta = meta_len + (8 - meta_len % 8) % 8
        var metadata_length = Int32(8 + padded_meta)
        var body_length = Int64(len(body))
        var msg = _IpcEncoder.frame_message(rb_meta, body)
        self.nodes = List[_FieldNode]()
        self.raw_bufs = List[List[UInt8]]()
        self.variadic_counts = List[Int64]()
        return _EncodedBatch(msg^, metadata_length, body_length)

    def encode_dict_message(
        self, dict_id: Int64, values: DynArray
    ) raises -> _EncodedBatch:
        """Encode a dictionary values array as a DictionaryBatch IPC message,
        compressed as this encoder's batches are."""
        var benc = _BatchEncoder(self.compression, self.native_codecs)
        benc.write_array(_BatchEncoder.dense(values))
        var buf_meta = List[_BodyBuffer]()
        var body = List[UInt8]()
        benc._build_body(buf_meta, body)

        var enc = _IpcEncoder(512)
        var nodes_vec = enc._write_field_nodes_vec(benc.nodes)
        var bufs_vec = enc._write_body_buffers_vec(buf_meta)
        var rb_pos = enc._write_record_batch_table(
            Int64(values.length()),
            nodes_vec,
            bufs_vec,
            benc.variadic_counts,
            self.compression,
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
# Batch body decoder: reconstructs DynArray from raw bytes + cursor state
# ---------------------------------------------------------------------------


struct _BatchDecoder(Movable):
    """Reconstructs DynArray values from a record batch body using node/buffer cursors.
    """

    var body: List[UInt8]
    var body_offset: Int
    var nodes: List[_FieldNode]
    var bufs: List[_BodyBuffer]
    var variadic_counts: List[Int64]
    var node_idx: Int
    var buf_idx: Int
    var variadic_idx: Int
    var compression: Optional[BodyCompression]
    var native_codecs: Bool
    var dict_values: List[DynArray]

    def __init__(
        out self,
        var body: List[UInt8],
        body_offset: Int,
        var nodes: List[_FieldNode],
        var bufs: List[_BodyBuffer],
        var variadic_counts: List[Int64],
        compression: Optional[BodyCompression],
        native_codecs: Bool,
        dict_values: List[DynArray] = List[DynArray](),
    ):
        self.body = body^
        self.body_offset = body_offset
        self.nodes = nodes^
        self.bufs = bufs^
        self.variadic_counts = variadic_counts^
        self.node_idx = 0
        self.buf_idx = 0
        self.variadic_idx = 0
        self.compression = compression
        self.native_codecs = native_codecs
        self.dict_values = dict_values.copy()

    def read_array(
        mut self, dtype: dt.DynType, ipc_info: _FieldIpcInfo
    ) raises -> DynArray:
        if self.node_idx >= len(self.nodes):
            raise CorruptError(
                t"ipc: the batch has {len(self.nodes)} field nodes, fewer than"
                t" its schema needs"
            )
        var node = self.nodes[self.node_idx]
        self.node_idx += 1

        var length = Int(node.length)
        var null_count = Int(node.null_count)

        # Null type: FieldNode is consumed but no body buffers — neither validity
        # nor data — per Arrow spec.
        if dtype.is_null():
            if length < 0:
                raise CorruptError(t"ipc: a null column of length {length}")
            return NullArray(length)

        var validity_buf = self._next_buffer()

        var bitmap: Optional[Bitmap[mut=False]] = None
        if null_count > 0 and validity_buf.length > 0:
            var off = Int(validity_buf.offset) + self.body_offset
            var n_bytes = Int(validity_buf.length)
            bitmap = Bitmap[mut=False](
                self._body_buffer(off, n_bytes), length=length
            )

        var layout = dtype.layout()
        var data_buffers = List[Buffer[mut=False]]()
        for _ in range(layout.num_buffers()):
            self._consume_buffer(data_buffers)
        var children = List[ArrayData]()

        # Dictionary: the indices are read; the values come from dict_values.
        # The dict_id comes from ipc_info (not the logical type) so that the type
        # system remains free of IPC metadata.
        if dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var dict_id = ipc_info.dict_id
            if dict_id < 0 or dict_id >= len(self.dict_values):
                raise CorruptError(
                    t"_BatchDecoder: no values for dict_id {dict_id}"
                )
            var values = self.dict_values[dict_id].copy()
            # Fields sharing an id may declare different value types.
            if values.dtype() != d.value_type():
                raise CorruptError(
                    t"_BatchDecoder: dictionary {dict_id} holds"
                    t" {values.dtype()} values, the field declares"
                    t" {d.value_type()}"
                )
            children.append(values.to_data())
        else:
            if layout.kind == dt.ArrayLayout.VIEW:
                # The data buffers after the views are counted per node, in
                # `variadicBufferCounts`.
                if self.variadic_idx >= len(self.variadic_counts):
                    raise CorruptError(
                        t"_BatchDecoder: {dtype} column without a variadic"
                        t" count"
                    )
                var n_data = Int(self.variadic_counts[self.variadic_idx])
                self.variadic_idx += 1
                for _ in range(n_data):
                    self._consume_buffer(data_buffers)
            for i in range(layout.num_children):
                var child_ipc = (
                    ipc_info.children[i].copy() if i
                    < len(ipc_info.children) else _FieldIpcInfo()
                )
                children.append(
                    self.read_array(dtype.child_type(i), child_ipc).to_data()
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
        # Each child was validated when it was read.
        try:
            ad.validate_node(full=True)
        except e:
            raise CorruptError(t"ipc: {e.message()}")
        return DynArray.from_data(ad)

    def _body_buffer(
        self, off: Int, n_bytes: Int
    ) raises DynError -> Buffer[mut=False]:
        """The `n_bytes` of the body at `off`, decompressed if the batch is."""
        if off < 0 or n_bytes < 0 or n_bytes > len(self.body) - off:
            raise CorruptError(
                t"ipc: a {n_bytes}-byte buffer at {off} is outside the"
                t" {len(self.body)}-byte body"
            )
        var src = Span(self.body)[off : off + n_bytes]
        if self.compression:
            return self.compression.value().decompress(src, self.native_codecs)
        var buf = Buffer.alloc_uninit[DType.uint8](n_bytes)
        buf.view(0, n_bytes).copy_from(BufferView(src), n_bytes)
        return buf^.to_immutable()

    def _next_buffer(mut self) raises CorruptError -> _BodyBuffer:
        """The next buffer the message describes; raises when it has run out."""
        if self.buf_idx >= len(self.bufs):
            raise CorruptError(
                t"ipc: the batch has {len(self.bufs)} buffers, fewer than its"
                t" schema needs"
            )
        var bb = self.bufs[self.buf_idx]
        self.buf_idx += 1
        return bb

    def _consume_buffer(mut self, mut out: List[Buffer[mut=False]]) raises:
        var bb = self._next_buffer()
        out.append(
            self._body_buffer(Int(bb.offset) + self.body_offset, Int(bb.length))
        )


# ---------------------------------------------------------------------------
# Public: incremental file and stream writers
# ---------------------------------------------------------------------------


struct RecordBatchFileWriter[S: ByteSink = FileSink](Movable):
    """Incremental writer for the Arrow IPC file format.

    Write batches with `write_batch`, then call `close()` to append the footer
    and commit. Nothing is published until `close()` succeeds.

    Batches are handed to the sink as they are written, so peak residency is
    one message rather than the whole file. That is safe here and is not in the
    Parquet writer: the only absolute offsets this format records are the
    `_Block` positions below, which come from `tell()`, while an encoded
    message is self-contained.
    """

    var _out: BufferedSink[Self.S]
    var _schema: Schema
    var _dict_blocks: List[_Block]
    var _blocks: List[_Block]
    var _enc: _BatchEncoder
    var _dicts: List[DynArray]
    var _closed: Bool

    def __init__(
        out self: RecordBatchFileWriter[FileSink],
        path: String,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """Write to a local file — the convenience that pins `S == FileSink`."""
        self = RecordBatchFileWriter[FileSink](
            FileSink(path), schema, compression, native_codecs
        )

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """`compression` compresses every body buffer, dictionaries'
        included -- in Mojo, or with `native_codecs=False` through liblz4
        and libzstd."""
        self._out = BufferedSink(sink^)
        self._schema = Schema(copy=schema)
        self._dict_blocks = List[_Block]()
        self._blocks = List[_Block]()
        self._enc = _BatchEncoder(compression, native_codecs)
        self._dicts = []
        self._closed = False

        for b in _magic():
            self._out.buffer().append(b)
        var schema_msg = _IpcEncoder.frame_message(
            _IpcEncoder.encode_schema_message(self._schema), List[UInt8]()
        )
        self._out.write(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise InvalidError("RecordBatchFileWriter: writer is closed")
        # Collect all (dict_id, values) pairs in DFS inner-first order; ids
        # are dense, so `_dicts[id]` is what was written for an id. The FILE
        # format holds one dictionary per id and this writer emits no deltas,
        # so a later batch must carry the same dictionary.
        var pairs = List[_DictPair]()
        var next_id = 0
        for col in batch.columns:
            _BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for ref pair in pairs:
            var did = pair.dict_id
            if did < len(self._dicts):
                if not KeyCompare.equals(self._dicts[did], pair.values):
                    raise InvalidError(
                        t"RecordBatchFileWriter: dictionary {did} differs from"
                        t" the one already written"
                    )
                continue
            var dict_blk_start = Int64(self._out.tell())
            var eb = self._enc.encode_dict_message(Int64(did), pair.values)
            self._out.write(Span(eb.msg))
            self._dict_blocks.append(
                _Block(dict_blk_start, eb.metadata_length, eb.body_length)
            )
            self._dicts.append(pair.values.copy())
        var blk_start = Int64(self._out.tell())
        var eb = self._enc.encode(batch)
        self._out.write(Span(eb.msg))
        self._blocks.append(
            _Block(blk_start, eb.metadata_length, eb.body_length)
        )
        # One message resident at a time. `tell()` stays absolute across this,
        # which is what keeps the `_Block` offsets above meaningful.
        self._out.flush()

    def close(mut self) raises:
        if self._closed:
            return
        self._out.pad_to(8)
        var footer_bytes = _IpcEncoder.encode_footer(
            self._schema, self._dict_blocks, self._blocks
        )
        self._out.write(Span(footer_bytes))
        LittleEndian.append[DType.int32](
            self._out.buffer(), Int32(len(footer_bytes))
        )
        var magic = _magic()
        for i in range(6):
            self._out.buffer().append(magic[i])
        self._out.close()
        self._closed = True


struct RecordBatchStreamWriter[S: ByteSink = FileSink](Movable):
    """Incremental writer for the Arrow IPC stream format.

    Write batches with `write_batch`, then call `close()` to write the EOS
    marker and commit. A stream records no offsets at all, so each message goes
    to the sink as soon as it is encoded.
    """

    var _out: BufferedSink[Self.S]
    var _enc: _BatchEncoder
    var _closed: Bool

    def __init__(
        out self: RecordBatchStreamWriter[FileSink],
        path: String,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """Write to a local file — the convenience that pins `S == FileSink`."""
        self = RecordBatchStreamWriter[FileSink](
            FileSink(path), schema, compression, native_codecs
        )

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """`compression` compresses every body buffer, dictionaries'
        included -- in Mojo, or with `native_codecs=False` through liblz4
        and libzstd."""
        self._out = BufferedSink(sink^)
        self._enc = _BatchEncoder(compression, native_codecs)
        self._closed = False

        var schema_msg = _IpcEncoder.frame_message(
            _IpcEncoder.encode_schema_message(schema),
            List[UInt8](),
        )
        self._out.write(Span(schema_msg))

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise InvalidError("RecordBatchStreamWriter: writer is closed")
        # Stream format sends all dicts before each record batch.
        var pairs = List[_DictPair]()
        var next_id = 0
        for col in batch.columns:
            _BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for j in range(len(pairs)):
            var eb = self._enc.encode_dict_message(
                Int64(pairs[j].dict_id), pairs[j].values
            )
            self._out.write(Span(eb.msg))
        self._out.write(Span(self._enc.encode(batch).msg))
        self._out.flush()

    def close(mut self) raises:
        if self._closed:
            return
        LittleEndian.append[DType.uint32](
            self._out.buffer(), UInt32(0xFFFFFFFF)
        )
        LittleEndian.append[DType.int32](self._out.buffer(), Int32(0))
        self._out.close()
        self._closed = True


# ---------------------------------------------------------------------------
# Public: file and stream readers
# ---------------------------------------------------------------------------


struct RecordBatchFileReader[S: ByteSource = BufferSource](Movable):
    """Reader for the Arrow IPC file format with random-access batch reads.

    Opens a file by reading its **tail**, not its whole extent: the footer sits
    at the end, so one speculative `FOOTER_READ_SIZE` read finds it, and the
    read is repeated only when the footer overruns that window.
    """

    var schema: Schema
    var _ipc_infos: List[_FieldIpcInfo]
    var _blocks: List[_Block]
    var _src: Self.S
    var _dict_values: List[DynArray]
    var _native_codecs: Bool
    """Whether compressed bodies decode in Mojo, the default, or through
    liblz4 and libzstd."""

    def __init__(
        out self: RecordBatchFileReader[BufferSource],
        path: String,
        native_codecs: Bool = True,
    ) raises:
        """Open a local file as a memory map — the convenience that pins
        `S == BufferSource`."""
        self = RecordBatchFileReader[BufferSource](
            BufferSource(path), native_codecs
        )

    def __init__(
        out self, var source: Self.S, native_codecs: Bool = True
    ) raises:
        self._native_codecs = native_codecs
        var n = source.size()
        if n < 14:
            raise CorruptError("IPC file too short")
        var magic = _magic()

        var head = source.read_at(0, 8)
        for i in range(8):
            if head[i] != magic[i]:
                raise CorruptError("IPC file: bad magic bytes")

        # One tail read serves the trailing magic, the footer length and, all
        # but always, the footer itself.
        var want = min(FOOTER_READ_SIZE, n)
        var tail = source.read_at(n - want, want)
        for i in range(6):
            if tail[want - 6 + i] != magic[i]:
                raise CorruptError("IPC file: bad trailing magic")

        var footer_size = Int(
            LittleEndian.checked[DType.int32](tail, want - 10)
        )
        if footer_size < 0 or footer_size + 10 > n:
            raise CorruptError(t"IPC file: bad footer length {footer_size}")

        var footer_bytes: List[UInt8]
        if footer_size + 10 <= want:
            var at = want - 10 - footer_size
            footer_bytes = List[UInt8](tail[at : at + footer_size])
        else:
            # The speculative window was too small; ask for exactly the footer.
            var start = n - 10 - footer_size
            footer_bytes = List[UInt8](source.read_at(start, footer_size))

        var dec = _IpcDecoder(footer_bytes^)
        var dict_blocks = List[_Block]()
        var blocks = List[_Block]()
        var ipc_infos = List[_FieldIpcInfo]()
        self.schema = dec.read_footer(dict_blocks, blocks, ipc_infos)
        self._ipc_infos = ipc_infos^
        self._blocks = blocks^
        self._src = source^
        self._dict_values = List[DynArray]()

        # Load dictionary values from their footer-registered blocks.
        # dict_values is indexed by dict_id; pass partial list to decode_dict_batch
        # so that nested dicts (already loaded at lower ids) can be resolved.
        for di in range(len(dict_blocks)):
            var pos = Int(dict_blocks[di].offset)
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not _read_message(self._src, pos, meta, body):
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
                    self._native_codecs,
                )
                while len(self._dict_values) <= dict_id:
                    self._dict_values.append(NullArray(0))
                self._dict_values[dict_id] = values^

    def num_record_batches(self) -> Int:
        return len(self._blocks)

    def read_batch(ref self, i: Int) raises -> RecordBatch:
        """One batch, by index. `ref self` rather than `mut self`: this format is
        random-access -- the footer's `_Block` table holds every offset -- so the
        cursor is a local, and several readers of one file do not contend."""
        if i < 0 or i >= len(self._blocks):
            raise IndexError("RecordBatchFileReader: batch index out of range")
        var pos = Int(self._blocks[i].offset)
        var meta = List[UInt8]()
        var body = List[UInt8]()
        var _ok = _read_message(self._src, pos, meta, body)
        var dec = _IpcDecoder(meta^)
        return dec.decode_record_batch(
            self.schema,
            self._ipc_infos,
            body^,
            self._dict_values,
            self._native_codecs,
        )

    def read_all(mut self) raises -> List[RecordBatch]:
        # Footer only lists record-batch blocks, so no header check needed.
        var batches = List[RecordBatch]()
        for i in range(len(self._blocks)):
            batches.append(self.read_batch(i))
        return batches^


struct RecordBatchStreamReader[S: ByteSource = BufferSource](Movable):
    """Reader for the Arrow IPC stream format.

    Framing stays strictly sequential, because a stream carries no index. That
    is honest but not ideal over an object store, which wants a prefetch window
    rather than a round trip per message; the seam admits such a source, and
    this reader does not add one.
    """

    var schema: Schema
    var _ipc_infos: List[_FieldIpcInfo]
    var _src: Self.S
    var _pos: Int
    var _native_codecs: Bool
    """Whether compressed bodies decode in Mojo, the default, or through
    liblz4 and libzstd."""

    def __init__(
        out self: RecordBatchStreamReader[BufferSource],
        path: String,
        native_codecs: Bool = True,
    ) raises:
        """Open a local file as a memory map — the convenience that pins
        `S == BufferSource`."""
        self = RecordBatchStreamReader[BufferSource](
            BufferSource(path), native_codecs
        )

    def __init__(
        out self, var source: Self.S, native_codecs: Bool = True
    ) raises:
        self._native_codecs = native_codecs
        self._src = source^
        self._pos = 0
        var meta = List[UInt8]()
        var body = List[UInt8]()
        if not _read_message(self._src, self._pos, meta, body):
            raise CorruptError(
                "RecordBatchStreamReader: missing schema message"
            )
        var ipc_infos = List[_FieldIpcInfo]()
        var dec = _IpcDecoder(meta^)
        self.schema = dec.decode_schema(ipc_infos)
        self._ipc_infos = ipc_infos^

    def read_all(mut self) raises -> List[RecordBatch]:
        var dict_values = List[DynArray]()
        var batches = List[RecordBatch]()
        while True:
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not _read_message(self._src, self._pos, meta, body):
                break
            var header_type: UInt8
            var peek = _IpcDecoder(meta.copy())
            header_type = peek.peek_header()
            if Int(header_type) == Int(_HEADER_DICTIONARY_BATCH):
                var dict_id = _IpcDecoder(meta.copy()).peek_dict_id()
                var is_delta = _IpcDecoder(meta.copy()).peek_is_delta()
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
                        self._native_codecs,
                    )
                    while len(dict_values) <= dict_id:
                        dict_values.append(NullArray(0))
                    if is_delta and len(dict_values[dict_id]) > 0:
                        # a delta carries only the newly-appended values
                        var merged = List[DynArray]()
                        merged.append(dict_values[dict_id].copy())
                        merged.append(values^)
                        dict_values[dict_id] = _concat(merged)
                    else:
                        dict_values[dict_id] = values^
            elif Int(header_type) == Int(_HEADER_RECORD_BATCH):
                var dec = _IpcDecoder(meta^)
                batches.append(
                    dec.decode_record_batch(
                        self.schema,
                        self._ipc_infos,
                        body^,
                        dict_values,
                        self._native_codecs,
                    )
                )
        return batches^


# ---------------------------------------------------------------------------
# Public top-level functions
# ---------------------------------------------------------------------------


def write_ipc_file(
    uri: String,
    schema: Schema,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC file with an explicit schema;
    `compression` runs in Mojo, or with `native_codecs=False` through liblz4
    and libzstd."""
    var w = RecordBatchFileWriter(
        DynSink.open(uri, options), schema, compression, native_codecs
    )
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_file(
    uri: String,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC file."""
    if len(batches) == 0:
        raise InvalidError(
            "write_ipc_file: no batches; use write_ipc_file(path, schema, "
            "batches) for schema-only files"
        )
    write_ipc_file(
        uri, batches[0].schema, batches, options, compression, native_codecs
    )


def write_ipc_stream(
    uri: String,
    schema: Schema,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC stream with an explicit schema;
    `compression` runs as `write_ipc_file` says."""
    var w = RecordBatchStreamWriter(
        DynSink.open(uri, options), schema, compression, native_codecs
    )
    for batch in batches:
        w.write_batch(batch)
    w.close()


def write_ipc_stream(
    uri: String,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC stream."""
    if len(batches) == 0:
        raise InvalidError(
            "write_ipc_stream: no batches; use write_ipc_stream(path, schema, "
            "batches) for schema-only streams"
        )
    write_ipc_stream(
        uri, batches[0].schema, batches, options, compression, native_codecs
    )


def read_ipc_file(
    uri: String,
    options: StorageOptions = StorageOptions(),
    native_codecs: Bool = True,
) raises -> List[RecordBatch]:
    """Read an Arrow IPC file and return all RecordBatches; compressed
    bodies decode in Mojo, or with `native_codecs=False` through liblz4 and
    libzstd."""
    var r = RecordBatchFileReader(DynSource.open(uri, options), native_codecs)
    return r.read_all()


def read_ipc_stream(
    uri: String,
    options: StorageOptions = StorageOptions(),
    native_codecs: Bool = True,
) raises -> List[RecordBatch]:
    """Read an Arrow IPC stream and return all RecordBatches; compressed
    bodies decode as `read_ipc_file` says."""
    var r = RecordBatchStreamReader(DynSource.open(uri, options), native_codecs)
    return r.read_all()


def read_ipc_file_schema(
    uri: String, options: StorageOptions = StorageOptions()
) raises -> RecordBatch:
    """Read the schema from an Arrow IPC file; return a 0-row RecordBatch."""
    var r = RecordBatchFileReader(DynSource.open(uri, options))
    return RecordBatch.empty(r.schema)


def read_ipc_stream_schema(
    uri: String, options: StorageOptions = StorageOptions()
) raises -> RecordBatch:
    """Read the schema from an Arrow IPC stream; return a 0-row RecordBatch."""
    var r = RecordBatchStreamReader(DynSource.open(uri, options))
    return RecordBatch.empty(r.schema)
