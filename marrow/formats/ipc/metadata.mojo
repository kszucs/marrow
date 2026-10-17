# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""IPC metadata: schemas, record batch and dictionary messages, and the
file footer, encoded to and decoded from FlatBuffers."""

from ...errors import CorruptError, NotImplementedError
from ...arrays import DynArray
from ...schema import Schema
from ...tabular import RecordBatch
from ...utils.byteorder import LittleEndian
from ... import dtypes as dt
from .body import BatchDecoder
from .compression import BodyCompression
from .flatbuf import FieldOffset, FlatbufReader, FlatbufWriter
from .format import (
    Block,
    BodyBuffer,
    DATE_UNIT_DAY,
    DATE_UNIT_MILLISECOND,
    ENDIANNESS_LITTLE,
    FieldIpcInfo,
    FieldNode,
    HEADER_RECORD_BATCH,
    HEADER_SCHEMA,
    INTERVAL_UNIT_DAY_TIME,
    INTERVAL_UNIT_MONTH_DAY_NANO,
    INTERVAL_UNIT_YEAR_MONTH,
    MAX_NESTING_DEPTH,
    METADATA_VERSION_V5,
    PRECISION_DOUBLE,
    PRECISION_HALF,
    PRECISION_SINGLE,
    TIME_UNIT_MILLISECOND,
    TIME_UNIT_SECOND,
    TYPE_BINARY,
    TYPE_BINARY_VIEW,
    TYPE_BOOL,
    TYPE_DATE,
    TYPE_DECIMAL,
    TYPE_DURATION,
    TYPE_FIXED_SIZE_BINARY,
    TYPE_FIXED_SIZE_LIST,
    TYPE_FLOATING_POINT,
    TYPE_INT,
    TYPE_INTERVAL,
    TYPE_LARGE_BINARY,
    TYPE_LARGE_LIST,
    TYPE_LARGE_UTF8,
    TYPE_LIST,
    TYPE_MAP,
    TYPE_NULL,
    TYPE_STRUCT,
    TYPE_TIME,
    TYPE_TIMESTAMP,
    TYPE_UTF8,
    TYPE_UTF8_VIEW,
    pad_to,
)


# ---------------------------------------------------------------------------
# Arrow IPC metadata encoder
# ---------------------------------------------------------------------------


struct IpcEncoder(Movable):
    """Encodes Arrow IPC metadata (schema, record batch, footer) into FlatBuffers.
    """

    var _fb: FlatbufWriter

    def __init__(out self, capacity: Int = 512):
        self._fb = FlatbufWriter(capacity)

    @staticmethod
    def encode_schema(schema: Schema) raises -> List[UInt8]:
        var enc = IpcEncoder(256)
        var schema_pos = enc._write_schema_table(schema)
        return enc._finish(schema_pos)

    @staticmethod
    def encode_schema_message(schema: Schema) raises -> List[UInt8]:
        var enc = IpcEncoder(512)
        var schema_pos = enc._write_schema_table(schema)
        var msg_pos = enc._write_message_table(
            HEADER_SCHEMA, schema_pos, Int64(0)
        )
        return enc._finish(msg_pos)

    @staticmethod
    def encode_record_batch(
        length: Int64,
        nodes: List[FieldNode],
        buffers: List[BodyBuffer],
        variadic_counts: List[Int64],
        compression: Optional[BodyCompression],
    ) raises -> List[UInt8]:
        var enc = IpcEncoder(512)
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
            HEADER_RECORD_BATCH, rb_pos, body_len
        )
        return enc._finish(msg_pos)

    @staticmethod
    def encode_footer(
        schema: Schema,
        dict_blocks: List[Block],
        blocks: List[Block],
    ) raises -> List[UInt8]:
        var enc = IpcEncoder(512)
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
        pad_to(out, 8)
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
            var bflds = List[FieldOffset]()
            bflds.append(FieldOffset(0, codec_at))
            bflds.append(FieldOffset(1, method_at))
            bc_pos = self._fb.write_table(bflds, bts)
        var ts = self._fb.offset()
        var flds = List[FieldOffset]()
        var vc_at = UInt32(0)
        if len(variadic_counts) > 0:
            vc_at = self._fb.prepend_uoffset(vc_vec)
        if compression:
            flds.append(FieldOffset(3, self._fb.prepend_uoffset(bc_pos)))
        var bv_at = self._fb.prepend_uoffset(bufs_vec)
        var nv_at = self._fb.prepend_uoffset(nodes_vec)
        var ln_at = self._fb.prepend_i64(length)
        flds.append(FieldOffset(0, ln_at))
        flds.append(FieldOffset(1, nv_at))
        flds.append(FieldOffset(2, bv_at))
        if len(variadic_counts) > 0:
            flds.append(FieldOffset(4, vc_at))
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
        var ver_at = self._fb.prepend_i16(METADATA_VERSION_V5)
        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, ver_at))
        flds.append(FieldOffset(1, sc_at))
        flds.append(FieldOffset(2, dv_at))
        flds.append(FieldOffset(3, bv_at))
        return self._fb.write_table(flds, ts)

    def _write_field_nodes_vec(
        mut self, nodes: List[FieldNode]
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
        mut self, bufs: List[BodyBuffer]
    ) raises -> UInt32:
        var n = len(bufs)
        var data = List[UInt8](capacity=n * 16)
        for _ in range(n * 16):
            data.append(UInt8(0))
        for i in range(n):
            LittleEndian.write[DType.int64](data, i * 16, bufs[i].offset)
            LittleEndian.write[DType.int64](data, i * 16 + 8, bufs[i].length)
        return self._fb.create_vector_structs(data, n, 16, 8)

    def _write_blocks_vec(mut self, blocks: List[Block]) raises -> UInt32:
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
            return TYPE_NULL
        elif dtype.is_bool():
            return TYPE_BOOL
        elif dtype.is_integer():
            return TYPE_INT
        elif dtype.is_floating_point():
            return TYPE_FLOATING_POINT
        elif dtype.is_binary():
            return TYPE_BINARY
        elif dtype.is_large_binary():
            return TYPE_LARGE_BINARY
        elif dtype.is_string():
            return TYPE_UTF8
        elif dtype.is_large_string():
            return TYPE_LARGE_UTF8
        elif dtype.is_binary_view():
            return TYPE_BINARY_VIEW
        elif dtype.is_string_view():
            return TYPE_UTF8_VIEW
        elif dtype.is_map():
            # Before `is_list()`: a map is a list of entry structs, and if
            # `is_list()` answered first a map would be written as a plain list
            # and read back as one.
            return TYPE_MAP
        elif dtype.is_list():
            return TYPE_LIST
        elif dtype.is_large_list():
            return TYPE_LARGE_LIST
        elif dtype.is_fixed_size_list():
            return TYPE_FIXED_SIZE_LIST
        elif dtype.is_fixed_size_binary():
            return TYPE_FIXED_SIZE_BINARY
        elif dtype.is_date32() or dtype.is_date64():
            return TYPE_DATE
        elif dtype.is_time32() or dtype.is_time64():
            return TYPE_TIME
        elif dtype.is_timestamp():
            return TYPE_TIMESTAMP
        elif dtype.is_duration():
            return TYPE_DURATION
        elif dtype.is_interval():
            return TYPE_INTERVAL
        elif dtype.is_decimal():
            return TYPE_DECIMAL
        elif dtype.is_struct():
            return TYPE_STRUCT
        elif dtype.is_dictionary():
            # Schema encodes the value type; DictionaryEncoding carries index type.
            return self._type_code(dtype.as_dictionary().value_type())
        else:
            raise NotImplementedError(t"IpcEncoder: unsupported dtype: {dtype}")

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
            return self._fb.write_table(List[FieldOffset](), ts)
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
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, bw_at))
            flds.append(FieldOffset(1, signed_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_floating_point():
            var prec: UInt16
            if dtype == dt.float16:
                prec = PRECISION_HALF
            elif dtype == dt.float32:
                prec = PRECISION_SINGLE
            else:
                prec = PRECISION_DOUBLE
            var ts = self._fb.offset()
            var prec_at = self._fb.prepend_u16(prec)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, prec_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_map():
            ref m = dtype.as_map()
            var ts = self._fb.offset()
            var ks_at = self._fb.prepend_bool(m.keys_sorted)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, ks_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_fixed_size_list():
            ref fsl = dtype.as_fixed_size_list()
            var ts = self._fb.offset()
            var sz_at = self._fb.prepend_i32(Int32(fsl.size))
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, sz_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_fixed_size_binary():
            ref fsb = dtype.as_fixed_size_binary()
            var ts = self._fb.offset()
            var bw_at = self._fb.prepend_i32(Int32(fsb.byte_width))
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, bw_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_date32():
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(DATE_UNIT_DAY)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_date64():
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(DATE_UNIT_MILLISECOND)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
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
            var ipc_unit = UInt16(unit.value)
            var ts = self._fb.offset()
            var bw_at = self._fb.prepend_i32(bw)
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
            flds.append(FieldOffset(1, bw_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_timestamp():
            ref tstype = dtype.as_timestamp()
            var ipc_unit = UInt16(tstype.unit.value)
            var ts = self._fb.offset()
            var tz_at: Optional[UInt32] = None
            if tstype.timezone:
                var tz_str_pos = self._fb.create_string(tstype.timezone)
                tz_at = self._fb.prepend_uoffset(tz_str_pos)
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
            if tz_at:
                flds.append(FieldOffset(1, tz_at.value()))
            return self._fb.write_table(flds, ts)
        elif dtype.is_duration():
            var unit = dtype.as_duration().unit
            var ipc_unit = UInt16(unit.value)
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(ipc_unit)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_interval():
            var unit: UInt16
            if dtype.is_year_month_interval():
                unit = INTERVAL_UNIT_YEAR_MONTH
            elif dtype.is_day_time_interval():
                unit = INTERVAL_UNIT_DAY_TIME
            else:
                unit = INTERVAL_UNIT_MONTH_DAY_NANO
            var ts = self._fb.offset()
            var u_at = self._fb.prepend_u16(unit)
            var flds = List[FieldOffset]()
            flds.append(FieldOffset(0, u_at))
            return self._fb.write_table(flds, ts)
        elif dtype.is_decimal():

            def decimal[T: dt.DecimalType](d: T) raises {mut self} -> UInt32:
                var ts = self._fb.offset()
                var bw_at = self._fb.prepend_i32(Int32(d.bit_width()))
                var scale_at = self._fb.prepend_i32(Int32(d.scale()))
                var prec_at = self._fb.prepend_i32(Int32(d.precision()))
                var flds = List[FieldOffset]()
                flds.append(FieldOffset(0, prec_at))
                flds.append(FieldOffset(1, scale_at))
                flds.append(FieldOffset(2, bw_at))
                return self._fb.write_table(flds, ts)

            return dtype.try_visit[UInt32, dt.DecimalType](
                decimal, "dtype is not decimal"
            )
        elif dtype.is_dictionary():
            return self._write_type_table(dtype.as_dictionary().value_type())
        else:
            raise NotImplementedError(
                t"IpcEncoder: unsupported dtype for type table: {dtype}"
            )

    def _write_dictionary_encoding_table(
        mut self, dict_id: Int64, index_dtype: dt.DynType, ordered: Bool
    ) raises -> UInt32:
        var idx_type_pos = self._write_type_table(index_dtype)
        var ts = self._fb.offset()
        var ord_at = self._fb.prepend_bool(ordered)
        var idx_at = self._fb.prepend_uoffset(idx_type_pos)
        var id_at = self._fb.prepend_i64(dict_id)
        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, id_at))
        flds.append(FieldOffset(1, idx_at))
        flds.append(FieldOffset(2, ord_at))
        return self._fb.write_table(flds, ts)

    def _write_dictionary_batch_table(
        mut self, dict_id: Int64, is_delta: Bool, rb_pos: UInt32
    ) raises -> UInt32:
        var ts = self._fb.offset()
        var delta_at = self._fb.prepend_bool(is_delta)
        var data_at = self._fb.prepend_uoffset(rb_pos)
        var id_at = self._fb.prepend_i64(dict_id)
        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, id_at))
        flds.append(FieldOffset(1, data_at))
        flds.append(FieldOffset(2, delta_at))
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
            var kv_flds = List[FieldOffset]()
            kv_flds.append(FieldOffset(0, key_at))
            kv_flds.append(FieldOffset(1, val_at))
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

        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, nm_at))
        flds.append(FieldOffset(1, nb_at))
        flds.append(FieldOffset(2, tc_at))
        flds.append(FieldOffset(3, tp_at))
        if dict_enc_pos:
            flds.append(FieldOffset(4, de_at))
        if children_vec_pos:
            flds.append(FieldOffset(5, ch_at))
        if meta_vec_pos:
            flds.append(FieldOffset(6, meta_at))
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
        var en_at = self._fb.prepend_i16(ENDIANNESS_LITTLE)
        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, en_at))
        flds.append(FieldOffset(1, fv_at))
        if meta_vec_pos:
            flds.append(FieldOffset(2, meta_at))
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
        var ver_at = self._fb.prepend_i16(METADATA_VERSION_V5)
        var flds = List[FieldOffset]()
        flds.append(FieldOffset(0, ver_at))
        flds.append(FieldOffset(1, ht_at))
        flds.append(FieldOffset(2, hdr_at))
        flds.append(FieldOffset(3, bl_at))
        return self._fb.write_table(flds, ts)


# ---------------------------------------------------------------------------
# Arrow IPC metadata decoder
# ---------------------------------------------------------------------------


struct IpcDecoder(Movable):
    """Decodes Arrow IPC metadata (schema, record batch, footer) from FlatBuffers.
    """

    var _r: FlatbufReader

    def __init__(out self, var buf: List[UInt8]):
        self._r = FlatbufReader(buf^)

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
    def _wire_to_time_unit(v: UInt16) raises CorruptError -> dt.TimeUnit:
        # The flatbuffer `TimeUnit` enum numbers the units as Arrow does.
        var unit = dt.TimeUnit.from_value(Int(v))
        if not unit:
            raise CorruptError(t"ipc: unknown time unit {v}")
        return unit.value()

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

    def decode_schema(self, mut out_ipc: List[FieldIpcInfo]) raises -> Schema:
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
        values_ipc_info: FieldIpcInfo,
        var body: List[UInt8],
        dict_values: List[DynArray] = List[DynArray](),
        native_codecs: Bool = True,
    ) raises -> DynArray:
        var msg_tp = self._r.root()
        var db_pos = self._r.read_table(msg_tp, 2)
        var rb_pos = self._r.read_table(db_pos, 1)
        var nodes = List[FieldNode]()
        var bufs = List[BodyBuffer]()
        var variadic = List[Int64]()
        var compression = self._read_record_batch_meta(
            rb_pos, nodes, bufs, variadic
        )
        var batch_dec = BatchDecoder(
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
        ipc_infos: List[FieldIpcInfo],
        var body: List[UInt8],
        dict_values: List[DynArray] = List[DynArray](),
        native_codecs: Bool = True,
    ) raises -> RecordBatch:
        var msg_tp = self._r.root()
        var hdr_type = self._r.read_u8(msg_tp, 1, 0)
        if Int(hdr_type) != Int(HEADER_RECORD_BATCH):
            raise CorruptError(
                t"IpcDecoder: expected record-batch header, got {Int(hdr_type)}"
            )
        var rb_pos = self._r.read_table(msg_tp, 2)
        var nodes = List[FieldNode]()
        var bufs = List[BodyBuffer]()
        var variadic = List[Int64]()
        var compression = self._read_record_batch_meta(
            rb_pos, nodes, bufs, variadic
        )
        var batch_dec = BatchDecoder(
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
                ipc_infos[i].copy() if i < len(ipc_infos) else FieldIpcInfo()
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
        mut dict_blocks: List[Block],
        mut blocks: List[Block],
        mut out_ipc: List[FieldIpcInfo],
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
                Block(
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
                Block(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int32](sb, 8),
                    LittleEndian.checked[DType.int64](sb, 16),
                )
            )
        return Schema(fields=fields^, metadata=metadata^)

    def _decode_schema_fields(
        self, schema_pos: UInt32, mut out_ipc: List[FieldIpcInfo]
    ) raises -> List[dt.Field]:
        # Field 0 is the endianness of every buffer that follows; reading
        # big-endian data as little-endian would answer wrong values silently.
        if self._r.read_u16(schema_pos, 0) != UInt16(ENDIANNESS_LITTLE):
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
        mut nodes: List[FieldNode],
        mut bufs: List[BodyBuffer],
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
                FieldNode(
                    LittleEndian.checked[DType.int64](sb, 0),
                    LittleEndian.checked[DType.int64](sb, 8),
                )
            )

        var bufs_vec = self._r.read_vector(rb_pos, 2)
        var nb = Int(self._r.vector_len(bufs_vec))
        for i in range(nb):
            var sb = self._r.vec_struct_bytes(bufs_vec, UInt32(i), 16)
            bufs.append(
                BodyBuffer(
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
        mut out_ipc: List[FieldIpcInfo],
        mut budget: Int,
        depth: Int,
    ) raises -> dt.Field:
        # Child offsets may point back at an enclosing table, or several at
        # one table, so both the nesting and the number of fields read are
        # bounded rather than trusted. A tree of fields spends at least one
        # 4-byte offset per field, which is what `budget` counts down.
        if depth >= MAX_NESTING_DEPTH:
            raise CorruptError(
                t"ipc: fields nested deeper than {MAX_NESTING_DEPTH} levels"
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
        var child_ipc = List[FieldIpcInfo]()
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
        if type_type == TYPE_NULL:
            dtype = dt.null
        elif type_type == TYPE_BOOL:
            dtype = dt.bool_
        elif type_type == TYPE_DECIMAL:
            var tp = self._r.read_table(fp, 3)
            var precision = Int(self._r.read_i32(tp, 0, 0))
            var scale = Int(self._r.read_i32(tp, 1, 0))
            var bit_width = Int(self._r.read_i32(tp, 2, 128))
            if bit_width == 32:
                dtype = dt.decimal32(precision, scale)
            elif bit_width == 64:
                dtype = dt.decimal64(precision, scale)
            elif bit_width == 128:
                dtype = dt.decimal128(precision, scale)
            elif bit_width == 256:
                dtype = dt.decimal256(precision, scale)
            else:
                raise CorruptError(t"ipc: no {bit_width}-bit decimal type")
        elif type_type == TYPE_INT:
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
        elif type_type == TYPE_FLOATING_POINT:
            var tp = self._r.read_table(fp, 3)
            var prec = self._r.read_u16(tp, 0, PRECISION_HALF)
            if prec == PRECISION_HALF:
                dtype = dt.float16
            elif prec == PRECISION_SINGLE:
                dtype = dt.float32
            else:
                dtype = dt.float64
        elif type_type == TYPE_BINARY:
            dtype = dt.binary
        elif type_type == TYPE_LARGE_BINARY:
            dtype = dt.large_binary
        elif type_type == TYPE_UTF8:
            dtype = dt.string
        elif type_type == TYPE_LARGE_UTF8:
            dtype = dt.large_string
        elif type_type == TYPE_BINARY_VIEW:
            dtype = dt.binary_view
        elif type_type == TYPE_UTF8_VIEW:
            dtype = dt.string_view
        elif type_type == TYPE_MAP:
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
        elif type_type == TYPE_LIST:
            if len(children) == 0:
                raise CorruptError("list Field must have 1 child, got 0")
            # Preserve the child Field as-is (its name may not be the default
            # "item" — e.g. arrow-rs uses "inner_list" for nested lists).
            dtype = dt.ListType(children[0].copy()).to_dyn()
        elif type_type == TYPE_LARGE_LIST:
            if len(children) == 0:
                raise CorruptError("large_list Field must have 1 child, got 0")
            dtype = dt.LargeListType(children[0].copy()).to_dyn()
        elif type_type == TYPE_FIXED_SIZE_LIST:
            var tp = self._r.read_table(fp, 3)
            var list_size = Int(self._r.read_i32(tp, 0, 0))
            if len(children) == 0:
                raise CorruptError(
                    "fixed_size_list Field must have 1 child, got 0"
                )
            dtype = dt.FixedSizeListType(children[0].copy(), list_size).to_dyn()
        elif type_type == TYPE_FIXED_SIZE_BINARY:
            var tp = self._r.read_table(fp, 3)
            var byte_width = Int(self._r.read_i32(tp, 0, 0))
            dtype = dt.FixedSizeBinaryType(byte_width).to_dyn()
        elif type_type == TYPE_DATE:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, DATE_UNIT_MILLISECOND)
            if unit_v == DATE_UNIT_DAY:
                dtype = dt.date32()
            else:
                dtype = dt.date64()
        elif type_type == TYPE_TIME:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, TIME_UNIT_MILLISECOND)
            var bw = Int(self._r.read_i32(tp, 1, 32))
            var unit = IpcDecoder._wire_to_time_unit(unit_v)
            var wide = unit == dt.microsecond or unit == dt.nanosecond
            if bw != (64 if wide else 32):
                raise CorruptError(t"ipc: a {bw}-bit time in {unit}")
            if wide:
                dtype = dt.time64(unit)
            else:
                dtype = dt.time32(unit)
        elif type_type == TYPE_TIMESTAMP:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, TIME_UNIT_SECOND)
            var tz = self._r.read_string(tp, 1)
            dtype = dt.timestamp(
                IpcDecoder._wire_to_time_unit(unit_v), tz
            ).to_dyn()
        elif type_type == TYPE_DURATION:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, TIME_UNIT_MILLISECOND)
            dtype = dt.duration(IpcDecoder._wire_to_time_unit(unit_v)).to_dyn()
        elif type_type == TYPE_INTERVAL:
            var tp = self._r.read_table(fp, 3)
            var unit_v = self._r.read_u16(tp, 0, INTERVAL_UNIT_YEAR_MONTH)
            if unit_v == INTERVAL_UNIT_DAY_TIME:
                dtype = dt.day_time_interval().to_dyn()
            elif unit_v == INTERVAL_UNIT_MONTH_DAY_NANO:
                dtype = dt.month_day_nano_interval().to_dyn()
            else:
                dtype = dt.year_month_interval().to_dyn()
        elif type_type == TYPE_STRUCT:
            dtype = dt.struct_(children^)
        else:
            raise NotImplementedError(
                t"IpcDecoder: unsupported type_type: {Int(type_type)}"
            )

        # DictionaryEncoding, slot 4, wraps the value type in DictionaryType.
        # The dict_id is stored in FieldIpcInfo rather than on the type itself.
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

        out_ipc.append(FieldIpcInfo(own_dict_id, child_ipc^))
        return dt.Field(name, dtype^, nullable, metadata^)
