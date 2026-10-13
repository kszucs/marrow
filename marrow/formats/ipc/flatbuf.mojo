# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""A minimal FlatBuffers builder and reader, the encoding every IPC
metadata message uses."""

from ...errors import CorruptError, InternalError, InvalidError
from ...utils.byteorder import LittleEndian
from .format import padding_to


# ---------------------------------------------------------------------------
# Generic FlatBuffers codec
# ---------------------------------------------------------------------------


@fieldwise_init
struct FieldOffset(Copyable, Movable):
    """Slot index and tail-distance recorded after prepending a field."""

    var slot: Int
    var at: UInt32


struct FlatbufWriter(Movable):
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
        var pad = padding_to(written + needed, align)
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
        fields: List[FieldOffset],
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


struct FlatbufReader(Movable):
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
