# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Record batch bodies: an array tree flattened into field nodes and
buffers, and rebuilt from them."""

from std.math import ceildiv
from ...errors import CorruptError, DynError
from ...arrays import (
    ArrayData,
    BinaryViewArray,
    DynArray,
    NullArray,
    StringViewArray,
)
from ...buffers import Bitmap, Buffer
from ...views import BufferView
from ...tabular import RecordBatch
from ...builders import DynBuilder
from ... import dtypes as dt
from .compression import BodyCompression
from .format import (
    BodyBuffer,
    FieldIpcInfo,
    FieldNode,
    HEADER_DICTIONARY_BATCH,
    pad_to,
)
from .metadata import IpcEncoder


struct DictPair(Copyable, Movable):
    """A (dict_id, values) pair collected from an ArrayData tree."""

    var dict_id: Int
    var values: DynArray

    def __init__(out self, dict_id: Int, var values: DynArray):
        self.dict_id = dict_id
        self.values = values^

    def __init__(out self, *, copy: Self):
        self.dict_id = copy.dict_id
        self.values = copy.values.copy()


# ---------------------------------------------------------------------------
# Batch body encoder: traverses ArrayData trees to collect raw bytes
# ---------------------------------------------------------------------------


@fieldwise_init
struct _EncodedBatch(Movable):
    var msg: List[UInt8]
    var metadata_length: Int32
    var body_length: Int64


struct BatchEncoder(Movable):
    """Traverses an ArrayData tree collecting FieldNode metadata and raw buffer bytes.
    """

    var nodes: List[FieldNode]
    var raw_bufs: List[List[UInt8]]
    var variadic_counts: List[Int64]
    var compression: Optional[BodyCompression]
    var native_codecs: Bool

    def __init__(
        out self, compression: Optional[BodyCompression], native_codecs: Bool
    ):
        self.nodes = List[FieldNode]()
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
        elements. A builder copies the slice densely for every layout, nested
        children included, and is paid only when `offset != 0`.
        """
        var data = arr.to_data()
        if data.offset == 0:
            return data^
        var b = DynBuilder(arr.dtype(), len(arr))
        b.extend(arr)
        return b.finish().to_data()

    def write_array(mut self, root: ArrayData) raises:
        var stack = List[ArrayData]()
        stack.append(root.copy())
        while len(stack) > 0:
            var data = stack.pop()

            self.nodes.append(FieldNode(Int64(data.length), Int64(data.nulls)))

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
        data: ArrayData, mut pairs: List[DictPair], mut next_id: Int
    ) raises:
        """DFS inner-first: append (dict_id, values) for every dict array in the tree.
        """
        if data.dtype.is_dictionary():
            BatchEncoder.collect_dict_pairs(data.children[0], pairs, next_id)
            pairs.append(
                DictPair(next_id, DynArray.from_data(data.children[0]))
            )
            next_id += 1
        else:
            for child in data.children:
                BatchEncoder.collect_dict_pairs(child, pairs, next_id)

    def _build_body(
        mut self, mut buf_meta: List[BodyBuffer], mut body: List[UInt8]
    ) raises:
        """Assemble the buffers into a padded body, populating buf_meta
        offsets -- each non-empty one compressed if the batch is."""
        for buf in self.raw_bufs:
            pad_to(body, 8)
            var at = len(body)
            if self.compression:
                self.compression.value().compress(
                    Span(buf), body, self.native_codecs
                )
            else:
                body.extend(Span(buf))
            buf_meta.append(BodyBuffer(Int64(at), Int64(len(body) - at)))
        pad_to(body, 8)

    def encode(mut self, batch: RecordBatch) raises -> _EncodedBatch:
        for col in batch.columns:
            self.write_array(BatchEncoder.dense(col))
        var buf_meta = List[BodyBuffer]()
        var body = List[UInt8]()
        self._build_body(buf_meta, body)
        var rb_meta = IpcEncoder.encode_record_batch(
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
        var msg = IpcEncoder.frame_message(rb_meta, body)
        self.nodes = List[FieldNode]()
        self.raw_bufs = List[List[UInt8]]()
        self.variadic_counts = List[Int64]()
        return _EncodedBatch(msg^, metadata_length, body_length)

    def encode_dict_message(
        self, dict_id: Int64, values: DynArray
    ) raises -> _EncodedBatch:
        """Encode a dictionary values array as a DictionaryBatch IPC message,
        compressed as this encoder's batches are."""
        var benc = BatchEncoder(self.compression, self.native_codecs)
        benc.write_array(BatchEncoder.dense(values))
        var buf_meta = List[BodyBuffer]()
        var body = List[UInt8]()
        benc._build_body(buf_meta, body)

        var enc = IpcEncoder(512)
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
            HEADER_DICTIONARY_BATCH, db_pos, body_len
        )
        var meta = enc._finish(msg_pos)
        var meta_len = len(meta)
        var padded_meta = meta_len + (8 - meta_len % 8) % 8
        var metadata_length = Int32(8 + padded_meta)
        var body_length = Int64(len(body))
        var msg = IpcEncoder.frame_message(meta, body)
        return _EncodedBatch(msg^, metadata_length, body_length)


# ---------------------------------------------------------------------------
# Batch body decoder: reconstructs DynArray from raw bytes + cursor state
# ---------------------------------------------------------------------------


struct BatchDecoder(Movable):
    """Reconstructs DynArray values from a record batch body using node/buffer cursors.
    """

    var body: List[UInt8]
    var body_offset: Int
    var nodes: List[FieldNode]
    var bufs: List[BodyBuffer]
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
        var nodes: List[FieldNode],
        var bufs: List[BodyBuffer],
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
        mut self, dtype: dt.DynType, ipc_info: FieldIpcInfo
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
                    t"BatchDecoder: no values for dict_id {dict_id}"
                )
            var values = self.dict_values[dict_id].copy()
            # Fields sharing an id may declare different value types.
            if values.dtype() != d.value_type():
                raise CorruptError(
                    t"BatchDecoder: dictionary {dict_id} holds"
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
                        t"BatchDecoder: {dtype} column without a variadic count"
                    )
                var n_data = Int(self.variadic_counts[self.variadic_idx])
                self.variadic_idx += 1
                for _ in range(n_data):
                    self._consume_buffer(data_buffers)
            for i in range(layout.num_children):
                var child_ipc = (
                    ipc_info.children[i].copy() if i
                    < len(ipc_info.children) else FieldIpcInfo()
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

    def _next_buffer(mut self) raises CorruptError -> BodyBuffer:
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
