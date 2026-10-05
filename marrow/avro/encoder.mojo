# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow arrays into Avro records.

The inverse of `decoder`: `RecordEncoder` compiles a record schema against a
batch's columns into a tree of nodes, each holding the array it reads, and
writes the batch row by row -- every field of row 0, then of row 1, and so on,
since that is the order Avro's encoding has.
"""

from ..arrays import DynArray
from ..dtypes import DynType
from ..errors import InvalidError, TypeError
from ..tabular import RecordBatch
from .binary import AvroBytes
from .mapping import (
    LEAF_BOOL,
    LEAF_INT32,
    LEAF_INT64,
    LEAF_FLOAT32,
    LEAF_FLOAT64,
    LEAF_STRING,
    LEAF_LARGE_STRING,
    LEAF_BINARY,
    LEAF_LARGE_BINARY,
    LEAF_FIXED,
    LEAF_UUID_TEXT,
    LEAF_DATE32,
    LEAF_TIME32,
    LEAF_TIME64,
    LEAF_TIMESTAMP,
    LEAF_DECIMAL128,
    LEAF_DECIMAL256,
    LEAF_DURATION,
    leaf_of,
    is_logical_map,
    writes_as,
)
from .schema import AvroKind, AvroSchema

comptime _NULL = 100
comptime _LIST = 101
comptime _LARGE_LIST = 102
comptime _MAP = 103
comptime _STRUCT = 104
comptime _UNION = 105
comptime _DICT = 106


struct _Node(Copyable, Movable):
    """One schema node: the array its values come from (`op` says how to read
    it), how to write them (`kind`, `size`, `symbols`) and its sub-nodes."""

    var op: Int
    var kind: AvroKind
    var size: Int
    var array: DynArray
    var checks_nulls: Bool
    """Whether a value must be checked for null before it is written: the
    array has nulls, and no union above has already checked."""
    var symbols: List[String]
    var children: List[_Node]
    var null_index: Int
    var value_index: Int

    def __init__(out self, op: Int, kind: AvroKind, array: DynArray):
        self.op = op
        self.kind = kind
        self.size = 0
        self.array = array.copy()
        self.checks_nulls = array.null_count() > 0
        self.symbols = List[String]()
        self.children = List[_Node]()
        self.null_index = -1
        self.value_index = -1

    def __deinit__(deinit self):
        pass

    def encode(self, i: Int, mut out: AvroBytes) raises:
        """Write row `i` of this node's array."""
        var op = self.op
        if op == _UNION:
            if not self.checks_nulls or self.array.is_valid(i):
                out.long(Int64(self.value_index))
                self.children[0].encode(i, out)
            elif self.null_index >= 0:
                out.long(Int64(self.null_index))
            else:
                raise InvalidError(
                    t"avro: a null at row {i} where the schema has no null"
                    t" branch: {self.kind}"
                )
            return
        if op == _NULL:
            return
        if self.checks_nulls and not self.array.is_valid(i):
            raise InvalidError(
                t"avro: a null at row {i} of a non-nullable {self.kind}"
            )
        if op == LEAF_BOOL:
            out.boolean(self.array.as_bool().values().test(i))
        elif op == LEAF_INT32:
            out.int(self.array.as_int32().unsafe_get(i))
        elif op == LEAF_INT64:
            out.long(self.array.as_int64().unsafe_get(i))
        elif op == LEAF_FLOAT32:
            out.float(self.array.as_float32().unsafe_get(i))
        elif op == LEAF_FLOAT64:
            out.double(self.array.as_float64().unsafe_get(i))
        elif op == LEAF_STRING:
            self._text(self.array.as_string().unsafe_get(UInt(i)), out)
        elif op == LEAF_LARGE_STRING:
            self._text(self.array.as_large_string().unsafe_get(UInt(i)), out)
        elif op == LEAF_BINARY:
            out.bytes(self.array.as_binary().unsafe_get(UInt(i)).as_bytes())
        elif op == LEAF_LARGE_BINARY:
            out.bytes(
                self.array.as_large_binary().unsafe_get(UInt(i)).as_bytes()
            )
        elif op == LEAF_FIXED:
            out.fixed(self._fixed_bytes(i))
        elif op == LEAF_UUID_TEXT:
            out.uuid_text(self._fixed_bytes(i))
        elif op == LEAF_DATE32:
            out.int(self.array.as_date32().unsafe_get(i))
        elif op == LEAF_TIME32:
            out.int(self.array.as_time32().unsafe_get(i))
        elif op == LEAF_TIME64:
            out.long(self.array.as_time64().unsafe_get(i))
        elif op == LEAF_TIMESTAMP:
            out.long(self.array.as_timestamp().unsafe_get(i))
        elif op == LEAF_DECIMAL128:
            out.decimal(self.array.as_decimal128().unsafe_get(i), self.size)
        elif op == LEAF_DECIMAL256:
            out.decimal(self.array.as_decimal256().unsafe_get(i), self.size)
        elif op == LEAF_DURATION:
            out.duration(self.array.as_month_day_nano_interval().unsafe_get(i))
        elif op == _LIST:
            var start, end = self.array.as_list().child_range(i)
            self._items(start, end, out)
        elif op == _LARGE_LIST:
            var start, end = self.array.as_large_list().child_range(i)
            self._items(start, end, out)
        elif op == _MAP:
            var start, end = self.array.as_map().child_range(i)
            self._items(start, end, out)
        elif op == _DICT:
            self.children[0].encode(self._index(i), out)
        else:  # _STRUCT
            for ref c in self.children:
                c.encode(i, out)

    def _items(self, start: Int, end: Int, mut out: AvroBytes) raises:
        """An array's items -- or a map's key-value pairs -- as one block,
        then the empty block that ends it."""
        if end > start:
            out.long(Int64(end - start))
            for j in range(start, end):
                for ref c in self.children:
                    c.encode(j, out)
        out.long(0)

    def _fixed_bytes(self, i: Int) -> Span[UInt8, origin_of(self.array)]:
        ref a = self.array.as_fixed_size_binary()
        var w = a.byte_width
        return rebind[Span[UInt8, origin_of(self.array)]](
            a.buffer.slice((a.offset + i) * w, w).as_span()
        )

    def _text(self, s: StringSlice[_], mut out: AvroBytes) raises:
        """A string -- or, for an enum, the index of the symbol it names."""
        if self.kind == AvroKind.ENUM:
            for k in range(len(self.symbols)):
                if StringSlice(self.symbols[k]) == s:
                    out.int(Int32(k))
                    return
            raise InvalidError(t"avro: '{s}' is not a symbol of the enum")
        out.bytes(s.as_bytes())

    def _index(self, i: Int) -> Int:
        """Row `i` of a dictionary column's indices, whichever integer type
        they are."""
        ref a = self.array
        var t = a.dtype()
        if t.is_int8():
            return Int(a.as_int8().unsafe_get(i))
        elif t.is_int16():
            return Int(a.as_int16().unsafe_get(i))
        elif t.is_int32():
            return Int(a.as_int32().unsafe_get(i))
        elif t.is_int64():
            return Int(a.as_int64().unsafe_get(i))
        elif t.is_uint8():
            return Int(a.as_uint8().unsafe_get(i))
        elif t.is_uint16():
            return Int(a.as_uint16().unsafe_get(i))
        elif t.is_uint32():
            return Int(a.as_uint32().unsafe_get(i))
        return Int(a.as_uint64().unsafe_get(i))


def _mismatch(avro: AvroSchema, dtype: DynType) -> TypeError:
    return TypeError(t"avro: a {dtype} column cannot be written as {avro}")


def _compile(avro: AvroSchema, array: DynArray) raises -> _Node:
    """The node writing `array` as `avro`."""
    ref k = avro.kind
    var dtype = array.dtype()
    if k == AvroKind.UNION:
        var v = avro.value_index()
        var n = _Node(_UNION, k, array)
        n.null_index = avro.null_index()
        n.value_index = v
        if v < 0:
            n.children.append(_Node(_NULL, AvroKind.NULL, array))
        else:
            n.children.append(_compile(avro.children[v], array))
        # The union reads the validity; below it every value is present.
        n.children[0].checks_nulls = False
        return n^
    if dtype.is_dictionary():
        # Each row is written as the dictionary value its index points at.
        ref d = array.as_dictionary()
        var n = _Node(_DICT, k, d.indices())
        n.children.append(_compile(avro, d.dictionary()))
        return n^
    if k == AvroKind.NULL:
        if not dtype.is_null():
            raise _mismatch(avro, dtype)
        return _Node(_NULL, k, array)
    if k == AvroKind.RECORD:
        if not dtype.is_struct():
            raise _mismatch(avro, dtype)
        ref s = array.as_struct()
        if len(s.children) != len(avro.children):
            raise _mismatch(avro, dtype)
        var n = _Node(_STRUCT, k, array)
        for i in range(len(avro.children)):
            n.children.append(_compile(avro.children[i], s.field(i)))
        return n^
    if k == AvroKind.MAP or is_logical_map(avro):
        if not dtype.is_map():
            raise _mismatch(avro, dtype)
        ref entries = array.as_map().values().as_struct()
        var n = _Node(_MAP, k, array)
        if k == AvroKind.MAP:
            n.children.append(
                _compile(AvroSchema(AvroKind.STRING), entries.field(0))
            )
            n.children.append(_compile(avro.children[0], entries.field(1)))
        else:
            ref items = avro.children[0]
            n.children.append(_compile(items.children[0], entries.field(0)))
            n.children.append(_compile(items.children[1], entries.field(1)))
        return n^
    if k == AvroKind.ARRAY:
        var n: _Node
        if dtype.is_list():
            n = _Node(_LIST, k, array)
            n.children.append(
                _compile(avro.children[0], array.as_list().values())
            )
        elif dtype.is_large_list():
            n = _Node(_LARGE_LIST, k, array)
            n.children.append(
                _compile(avro.children[0], array.as_large_list().values())
            )
        else:
            raise _mismatch(avro, dtype)
        return n^
    if not writes_as(dtype, avro):
        raise _mismatch(avro, dtype)
    var op = leaf_of(dtype, k)
    if op == 0:
        raise _mismatch(avro, dtype)
    var n = _Node(op, k, array)
    n.size = avro.fixed_size()
    n.symbols = avro.symbols.copy()
    return n^


struct RecordEncoder(Movable):
    """Encodes the rows of a batch as records of one schema: `bind` a batch,
    then `encode` its rows, one call each."""

    var _avro: AvroSchema
    var _nodes: List[_Node]

    def __init__(out self, avro: AvroSchema) raises:
        if avro.kind != AvroKind.RECORD:
            raise TypeError(
                t"avro: only a record schema writes a table, got {avro.kind}"
            )
        self._avro = avro.copy()
        self._nodes = List[_Node]()

    def bind(mut self, batch: RecordBatch) raises:
        """Read rows from `batch` from now on. Its columns are matched to the
        record's fields by position, and each must have the type its field
        reads back as."""
        if len(batch.columns) != len(self._avro.children):
            raise TypeError(
                t"avro: batch has {len(batch.columns)} columns, the schema"
                t" {len(self._avro.children)} fields"
            )
        self._nodes = List[_Node]()
        for i in range(len(batch.columns)):
            self._nodes.append(
                _compile(self._avro.children[i], batch.columns[i])
            )

    def encode(self, row: Int, mut out: AvroBytes) raises:
        """Append row `row` of the bound batch."""
        for ref n in self._nodes:
            n.encode(row, out)
