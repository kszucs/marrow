# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro records into Arrow arrays.

Avro is row-oriented: a record's fields follow one another, so a column cannot
be decoded on its own. `RecordDecoder` compiles the writer's schema once into a
tree of nodes, each holding the builder its values go to, and then interprets
that tree per row -- a switch on each node's opcode, appending into the
builders. A column left out of the projection is only read past.
"""

from ..arrays import DynArray, StringArray
from ..builders import DynBuilder, MapBuilder
from ..dtypes import Field
from ..errors import CorruptError, InternalError, KeyError
from ..schema import Schema
from ..tabular import RecordBatch
from .binary import AvroCursor
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
    to_arrow,
)
from .schema import AvroKind, AvroSchema

comptime _NULL = 100
comptime _ENUM = 101
comptime _LIST = 102
comptime _MAP = 103
comptime _STRUCT = 104
comptime _UNION = 105


struct _Node(Copyable, Movable):
    """One schema node: what to read (`kind`, `size`), where it goes (`op`,
    `builder`) and its sub-nodes. The builder is a shared handle -- a struct's
    node and its fields' nodes hold the same builders the struct builder
    does."""

    var op: Int
    var kind: AvroKind
    var size: Int
    """A fixed's byte width, or an enum's symbol count."""
    var builder: DynBuilder
    var children: List[_Node]
    var null_index: Int
    var value_index: Int
    """A union's branches: `children[0]` decodes `value_index`, and
    `null_index` is null. Either is -1 when the union has no such branch."""

    def __init__(out self, op: Int, kind: AvroKind, var builder: DynBuilder):
        self.op = op
        self.kind = kind
        self.size = 0
        self.builder = builder^
        self.children = List[_Node]()
        self.null_index = -1
        self.value_index = -1

    def __deinit__(deinit self):
        pass

    def decode(mut self, mut cur: AvroCursor[_]) raises:
        """Read one value and append it."""
        var op = self.op
        # Unions first: an Iceberg schema makes nearly every field one.
        if op == _UNION:
            var i = Int(cur.long())
            if i >= 0 and i == self.value_index:
                self.children[0].decode(cur)
            elif i == self.null_index:
                self.children[0].append_null()
            else:
                raise CorruptError(t"avro: union branch {i} out of range")
        elif op == LEAF_BOOL:
            self.builder.as_bool().append(cur.boolean())
        elif op == LEAF_INT32:
            self.builder.as_int32().append(cur.int())
        elif op == LEAF_INT64:
            self.builder.as_int64().append(cur.long())
        elif op == LEAF_FLOAT32:
            self.builder.as_float32().append(cur.float())
        elif op == LEAF_FLOAT64:
            self.builder.as_float64().append(cur.double())
        elif op == LEAF_STRING:
            self.builder.as_string().append(
                StringSlice(unsafe_from_utf8=cur.bytes())
            )
        elif op == LEAF_BINARY:
            self.builder.as_binary().append(
                StringSlice(unsafe_from_utf8=cur.bytes())
            )
        elif op == LEAF_FIXED:
            self.builder.as_fixed_size_binary().append(cur.fixed(self.size))
        elif op == LEAF_UUID_TEXT:
            var uuid = cur.uuid_text()
            self.builder.as_fixed_size_binary().append(Span(uuid))
        elif op == LEAF_DATE32:
            self.builder.as_date32().append(cur.int())
        elif op == LEAF_TIME32:
            self.builder.as_time32().append(cur.int())
        elif op == LEAF_TIME64:
            self.builder.as_time64().append(cur.long())
        elif op == LEAF_TIMESTAMP:
            self.builder.as_timestamp().append(cur.long())
        elif op == LEAF_DECIMAL128:
            self.builder.as_decimal128().append(
                cur.decimal[DType.int128](self.size)
            )
        elif op == LEAF_DECIMAL256:
            self.builder.as_decimal256().append(
                cur.decimal[DType.int256](self.size)
            )
        elif op == LEAF_DURATION:
            self.builder.as_month_day_nano_interval().append(cur.duration())
        elif op == _ENUM:
            var i = Int(cur.int())
            if i < 0 or i >= self.size:
                raise CorruptError(
                    t"avro: enum index {i} out of range for {self.size} symbols"
                )
            self.builder.as_dictionary().append(i)
        elif op == _LIST:
            while True:
                var count = cur.block()[0]
                if count == 0:
                    break
                for _ in range(count):
                    self.children[0].decode(cur)
            self.builder.as_list().append_valid()
        elif op == _MAP:
            var entries = self.builder.as_type[MapBuilder]().entries()
            while True:
                var count = cur.block()[0]
                if count == 0:
                    break
                # An Avro map's string key then its value, or -- for
                # `logicalType: map` -- a record{key, value}: the same bytes.
                for _ in range(count):
                    self.children[0].decode(cur)
                    self.children[1].decode(cur)
                    entries.as_struct().append_valid()
            self.builder.as_type[MapBuilder]().append_valid()
        elif op == _STRUCT:
            for ref c in self.children:
                c.decode(cur)
            self.builder.as_struct().append_valid()
        else:  # _NULL
            self.builder.append_null()

    def append_null(mut self) raises:
        if self.op == _UNION:
            self.children[0].append_null()
        else:
            self.builder.append_null()


def _skip(avro: AvroSchema, mut cur: AvroCursor[_]) raises:
    """Read past one value of `avro`, building nothing -- what an unselected
    column decodes to."""
    var k = avro.kind
    if k == AvroKind.NULL:
        pass
    elif k == AvroKind.BOOLEAN:
        cur.skip(1)
    elif k == AvroKind.INT or k == AvroKind.LONG or k == AvroKind.ENUM:
        _ = cur.long()
    elif k == AvroKind.FLOAT:
        cur.skip(4)
    elif k == AvroKind.DOUBLE:
        cur.skip(8)
    elif k == AvroKind.BYTES or k == AvroKind.STRING:
        _ = cur.bytes()
    elif k == AvroKind.FIXED:
        cur.skip(avro.size)
    elif k == AvroKind.RECORD:
        for ref c in avro.children:
            _skip(c, cur)
    elif k == AvroKind.ARRAY or k == AvroKind.MAP:
        while True:
            var count, size = cur.block()
            if count == 0:
                break
            if size >= 0:
                cur.skip(size)
                continue
            for _ in range(count):
                if k == AvroKind.MAP:
                    _ = cur.bytes()
                _skip(avro.children[0], cur)
    else:  # union
        var i = Int(cur.long())
        if i < 0 or i >= len(avro.children):
            raise CorruptError(t"avro: union branch {i} out of range")
        _skip(avro.children[i], cur)


def _compile(avro: AvroSchema, builder: DynBuilder) raises -> _Node:
    """The node decoding `avro` into `builder`, a `DynBuilder` of the type
    `avro` maps to in Arrow; a nested node takes its builder from the
    parent's."""
    ref k = avro.kind
    if k == AvroKind.UNION:
        var v = avro.value_index()
        var n = _Node(_UNION, k, builder)
        n.null_index = avro.null_index()
        n.value_index = v
        if v < 0:
            n.children.append(_Node(_NULL, AvroKind.NULL, builder))
        else:
            n.children.append(_compile(avro.children[v], builder))
        return n^
    var dtype = builder.dtype()
    if dtype.is_null():
        return _Node(_NULL, k, builder)
    if dtype.is_struct():
        var n = _Node(_STRUCT, k, builder)
        for i in range(len(avro.children)):
            n.children.append(
                _compile(avro.children[i], builder.as_struct().field_builder(i))
            )
        return n^
    if dtype.is_list():
        var n = _Node(_LIST, k, builder)
        n.children.append(
            _compile(avro.children[0], builder.as_list().values())
        )
        return n^
    if dtype.is_map():
        var entries = builder.as_type[MapBuilder]().entries()
        ref fields = entries.as_struct()
        var n = _Node(_MAP, k, builder)
        if k == AvroKind.MAP:
            n.children.append(
                _Node(LEAF_STRING, AvroKind.STRING, fields.field_builder(0))
            )
            n.children.append(
                _compile(avro.children[0], fields.field_builder(1))
            )
        else:  # `logicalType: map` -- an array of record{key, value}
            ref items = avro.children[0]
            n.children.append(
                _compile(items.children[0], fields.field_builder(0))
            )
            n.children.append(
                _compile(items.children[1], fields.field_builder(1))
            )
        return n^
    if dtype.is_dictionary():
        var symbols = List[Optional[String]]()
        for ref s in avro.symbols:
            symbols.append(s)
        builder.as_dictionary().set_dictionary(StringArray.from_values(symbols))
        var n = _Node(_ENUM, k, builder)
        n.size = len(avro.symbols)
        return n^
    var op = leaf_of(dtype, k)
    if op == 0:
        raise InternalError(t"avro: no leaf decodes {avro} as {dtype}")
    var n = _Node(op, k, builder)
    n.size = avro.fixed_size()
    return n^


struct RecordDecoder(Movable):
    """Decodes rows of one record schema into a `RecordBatch`.

    `columns` selects top-level fields by name, in the order given; the rest
    are skipped over. Rows accumulate across `decode` calls until `finish`.
    """

    var _avro: AvroSchema
    var _fields: List[Field]
    """Every record field as an Arrow field, selected or not."""
    var _selected: List[Int]
    """For each output column, its field index in the record."""
    var _schema: Schema
    var _nodes: List[_Node]
    """One per selected column, in output order."""
    var _plan: List[Int]
    """Per record field, its node in `_nodes`, or -1 to skip it."""
    var _rows: Int

    def __init__(
        out self, avro: AvroSchema, columns: Optional[List[String]] = None
    ) raises:
        var full = to_arrow(avro)
        var selected = List[Int]()
        if columns:
            for ref name in columns.value():
                var i = full.get_field_index(name)
                if i < 0:
                    raise KeyError(t"avro: no column named '{name}'")
                selected.append(i)
        else:
            for i in range(len(full.fields)):
                selected.append(i)
        var fields = List[Field]()
        for i in selected:
            fields.append(full.fields[i].copy())
        self._avro = avro.copy()
        self._fields = full.fields.copy()
        self._selected = selected^
        self._schema = Schema(fields=fields^)
        self._nodes = List[_Node]()
        self._plan = [-1 for _ in range(len(avro.children))]
        for j in range(len(self._selected)):
            self._plan[self._selected[j]] = j
        self._rows = 0
        self._reset()

    def _reset(mut self) raises:
        """Fresh builders, and the nodes over them."""
        self._nodes = List[_Node]()
        for i in self._selected:
            self._nodes.append(
                _compile(
                    self._avro.children[i], DynBuilder(self._fields[i].dtype)
                )
            )
        self._rows = 0

    def schema(self) -> Schema:
        return self._schema.copy()

    def rows(self) -> Int:
        """Rows decoded since the last `finish`."""
        return self._rows

    def decode(mut self, mut cur: AvroCursor[_], count: Int) raises:
        """Decode `count` records from `cur`."""
        for _ in range(count):
            for i in range(len(self._plan)):
                var j = self._plan[i]
                if j >= 0:
                    self._nodes[j].decode(cur)
                else:
                    _skip(self._avro.children[i], cur)
        self._rows += count

    def finish(mut self) raises -> RecordBatch:
        """The rows decoded so far, as a batch; the decoder starts over
        empty."""
        var columns = List[DynArray]()
        for ref n in self._nodes:
            columns.append(n.builder.finish())
        var batch = RecordBatch(self._schema, columns^)
        self._reset()
        return batch^
