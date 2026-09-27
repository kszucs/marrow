# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON into Arrow — `read_json` and the streaming `JsonReader`.

The input is read through a `ByteSource` in blocks, each cut at the last newline
inside it, so a local file stays a zero-copy memory map and a remote one is
fetched a block at a time. Within a block, rows are a whitespace-separated
stream of objects, parsed by EmberJson's `Parser` token by token straight into
marrow builders: nothing builds a JSON document first.

A read is two passes (see `infer.mojo`): pass 1 settles the schema, pass 2
parses each block into a `RecordBatch` of that schema. `read_json` infers over
the whole input, so every block agrees; `JsonReader`, like Arrow C++'s
streaming reader, infers from the first block and is strict after it.

Behaviour follows `pyarrow.json.read_json`, down to the error messages, with
two deliberate differences: the result has one chunk per *block* but a row
longer than a block grows the read instead of failing, and `NaN`/`Infinity`
literals are not JSON and are rejected.
"""

from emberjson import Parser, is_valid_utf8

from ..builders import DynBuilder
from ..dtypes import (
    DynType,
    bool_,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    microsecond,
    millisecond,
    nanosecond,
    second,
    struct_,
    uint8,
    uint16,
    uint32,
    uint64,
)
from ..io import ByteSource, DynSource, StorageOptions
from ..schema import Schema
from ..tabular import RecordBatch, Table
from ..utils.datetime import parse_iso8601

from .infer import (
    BOOL,
    INT,
    LIST,
    NULL,
    STRING,
    STRUCT,
    Shape,
    changed,
    duplicate,
    family_of,
    index_of,
    infer_rows,
)
from .options import ParseOptions, ReadOptions, UnexpectedFieldBehavior

comptime _NULL = 0
comptime _BOOL = 1
comptime _I8 = 2
comptime _I16 = 3
comptime _I32 = 4
comptime _I64 = 5
comptime _U8 = 6
comptime _U16 = 7
comptime _U32 = 8
comptime _U64 = 9
comptime _F32 = 10
comptime _F64 = 11
comptime _STR = 12
comptime _LSTR = 13
comptime _TS_S = 14
comptime _TS_MS = 15
comptime _TS_US = 16
comptime _TS_NS = 17
comptime _LIST = 18
comptime _STRUCT = 19


def _code(dtype: DynType) raises -> Int:
    """The slot code for a type the reader can build, or raise."""
    if dtype.is_null():
        return _NULL
    elif dtype == bool_:
        return _BOOL
    elif dtype == int8:
        return _I8
    elif dtype == int16:
        return _I16
    elif dtype == int32:
        return _I32
    elif dtype == int64:
        return _I64
    elif dtype == uint8:
        return _U8
    elif dtype == uint16:
        return _U16
    elif dtype == uint32:
        return _U32
    elif dtype == uint64:
        return _U64
    elif dtype == float32:
        return _F32
    elif dtype == float64:
        return _F64
    elif dtype.is_string():
        return _STR
    elif dtype.is_large_string():
        return _LSTR
    elif dtype.is_timestamp():
        var unit = dtype.as_timestamp().unit
        if unit == second:
            return _TS_S
        elif unit == millisecond:
            return _TS_MS
        elif unit == microsecond:
            return _TS_US
        return _TS_NS
    elif dtype.is_list():
        return _LIST
    elif dtype.is_struct():
        return _STRUCT
    raise Error("JSON reader: unsupported type ", dtype)


def _family(dtype: DynType) -> Int:
    """The JSON value kind a column of `dtype` accepts."""
    if dtype.is_null():
        return NULL
    elif dtype == bool_:
        return BOOL
    elif dtype.is_string() or dtype.is_large_string() or dtype.is_timestamp():
        return STRING
    elif dtype.is_list():
        return LIST
    elif dtype.is_struct():
        return STRUCT
    return INT


struct _Slot(Movable):
    """Pass 2's view of one column: its builder and, for a nested type, the
    slots of its children, which share builders with the parent's."""

    var code: Int
    var family: Int
    var dtype: DynType
    var nullable: Bool
    var path: String
    """Where the column is, as Arrow's errors spell it: `/a/b`, `/a/[]`."""
    var builder: DynBuilder
    var names: List[String]
    var children: List[_Slot]
    var visits: Int
    """For a struct: objects seen so far, which stamps its children."""
    var last: Int
    """The parent's `visits` when this column was last given a value."""

    def __init__(
        out self,
        dtype: DynType,
        nullable: Bool,
        var path: String,
        builder: DynBuilder,
    ) raises:
        self.code = _code(dtype)
        self.family = _family(dtype)
        self.dtype = dtype.copy()
        self.nullable = nullable
        self.path = path^
        self.builder = builder
        self.names = []
        self.children = []
        self.visits = 0
        self.last = 0
        if self.code == _LIST:
            ref item = dtype.as_list().item[]
            self.children.append(
                _Slot(
                    item.dtype,
                    item.nullable,
                    self.path + "/[]",
                    builder.as_list().values(),
                )
            )
        elif self.code == _STRUCT:
            ref fields = dtype.as_struct().fields
            for i in range(len(fields)):
                self.names.append(fields[i].name)
                self.children.append(
                    _Slot(
                        fields[i].dtype,
                        fields[i].nullable,
                        self.path + "/" + fields[i].name,
                        builder.as_struct().field_builder(i),
                    )
                )

    # `children: List[_Slot]` cycles back through `_Slot`.
    def __deinit__(deinit self):
        pass


def _pad_null(mut slot: _Slot) raises:
    """A null here, and in every child of a struct: `StructBuilder` leaves its
    children to the caller."""
    slot.builder.append_null()
    if slot.code == _STRUCT:
        for i in range(len(slot.children)):
            _pad_null(slot.children[i])


def _append_null(mut slot: _Slot, row: Int, absent: Bool) raises:
    if not slot.nullable:
        raise Error(
            "JSON parse error: Column(",
            slot.path,
            "): a required field was ",
            "absent" if absent else "null",
            " in row ",
            row,
        )
    _pad_null(slot)


def _number[dt: DType](mut p: Parser, dtype: DynType) raises -> Scalar[dt]:
    """The number at the cursor as `dt`, or Arrow's wording for one the
    column's type cannot hold."""
    var start = p.data
    try:
        comptime if dt.is_floating_point():
            return p.expect_float[dt]()
        else:
            return p.expect_int[dt]()
    except:
        p.data = start
        raise Error(
            "JSON parse error: Failed to convert JSON to ",
            dtype,
            ", couldn't parse:",
            StringSlice(unsafe_from_utf8=p.expect_float_bytes()),
        )


def _append_number(mut p: Parser, mut slot: _Slot) raises:
    ref b = slot.builder
    var code = slot.code
    if code == _I8:
        b.as_int8().append(_number[DType.int8](p, slot.dtype))
    elif code == _I16:
        b.as_int16().append(_number[DType.int16](p, slot.dtype))
    elif code == _I32:
        b.as_int32().append(_number[DType.int32](p, slot.dtype))
    elif code == _I64:
        b.as_int64().append(_number[DType.int64](p, slot.dtype))
    elif code == _U8:
        b.as_uint8().append(_number[DType.uint8](p, slot.dtype))
    elif code == _U16:
        b.as_uint16().append(_number[DType.uint16](p, slot.dtype))
    elif code == _U32:
        b.as_uint32().append(_number[DType.uint32](p, slot.dtype))
    elif code == _U64:
        b.as_uint64().append(_number[DType.uint64](p, slot.dtype))
    elif code == _F32:
        b.as_float32().append(_number[DType.float32](p, slot.dtype))
    else:
        b.as_float64().append(_number[DType.float64](p, slot.dtype))


def _append_timestamp[digits: Int](mut p: Parser, mut slot: _Slot) raises:
    var text = p.expect_string()
    var ticks = parse_iso8601[digits](text.as_bytes())
    if not ticks:
        raise Error(
            "JSON parse error: Failed to convert JSON to ",
            slot.dtype,
            ", couldn't parse:",
            text,
        )
    slot.builder.as_timestamp().append(Int64(ticks.value()))


def _append_value(
    mut p: Parser, mut slot: _Slot, row: Int, ignore_unexpected: Bool
) raises:
    p.skip_whitespace()
    var kind = family_of(p.peek())
    if kind == NULL:
        p.expect_null()
        _append_null(slot, row, False)
        return
    if slot.family != kind:
        raise changed(slot.path, slot.family, kind, row)
    var code = slot.code
    if code == _BOOL:
        slot.builder.as_bool().append(p.expect_bool())
    elif kind == INT:
        _append_number(p, slot)
    elif code == _STR:
        slot.builder.as_string().append(p.expect_string())
    elif code == _LSTR:
        slot.builder.as_large_string().append(p.expect_string())
    elif code == _TS_S:
        _append_timestamp[0](p, slot)
    elif code == _TS_MS:
        _append_timestamp[3](p, slot)
    elif code == _TS_US:
        _append_timestamp[6](p, slot)
    elif code == _TS_NS:
        _append_timestamp[9](p, slot)
    elif code == _LIST:
        p.expect_open(Byte(ord("[")))
        p.skip_whitespace()
        if p.peek() == Byte(ord("]")):
            p.expect(Byte(ord("]")))
        else:
            while True:
                _append_value(p, slot.children[0], row, ignore_unexpected)
                p.skip_whitespace()
                if p.peek() == Byte(ord(",")):
                    p.expect(Byte(ord(",")))
                else:
                    p.expect(Byte(ord("]")))
                    break
        slot.builder.as_list().append_valid()
    else:
        _append_object(p, slot, row, ignore_unexpected)


def _append_object(
    mut p: Parser, mut slot: _Slot, row: Int, ignore_unexpected: Bool
) raises:
    p.expect_open(Byte(ord("{")))
    slot.visits += 1
    var visit = slot.visits
    p.skip_whitespace()
    if p.peek() == Byte(ord("}")):
        p.expect(Byte(ord("}")))
    else:
        while True:
            var key = p.expect_string()
            p.expect(Byte(ord(":")))
            var idx = index_of(slot.names, key)
            if idx < 0:
                if not ignore_unexpected:
                    raise Error("JSON parse error: unexpected field")
                p.skip_value()
            else:
                ref child = slot.children[idx]
                if child.last == visit:
                    raise duplicate(child.path, row)
                child.last = visit
                _append_value(p, child, row, ignore_unexpected)
            p.skip_whitespace()
            if p.peek() == Byte(ord(",")):
                p.expect(Byte(ord(",")))
            else:
                p.expect(Byte(ord("}")))
                break
    for i in range(len(slot.children)):
        if slot.children[i].last != visit:
            _append_null(slot.children[i], row, True)
    slot.builder.as_struct().append_valid()


def _in_row(e: Error, row: Int) -> Error:
    """A tokenizer error, worded and located the way Arrow's are; the reader's
    own errors already are."""
    var message = String(e)
    if message.startswith("JSON parse error"):
        return e
    return Error("JSON parse error: ", message, " in row ", row)


def _check_utf8(block: Span[Byte, _]) raises:
    if not is_valid_utf8(block):
        raise Error("JSON parse error: invalid UTF-8")


def parse_block[
    origin: ImmOrigin
](
    block: Span[Byte, origin],
    schema: Schema,
    ignore_unexpected: Bool,
    validate: Bool,
    mut row: Int,
) raises -> RecordBatch:
    """Pass 2 over one block: its rows, as a batch of `schema`. `validate` is
    False only for a block pass 1 already checked for UTF-8."""
    if validate:
        _check_utf8(block)
    var dtype: DynType = struct_(schema.fields.copy())
    var builder = DynBuilder(dtype)
    var root = _Slot(dtype, False, "", builder)
    var p = Parser(block)
    p.skip_whitespace()
    while p.has_more():
        var byte = p.peek()
        if byte != Byte(ord("{")):
            raise changed("", STRUCT, family_of(byte), row)
        try:
            _append_object(p, root, row, ignore_unexpected)
        except e:
            raise _in_row(e, row)
        row += 1
        p.skip_whitespace()
    var rows = builder.finish()
    return RecordBatch(schema, rows.as_struct().children.copy())


def infer_block[
    origin: ImmOrigin
](block: Span[Byte, origin], mut root: Shape, mut row: Int) raises:
    """Pass 1 over one block: fold its rows into `root`."""
    _check_utf8(block)
    var p = Parser(block)
    try:
        infer_rows(p, root, row)
    except e:
        raise _in_row(e, row)


def block_end[
    S: ByteSource
](source: S, offset: Int, block_size: Int) raises -> Int:
    """Where the block starting at `offset` ends: just past the last newline
    within `block_size` bytes, the end of the input if that comes first, and a
    doubled read if a single row is longer than the block."""
    var size = source.size()
    var want = block_size
    while True:
        var n = min(want, size - offset)
        if offset + n >= size:
            return size
        var data = source.read_at(offset, n)
        var i = n - 1
        while i >= 0 and data[i] != Byte(ord("\n")):
            i -= 1
        if i >= 0:
            return offset + i + 1
        want *= 2


def infer_schema[
    S: ByteSource
](
    source: S,
    parse_options: ParseOptions,
    block_size: Int,
    whole: Bool,
    mut inferred: Int,
) raises -> Schema:
    """The schema a read settles on: the explicit one when nothing is left to
    infer, otherwise inferred over the first block, or every block when
    `whole`. `inferred` answers how many leading bytes pass 1 read — and so
    checked for UTF-8."""
    if source.size() == 0:
        raise Error("Empty JSON file")
    if parse_options.behavior() != UnexpectedFieldBehavior.INFER:
        return parse_options.explicit_schema.value().copy()
    var root = Shape.of(
        parse_options.explicit_schema.value()
    ) if parse_options.explicit_schema else Shape.of(Schema())
    var offset = 0
    var row = 0
    while offset < source.size():
        var end = block_end(source, offset, block_size)
        infer_block(source.read_at(offset, end - offset), root, row)
        offset = end
        if not whole:
            break
    inferred = offset
    return root.schema()


struct JsonReader[S: ByteSource](Movable):
    """Newline-delimited JSON, one `RecordBatch` per block.

    Mirrors Arrow C++'s streaming reader: the schema is the explicit one when
    nothing is left to infer, and otherwise is inferred from the *first* block
    and fixed from then on. A later block with a key the schema lacks, or a
    value its type cannot hold, raises rather than widening what earlier
    batches already committed to.
    """

    var schema: Schema
    var _source: Self.S
    var _block_size: Int
    var _ignore_unexpected: Bool
    var _inferred: Int
    """Leading bytes pass 1 already checked for UTF-8."""
    var _offset: Int
    var _row: Int

    def __init__(
        out self,
        var source: Self.S,
        read_options: ReadOptions = ReadOptions(),
        parse_options: ParseOptions = ParseOptions(),
    ) raises:
        var inferred = 0
        var schema = infer_schema(
            source, parse_options, read_options.block_size, False, inferred
        )
        self = Self(source^, schema, read_options, parse_options, inferred)

    def __init__(
        out self,
        var source: Self.S,
        schema: Schema,
        read_options: ReadOptions,
        parse_options: ParseOptions,
        inferred: Int,
    ):
        """A reader of a schema already settled, by `infer_schema` over the
        first `inferred` bytes."""
        self.schema = schema.copy()
        self._source = source^
        self._block_size = read_options.block_size
        # After pass 1 has run with `INFER` there is no key left to meet, so
        # meeting one is an error; only an explicit `IGNORE` skips.
        self._ignore_unexpected = (
            parse_options.behavior() == UnexpectedFieldBehavior.IGNORE
        )
        self._inferred = inferred
        self._offset = 0
        self._row = 0

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The next block's rows, or `None` at the end of the input."""
        if self._offset >= self._source.size():
            return None
        var end = block_end(self._source, self._offset, self._block_size)
        var batch = parse_block(
            self._source.read_at(self._offset, end - self._offset),
            self.schema,
            self._ignore_unexpected,
            end > self._inferred,
            self._row,
        )
        self._offset = end
        return batch^

    def read_all(mut self) raises -> Table:
        """Every remaining block, as a table with one chunk per block."""
        var batches = List[RecordBatch]()
        while True:
            var batch = self.read_next_batch()
            if not batch:
                break
            batches.append(batch.take())
        return Table.from_batches(self.schema, batches)


def read_json[
    S: ByteSource
](
    var source: S,
    read_options: ReadOptions = ReadOptions(),
    parse_options: ParseOptions = ParseOptions(),
) raises -> Table:
    """Read newline-delimited JSON from `source` into a `Table`, inferring over
    the whole input so every block agrees on one schema."""
    var inferred = 0
    var schema = infer_schema(
        source, parse_options, read_options.block_size, True, inferred
    )
    var reader = JsonReader[S](
        source^, schema, read_options, parse_options, inferred
    )
    return reader.read_all()


def read_json(
    uri: String,
    read_options: ReadOptions = ReadOptions(),
    parse_options: ParseOptions = ParseOptions(),
    options: StorageOptions = StorageOptions(),
) raises -> Table:
    """Read a newline-delimited JSON file into a `Table` (mirrors
    `pyarrow.json.read_json`).

    `uri` is a path or a URL, opened like `marrow.parquet.read_table`'s: a bare
    path is a zero-copy memory map, anything else goes through OpenDAL with
    `options` as its service configuration.
    """
    return read_json(DynSource.open(uri, options), read_options, parse_options)


def open_json(
    uri: String,
    read_options: ReadOptions = ReadOptions(),
    parse_options: ParseOptions = ParseOptions(),
    options: StorageOptions = StorageOptions(),
) raises -> JsonReader[DynSource]:
    """A streaming reader over a newline-delimited JSON file (mirrors
    `pyarrow.json.open_json`)."""
    return JsonReader[DynSource](
        DynSource.open(uri, options), read_options, parse_options
    )
