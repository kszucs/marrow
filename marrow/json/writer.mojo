# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow into newline-delimited JSON — `write_json` and `JsonWriter`.

One object per row, keys in schema order, every column present: a null is
written as `null`, not dropped, so reading the output back yields the same
columns. What `read_json` reads, this writes, and the round trip keeps each
value and every inferable type — `int64`, `double`, `bool`, `utf8`, lists,
structs, and `timestamp[s]`. A type the reader has to be told (a narrower
integer, `float32`, a sub-second timestamp) comes back through an explicit
schema.

Strings are escaped and floats written in their shortest round-trip form by
EmberJson, which `read_json` parses with; a NaN or an infinity has no JSON
spelling and is written as `null`. Timestamps and dates are ISO-8601 strings.
"""

from emberjson.teju import write_float
from emberjson.utils import write_escaped_string

from ..arrays import DynArray
from ..dtypes import (
    DynType,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    millisecond,
    microsecond,
    second,
    uint8,
    uint16,
    uint32,
    uint64,
)
from ..io import ByteSink, DynSink, StorageOptions
from ..tabular import RecordBatch, Table
from ..utils.datetime import CivilDate, Epoch, floor_div, write_iso8601

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
comptime _F16 = 10
comptime _F32 = 11
comptime _F64 = 12
comptime _STR = 13
comptime _LSTR = 14
comptime _TS_S = 15
comptime _TS_MS = 16
comptime _TS_US = 17
comptime _TS_NS = 18
comptime _DATE32 = 19
comptime _DATE64 = 20
comptime _LIST = 21
comptime _LLIST = 22
comptime _STRUCT = 23


def _code(dtype: DynType) raises -> Int:
    """The column code for a type the writer can write, or raise."""
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
    elif dtype == float16:
        return _F16
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
    elif dtype.is_date32():
        return _DATE32
    elif dtype.is_date64():
        return _DATE64
    elif dtype.is_list():
        return _LIST
    elif dtype.is_large_list():
        return _LLIST
    elif dtype.is_struct():
        return _STRUCT
    raise Error("JSON writer: unsupported type ", dtype)


struct _Column(Movable):
    """One array and how to write each of its values; a struct's fields and a
    list's items are columns of their own."""

    var code: Int
    var array: DynArray
    var keys: List[String]
    """For a struct: each field's `"name":`, escaped once rather than per
    row."""
    var children: List[_Column]
    """For a struct: its fields, with its slice applied. For a list: its
    items, the whole child array, indexed through the list's offsets."""

    def __init__(out self, array: DynArray) raises:
        var dtype = array.dtype()
        self.code = _code(dtype)
        self.array = array.copy()
        self.keys = []
        self.children = []
        if self.code == _STRUCT:
            ref s = array.as_struct()
            ref fields = dtype.as_struct().fields
            for i in range(len(fields)):
                self.keys.append(_key(fields[i].name))
                self.children.append(_Column(s.field(i)))
        elif self.code == _LIST:
            self.children.append(_Column(array.as_list().values()))
        elif self.code == _LLIST:
            self.children.append(_Column(array.as_large_list().values()))

    # `children: List[_Column]` cycles back through `_Column`.
    def __deinit__(deinit self):
        pass

    def is_valid(self, i: Int) -> Bool:
        return self.code != _NULL and self.array.is_valid(i)

    def write(self, i: Int, mut out: String) raises:
        """The value at row `i` of this column, as JSON."""
        if not self.is_valid(i):
            out.write("null")
            return
        var code = self.code
        ref a = self.array
        if code == _BOOL:
            out.write("true" if a.as_bool().values().test(i) else "false")
        elif code == _I8:
            out.write(a.as_int8().unsafe_get(i))
        elif code == _I16:
            out.write(a.as_int16().unsafe_get(i))
        elif code == _I32:
            out.write(a.as_int32().unsafe_get(i))
        elif code == _I64:
            out.write(a.as_int64().unsafe_get(i))
        elif code == _U8:
            out.write(a.as_uint8().unsafe_get(i))
        elif code == _U16:
            out.write(a.as_uint16().unsafe_get(i))
        elif code == _U32:
            out.write(a.as_uint32().unsafe_get(i))
        elif code == _U64:
            out.write(a.as_uint64().unsafe_get(i))
        elif code == _F16:
            write_float(a.as_float16().unsafe_get(i), out)
        elif code == _F32:
            write_float(a.as_float32().unsafe_get(i), out)
        elif code == _F64:
            write_float(a.as_float64().unsafe_get(i), out)
        elif code == _STR:
            write_escaped_string(a.as_string().unsafe_get(UInt(i)), out)
        elif code == _LSTR:
            write_escaped_string(a.as_large_string().unsafe_get(UInt(i)), out)
        elif code == _TS_S:
            self._timestamp[0](i, out)
        elif code == _TS_MS:
            self._timestamp[3](i, out)
        elif code == _TS_US:
            self._timestamp[6](i, out)
        elif code == _TS_NS:
            self._timestamp[9](i, out)
        elif code == _DATE32:
            self._date(Int(a.as_date32().unsafe_get(i)), out)
        elif code == _DATE64:
            var millis = Int(a.as_date64().unsafe_get(i))
            self._date(floor_div(millis, Epoch.MILLIS_PER_DAY), out)
        elif code == _STRUCT:
            out.write("{")
            for f in range(len(self.children)):
                if f > 0:
                    out.write(",")
                out.write(self.keys[f])
                self.children[f].write(i, out)
            out.write("}")
        else:
            var bounds = a.as_list().child_range(
                i
            ) if code == _LIST else a.as_large_list().child_range(i)
            var start = bounds[0]
            var end = bounds[1]
            out.write("[")
            for j in range(start, end):
                if j > start:
                    out.write(",")
                self.children[0].write(j, out)
            out.write("]")

    def _timestamp[digits: Int](self, i: Int, mut out: String):
        out.write('"')
        write_iso8601[digits](Int(self.array.as_timestamp().unsafe_get(i)), out)
        out.write('"')

    @staticmethod
    def _date(days: Int, mut out: String):
        out.write('"', CivilDate.from_days(days), '"')


def _key(name: String) -> String:
    """`"name":`, escaped."""
    var out = String()
    write_escaped_string(name, out)
    out.write(":")
    return out^


def render_json(batch: RecordBatch) raises -> String:
    """`batch` as newline-delimited JSON: one object per row, each ending in a
    newline."""
    var columns = List[_Column](capacity=len(batch.columns))
    var keys = List[String](capacity=len(batch.columns))
    for i in range(len(batch.columns)):
        columns.append(_Column(batch.columns[i]))
        keys.append(_key(batch.schema.fields[i].name))
    var out = String()
    for row in range(batch.num_rows()):
        out.write("{")
        for c in range(len(columns)):
            if c > 0:
                out.write(",")
            out.write(keys[c])
            columns[c].write(row, out)
        out.write("}\n")
    return out^


struct JsonWriter[S: ByteSink](Movable):
    """Writes batches to a sink as newline-delimited JSON. Nothing is
    published until `close`, which is the sink's contract."""

    var _sink: Self.S

    def __init__(out self, var sink: Self.S):
        self._sink = sink^

    def write(mut self, batch: RecordBatch) raises:
        var text = render_json(batch)
        self._sink.write(text.as_bytes())

    def write(mut self, table: Table) raises:
        for ref batch in table.to_batches():
            self.write(batch)

    def close(mut self) raises:
        self._sink.close()


def write_json(
    table: Table, uri: String, options: StorageOptions = StorageOptions()
) raises:
    """Write `table` as newline-delimited JSON to `uri` — a path or a URL,
    opened like `marrow.parquet.write_table`'s."""
    var writer = JsonWriter[DynSink](DynSink.open(uri, options))
    writer.write(table)
    writer.close()


def write_json(
    batch: RecordBatch, uri: String, options: StorageOptions = StorageOptions()
) raises:
    """Write one batch as newline-delimited JSON to `uri`."""
    var writer = JsonWriter[DynSink](DynSink.open(uri, options))
    writer.write(batch)
    writer.close()
