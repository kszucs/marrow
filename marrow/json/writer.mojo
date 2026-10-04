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
from ..errors import NotImplementedError
from ..io import ByteSink, DynSink, StorageOptions
from ..tabular import RecordBatch, Table
from ..utils.datetime import CivilDate, Epoch, Iso8601, floor_div

from .types import JsonType


struct _Column(Movable):
    """One array and how to write each of its values; a struct's fields and a
    list's items are columns of their own."""

    var json_type: JsonType
    var digits: Int
    """For a timestamp: its unit's sub-second digits, 0, 3, 6 or 9."""
    var array: DynArray
    var keys: List[String]
    """For a struct: each field's `"name":`, escaped once rather than per
    row."""
    var children: List[_Column]
    """For a struct: its fields, with its slice applied. For a list: its
    items, the whole child array, indexed through the list's offsets."""

    def __init__(out self, array: DynArray) raises:
        var dtype = array.dtype()
        var json_type = JsonType.of(dtype)
        if not json_type:
            raise NotImplementedError(t"JSON writer: unsupported type {dtype}")
        self.json_type = json_type.value()
        self.digits = (
            dtype.as_timestamp().unit.fraction_digits() if dtype.is_timestamp() else 0
        )
        self.array = array.copy()
        self.keys = []
        self.children = []
        if self.json_type == JsonType.STRUCT:
            ref s = array.as_struct()
            ref fields = dtype.as_struct().fields
            for i in range(len(fields)):
                self.keys.append(Self.key(fields[i].name))
                self.children.append(_Column(s.field(i)))
        elif self.json_type == JsonType.LIST:
            self.children.append(_Column(array.as_list().values()))
        elif self.json_type == JsonType.LARGE_LIST:
            self.children.append(_Column(array.as_large_list().values()))

    # `children: List[_Column]` cycles back through `_Column`.
    def __deinit__(deinit self):
        pass

    @staticmethod
    def key(name: String) -> String:
        """`"name":`, escaped."""
        var out = String()
        write_escaped_string(name, out)
        out.write(":")
        return out^

    def write(self, i: Int, mut out: String) raises:
        """The value at row `i` of this column, as JSON."""
        if not self.array.is_valid(i):
            out.write("null")
            return
        var t = self.json_type
        ref a = self.array
        if t == JsonType.BOOL:
            out.write("true" if a.as_bool().values().test(i) else "false")
        elif t == JsonType.INT8:
            out.write(a.as_int8().unsafe_get(i))
        elif t == JsonType.INT16:
            out.write(a.as_int16().unsafe_get(i))
        elif t == JsonType.INT32:
            out.write(a.as_int32().unsafe_get(i))
        elif t == JsonType.INT64:
            out.write(a.as_int64().unsafe_get(i))
        elif t == JsonType.UINT8:
            out.write(a.as_uint8().unsafe_get(i))
        elif t == JsonType.UINT16:
            out.write(a.as_uint16().unsafe_get(i))
        elif t == JsonType.UINT32:
            out.write(a.as_uint32().unsafe_get(i))
        elif t == JsonType.UINT64:
            out.write(a.as_uint64().unsafe_get(i))
        elif t == JsonType.FLOAT16:
            write_float(a.as_float16().unsafe_get(i), out)
        elif t == JsonType.FLOAT32:
            write_float(a.as_float32().unsafe_get(i), out)
        elif t == JsonType.FLOAT64:
            write_float(a.as_float64().unsafe_get(i), out)
        elif t == JsonType.STRING:
            write_escaped_string(a.as_string().unsafe_get(UInt(i)), out)
        elif t == JsonType.LARGE_STRING:
            write_escaped_string(a.as_large_string().unsafe_get(UInt(i)), out)
        elif t == JsonType.TIMESTAMP:
            var ticks = Int(a.as_timestamp().unsafe_get(i))
            out.write('"')
            comptime for d in range(4):
                comptime digits = 3 * d
                if self.digits == digits:
                    Iso8601[digits].write(ticks, out)
            out.write('"')
        elif t == JsonType.DATE32:
            out.write(
                '"', CivilDate.from_days(Int(a.as_date32().unsafe_get(i))), '"'
            )
        elif t == JsonType.DATE64:
            var millis = Int(a.as_date64().unsafe_get(i))
            out.write(
                '"',
                CivilDate.from_days(floor_div(millis, Epoch.MILLIS_PER_DAY)),
                '"',
            )
        elif t == JsonType.STRUCT:
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
            ) if t == JsonType.LIST else a.as_large_list().child_range(i)
            var start = bounds[0]
            var end = bounds[1]
            out.write("[")
            for j in range(start, end):
                if j > start:
                    out.write(",")
                self.children[0].write(j, out)
            out.write("]")


def render_json(batch: RecordBatch) raises -> String:
    """`batch` as newline-delimited JSON: one object per row, each ending in a
    newline."""
    var rows = _Column(batch.to_struct_array().to_dyn())
    var out = String()
    for row in range(batch.num_rows()):
        rows.write(row, out)
        out.write("\n")
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
