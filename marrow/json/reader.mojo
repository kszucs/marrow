# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON into Arrow — `read_json` and the streaming `JsonReader`.

The input is read through a `ByteSource` in `Blocks`, each cut at the last
newline inside it, so a local file stays a zero-copy memory map and a remote one
is fetched a block at a time. Within a block, rows are a whitespace-separated
stream of objects that a `JsonCursor` walks with EmberJson's pull `Parser`:
nothing builds a JSON document first.

A read is two passes. Pass 1 settles a `ColumnShape` per column without
building anything; pass 2 parses each block into builders of the type it
settled on, so promotion never rewrites a half-built column, and an explicit
schema that leaves nothing to infer skips pass 1. The rules are Arrow C++'s
(`arrow/json/converter.cc`, `GetPromotionGraph`):

- a column is `null` until its first non-null value;
- a number is `int64`, widened to `float64` by any non-integer or by an integer
  that does not fit;
- a string is `timestamp[s]` while every string so far parses as one
  (`Iso8601`), and `utf8` from the first that does not;
- an array is a `list` of its items' shape, an object a `struct` of its keys'
  in first-seen order;
- any other change of kind is an error, worded as Arrow words it:
  `Column(/a) changed from number to string in row 1`.

`read_json` infers over the whole input, so every block agrees; `JsonReader`,
like Arrow C++'s streaming reader, infers from the first block and is strict
after it. Behaviour follows `pyarrow.json.read_json`, down to the error
messages, with two deliberate differences: a row longer than a block grows the
read instead of failing, and `NaN`/`Infinity` literals are not JSON and are
rejected.
"""

from emberjson import Parser, is_valid_utf8

from ..builders import DynBuilder
from ..dtypes import (
    DynType,
    Field,
    bool_,
    field,
    float64,
    int64,
    ListType,
    null,
    second,
    string,
    struct_,
    timestamp,
)
from ..errors import DynError, InvalidError, NotImplementedError
from ..execution import ExecContext
from ..io import ByteSource, DynSource, Fetched, StorageOptions
from ..schema import Schema
from ..tabular import RecordBatch, Table
from ..utils.datetime import Iso8601

from .types import JsonKind, JsonType


# ---------------------------------------------------------------------------
# Options — what a read is asked to do, as `pyarrow.json` spells it
# ---------------------------------------------------------------------------


@fieldwise_init
struct UnexpectedFieldBehavior(Equatable, ImplicitlyCopyable, Movable):
    """What to do with a key the explicit schema does not name.

    Without an explicit schema every key is unexpected, so the behaviour is
    forced to `INFER`, as in Arrow C++.
    """

    var code: Int

    comptime IGNORE = Self(0)
    """Skip the value."""
    comptime ERROR = Self(1)
    """Raise."""
    comptime INFER = Self(2)
    """Infer a type for it and add the column after the schema's own."""

    @staticmethod
    def parse(name: String) raises InvalidError -> Self:
        """`"ignore"`, `"error"` or `"infer"`, as pyarrow spells them."""
        if name == "ignore":
            return Self.IGNORE
        elif name == "error":
            return Self.ERROR
        elif name == "infer":
            return Self.INFER
        raise InvalidError(
            t"unexpected_field_behavior must be 'ignore', 'error' or 'infer',"
            t" got '{name}'"
        )

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


@fieldwise_init
struct ReadOptions(Copyable, Movable):
    """How the input is read."""

    var block_size: Int
    """Bytes per block. A block ends at the last newline inside it, and a row
    longer than a block grows the read rather than failing, where Arrow C++
    raises "straddling object"."""

    def __init__(out self):
        self.block_size = 1 << 20


struct ParseOptions(Copyable, Movable):
    """How each value is turned into a column."""

    var explicit_schema: Optional[Schema]
    """Columns and types to read, in this order; `None` infers everything."""
    var unexpected_field_behavior: UnexpectedFieldBehavior
    """What a key outside `explicit_schema` does."""

    def __init__(out self):
        self.explicit_schema = None
        self.unexpected_field_behavior = UnexpectedFieldBehavior.INFER

    def __init__(
        out self,
        explicit_schema: Schema,
        unexpected_field_behavior: UnexpectedFieldBehavior = (
            UnexpectedFieldBehavior.INFER
        ),
    ):
        self.explicit_schema = explicit_schema.copy()
        self.unexpected_field_behavior = unexpected_field_behavior

    def behavior(self) -> UnexpectedFieldBehavior:
        """The behaviour in force: `INFER` whenever there is no schema."""
        if not self.explicit_schema:
            return UnexpectedFieldBehavior.INFER
        return self.unexpected_field_behavior


# ---------------------------------------------------------------------------
# Input — blocks of the source, and a cursor over one
# ---------------------------------------------------------------------------


struct Blocks[S: ByteSource](Movable):
    """A source cut into blocks, each ending just past the last newline within
    `block_size` bytes, or at the end of the input.

    A row longer than a block grows that block's read rather than failing,
    where Arrow C++ raises "straddling object". Each read is one `read_ranges`
    into a `Fetched` this owns, and the block is a view of it: on a memory map
    that is zero-copy, and on an object store one fetch per block that is
    released when the next replaces it — `read_at` there would keep every
    fetch for as long as the source lives.
    """

    var _source: Self.S
    var _block_size: Int
    var _start: Int
    var _stop: Int
    var _window: Fetched
    """The current read; the block is its first `_stop - _start` bytes."""
    var _validated: Int
    """Leading bytes pass 1 has read, and so checked for UTF-8."""

    def __init__(out self, var source: Self.S, block_size: Int) raises:
        if block_size <= 0:
            raise InvalidError(
                t"ReadOptions.block_size must be positive, got {block_size}"
            )
        self._source = source^
        self._block_size = block_size
        self._start = 0
        self._stop = 0
        self._window = Fetched()
        self._validated = 0

    def is_empty(self) -> Bool:
        return self._source.size() == 0

    def next(mut self) raises -> Bool:
        """Move to the next block; `False` at the end of the input."""
        var size = self._source.size()
        self._start = self._stop
        if self._start >= size:
            return False
        var want = self._block_size
        while True:
            var n = min(want, size - self._start)
            self._window = self._source.read_ranges(
                [(self._start, n)], ExecContext.serial()
            )
            if self._start + n >= size:
                self._stop = size
                return True
            var end = self._last_newline(n)
            if end >= 0:
                self._stop = self._start + end + 1
                return True
            want *= 2

    def _last_newline(self, n: Int) raises -> Int:
        """The offset of the last newline in the window's first `n` bytes, or
        -1."""
        var data = self._window.span(0)
        var i = n - 1
        while i >= 0 and data[i] != Byte(ord("\n")):
            i -= 1
        return i

    def block(ref self) raises -> Span[UInt8, origin_of(self._window)]:
        """The current block's bytes."""
        return self._window.span(0)[: self._stop - self._start]

    def is_validated(self) -> Bool:
        """Whether pass 1 already checked the current block for UTF-8."""
        return self._stop <= self._validated

    def rewind(mut self):
        """Back to the first block, for pass 2 to read what pass 1 did."""
        self._validated = max(self._validated, self._stop)
        self._start = 0
        self._stop = 0


@fieldwise_init
struct JsonString[origin: ImmOrigin](ImplicitlyCopyable, Movable):
    """A string token as it sits in the block: its bytes, quotes included and
    already validated, and whether they hold an escape.

    Nearly every key and value has none, and is then used in place — matched
    against a field name, appended to a builder, parsed as a timestamp —
    without allocating. An escaped one is decoded by running EmberJson over
    the token alone.
    """

    var quoted: Span[Byte, Self.origin]
    var escaped: Bool

    @always_inline
    def text(self) -> StringSlice[Self.origin]:
        """The characters between the quotes, for an unescaped token."""
        return StringSlice(
            unsafe_from_utf8=self.quoted[1 : len(self.quoted) - 1]
        )

    def decoded(self) raises -> String:
        """The token with its escapes decoded."""
        var parser = Parser[Self.origin](self.quoted)
        return parser.expect_string()

    def to_string(self) raises -> String:
        if self.escaped:
            return self.decoded()
        return String(self.text())

    @always_inline
    def matches(self, name: String) raises -> Bool:
        if self.escaped:
            return self.decoded() == name
        return self.text() == StringSlice(name)

    def index_in(self, names: List[String], hint: Int) raises -> Int:
        """Where this key is among `names`, or -1: `names[hint]` first, since
        rows mostly repeat their key order, then a linear scan — objects are
        narrow, and a map would cost more to build per block than it
        saves."""
        if hint < len(names) and self.matches(names[hint]):
            return hint
        for i in range(len(names)):
            if self.matches(names[i]):
                return i
        return -1

    @always_inline
    def iso8601[fraction_digits: Int](self) raises -> Optional[Int]:
        """The token as ISO-8601 ticks; see `Iso8601.parse`."""
        if self.escaped:
            return Iso8601[fraction_digits].parse(self.decoded().as_bytes())
        return Iso8601[fraction_digits].parse(self.text().as_bytes())


struct JsonCursor[origin: ImmOrigin](Movable):
    """EmberJson's `Parser` over one block, plus the row it is in.

    The row is what every error message ends with, so the errors are built
    here. `ignore_unexpected` is pass 2's: skip a key the schema lacks rather
    than raise.
    """

    var parser: Parser[Self.origin]
    var row: Int
    """The row being read, counted from the start of the input."""
    var ignore_unexpected: Bool

    def __init__(
        out self,
        block: Span[Byte, Self.origin],
        row: Int,
        validate: Bool,
        ignore_unexpected: Bool = False,
    ) raises:
        """A cursor at the start of `block`. EmberJson does not check UTF-8,
        so `validate` does, unless pass 1 already has."""
        if validate and not is_valid_utf8(block):
            raise InvalidError(t"JSON parse error: invalid UTF-8")
        self.parser = Parser[Self.origin](block)
        self.row = row
        self.ignore_unexpected = ignore_unexpected

    # -- structure -----------------------------------------------------------

    @always_inline
    def next_row(mut self) raises -> Bool:
        """Whether another row starts here; a row must be an object."""
        self.parser.skip_whitespace()
        if not self.parser.has_more():
            return False
        var kind = self.kind()
        if kind != JsonKind.OBJECT:
            raise self.changed("", JsonKind.OBJECT, kind)
        return True

    @always_inline
    def kind(mut self) raises -> JsonKind:
        """The kind of the next value, which is not consumed."""
        self.parser.skip_whitespace()
        var kind = JsonKind.of(self.parser.peek())
        if not kind:
            raise InvalidError(
                t"JSON parse error: Invalid value. in row {self.row}"
            )
        return kind.value()

    @always_inline
    def enter_object(mut self) raises -> Bool:
        """Consume `{`. `False` for an empty object, whose `}` is consumed
        too; otherwise read `key()`s until `next_member()` is `False`."""
        return self._enter(Byte(ord("{")), Byte(ord("}")))

    @always_inline
    def key(mut self) raises -> JsonString[Self.origin]:
        """A member's key and its `:`."""
        var key = self.string()
        self.parser.expect(Byte(ord(":")))
        return key

    @always_inline
    def next_member(mut self) raises -> Bool:
        """After a member's value: `True` past a `,`, `False` past the `}`."""
        return self._next(Byte(ord("}")))

    @always_inline
    def enter_array(mut self) raises -> Bool:
        """Consume `[`. `False` for an empty array, whose `]` is consumed
        too; otherwise read items until `next_item()` is `False`."""
        return self._enter(Byte(ord("[")), Byte(ord("]")))

    @always_inline
    def next_item(mut self) raises -> Bool:
        """After an item: `True` past a `,`, `False` past the `]`."""
        return self._next(Byte(ord("]")))

    @always_inline
    def _enter(mut self, open: Byte, close: Byte) raises -> Bool:
        self.parser.expect_open(open)
        self.parser.skip_whitespace()
        if self.parser.peek() == close:
            self.parser.expect(close)
            return False
        return True

    @always_inline
    def _next(mut self, close: Byte) raises -> Bool:
        self.parser.skip_whitespace()
        if self.parser.peek() == Byte(ord(",")):
            self.parser.expect(Byte(ord(",")))
            return True
        self.parser.expect(close)
        return False

    # -- values --------------------------------------------------------------

    @always_inline
    def skip(mut self) raises:
        self.parser.skip_value()

    @always_inline
    def null(mut self) raises:
        self.parser.expect_null()

    @always_inline
    def boolean(mut self) raises -> Bool:
        return self.parser.expect_bool()

    @always_inline
    def string(mut self) raises -> JsonString[Self.origin]:
        """The string at the cursor, validated but not decoded."""
        self.parser.skip_whitespace()
        if self.parser.peek() != Byte(ord('"')):
            # Not a string: EmberJson raises, worded as it words one.
            _ = self.parser.expect_string()
        var quoted = self.parser.expect_string_bytes()
        var inner = StringSlice(unsafe_from_utf8=quoted[1 : len(quoted) - 1])
        return JsonString(quoted, inner.find("\\") >= 0)

    @always_inline
    def is_integer(mut self) raises -> Bool:
        """Consume a number; whether it is an integer that fits `int64`. A
        fraction, an exponent or a wider integer is re-read as a validated
        float."""
        var start = self.parser.data
        try:
            _ = self.parser.expect_int[DType.int64]()
            return True
        except:
            self.parser.data = start
            _ = self.parser.expect_float_bytes()
            return False

    @always_inline
    def skip_number(mut self) raises:
        """Consume and validate a number."""
        _ = self.parser.expect_float_bytes()

    def number[dt: DType](mut self, dtype: DynType) raises -> Scalar[dt]:
        """The number here as `dt`, or Arrow's wording for one a column of
        `dtype` cannot hold."""
        var start = self.parser.data
        try:
            comptime if dt.is_floating_point():
                return self.parser.expect_float[dt]()
            else:
                return self.parser.expect_int[dt]()
        except:
            self.parser.data = start
            var text = StringSlice(
                unsafe_from_utf8=self.parser.expect_float_bytes()
            )
            raise InvalidError(
                t"JSON parse error: Failed to convert JSON to {dtype}, couldn't"
                t" parse:{text}"
            )

    def timestamp[
        fraction_digits: Int
    ](mut self, dtype: DynType) raises -> Int64:
        """The ISO-8601 string here as ticks of `dtype`'s unit."""
        var token = self.string()
        var ticks = token.iso8601[fraction_digits]()
        if not ticks:
            var text = token.to_string()
            raise InvalidError(
                t"JSON parse error: Failed to convert JSON to {dtype},"
                t" couldn't parse:{text}"
            )
        return Int64(ticks.value())

    # -- errors, worded as Arrow words them ----------------------------------

    def changed(
        self, path: String, was: JsonKind, now: JsonKind
    ) -> InvalidError:
        return InvalidError(
            t"JSON parse error: Column({path}) changed from {was} to {now} in"
            t" row {self.row}"
        )

    def duplicate(self, path: String) -> InvalidError:
        return InvalidError(
            t"JSON parse error: Column({path}) was specified twice in row"
            t" {self.row}"
        )

    @staticmethod
    def unexpected() -> InvalidError:
        return InvalidError(t"JSON parse error: unexpected field")

    def missing(self, path: String, absent: Bool) -> InvalidError:
        """A required field absent from its object, or present as null."""
        var what = "absent" if absent else "null"
        return InvalidError(
            t"JSON parse error: Column({path}): a required field was {what} in"
            t" row {self.row}"
        )

    def located(self, e: Error) -> DynError:
        """`e` as the read reports it: a tokenizer error is worded and located
        the way Arrow's are; an error raised above, which carries a kind,
        already is."""
        var err = DynError(e)
        if err.kind:
            return err^
        return InvalidError(t"JSON parse error: {e} in row {self.row}")


# ---------------------------------------------------------------------------
# Pass 1 — the schema
# ---------------------------------------------------------------------------


struct ColumnShape(Copyable, Movable):
    """What pass 1 knows about one column, recursively."""

    var kind: JsonKind
    """The kind of every non-null value so far; `NULL` until the first."""
    var float: Bool
    """For a `NUMBER`: some value was not an `int64`."""
    var timestamp: Bool
    """For a `STRING`: every string so far parses as `timestamp[s]`."""
    var fixed: Optional[DynType]
    """The explicit, non-nested type, whose values this pass only skips."""
    var nullable: Bool
    var path: String
    """Where the column is, as Arrow's errors spell it: `/a/b`, `/a/[]`."""
    var names: List[String]
    """For an `OBJECT`: field names, first-seen order."""
    var children: List[ColumnShape]
    """For an `OBJECT`: one per name. For an `ARRAY`: the single item shape."""
    var item_name: String
    """For an `ARRAY`: the item field's name — `item` unless an explicit list
    type names it otherwise."""
    var visits: Int
    """For an `OBJECT`: objects seen so far, which stamps its children."""
    var last: Int
    """The parent's `visits` when this column was last seen: a key seen twice
    in one object, or not at all, is a stamp compare rather than a list."""

    def __init__(out self, var path: String):
        self.kind = JsonKind.NULL
        self.float = False
        self.timestamp = False
        self.fixed = None
        self.nullable = True
        self.path = path^
        self.names = []
        self.children = []
        self.item_name = "item"
        self.visits = 0
        self.last = 0

    # `children: List[ColumnShape]` cycles back through `ColumnShape`; an
    # explicit destructor keeps it `Deinitable`, as `StructBuilder` does.
    def __deinit__(deinit self):
        pass

    @staticmethod
    def of(dtype: DynType, var path: String, nullable: Bool = True) -> Self:
        """The shape an explicit type seeds: structs and lists stay open to
        inference inside them, every other type is fixed."""
        var shape = Self(path^)
        shape.nullable = nullable
        if dtype.is_struct():
            shape.kind = JsonKind.OBJECT
            for f in dtype.as_struct().fields:
                shape.add(
                    f.name,
                    Self.of(f.dtype, shape.path + "/" + f.name, f.nullable),
                )
        elif dtype.is_list():
            shape.kind = JsonKind.ARRAY
            ref item = dtype.as_list().item[]
            shape.item_name = item.name
            shape.children.append(
                Self.of(item.dtype, shape.path + "/[]", item.nullable)
            )
        else:
            shape.fixed = dtype.copy()
        return shape^

    @staticmethod
    def infer[
        S: ByteSource
    ](
        mut blocks: Blocks[S], parse_options: ParseOptions, *, all_blocks: Bool
    ) raises -> Schema:
        """The schema a read settles on: the explicit one when nothing is left
        to infer, otherwise inferred over the first block, or over every block
        when `all_blocks`. `blocks` is left rewound, remembering how far it
        was checked for UTF-8."""
        if blocks.is_empty():
            raise InvalidError(t"Empty JSON file")
        if parse_options.behavior() != UnexpectedFieldBehavior.INFER:
            return parse_options.explicit_schema.value().copy()
        var root = Self.of(
            struct_(
                parse_options.explicit_schema.value().fields.copy()
            ) if parse_options.explicit_schema else struct_(List[Field]()),
            "",
        )
        var row = 0
        while blocks.next():
            var cur = JsonCursor(blocks.block(), row, validate=True)
            try:
                while cur.next_row():
                    root.fold_object(cur)
                    cur.row += 1
            except e:
                raise cur.located(e)
            row = cur.row
            if not all_blocks:
                break
        blocks.rewind()
        return Schema(fields=root.fields())

    def add(mut self, name: String, var child: ColumnShape):
        self.names.append(name)
        self.children.append(child^)

    def dtype(self) -> DynType:
        """The type this shape settled on."""
        if self.fixed:
            return self.fixed.value().copy()
        elif self.kind == JsonKind.BOOL:
            return bool_
        elif self.kind == JsonKind.NUMBER:
            return float64 if self.float else int64
        elif self.kind == JsonKind.STRING:
            if self.timestamp:
                return timestamp(second)
            return string
        elif self.kind == JsonKind.ARRAY:
            ref item = self.children[0]
            return ListType(field(self.item_name, item.dtype(), item.nullable))
        elif self.kind == JsonKind.OBJECT:
            return struct_(self.fields())
        return null

    def fields(self) -> List[Field]:
        """For an `OBJECT`: its fields, in first-seen order."""
        var out = List[Field](capacity=len(self.names))
        for i in range(len(self.names)):
            out.append(
                field(
                    self.names[i],
                    self.children[i].dtype(),
                    self.children[i].nullable,
                )
            )
        return out^

    def fold_object(mut self, mut cur: JsonCursor) raises:
        """Fold the object at the cursor into this `OBJECT` shape."""
        self.visits += 1
        var visit = self.visits
        if not cur.enter_object():
            return
        var hint = 0
        while True:
            var key = cur.key()
            var idx = key.index_in(self.names, hint)
            if idx < 0:
                var name = key.to_string()
                self.add(name, ColumnShape(self.path + "/" + name))
                idx = len(self.names) - 1
            hint = idx + 1
            ref child = self.children[idx]
            if child.last == visit:
                raise cur.duplicate(child.path)
            child.last = visit
            child.fold_value(cur)
            if not cur.next_member():
                return

    def fold_value(mut self, mut cur: JsonCursor) raises:
        """Fold the value at the cursor into this shape, widening it by
        Arrow's rules."""
        if self.fixed:
            cur.skip()
            return
        var kind = cur.kind()
        if kind == JsonKind.NULL:
            cur.null()
            return
        if self.kind == JsonKind.NULL:
            self.kind = kind
            if kind == JsonKind.STRING:
                self.timestamp = True
            elif kind == JsonKind.ARRAY:
                self.children.append(ColumnShape(self.path + "/[]"))
        elif self.kind != kind:
            raise cur.changed(self.path, self.kind, kind)

        if kind == JsonKind.NUMBER:
            if self.float:
                cur.skip_number()
            else:
                self.float = not cur.is_integer()
        elif kind == JsonKind.STRING:
            if self.timestamp:
                self.timestamp = Bool(cur.string().iso8601[0]())
            else:
                cur.skip()
        elif kind == JsonKind.BOOL:
            _ = cur.boolean()
        elif kind == JsonKind.ARRAY:
            if cur.enter_array():
                while True:
                    self.children[0].fold_value(cur)
                    if not cur.next_item():
                        return
        else:
            self.fold_object(cur)


# ---------------------------------------------------------------------------
# Pass 2 — the batches
# ---------------------------------------------------------------------------


struct _Slot(Movable):
    """Pass 2's view of one column: its builder and, for a nested type, the
    slots of its children, which share builders with the parent's."""

    var json_type: JsonType
    var digits: Int
    """For a timestamp: its unit's sub-second digits, 0, 3, 6 or 9."""
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
        var json_type = JsonType.of(dtype)
        if not json_type or not json_type.value().is_readable():
            raise NotImplementedError(
                t"JSON conversion to {dtype} is not supported"
            )
        self.json_type = json_type.value()
        self.digits = (
            dtype.as_timestamp().unit.fraction_digits() if dtype.is_timestamp() else 0
        )
        self.dtype = dtype.copy()
        self.nullable = nullable
        self.path = path^
        self.builder = builder
        self.names = []
        self.children = []
        self.visits = 0
        self.last = 0
        if self.json_type == JsonType.LIST:
            ref item = dtype.as_list().item[]
            self.children.append(
                _Slot(
                    item.dtype,
                    item.nullable,
                    self.path + "/[]",
                    builder.as_list().values(),
                )
            )
        elif self.json_type == JsonType.STRUCT:
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

    def append_null(mut self, cur: JsonCursor, absent: Bool) raises:
        """A null — given, or `absent` from its object — unless the field is
        required."""
        if not self.nullable:
            raise cur.missing(self.path, absent)
        self._pad_null()

    def _pad_null(mut self) raises:
        """A null here, and in every child of a struct: `StructBuilder` leaves
        its children to the caller."""
        self.builder.append_null()
        if self.json_type == JsonType.STRUCT:
            for i in range(len(self.children)):
                self.children[i]._pad_null()

    def append_value(mut self, mut cur: JsonCursor) raises:
        """The value at the cursor, into this column."""
        var kind = cur.kind()
        if kind == JsonKind.NULL:
            cur.null()
            self.append_null(cur, absent=False)
            return
        if self.json_type.kind() != kind:
            raise cur.changed(self.path, self.json_type.kind(), kind)
        var t = self.json_type
        if t == JsonType.BOOL:
            self.builder.as_bool().append(cur.boolean())
        elif kind == JsonKind.NUMBER:
            self._append_number(cur)
        elif t == JsonType.STRING:
            var token = cur.string()
            if token.escaped:
                self.builder.as_string().append(token.decoded())
            else:
                self.builder.as_string().append(token.text())
        elif t == JsonType.LARGE_STRING:
            var token = cur.string()
            if token.escaped:
                self.builder.as_large_string().append(token.decoded())
            else:
                self.builder.as_large_string().append(token.text())
        elif t == JsonType.TIMESTAMP:
            comptime for i in range(4):
                comptime digits = 3 * i
                if self.digits == digits:
                    self.builder.as_timestamp().append(
                        cur.timestamp[digits](self.dtype)
                    )
        elif t == JsonType.LIST:
            if cur.enter_array():
                while True:
                    self.children[0].append_value(cur)
                    if not cur.next_item():
                        break
            self.builder.as_list().append_valid()
        else:
            self.append_object(cur)

    def append_object(mut self, mut cur: JsonCursor) raises:
        """The object at the cursor, into this struct column: a key the schema
        lacks is skipped or an error, a field the object lacks is null."""
        self.visits += 1
        var visit = self.visits
        if cur.enter_object():
            var hint = 0
            while True:
                var key = cur.key()
                var idx = key.index_in(self.names, hint)
                hint = idx + 1
                if idx < 0:
                    if not cur.ignore_unexpected:
                        raise cur.unexpected()
                    cur.skip()
                else:
                    ref child = self.children[idx]
                    if child.last == visit:
                        raise cur.duplicate(child.path)
                    child.last = visit
                    child.append_value(cur)
                if not cur.next_member():
                    break
        for i in range(len(self.children)):
            if self.children[i].last != visit:
                self.children[i].append_null(cur, absent=True)
        self.builder.as_struct().append_valid()

    def _append_number(mut self, mut cur: JsonCursor) raises:
        ref b = self.builder
        var t = self.json_type
        if t == JsonType.INT8:
            b.as_int8().append(cur.number[DType.int8](self.dtype))
        elif t == JsonType.INT16:
            b.as_int16().append(cur.number[DType.int16](self.dtype))
        elif t == JsonType.INT32:
            b.as_int32().append(cur.number[DType.int32](self.dtype))
        elif t == JsonType.INT64:
            b.as_int64().append(cur.number[DType.int64](self.dtype))
        elif t == JsonType.UINT8:
            b.as_uint8().append(cur.number[DType.uint8](self.dtype))
        elif t == JsonType.UINT16:
            b.as_uint16().append(cur.number[DType.uint16](self.dtype))
        elif t == JsonType.UINT32:
            b.as_uint32().append(cur.number[DType.uint32](self.dtype))
        elif t == JsonType.UINT64:
            b.as_uint64().append(cur.number[DType.uint64](self.dtype))
        elif t == JsonType.FLOAT32:
            b.as_float32().append(cur.number[DType.float32](self.dtype))
        else:
            b.as_float64().append(cur.number[DType.float64](self.dtype))


struct JsonReader[S: ByteSource](Movable):
    """Newline-delimited JSON, one `RecordBatch` per block.

    Mirrors Arrow C++'s streaming reader: the schema is the explicit one when
    nothing is left to infer, and otherwise is inferred from the *first* block
    and fixed from then on. A later block with a key the schema lacks, or a
    value its type cannot hold, raises rather than widening what earlier
    batches already committed to.
    """

    var schema: Schema
    var _blocks: Blocks[Self.S]
    var _ignore_unexpected: Bool
    var _row: Int

    def __init__(
        out self,
        var source: Self.S,
        read_options: ReadOptions = ReadOptions(),
        parse_options: ParseOptions = ParseOptions(),
    ) raises:
        var blocks = Blocks(source^, read_options.block_size)
        var schema = ColumnShape.infer(blocks, parse_options, all_blocks=False)
        self = Self(blocks^, schema, parse_options)

    def __init__(
        out self,
        var blocks: Blocks[Self.S],
        schema: Schema,
        parse_options: ParseOptions,
    ):
        """A reader of `schema` over `blocks`, as `ColumnShape.infer` left
        them."""
        self.schema = schema.copy()
        self._blocks = blocks^
        # After pass 1 has run with `INFER` there is no key left to meet, so
        # meeting one is an error; only an explicit `IGNORE` skips.
        self._ignore_unexpected = (
            parse_options.behavior() == UnexpectedFieldBehavior.IGNORE
        )
        self._row = 0

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The next block's rows, or `None` at the end of the input."""
        if not self._blocks.next():
            return None
        var dtype: DynType = struct_(self.schema.fields.copy())
        var builder = DynBuilder(dtype)
        var root = _Slot(dtype, False, "", builder)
        var cur = JsonCursor(
            self._blocks.block(),
            self._row,
            validate=not self._blocks.is_validated(),
            ignore_unexpected=self._ignore_unexpected,
        )
        try:
            while cur.next_row():
                root.append_object(cur)
                cur.row += 1
        except e:
            raise cur.located(e)
        self._row = cur.row
        var rows = builder.finish()
        return RecordBatch(self.schema, rows.as_struct().children.copy())

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
    var blocks = Blocks(source^, read_options.block_size)
    var schema = ColumnShape.infer(blocks, parse_options, all_blocks=True)
    var reader = JsonReader[S](blocks^, schema, parse_options)
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
