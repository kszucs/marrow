# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Pass 1: what type each column is, read without building anything.

The reader makes two passes over the bytes rather than promoting builders as it
goes. This one walks the tokens and settles a `Shape` per column, and the
second parses into builders of the type it settled on. Promotion then never
has to rewrite a half-built column, and an explicit schema that leaves nothing
to infer skips this pass entirely.

The rules are Arrow C++'s (`arrow/json/converter.cc`, `GetPromotionGraph`):

- a column is `null` until its first non-null value;
- a number is `int64`, widened to `float64` by any non-integer or by an integer
  that does not fit;
- a string is `timestamp[s]` while every string so far parses as one
  (`parse_iso8601`), and `utf8` from the first that does not;
- an array is a `list` of its items' shape, an object a `struct` of its keys'
  in first-seen order;
- any other change of kind is an error, worded as Arrow words it:
  `Column(/a) changed from number to string in row 1`.
"""

from emberjson import Parser

from ..dtypes import (
    DynType,
    Field,
    bool_,
    field,
    float64,
    int64,
    list_,
    null,
    second,
    string,
    struct_,
    timestamp,
)
from ..schema import Schema
from ..utils.datetime import parse_iso8601

comptime NULL = 0
comptime BOOL = 1
comptime INT = 2
comptime FLOAT = 3
comptime STRING = 4
comptime LIST = 5
comptime STRUCT = 6
comptime FIXED = 7
"""A column whose type the explicit schema gives: pass 1 only skips it."""


def family(kind: Int) -> Int:
    """The JSON value kind a column kind accepts: both numbers are `INT`."""
    return INT if kind == FLOAT else kind


def family_name(kind: Int) -> String:
    """The JSON value kind, in the words of Arrow's error messages."""
    var f = family(kind)
    if f == NULL:
        return "null"
    elif f == BOOL:
        return "boolean"
    elif f == INT:
        return "number"
    elif f == STRING:
        return "string"
    elif f == LIST:
        return "array"
    return "object"


def family_of(byte: Byte) raises -> Int:
    """The kind of the value starting with `byte`; a number is `INT`."""
    if byte == Byte(ord('"')):
        return STRING
    elif byte == Byte(ord("{")):
        return STRUCT
    elif byte == Byte(ord("[")):
        return LIST
    elif byte == Byte(ord("t")) or byte == Byte(ord("f")):
        return BOOL
    elif byte == Byte(ord("n")):
        return NULL
    elif byte == Byte(ord("-")) or (
        byte >= Byte(ord("0")) and byte <= Byte(ord("9"))
    ):
        return INT
    raise Error("Invalid value.")


def changed(path: String, was: Int, now: Int, row: Int) -> Error:
    return Error(
        "JSON parse error: Column(",
        path,
        ") changed from ",
        family_name(was),
        " to ",
        family_name(now),
        " in row ",
        row,
    )


def duplicate(path: String, row: Int) -> Error:
    return Error(
        "JSON parse error: Column(", path, ") was specified twice in row ", row
    )


def index_of(names: List[String], name: String) -> Int:
    """Where `name` is in `names`, or -1. A linear scan: objects are narrow,
    and a map would cost more to build per block than it saves."""
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


struct Shape(Copyable, Movable):
    """What pass 1 knows about one column, recursively."""

    var kind: Int
    var timestamp: Bool
    """For `STRING`: every string seen so far parses as `timestamp[s]`."""
    var fixed: DynType
    """For `FIXED`: the explicit type."""
    var nullable: Bool
    var path: String
    """Where the column is, as Arrow's errors spell it: `/a/b`, `/a/[]`."""
    var names: List[String]
    """For `STRUCT`: field names, first-seen order."""
    var children: List[Shape]
    """For `STRUCT`: one per name. For `LIST`: the single item shape."""
    var visits: Int
    """For `STRUCT`: objects seen so far, which stamps its children."""
    var last: Int
    """The parent's `visits` when this column was last seen: a key seen twice
    in one object, or not at all, is a stamp compare rather than a list."""

    def __init__(out self, var path: String):
        self.kind = NULL
        self.timestamp = False
        self.fixed = null
        self.nullable = True
        self.path = path^
        self.names = []
        self.children = []
        self.visits = 0
        self.last = 0

    # `children: List[Shape]` cycles back through `Shape`; an explicit
    # destructor keeps it `Deinitable`, as `StructBuilder` does.
    def __deinit__(deinit self):
        pass

    @staticmethod
    def of(dtype: DynType, var path: String, nullable: Bool = True) -> Self:
        """The shape an explicit type seeds: structs and lists stay open to
        inference inside them, every other type is `FIXED`."""
        var shape = Self(path^)
        shape.nullable = nullable
        if dtype.is_struct():
            shape.kind = STRUCT
            for f in dtype.as_struct().fields:
                shape.add(
                    f.name,
                    Self.of(f.dtype, shape.path + "/" + f.name, f.nullable),
                )
        elif dtype.is_list():
            shape.kind = LIST
            shape.children.append(
                Self.of(dtype.as_list().value_type(), shape.path + "/[]")
            )
        else:
            shape.kind = FIXED
            shape.fixed = dtype.copy()
        return shape^

    @staticmethod
    def of(schema: Schema) -> Self:
        """The row shape an explicit schema seeds."""
        return Self.of(struct_(schema.fields.copy()), "")

    def add(mut self, name: String, var child: Shape):
        self.names.append(name)
        self.children.append(child^)

    def dtype(self) -> DynType:
        """The type this shape settled on."""
        if self.kind == FIXED:
            return self.fixed.copy()
        elif self.kind == BOOL:
            return bool_
        elif self.kind == INT:
            return int64
        elif self.kind == FLOAT:
            return float64
        elif self.kind == STRING:
            if self.timestamp:
                return timestamp(second)
            return string
        elif self.kind == LIST:
            return list_(self.children[0].dtype())
        elif self.kind == STRUCT:
            return struct_(self.fields())
        return null

    def fields(self) -> List[Field]:
        """For a `STRUCT`: its fields, in first-seen order."""
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

    def schema(self) -> Schema:
        """For the row `STRUCT`: the table schema."""
        return Schema(fields=self.fields())


def infer_rows(mut p: Parser, mut root: Shape, mut row: Int) raises:
    """Fold every row of one block into `root`, counting rows into `row`."""
    p.skip_whitespace()
    while p.has_more():
        var byte = p.peek()
        if byte != Byte(ord("{")):
            raise changed("", STRUCT, family_of(byte), row)
        infer_object(p, root, row)
        row += 1
        p.skip_whitespace()


def infer_object(mut p: Parser, mut shape: Shape, row: Int) raises:
    p.expect_open(Byte(ord("{")))
    shape.visits += 1
    var visit = shape.visits
    p.skip_whitespace()
    if p.peek() == Byte(ord("}")):
        p.expect(Byte(ord("}")))
        return
    while True:
        var key = p.expect_string()
        p.expect(Byte(ord(":")))
        var idx = index_of(shape.names, key)
        if idx < 0:
            shape.add(key, Shape(shape.path + "/" + key))
            idx = len(shape.names) - 1
        ref child = shape.children[idx]
        if child.last == visit:
            raise duplicate(child.path, row)
        child.last = visit
        infer_value(p, child, row)
        p.skip_whitespace()
        if p.peek() == Byte(ord(",")):
            p.expect(Byte(ord(",")))
        else:
            p.expect(Byte(ord("}")))
            return


def infer_value(mut p: Parser, mut shape: Shape, row: Int) raises:
    p.skip_whitespace()
    if shape.kind == FIXED:
        p.skip_value()
        return
    var kind = family_of(p.peek())
    if kind == NULL:
        p.expect_null()
        return
    if shape.kind == NULL:
        shape.kind = kind
        if kind == STRING:
            shape.timestamp = True
        elif kind == LIST:
            shape.children.append(Shape(shape.path + "/[]"))
    elif family(shape.kind) != kind:
        raise changed(shape.path, shape.kind, kind, row)

    if kind == INT:
        infer_number(p, shape)
    elif kind == STRING:
        if shape.timestamp:
            var text = p.expect_string()
            shape.timestamp = Bool(parse_iso8601[0](text.as_bytes()))
        else:
            p.skip_value()
    elif kind == BOOL:
        _ = p.expect_bool()
    elif kind == LIST:
        p.expect_open(Byte(ord("[")))
        p.skip_whitespace()
        if p.peek() == Byte(ord("]")):
            p.expect(Byte(ord("]")))
            return
        while True:
            infer_value(p, shape.children[0], row)
            p.skip_whitespace()
            if p.peek() == Byte(ord(",")):
                p.expect(Byte(ord(",")))
            else:
                p.expect(Byte(ord("]")))
                return
    else:
        infer_object(p, shape, row)


def infer_number(mut p: Parser, mut shape: Shape) raises:
    """Widen an `INT` column to `FLOAT` on a fraction, an exponent, or an
    integer outside `int64`; the value is consumed either way.

    An `INT` column parses the token as an integer, which fails on exactly
    those; only then is it re-read as a validated float. A `FLOAT` column only
    validates.
    """
    if shape.kind == INT:
        var start = p.data
        try:
            _ = p.expect_int[DType.int64]()
            return
        except:
            p.data = start
            shape.kind = FLOAT
    _ = p.expect_float_bytes()
