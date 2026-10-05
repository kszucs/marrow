# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro schemas, parsed from and written to JSON.

`AvroSchema` is the parsed tree; `marrow.avro.mapping` maps it to and from
Arrow. The ids Iceberg puts on a schema -- a record field's `field-id`, an
array's `element-id`, a map's `key-id` and `value-id` -- are parsed into
typed fields here, so nothing downstream reads them out of JSON.
"""

from emberjson import (
    Array as JsonArray,
    Object as JsonObject,
    Value as Json,
)

from ..errors import InvalidError, NotImplementedError


@fieldwise_init
struct AvroKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The type of an Avro schema node."""

    var code: Int

    comptime NULL = Self(0)
    comptime BOOLEAN = Self(1)
    comptime INT = Self(2)
    comptime LONG = Self(3)
    comptime FLOAT = Self(4)
    comptime DOUBLE = Self(5)
    comptime BYTES = Self(6)
    comptime STRING = Self(7)
    comptime RECORD = Self(8)
    comptime ENUM = Self(9)
    comptime ARRAY = Self(10)
    comptime MAP = Self(11)
    comptime UNION = Self(12)
    comptime FIXED = Self(13)

    @staticmethod
    def primitive(name: StringSlice) -> Optional[Self]:
        """The primitive type spelled `name`, or `None`."""
        for i in range(Self.RECORD.code):
            if name == Self(i).name():
                return Self(i)
        return None

    def name(self) -> StaticString:
        var c = self.code
        if c == 0:
            return "null"
        elif c == 1:
            return "boolean"
        elif c == 2:
            return "int"
        elif c == 3:
            return "long"
        elif c == 4:
            return "float"
        elif c == 5:
            return "double"
        elif c == 6:
            return "bytes"
        elif c == 7:
            return "string"
        elif c == 8:
            return "record"
        elif c == 9:
            return "enum"
        elif c == 10:
            return "array"
        elif c == 11:
            return "map"
        elif c == 12:
            return "union"
        return "fixed"

    def is_named(self) -> Bool:
        return self == Self.RECORD or self == Self.ENUM or self == Self.FIXED

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name())


struct AvroSchema(Copyable, Movable, Writable):
    """A parsed Avro schema node.

    One struct for every kind, rather than a variant, so a `List` of them is
    safe to grow. What a field means depends on `kind`: `children` are a
    record's field types, an array's items, a map's values or a union's
    branches; `field_names`, `field_ids` and `field_props` belong to a record,
    `element_id` to an array, `key_id` and `value_id` to a map, `symbols` to an
    enum, `size` to a fixed.

    Attributes this module does not interpret -- `doc`, `default`, `aliases`,
    and the rest -- are kept as JSON text in `props` (for the node) and
    `field_props` (per record field), so a schema writes back out with them.

    A named type referenced by name after its definition is a copy of that
    definition; `to_json` writes its first occurrence in full and the rest by
    name.
    """

    var kind: AvroKind
    var name: String
    """The full name of a record, enum or fixed; empty otherwise."""
    var children: List[AvroSchema]
    var field_names: List[String]
    var field_ids: List[Optional[Int]]
    var field_props: List[Dict[String, String]]
    var element_id: Optional[Int]
    var key_id: Optional[Int]
    var value_id: Optional[Int]
    var symbols: List[String]
    var size: Int
    var logical: String
    """The `logicalType`, or empty."""
    var precision: Int
    var scale: Int
    var props: Dict[String, String]

    def __init__(
        out self, kind: AvroKind, name: String = "", *, logical: String = ""
    ):
        self.kind = kind
        self.name = name
        self.children = List[AvroSchema]()
        self.field_names = List[String]()
        self.field_ids = List[Optional[Int]]()
        self.field_props = List[Dict[String, String]]()
        self.element_id = None
        self.key_id = None
        self.value_id = None
        self.symbols = List[String]()
        self.size = 0
        self.logical = logical
        self.precision = 0
        self.scale = 0
        self.props = Dict[String, String]()

    # `children` makes the type recursive: an explicit destructor keeps it
    # Deinitable, and the fields are still destroyed after it runs.
    def __deinit__(deinit self):
        pass

    # --- construction ---------------------------------------------------

    @staticmethod
    def record(
        name: String,
        var field_names: List[String],
        var children: List[AvroSchema],
        var field_ids: List[Optional[Int]],
        var field_props: List[Dict[String, String]],
    ) -> AvroSchema:
        var s = AvroSchema(AvroKind.RECORD, name)
        s.field_names = field_names^
        s.children = children^
        s.field_ids = field_ids^
        s.field_props = field_props^
        return s^

    @staticmethod
    def array(var items: AvroSchema) -> AvroSchema:
        var s = AvroSchema(AvroKind.ARRAY)
        s.children.append(items^)
        return s^

    @staticmethod
    def map(var values: AvroSchema) -> AvroSchema:
        var s = AvroSchema(AvroKind.MAP)
        s.children.append(values^)
        return s^

    @staticmethod
    def union(var branches: List[AvroSchema]) -> AvroSchema:
        var s = AvroSchema(AvroKind.UNION)
        s.children = branches^
        return s^

    @staticmethod
    def optional(var value: AvroSchema) -> AvroSchema:
        """`["null", value]`."""
        return AvroSchema.union([AvroSchema(AvroKind.NULL), value^])

    @staticmethod
    def fixed(name: String, size: Int) -> AvroSchema:
        var s = AvroSchema(AvroKind.FIXED, name)
        s.size = size
        return s^

    @staticmethod
    def enum_(name: String, var symbols: List[String]) -> AvroSchema:
        var s = AvroSchema(AvroKind.ENUM, name)
        s.symbols = symbols^
        return s^

    def fixed_size(self) -> Int:
        """A `fixed`'s byte width, or -1 for any other kind -- which for a
        decimal means `bytes`."""
        return self.size if self.kind == AvroKind.FIXED else -1

    # --- unions -----------------------------------------------------------

    def null_index(self) -> Int:
        """The branch of a union that is `null`, or -1."""
        if self.kind == AvroKind.UNION:
            for i in range(len(self.children)):
                if self.children[i].kind == AvroKind.NULL:
                    return i
        return -1

    def value_index(self) raises -> Int:
        """The one non-null branch of a union, or -1 when every branch is
        `null`. Raises for a union of several non-null branches."""
        var found = -1
        for i in range(len(self.children)):
            if self.children[i].kind != AvroKind.NULL:
                if found >= 0:
                    raise NotImplementedError(
                        t"avro: a union of several non-null types has no Arrow"
                        t" form: {self}"
                    )
                found = i
        return found

    # --- JSON -------------------------------------------------------------

    @staticmethod
    def parse(text: StringSlice) raises -> AvroSchema:
        """Parse a schema from its JSON text."""
        var json: Json
        try:
            json = Json(parse_bytes=text.as_bytes())
        except e:
            raise InvalidError(t"avro: schema is not valid JSON: {e}")
        var parser = _Parser()
        return parser.parse(json, "")

    def to_json(self) raises -> String:
        """The schema as JSON text."""
        var seen = List[String]()
        return String(self._json(seen))

    def _json(self, mut seen: List[String]) raises -> Json:
        if self.kind.is_named():
            for ref n in seen:
                if n == self.name:
                    return Json(self.name)
            seen.append(self.name)
        if self.kind == AvroKind.UNION:
            var branches = JsonArray()
            for ref c in self.children:
                branches.append(c._json(seen))
            return Json(branches^)
        var bare = (
            len(self.props) == 0
            and self.logical == ""
            and self.kind.code < AvroKind.RECORD.code
        )
        if bare:
            return Json(String(self.kind.name()))
        var o = JsonObject()
        o["type"] = Json(String(self.kind.name()))
        if self.kind.is_named():
            o["name"] = Json(self.name)
        if self.kind == AvroKind.RECORD:
            var fields = JsonArray()
            for i in range(len(self.children)):
                var f = JsonObject()
                f["name"] = Json(self.field_names[i])
                f["type"] = self.children[i]._json(seen)
                _put_id(f, "field-id", self.field_ids[i])
                for entry in self.field_props[i].items():
                    f[entry.key] = _json_text(entry.value)
                fields.append(Json(f^))
            o["fields"] = Json(fields^)
        elif self.kind == AvroKind.ENUM:
            var symbols = JsonArray()
            for ref s in self.symbols:
                symbols.append(Json(s))
            o["symbols"] = Json(symbols^)
        elif self.kind == AvroKind.ARRAY:
            o["items"] = self.children[0]._json(seen)
            _put_id(o, "element-id", self.element_id)
        elif self.kind == AvroKind.MAP:
            o["values"] = self.children[0]._json(seen)
            _put_id(o, "key-id", self.key_id)
            _put_id(o, "value-id", self.value_id)
        elif self.kind == AvroKind.FIXED:
            o["size"] = Json(self.size)
        if self.logical != "":
            o["logicalType"] = Json(self.logical)
            if self.logical == "decimal":
                o["precision"] = Json(self.precision)
                o["scale"] = Json(self.scale)
        for entry in self.props.items():
            o[entry.key] = _json_text(entry.value)
        return Json(o^)

    def write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(self.to_json())
        except e:
            writer.write("<avro schema: ", e, ">")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("AvroSchema(")
        self.write_to(writer)
        writer.write(")")


# ---------------------------------------------------------------------------
# JSON -> AvroSchema
# ---------------------------------------------------------------------------


struct _Parser(Movable):
    """Named types defined so far, by full name, and the records still being
    defined -- a reference to one of those is a recursive type."""

    var named: Dict[String, AvroSchema]
    var pending: List[String]

    def __init__(out self):
        self.named = Dict[String, AvroSchema]()
        self.pending = List[String]()

    def parse(mut self, json: Json, namespace: String) raises -> AvroSchema:
        if json.is_string():
            var prim = AvroKind.primitive(json.string())
            if prim:
                return AvroSchema(prim.value())
            return self._name(json.string(), namespace)
        if json.is_array():
            var branches = List[AvroSchema]()
            for ref b in json.array():
                var branch = self.parse(b, namespace)
                if branch.kind == AvroKind.UNION:
                    raise InvalidError("avro: a union may not contain a union")
                branches.append(branch^)
            return AvroSchema.union(branches^)
        if not json.is_object():
            raise InvalidError(t"avro: not a schema: {json}")
        ref o = json.object()
        if "type" not in o:
            raise InvalidError(t"avro: schema object without a type: {json}")
        ref t = o["type"]
        if not t.is_string():
            # `{"type": {...}}`: a schema nested where a type name belongs.
            return self.parse(t, namespace)
        var type_name = t.string()
        var prim = AvroKind.primitive(type_name)
        var s: AvroSchema
        if prim:
            s = AvroSchema(prim.value())
        elif type_name == "record" or type_name == "error":
            return self._record(o, namespace)
        elif type_name == "enum":
            s = AvroSchema(AvroKind.ENUM, self._define(o, namespace))
            var symbols = _required(o, "symbols")
            if not symbols.is_array():
                raise InvalidError(t"avro: enum symbols are not an array")
            for ref sym in symbols.array():
                if not sym.is_string():
                    raise InvalidError(t"avro: invalid enum symbol: {sym}")
                s.symbols.append(sym.string())
            self.named[s.name] = s.copy()
            _keep_props(o, s, ["type", "name", "namespace", "symbols"])
            return s^
        elif type_name == "array":
            s = AvroSchema.array(self.parse(_required(o, "items"), namespace))
            s.element_id = _id(o, "element-id")
        elif type_name == "map":
            s = AvroSchema.map(self.parse(_required(o, "values"), namespace))
            s.key_id = _id(o, "key-id")
            s.value_id = _id(o, "value-id")
        elif type_name == "fixed":
            s = AvroSchema(AvroKind.FIXED, self._define(o, namespace))
            var size = _required(o, "size")
            if not size.is_int() or size.int() < 0:
                raise InvalidError(t"avro: invalid fixed size: {size}")
            s.size = Int(size.int())
        else:
            return self._name(type_name, namespace)
        _keep_props(
            o,
            s,
            [
                "type",
                "name",
                "namespace",
                "items",
                "values",
                "size",
                "logicalType",
                "element-id",
                "key-id",
                "value-id",
            ],
        )
        if "logicalType" in o and o["logicalType"].is_string():
            s.logical = o["logicalType"].string()
            if s.logical == "decimal":
                s.precision = _int_prop(o, "precision", -1)
                s.scale = _int_prop(o, "scale", 0)
                _ = s.props.pop("precision", "")
                _ = s.props.pop("scale", "")
        if s.kind.is_named():
            self.named[s.name] = s.copy()
        return s^

    def _record(
        mut self, o: JsonObject, namespace: String
    ) raises -> AvroSchema:
        var name = self._define(o, namespace)
        var inner = _namespace_of(name)
        self.pending.append(name)
        var names = List[String]()
        var children = List[AvroSchema]()
        var ids = List[Optional[Int]]()
        var props = List[Dict[String, String]]()
        var fields = _required(o, "fields")
        if not fields.is_array():
            raise InvalidError(
                t"avro: record fields are not an array: {fields}"
            )
        for ref f in fields.array():
            if not f.is_object():
                raise InvalidError(t"avro: record field is not an object: {f}")
            ref fo = f.object()
            var fname = _required(fo, "name")
            if not fname.is_string():
                raise InvalidError(t"avro: invalid field name: {fname}")
            names.append(fname.string())
            children.append(self.parse(_required(fo, "type"), inner))
            ids.append(_id(fo, "field-id"))
            var p = Dict[String, String]()
            for entry in fo.items():
                if (
                    entry.key != "name"
                    and entry.key != "type"
                    and entry.key != "field-id"
                ):
                    p[entry.key] = String(entry.value)
            props.append(p^)
        _ = self.pending.pop()
        var s = AvroSchema.record(name, names^, children^, ids^, props^)
        _keep_props(o, s, ["type", "name", "namespace", "fields"])
        self.named[name] = s.copy()
        return s^

    def _define(self, o: JsonObject, namespace: String) raises -> String:
        """The full name a named type is defined under, checked unique."""
        var name = _required(o, "name")
        if not name.is_string():
            raise InvalidError(t"avro: invalid type name: {name}")
        var full = String(name.string())
        if "." not in full:
            var ns = namespace
            if "namespace" in o:
                ref nso = o["namespace"]
                ns = nso.string() if nso.is_string() else ""
            if ns != "":
                full = ns + "." + full
        if full in self.named or full in self.pending:
            raise InvalidError(t"avro: type '{full}' is defined twice")
        return full^

    def _name(self, name: String, namespace: String) raises -> AvroSchema:
        """A reference to a named type, by full or namespace-relative name."""
        var candidates = List[String]()
        if "." not in name and namespace != "":
            candidates.append(namespace + "." + name)
        candidates.append(name)
        for ref c in candidates:
            for ref p in self.pending:
                if p == c:
                    raise NotImplementedError(
                        t"avro: recursive type '{c}' has no Arrow form"
                    )
            if c in self.named:
                return self.named[c].copy()
        raise InvalidError(t"avro: unknown type '{name}'")


def _required(o: JsonObject, key: String) raises -> Json:
    if key not in o:
        raise InvalidError(t"avro: schema is missing '{key}': {o}")
    return o[key].copy()


def _int_prop(o: JsonObject, key: String, default: Int) raises -> Int:
    if key not in o:
        return default
    ref v = o[key]
    if not v.is_int():
        raise InvalidError(t"avro: '{key}' is not an integer: {v}")
    return Int(v.int())


def _id(o: JsonObject, key: String) raises -> Optional[Int]:
    """An Iceberg id attribute -- an integer, if present."""
    if key not in o:
        return None
    return _int_prop(o, key, 0)


def _put_id(mut o: JsonObject, key: String, id: Optional[Int]):
    if id:
        o[key] = Json(id.value())


def _keep_props(o: JsonObject, mut s: AvroSchema, known: List[String]):
    """Every attribute of `o` not in `known`, as JSON text."""
    for entry in o.items():
        var is_known = False
        for ref k in known:
            if k == entry.key:
                is_known = True
        if not is_known:
            s.props[entry.key] = String(entry.value)


def _json_text(text: String) raises -> Json:
    return Json(parse_string=text)


def _namespace_of(full: String) -> String:
    var dot = full.rfind(".")
    if dot < 0:
        return ""
    return String(full[byte=:dot])


def check_name(name: String, what: StaticString) raises:
    """Avro names match `[A-Za-z_][A-Za-z0-9_]*`, dot-separated for a full
    name."""
    # Java's implementation, among others, refuses anything else.
    var parts = name.split(".")
    for part in parts:
        var bytes = part.as_bytes()
        var ok = len(bytes) > 0
        for i in range(len(bytes)):
            var b = Int(bytes[i])
            var alpha = (b >= ord("A") and b <= ord("Z")) or (
                b >= ord("a") and b <= ord("z")
            )
            var digit = b >= ord("0") and b <= ord("9")
            if not (alpha or b == ord("_") or (digit and i > 0)):
                ok = False
        if not ok:
            raise InvalidError(t"avro: '{name}' is not a valid {what} name")
