# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""A data file's columns read as an Iceberg table's current schema.

Iceberg identifies a column by its field id, never by its name: a rename
changes the name and keeps the id, so a data file written before the rename
still holds the column under its old name. Every `Field` carries its id under
`FIELD_ID_KEY`, in the table schema and in the file schema alike, and
`SchemaProjection` resolves each table field against the file by that id
(the spec's "Column Projection"):

- present with the same type: read it, renamed to the table's name;
- present with a promotable type (`int -> long`, `float -> double`,
  `decimal(P, S) -> decimal(P', S)` with `P' >= P`): read it and cast;
- absent: an identity-partition constant if one was supplied, else nulls —
  and a required field with neither is an error, since v3 initial defaults
  are not implemented;
- nested: struct children resolve by id the same way, at any depth, and list
  elements and map keys and values resolve through their container.

A file written without ids is given them by `NameMapping` (the table's
`schema.name-mapping.default`) before projecting; a column no name maps stays
id-less and is not read.
"""

from std.memory import ArcPointer

from emberjson import Value as Json

from ..arrays import ArrayData, DynArray
from ..builders import Int32Builder, nulls
from ..dtypes import (
    DynType,
    Field,
    field_index,
    LargeListType,
    ListType,
    MapType,
    struct_,
)
from ..errors import InvalidError, KeyError, TypeError
from ..kernels.cast import cast
from ..kernels.filter import take
from ..schema import Schema
from ..tabular import RecordBatch

from .document import (
    as_string,
    expect_array,
    expect_member,
    expect_object,
    has,
    optional_int,
    parse_json,
)


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------


struct MappedField(Copyable, Movable):
    """One entry of a name mapping: the names a field may appear under in a
    data file, the id they map to, and the mapping of its children.

    Children follow the spec's naming: a struct's by their own names, a list's
    element as `element`, a map's key and value as `key` and `value` —
    whatever the Arrow child fields happen to be called.
    """

    var field_id: Optional[Int]
    var names: List[String]
    var fields: List[ArcPointer[MappedField]]
    """Behind `ArcPointer` because a struct cannot hold a `List` of itself."""

    def __init__(out self, field_id: Optional[Int], var names: List[String]):
        self.field_id = field_id
        self.names = names^
        self.fields = []

    def __init__(
        out self,
        field_id: Optional[Int],
        var names: List[String],
        var fields: List[MappedField],
    ):
        self.field_id = field_id
        self.names = names^
        self.fields = _shared(fields^)


def _shared(var fields: List[MappedField]) -> List[ArcPointer[MappedField]]:
    var shared = List[ArcPointer[MappedField]](capacity=len(fields))
    for ref f in fields:
        shared.append(ArcPointer(f.copy()))
    return shared^


struct NameMapping(Copyable, Movable):
    """Iceberg's `schema.name-mapping.default`: field ids for a data file
    written without them, assigned by name at each nesting level."""

    var fields: List[ArcPointer[MappedField]]

    def __init__(out self, var fields: List[MappedField]):
        self.fields = _shared(fields^)

    @staticmethod
    def parse(text: StringSlice) raises -> Self:
        """The JSON the table property holds: a list of
        `{"field-id": id, "names": [...], "fields": [...]}`, where the id may
        be absent and `fields` maps a nested type's children."""
        var json = parse_json(text, "name mapping")
        return Self(_mapped_from_json(json))

    def apply(self, schema: Schema) raises -> Schema:
        """`schema` with an id on every field that lacked one and whose name
        the mapping lists at that level. A field that already has an id keeps
        it; a field no name matches stays without one."""
        var fields = List[Field](capacity=len(schema.fields))
        for ref f in schema.fields:
            fields.append(_map_field(f, f.name, self.fields))
        return Schema(fields=fields^, metadata=schema.metadata.copy())


def _mapped_from_json(json: Json) raises -> List[MappedField]:
    expect_array(json, "name mapping")
    var fields = List[MappedField]()
    for ref f in json.array():
        expect_object(f, "mapped field")
        ref o = f.object()
        expect_member(o, "names")
        ref names_json = o["names"]
        expect_array(names_json, "names")
        var names = List[String]()
        for ref n in names_json.array():
            names.append(as_string(n, "name"))
        var children = List[MappedField]()
        if has(o, "fields"):
            children = _mapped_from_json(o["fields"])
        fields.append(
            MappedField(optional_int(o, "field-id"), names^, children^)
        )
    return fields^


def _find_mapped(
    candidates: List[ArcPointer[MappedField]], name: String
) -> Int:
    for i in range(len(candidates)):
        for ref n in candidates[i][].names:
            if n == name:
                return i
    return -1


def _map_field(
    f: Field, name: String, candidates: List[ArcPointer[MappedField]]
) raises -> Field:
    """`f` and its children with mapped ids; `name` is what `f` is looked up
    as — its own name, or `element` / `key` / `value` inside a container."""
    var found = _find_mapped(candidates, name)
    var children = List[ArcPointer[MappedField]]()
    if found >= 0:
        children = candidates[found][].fields.copy()
    var mapped = Field(
        f.name, _map_dtype(f.dtype, children), f.nullable, f.metadata.copy()
    )
    if not f.field_id() and found >= 0 and candidates[found][].field_id:
        return mapped.with_field_id(candidates[found][].field_id.value())
    return mapped^


def _map_dtype(
    dtype: DynType, children: List[ArcPointer[MappedField]]
) raises -> DynType:
    if dtype.is_struct():
        var fields = List[Field]()
        for ref child in dtype.as_struct().fields:
            fields.append(_map_field(child, child.name, children))
        return struct_(fields^)
    elif dtype.is_list():
        return ListType(
            _map_field(dtype.as_list().value_field(), "element", children)
        )
    elif dtype.is_large_list():
        return LargeListType(
            _map_field(dtype.as_large_list().value_field(), "element", children)
        )
    elif dtype.is_map():
        ref m = dtype.as_map()
        var entries = m.entries_field()
        var kv = List[Field]()
        kv.append(_map_field(m.key_field(), "key", children))
        kv.append(_map_field(m.item_field(), "value", children))
        return MapType(
            Field(
                entries.name,
                struct_(kv^),
                entries.nullable,
                entries.metadata.copy(),
            ),
            m.keys_sorted,
        )
    return dtype.copy()


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


struct SchemaProjection(Copyable, Movable):
    """Reads batches of one data file as batches of the table schema.

    Every table field is matched to the file field with its id, at any
    nesting depth. Construction checks that each one resolves, so a type
    that does not promote or a required field the file lacks is reported
    before any data is read.
    """

    var _schema: Schema
    var _file: Schema
    var _constants: Dict[Int, DynArray]

    def __init__(
        out self,
        table_schema: Schema,
        file_schema: Schema,
        var constants: Dict[Int, DynArray] = {},
    ) raises:
        """Resolve every field of `table_schema` against `file_schema` by id.

        `constants` maps a field id to the value of an identity partition on
        it, as a one-row array; it fills that field when the file lacks it,
        and is ignored when the file has it. Raises `TypeError` for a type
        that does not promote and `InvalidError` for a required field the
        file cannot supply.
        """
        self._schema = table_schema
        self._file = file_schema
        self._constants = constants^
        _check_fields(
            table_schema.fields, file_schema.fields, "", self._constants
        )

    def file_columns(self) raises -> List[String]:
        """The top-level file columns the projection reads, in table order."""
        var names = List[String]()
        for ref target in self._schema.fields:
            var i = field_index(self._file.fields, _id(target))
            if i >= 0:
                names.append(self._file.fields[i].name)
        return names^

    def project(self, batch: RecordBatch) raises -> RecordBatch:
        """A batch of the data file — holding at least `file_columns()` —
        as a batch of the table schema, in table field order."""
        var columns = List[DynArray](capacity=len(self._schema.fields))
        for ref target in self._schema.fields:
            var i = field_index(self._file.fields, _id(target))
            if i < 0:
                columns.append(self._fill(target, batch.num_rows()))
            else:
                ref file = self._file.fields[i]
                var index = batch.schema.get_field_index(file.name)
                if index < 0:
                    raise KeyError(
                        t"iceberg projection: the batch has no column"
                        t" '{file.name}'"
                    )
                columns.append(self._read(target, file, batch.column(index)))
        return RecordBatch(self._schema, columns^)

    def _fill(self, target: Field, length: Int) raises -> DynArray:
        """A field the file lacks: its partition constant, or nulls."""
        var id = _id(target)
        if id in self._constants:
            var value = _broadcast(self._constants[id], length)
            if value.dtype() != target.dtype:
                return cast(value, target.dtype)
            return value^
        return nulls(length, target.dtype)

    def _read(
        self, target: Field, file: Field, array: DynArray
    ) raises -> DynArray:
        """`array`, the file's `file` column, as the table's `target`."""
        ref src = file.dtype
        ref dst = target.dtype
        if src == dst:
            return array.copy()
        elif not (dst.is_struct() and src.is_struct()) and not _same_container(
            dst, src
        ):
            # A promotion; construction checked that it is one.
            return cast(array, dst)
        var data = array.to_data()
        var children = List[ArrayData]()
        if dst.is_struct():
            # A struct's children are indexed through its offset.
            var length = data.offset + data.length
            ref sources = src.as_struct().fields
            for ref child in dst.as_struct().fields:
                var i = field_index(sources, _id(child))
                if i < 0:
                    children.append(self._fill(child, length).to_data())
                else:
                    var source = DynArray.from_data(data.children[i].copy())
                    children.append(
                        self._read(child, sources[i], source).to_data()
                    )
        else:
            var source = DynArray.from_data(data.children[0].copy())
            children.append(
                self._read(
                    _child_field(dst), _child_field(src), source
                ).to_data()
            )
        return DynArray.from_data(
            ArrayData(
                dtype=dst.copy(),
                length=data.length,
                nulls=data.nulls,
                offset=data.offset,
                bitmap=data.bitmap,
                buffers=data.buffers.copy(),
                children=children^,
            )
        )


def _id(field: Field) raises -> Int:
    """A table field's id, which construction checked is there."""
    return field.field_id().value()


def _path(parent: String, name: String) -> String:
    if parent == "":
        return name
    return parent + "." + name


def _broadcast(row: DynArray, length: Int) raises -> DynArray:
    """The one-row `row`, repeated `length` times."""
    var indices = Int32Builder(length)
    for _ in range(length):
        indices.unsafe_append(0)
    return take(row, indices.finish())


def _check_fields(
    targets: List[Field],
    sources: List[Field],
    parent: String,
    constants: Dict[Int, DynArray],
) raises:
    """That each target, matched by id among `sources`, can be read."""
    for ref target in targets:
        var path = _path(parent, target.name)
        var id = target.field_id()
        if not id:
            raise InvalidError(
                t"iceberg projection: table field '{path}' has no field id"
            )
        var i = field_index(sources, id.value())
        if i >= 0:
            _check(target, sources[i], path, constants)
        elif not target.nullable and id.value() not in constants:
            raise InvalidError(
                t"iceberg projection: required field '{path}' (id"
                t" {id.value()}) is not in the data file"
            )


def _check(
    target: Field, file: Field, path: String, constants: Dict[Int, DynArray]
) raises:
    """That the file's `file` reads as the table's `target`."""
    ref src = file.dtype
    ref dst = target.dtype
    if src == dst:
        pass
    elif dst.is_struct() and src.is_struct():
        _check_fields(
            dst.as_struct().fields, src.as_struct().fields, path, constants
        )
    elif _same_container(dst, src):
        # A map's entries struct has no id of its own; its key and value do.
        var suffix = "" if dst.is_map() else ".element"
        _check(_child_field(dst), _child_field(src), path + suffix, constants)
    elif not _promotes(src, dst):
        raise TypeError(
            t"iceberg projection: field '{path}' is {src} in the data file,"
            t" which does not promote to {dst}"
        )


def _same_container(dst: DynType, src: DynType) -> Bool:
    """Both a list, both a large list, or both a map."""
    return (
        (dst.is_list() and src.is_list())
        or (dst.is_large_list() and src.is_large_list())
        or (dst.is_map() and src.is_map())
    )


def _child_field(dtype: DynType) -> Field:
    """The one child of a list, large list or map: the element field, or the
    map's entries struct."""
    if dtype.is_list():
        return dtype.as_list().value_field().copy()
    elif dtype.is_large_list():
        return dtype.as_large_list().value_field().copy()
    return dtype.as_map().entries_field()


def _promotes(src: DynType, dst: DynType) raises -> Bool:
    """Iceberg's primitive type promotions."""
    if src.is_int32():
        return dst.is_int64()
    elif src.is_float32():
        return dst.is_float64()
    elif src.is_decimal() and dst.is_decimal():
        var from_ps = _precision_scale(src)
        var to_ps = _precision_scale(dst)
        return to_ps[1] == from_ps[1] and to_ps[0] >= from_ps[0]
    return False


def _precision_scale(dtype: DynType) -> Tuple[Int, Int]:
    """A decimal type's precision and scale, of any width: a Parquet file
    stores a narrow decimal in an INT32 or INT64."""
    if dtype.is_decimal32():
        ref d = dtype.as_decimal32()
        return (d.precision(), d.scale())
    elif dtype.is_decimal64():
        ref d = dtype.as_decimal64()
        return (d.precision(), d.scale())
    elif dtype.is_decimal128():
        ref d = dtype.as_decimal128()
        return (d.precision(), d.scale())
    ref d = dtype.as_decimal256()
    return (d.precision(), d.scale())
