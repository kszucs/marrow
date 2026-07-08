"""Mapping between the Parquet schema (a flat `SchemaElement` list) and Marrow's
Arrow type tree.

Milestone-1 covers flat columns: primitives, string/binary. Each top-level
column is one leaf; nullability maps to the `OPTIONAL`/`REQUIRED` repetition and
a max definition level of 1/0. Struct/list nesting is layered on in follow-ups.
"""

from ..dtypes import (
    AnyDataType,
    Field,
    struct_,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    string,
    binary,
    date32,
    timestamp,
    time32,
    time64,
    list_,
    TimeUnit,
    millisecond,
    microsecond,
    nanosecond,
)
from ..schema import Schema
from ..arrays import AnyArray, StructArray
from .nested import DecodedLeaf, assemble_list
from .format import (
    SchemaElement,
    FileMetaData,
    PT_BOOLEAN,
    PT_INT32,
    PT_INT64,
    PT_FLOAT,
    PT_DOUBLE,
    PT_BYTE_ARRAY,
    PT_FIXED_LEN_BYTE_ARRAY,
    REP_REQUIRED,
    REP_OPTIONAL,
    REP_REPEATED,
    CT_UTF8,
    CT_LIST,
    CT_MAP,
    CT_DATE,
    CT_TIME_MILLIS,
    CT_TIME_MICROS,
    CT_TIMESTAMP_MILLIS,
    CT_TIMESTAMP_MICROS,
    CT_INT_8,
    CT_INT_16,
    CT_INT_32,
    CT_INT_64,
    CT_UINT_8,
    CT_UINT_16,
    CT_UINT_32,
    CT_UINT_64,
    LT_STRING,
    LT_LIST,
    LT_MAP,
    LT_DATE,
    LT_TIME,
    LT_TIMESTAMP,
)

comptime NODE_LEAF: Int = 0
comptime NODE_STRUCT: Int = 1
comptime NODE_LIST: Int = 2


struct SchemaNode(Copyable, Movable):
    """A node of the Arrow type tree, tying nested structure back to flat leaves.

    Leaf nodes carry a `leaf_index` into the flat leaf-column list; struct nodes
    carry child nodes. The reader assembles arrays bottom-up from this tree.
    """

    var kind: Int
    var field: Field
    var children: List[SchemaNode]
    var leaf_index: Int

    def __init__(
        out self,
        kind: Int,
        var field: Field,
        var children: List[SchemaNode],
        leaf_index: Int,
    ):
        self.kind = kind
        self.field = field^
        self.children = children^
        self.leaf_index = leaf_index

    def __del__(deinit self):
        pass

    def assemble(self, ref decoded: List[DecodedLeaf]) raises -> AnyArray:
        """Reconstruct this node's Arrow array from the decoded leaf columns."""
        if self.kind == NODE_LEAF:
            return decoded[self.leaf_index].array.copy()
        elif self.kind == NODE_LIST:
            return assemble_list(decoded[self.children[0].leaf_index])
        elif self.kind == NODE_STRUCT:
            var children = List[AnyArray]()
            var fields = List[Field]()
            for ref c in self.children:
                children.append(c.assemble(decoded))
                fields.append(c.field.copy())
            var out: AnyArray = StructArray.from_arrays(children^, fields, None)
            return out^
        else:
            raise Error("parquet: unsupported schema node kind")

    def flatten(self, col: AnyArray, mut leaf_arrays: List[AnyArray]) raises:
        """Inverse of `assemble`: collect this node's leaf arrays in column order.
        """
        if self.kind == NODE_LEAF:
            leaf_arrays.append(col.copy())
        elif self.kind == NODE_STRUCT:
            ref sa = col.as_struct()
            for i in range(len(self.children)):
                self.children[i].flatten(sa.children[i], leaf_arrays)
        else:
            raise Error("parquet: unsupported schema node kind")


struct ParsedSchema(Movable):
    """Result of parsing a Parquet schema: the Arrow schema, the flat leaf
    descriptors (in column-chunk order), and the assembly tree."""

    var schema: Schema
    var leaves: List[LeafColumn]
    var nodes: List[SchemaNode]

    def __init__(
        out self,
        var schema: Schema,
        var leaves: List[LeafColumn],
        var nodes: List[SchemaNode],
    ):
        self.schema = schema^
        self.leaves = leaves^
        self.nodes = nodes^


struct LeafColumn(Copyable, Movable):
    """A single Parquet leaf column and how it maps to an Arrow value type."""

    var name: String
    var dtype: AnyDataType  # Arrow value type of the leaf
    var physical: Int  # PT_*
    var max_def: Int
    var max_rep: Int
    var nullable: Bool

    def __init__(
        out self,
        var name: String,
        var dtype: AnyDataType,
        physical: Int,
        max_def: Int,
        max_rep: Int,
        nullable: Bool,
    ):
        self.name = name^
        self.dtype = dtype^
        self.physical = physical
        self.max_def = max_def
        self.max_rep = max_rep
        self.nullable = nullable


def _time_unit(el: SchemaElement) -> TimeUnit:
    """The Arrow TimeUnit for a temporal leaf (logical TimeUnit, else converted).
    """
    if el.logical_unit == 1:
        return millisecond
    elif el.logical_unit == 2:
        return microsecond
    elif el.logical_unit == 3:
        return nanosecond
    elif (
        el.converted_type == CT_TIMESTAMP_MILLIS
        or el.converted_type == CT_TIME_MILLIS
    ):
        return millisecond
    else:
        return microsecond


def _leaf_arrow_dtype(el: SchemaElement) raises -> AnyDataType:
    """Arrow value type for a Parquet leaf `SchemaElement`."""
    var pt = el.type
    var ct = el.converted_type
    var lt = el.logical_type
    if pt == PT_BOOLEAN:
        return bool_
    elif pt == PT_INT32:
        if ct == CT_DATE or lt == LT_DATE:
            return date32()
        elif ct == CT_TIME_MILLIS or lt == LT_TIME:
            return time32(millisecond)
        elif ct == CT_INT_8:
            return int8
        elif ct == CT_INT_16:
            return int16
        elif ct == CT_UINT_8:
            return uint8
        elif ct == CT_UINT_16:
            return uint16
        elif ct == CT_UINT_32:
            return uint32
        else:
            return int32
    elif pt == PT_INT64:
        if (
            ct == CT_TIMESTAMP_MILLIS
            or ct == CT_TIMESTAMP_MICROS
            or (lt == LT_TIMESTAMP)
        ):
            return timestamp(
                _time_unit(el), "UTC" if el.logical_utc else String("")
            )
        elif ct == CT_TIME_MICROS or lt == LT_TIME:
            return time64(_time_unit(el))
        elif ct == CT_UINT_64:
            return uint64
        else:
            return int64
    elif pt == PT_FLOAT:
        return float32
    elif pt == PT_DOUBLE:
        return float64
    elif pt == PT_BYTE_ARRAY:
        if ct == CT_UTF8 or lt == LT_STRING:
            return string
        else:
            return binary
    else:
        raise Error("parquet: unsupported physical type " + String(pt))


def _parse_node(
    schema: List[SchemaElement],
    mut idx: Int,
    def_base: Int,
    rep_base: Int,
    mut leaves: List[LeafColumn],
) raises -> SchemaNode:
    ref el = schema[idx]
    idx += 1
    var rep = el.repetition_type
    var d = def_base + (1 if rep == REP_OPTIONAL else 0)
    var r = rep_base + (1 if rep == REP_REPEATED else 0)
    var nullable = rep != REP_REQUIRED

    if el.num_children == 0:
        var dtype = _leaf_arrow_dtype(el)
        var li = len(leaves)
        leaves.append(
            LeafColumn(
                el.name,
                dtype.copy(),
                physical=el.type,
                max_def=d,
                max_rep=r,
                nullable=nullable,
            )
        )
        return SchemaNode(
            NODE_LEAF, Field(el.name, dtype^, nullable), List[SchemaNode](), li
        )

    if el.converted_type == CT_MAP or el.logical_type == LT_MAP:
        raise Error(
            "parquet: map columns not supported yet (column '" + el.name + "')"
        )

    if el.converted_type == CT_LIST or el.logical_type == LT_LIST:
        # LIST = optional group(LIST) { repeated group { <element> } }. Skip the
        # repeated middle group (it adds one def + one rep level) and parse the
        # element as this list's single child.
        idx += 1  # consume the repeated middle group
        var elem = _parse_node(schema, idx, d + 1, r + 1, leaves)
        var item_dtype: AnyDataType = list_(elem.field.dtype.copy())
        var children = List[SchemaNode]()
        children.append(elem^)
        return SchemaNode(
            NODE_LIST, Field(el.name, item_dtype^, nullable), children^, -1
        )

    # plain group → Arrow struct
    var child_nodes = List[SchemaNode]()
    var child_fields = List[Field]()
    for _ in range(el.num_children):
        var cn = _parse_node(schema, idx, d, r, leaves)
        child_fields.append(cn.field.copy())
        child_nodes.append(cn^)
    var dtype = struct_(child_fields^)
    return SchemaNode(
        NODE_STRUCT, Field(el.name, dtype^, nullable), child_nodes^, -1
    )


def parquet_to_arrow(meta: FileMetaData) raises -> ParsedSchema:
    """Parse a Parquet schema into the Arrow schema, flat leaf descriptors, and
    an assembly tree. Supports flat columns and struct nesting."""
    var leaves = List[LeafColumn]()
    var nodes = List[SchemaNode]()
    var fields = List[Field]()
    var idx = 1  # schema[0] is the root group
    var n_top = meta.schema[0].num_children
    for _ in range(n_top):
        var node = _parse_node(meta.schema, idx, 0, 0, leaves)
        fields.append(node.field.copy())
        nodes.append(node^)
    return ParsedSchema(Schema(fields=fields^), leaves^, nodes^)


def _arrow_physical(dtype: AnyDataType) raises -> Tuple[Int, Int, Int]:
    """Return `(physical_type, converted_type, logical_type)` for an Arrow leaf
    value type. -1 denotes an absent annotation."""
    if dtype == bool_:
        return (PT_BOOLEAN, -1, -1)
    elif dtype == int8:
        return (PT_INT32, CT_INT_8, -1)
    elif dtype == int16:
        return (PT_INT32, CT_INT_16, -1)
    elif dtype == int32:
        return (PT_INT32, -1, -1)
    elif dtype == uint8:
        return (PT_INT32, CT_UINT_8, -1)
    elif dtype == uint16:
        return (PT_INT32, CT_UINT_16, -1)
    elif dtype == uint32:
        return (PT_INT32, CT_UINT_32, -1)
    elif dtype == int64:
        return (PT_INT64, -1, -1)
    elif dtype == uint64:
        return (PT_INT64, CT_UINT_64, -1)
    elif dtype == float32:
        return (PT_FLOAT, -1, -1)
    elif dtype == float64:
        return (PT_DOUBLE, -1, -1)
    elif dtype.is_string():
        return (PT_BYTE_ARRAY, CT_UTF8, LT_STRING)
    elif dtype.is_binary():
        return (PT_BYTE_ARRAY, -1, -1)
    else:
        raise Error("parquet: cannot write Arrow type " + String(dtype))


def _emit_node(
    field: Field,
    mut elems: List[SchemaElement],
    mut leaves: List[LeafColumn],
    def_base: Int,
) raises -> SchemaNode:
    if field.dtype.is_struct():
        ref st = field.dtype.as_struct()
        var el = SchemaElement()
        el.name = field.name
        el.repetition_type = REP_REQUIRED  # struct-level nulls not written yet
        el.num_children = len(st.fields)
        elems.append(el^)
        var child_nodes = List[SchemaNode]()
        for ref cf in st.fields:
            child_nodes.append(_emit_node(cf, elems, leaves, def_base))
        return SchemaNode(NODE_STRUCT, field.copy(), child_nodes^, -1)
    else:
        var phys, conv, logi = _arrow_physical(field.dtype)
        var el = SchemaElement()
        el.type = phys
        el.name = field.name
        el.repetition_type = REP_OPTIONAL if field.nullable else REP_REQUIRED
        el.converted_type = conv
        el.logical_type = logi
        elems.append(el^)
        var li = len(leaves)
        leaves.append(
            LeafColumn(
                field.name,
                field.dtype.copy(),
                physical=phys,
                max_def=def_base + (1 if field.nullable else 0),
                max_rep=0,
                nullable=field.nullable,
            )
        )
        return SchemaNode(NODE_LEAF, field.copy(), List[SchemaNode](), li)


def arrow_to_parquet(
    schema: Schema, mut leaves: List[LeafColumn], mut nodes: List[SchemaNode]
) raises -> List[SchemaElement]:
    """Build the Parquet `SchemaElement` list, leaf descriptors, and assembly
    tree from an Arrow `Schema`. Supports flat columns and (required) structs.
    """
    var elems = List[SchemaElement]()
    var root = SchemaElement()
    root.name = "schema"
    root.num_children = len(schema.fields)
    elems.append(root^)
    for ref f in schema.fields:
        nodes.append(_emit_node(f, elems, leaves, 0))
    return elems^
