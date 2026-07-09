"""Mapping between the Parquet schema (a flat `SchemaElement` list) and Marrow's
Arrow type tree.

`SchemaNode` is the shared tree (leaf / struct / list) with `assemble`/`flatten`
methods; `LeafColumn` is a flat leaf descriptor. The two conversions are objects
that own their traversal state: `_SchemaReader` (Parquet→Arrow, via
`ParsedSchema.from_metadata`) and `_SchemaWriter` (Arrow→Parquet, via
`ParquetSchema.from_arrow`). Covers flat columns, structs, and single-level lists.
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
    # List geometry (NODE_LIST only), in absolute definition levels:
    var list_def: Int  # def at/above which the list itself is non-null
    var element_floor: Int  # def at/above which the list holds an element
    var list_optional: Bool

    def __init__(
        out self,
        kind: Int,
        var field: Field,
        var children: List[SchemaNode],
        leaf_index: Int,
        list_def: Int = 0,
        element_floor: Int = 0,
        list_optional: Bool = False,
    ):
        self.kind = kind
        self.field = field^
        self.children = children^
        self.leaf_index = leaf_index
        self.list_def = list_def
        self.element_floor = element_floor
        self.list_optional = list_optional

    def __del__(deinit self):
        pass

    def first_leaf_index(self) -> Int:
        """The leftmost leaf under this node — all leaves in a list's subtree
        share the same Dremel levels, so any one supplies the list geometry."""
        if self.kind == NODE_LEAF:
            return self.leaf_index
        return self.children[0].first_leaf_index()

    def assemble(self, ref decoded: List[DecodedLeaf]) raises -> AnyArray:
        """Reconstruct this node's Arrow array from the decoded leaf columns."""
        if self.kind == NODE_LEAF:
            return decoded[self.leaf_index].array.copy()
        elif self.kind == NODE_LIST:
            if self.children[0].kind == NODE_LIST:
                raise Error(
                    "parquet: nested lists (list<list<...>>) not supported yet"
                )
            # The element is the list's child — a leaf column or an assembled
            # struct; both are one entry per present element. Any leaf under it
            # carries the shared rep/def levels this list folds into offsets.
            var element = self.children[0].assemble(decoded)
            var li = self.children[0].first_leaf_index()
            return assemble_list(
                element^,
                decoded[li].rep_levels,
                decoded[li].def_levels,
                self.list_def,
                self.element_floor,
                self.list_optional,
            )
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

    def collect_leaf_indices(self, mut out: List[Int]):
        """Append the flat leaf-column indices this node's subtree reads, in the
        same DFS/column-chunk order the schema was parsed in. Used by column
        projection to decide which column chunks to decode."""
        if self.kind == NODE_LEAF:
            out.append(self.leaf_index)
        else:
            for ref c in self.children:
                c.collect_leaf_indices(out)

    def remapped(self, mapping: List[Int]) -> SchemaNode:
        """A copy of this node with every leaf index rewritten through `mapping`
        (original flat index -> compact index), so a projected node assembles
        from a decoded list that holds only the selected columns."""
        var new_children = List[SchemaNode]()
        for ref c in self.children:
            new_children.append(c.remapped(mapping))
        var new_leaf = self.leaf_index
        if self.kind == NODE_LEAF:
            new_leaf = mapping[self.leaf_index]
        return SchemaNode(
            self.kind,
            self.field.copy(),
            new_children^,
            new_leaf,
            self.list_def,
            self.element_floor,
            self.list_optional,
        )


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

    @staticmethod
    def from_metadata(meta: FileMetaData) raises -> ParsedSchema:
        """Parse a Parquet schema into the Arrow schema, flat leaf descriptors,
        and the assembly tree (flat columns, structs, single-level lists)."""
        var reader = _SchemaReader(meta.schema.copy())
        var nodes = List[SchemaNode]()
        var fields = List[Field]()
        for _ in range(meta.schema[0].num_children):
            var node = reader.node(0, 0)
            fields.append(node.field.copy())
            nodes.append(node^)
        return ParsedSchema(
            Schema(fields=fields^), reader.leaves.copy(), nodes^
        )


struct LeafColumn(Copyable, Movable):
    """A single Parquet leaf column and how it maps to an Arrow value type."""

    var name: String
    var dtype: AnyDataType  # Arrow value type of the leaf
    var physical: Int  # PT_*
    var max_def: Int
    var max_rep: Int
    var nullable: Bool
    var rep_floor: Int  # def level at/above which this leaf's value slot exists

    def __init__(
        out self,
        var name: String,
        var dtype: AnyDataType,
        physical: Int,
        max_def: Int,
        max_rep: Int,
        nullable: Bool,
        rep_floor: Int = 0,
    ):
        self.name = name^
        self.dtype = dtype^
        self.physical = physical
        self.max_def = max_def
        self.max_rep = max_rep
        self.nullable = nullable
        self.rep_floor = rep_floor


# ---------------------------------------------------------------------------
# Parquet schema -> Arrow (parse)
# ---------------------------------------------------------------------------


struct _SchemaReader(Movable):
    """Depth-first walk over the flat Parquet `SchemaElement` list. Owns the
    cursor and the leaf accumulator, so the recursion carries no threaded
    out-params. Used via `ParsedSchema.from_metadata`."""

    var elements: List[SchemaElement]
    var idx: Int
    var leaves: List[LeafColumn]

    def __init__(out self, var elements: List[SchemaElement]):
        self.elements = elements^
        self.idx = 1  # schema[0] is the root group
        self.leaves = List[LeafColumn]()

    @staticmethod
    def _time_unit(el: SchemaElement) -> TimeUnit:
        """Arrow TimeUnit for a temporal leaf (logical unit, else converted)."""
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

    @staticmethod
    def _leaf_dtype(el: SchemaElement) raises -> AnyDataType:
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
                or lt == LT_TIMESTAMP
            ):
                return timestamp(
                    Self._time_unit(el),
                    "UTC" if el.logical_utc else String(""),
                )
            elif ct == CT_TIME_MICROS or lt == LT_TIME:
                return time64(Self._time_unit(el))
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

    def node(
        mut self, def_base: Int, rep_base: Int, rep_floor: Int = 0
    ) raises -> SchemaNode:
        """Consume the element at the cursor and return its Arrow node.

        `rep_floor` is the definition level at/above which this element's value
        slot exists — bumped past each enclosing list's repeated group so leaves
        under nested lists read the right present/absent slots."""
        var el = self.elements[self.idx].copy()
        self.idx += 1
        var rep = el.repetition_type
        var d = def_base + (1 if rep == REP_OPTIONAL else 0)
        var r = rep_base + (1 if rep == REP_REPEATED else 0)
        var nullable = rep != REP_REQUIRED

        if el.num_children == 0:
            var dtype = Self._leaf_dtype(el)
            var li = len(self.leaves)
            self.leaves.append(
                LeafColumn(
                    el.name,
                    dtype.copy(),
                    physical=el.type,
                    max_def=d,
                    max_rep=r,
                    nullable=nullable,
                    rep_floor=rep_floor,
                )
            )
            return SchemaNode(
                NODE_LEAF,
                Field(el.name, dtype^, nullable),
                List[SchemaNode](),
                li,
            )

        if el.converted_type == CT_MAP or el.logical_type == LT_MAP:
            raise Error(
                "parquet: map columns not supported yet (column '"
                + el.name
                + "')"
            )

        if el.converted_type == CT_LIST or el.logical_type == LT_LIST:
            # LIST = optional group(LIST) { repeated group { <element> } }. Skip
            # the repeated middle group (adds one def + one rep level) and parse
            # the element as this list's single child. The list holds an element
            # when the leaf def reaches `d + 1` (repeated group present); the
            # list itself is non-null at `d` (its own optional level).
            self.idx += 1
            var elem = self.node(d + 1, r + 1, rep_floor=d + 1)
            var item: AnyDataType = list_(elem.field.dtype.copy())
            var children = List[SchemaNode]()
            children.append(elem^)
            return SchemaNode(
                NODE_LIST,
                Field(el.name, item^, nullable),
                children^,
                -1,
                list_def=d,
                element_floor=d + 1,
                list_optional=rep == REP_OPTIONAL,
            )

        # plain group -> Arrow struct
        var child_nodes = List[SchemaNode]()
        var child_fields = List[Field]()
        for _ in range(el.num_children):
            var cn = self.node(d, r, rep_floor=rep_floor)
            child_fields.append(cn.field.copy())
            child_nodes.append(cn^)
        var dtype = struct_(child_fields^)
        return SchemaNode(
            NODE_STRUCT, Field(el.name, dtype^, nullable), child_nodes^, -1
        )


# ---------------------------------------------------------------------------
# Arrow schema -> Parquet (emit)
# ---------------------------------------------------------------------------


struct _SchemaWriter(Movable):
    """Depth-first walk over Arrow fields, accumulating the Parquet
    `SchemaElement` list and leaf descriptors. Structs are emitted as required
    groups (struct-level nulls are a follow-up). Used via `ParquetSchema`."""

    var elements: List[SchemaElement]
    var leaves: List[LeafColumn]

    def __init__(out self):
        self.elements = List[SchemaElement]()
        self.leaves = List[LeafColumn]()

    @staticmethod
    def _physical(dtype: AnyDataType) raises -> Tuple[Int, Int, Int]:
        """`(physical, converted, logical)` for an Arrow leaf; -1 = absent."""
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

    def emit(mut self, field: Field, def_base: Int) raises -> SchemaNode:
        if field.dtype.is_struct():
            ref st = field.dtype.as_struct()
            var el = SchemaElement()
            el.name = field.name
            el.repetition_type = REP_REQUIRED
            el.num_children = len(st.fields)
            self.elements.append(el^)
            var child_nodes = List[SchemaNode]()
            for ref cf in st.fields:
                child_nodes.append(self.emit(cf, def_base))
            return SchemaNode(NODE_STRUCT, field.copy(), child_nodes^, -1)
        else:
            var phys, conv, logi = Self._physical(field.dtype)
            var el = SchemaElement()
            el.type = phys
            el.name = field.name
            el.repetition_type = (
                REP_OPTIONAL if field.nullable else REP_REQUIRED
            )
            el.converted_type = conv
            el.logical_type = logi
            self.elements.append(el^)
            var li = len(self.leaves)
            self.leaves.append(
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


struct ParquetSchema(Movable):
    """The Parquet side of an Arrow schema: the flat `SchemaElement` list plus
    the leaf descriptors and node tree the writer uses to flatten columns."""

    var elements: List[SchemaElement]
    var leaves: List[LeafColumn]
    var nodes: List[SchemaNode]

    def __init__(
        out self,
        var elements: List[SchemaElement],
        var leaves: List[LeafColumn],
        var nodes: List[SchemaNode],
    ):
        self.elements = elements^
        self.leaves = leaves^
        self.nodes = nodes^

    @staticmethod
    def from_arrow(schema: Schema) raises -> ParquetSchema:
        var writer = _SchemaWriter()
        var root = SchemaElement()
        root.name = "schema"
        root.num_children = len(schema.fields)
        writer.elements.append(root^)
        var nodes = List[SchemaNode]()
        for ref f in schema.fields:
            nodes.append(writer.emit(f, 0))
        return ParquetSchema(
            writer.elements.copy(), writer.leaves.copy(), nodes^
        )
