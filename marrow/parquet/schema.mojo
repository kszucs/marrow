"""Mapping between the Parquet schema (a flat `SchemaElement` list) and Marrow's
Arrow type tree.

`SchemaMapping` is the bridge, built in either direction (`from_parquet` /
`from_arrow`); `SchemaNode` is the shared assembly tree (leaf / struct / list)
with `assemble` / `flatten`; `LeafColumn` is a flat leaf descriptor; `Projection`
is a read plan; `DecodedLeaf` is the reader's per-column output. Covers flat
columns, (nullable) structs, and nested lists. Each node carries its Dremel
geometry (`non_null_def`, `child_def`) computed once during the parse walk so
`assemble` stays a clean recursive walk.
"""

from .. import dtypes as dt
from ..schema import Schema
from ..arrays import AnyArray, StructArray, BoolArray, ListArray
from ..builders import BoolBuilder, PrimitiveBuilder
from .format import (
    SchemaElement,
    FileMetaData,
    PhysicalType,
    Repetition,
    ConvertedType,
    LogicalType,
)

comptime NODE_LEAF: Int = 0
comptime NODE_STRUCT: Int = 1
comptime NODE_LIST: Int = 2


struct NodeGeom(Copyable, Movable):
    """Dremel geometry for a schema node, in absolute definition/repetition
    levels — computed once during schema parsing so the assembler never
    re-derives thresholds. A struct uses `non_null_def` + `optional`; a list uses
    all fields, which lets its offset scan reconstruct any nesting depth on its
    own (no dependence on parent levels)."""

    var non_null_def: Int  # def at/above which this node is non-null
    var optional: Bool  # whether this node can be null
    var rep_level: Int  # NODE_LIST: repetition level of its repeated group
    var child_def: Int  # NODE_LIST: def at/above which it holds a child element
    var slot_def: Int  # NODE_LIST: def at/above which this list exists as an entry

    def __init__(
        out self,
        non_null_def: Int = 0,
        optional: Bool = False,
        rep_level: Int = 0,
        child_def: Int = 0,
        slot_def: Int = 0,
    ):
        self.non_null_def = non_null_def
        self.optional = optional
        self.rep_level = rep_level
        self.child_def = child_def
        self.slot_def = slot_def


struct SchemaNode(Copyable, Movable):
    """A node of the Arrow type tree, tying nested structure back to flat leaves.

    Leaf nodes carry a `leaf_index` into the flat leaf-column list; struct/list
    nodes carry child nodes and their `geom`. The reader assembles arrays
    bottom-up from this tree.
    """

    var kind: Int
    var field: dt.Field
    var children: List[SchemaNode]
    var leaf_index: Int
    var geom: NodeGeom

    def __init__(
        out self,
        kind: Int,
        var field: dt.Field,
        var children: List[SchemaNode],
        leaf_index: Int,
        var geom: NodeGeom = NodeGeom(),
    ):
        self.kind = kind
        self.field = field^
        self.children = children^
        self.leaf_index = leaf_index
        self.geom = geom^

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
            # Assemble the element (leaf, struct, or another list) — one entry
            # per present element — then fold this list's own offsets over those
            # entries. The recursion composes to any depth because each list's
            # geom carries every threshold its scan needs.
            var element = self.children[0].assemble(decoded)
            var li = self.first_leaf_index()
            return self._fold_list_offsets(
                element^, decoded[li].rep_levels, decoded[li].def_levels
            )
        elif self.kind == NODE_STRUCT:
            var children = List[AnyArray]()
            var fields = List[dt.Field]()
            for ref c in self.children:
                children.append(c.assemble(decoded))
                fields.append(c.field.copy())
            # A nullable struct is null wherever a representative leaf's def is
            # below the struct's present level. One validity bit per struct entry
            # — the records where this struct has a slot (`rep <= rep_level` and
            # `def >= slot_def`) — so it aligns with the child arrays at any
            # position (top level, holding a list, or as a list element).
            var mask: Optional[BoolArray] = None
            if self.geom.optional:
                var li = self.first_leaf_index()
                ref defs = decoded[li].def_levels
                ref reps = decoded[li].rep_levels
                if len(defs) > 0:
                    var mb = BoolBuilder(len(defs))
                    var any_null = False
                    for i in range(len(defs)):
                        var r = Int(reps[i]) if len(reps) > 0 else 0
                        var d = Int(defs[i])
                        if r <= self.geom.rep_level and (
                            d >= self.geom.slot_def
                        ):
                            var is_null = d < self.geom.non_null_def
                            mb.append(is_null)
                            any_null = any_null or is_null
                    if any_null:
                        mask = mb.finish()
            var out: AnyArray = StructArray.from_arrays(
                children^, fields, mask^
            )
            return out^
        else:
            raise Error("parquet: unsupported schema node kind")

    def _fold_list_offsets(
        self,
        var element: AnyArray,
        rep_levels: List[Int32],
        def_levels: List[Int32],
    ) raises -> AnyArray:
        """Fold a decoded element array + its Dremel levels into an Arrow
        `ListArray` for this (list) node, using only its own `geom`, to any
        nesting depth.

        Reading one leaf's `(rep, def)` records left to right:

        - a **new instance** of this list begins where `rep < rep_level` (the
          repetition is shallower than this list, so its parent moved on) *and*
          `def >= slot_def` (this list is actually reached — otherwise an
          ancestor is empty/null and this list has no entry here);
        - a **new child element** (one entry of `element`) is added where
          `rep <= rep_level` (a repetition at this level or shallower starts a
          fresh element) *and* `def >= child_def` (the element is present,
          not an empty/null structural slot);
        - the instance is **null** where `def < non_null_def`.

        Because every threshold lives in `geom`, the scan needs nothing from the
        parent, so nested lists compose by recursion: the child array is itself
        an assembled (possibly nested) list.
        """
        ref geom = self.geom
        var n = len(rep_levels)
        var offsets = PrimitiveBuilder[dt.Int32Type](n + 1)
        var mask = BoolBuilder(n)
        var any_null = False
        var child_idx = 0
        var started = False
        offsets.append(Int32(0))
        for i in range(n):
            var r = Int(rep_levels[i])
            var d = Int(def_levels[i])
            if r < geom.rep_level and d >= geom.slot_def:
                if started:
                    offsets.append(Int32(child_idx))  # close previous instance
                started = True
                var is_null = geom.optional and d < geom.non_null_def
                mask.append(is_null)
                any_null = any_null or is_null
            if r <= geom.rep_level and d >= geom.child_def:
                child_idx += 1
        offsets.append(Int32(child_idx))

        var offsets_arr = offsets.finish()
        var out: AnyArray
        if any_null:
            var m = mask.finish()
            out = ListArray.from_arrays(offsets_arr, element^, m^)
        else:
            out = ListArray.from_arrays(offsets_arr, element^, None)
        return out^

    def collect_leaf_arrays(
        self, col: AnyArray, mut leaf_arrays: List[AnyArray]
    ) raises:
        """Inverse of `assemble`: collect this node's leaf arrays in column order.
        """
        if self.kind == NODE_LEAF:
            leaf_arrays.append(col.copy())
        elif self.kind == NODE_STRUCT:
            ref sa = col.as_struct()
            for i in range(len(self.children)):
                self.children[i].collect_leaf_arrays(
                    sa.children[i], leaf_arrays
                )
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

    def with_remapped_leaves(self, mapping: List[Int]) -> SchemaNode:
        """A copy of this node with every leaf index rewritten through `mapping`
        (original flat index -> compact index), so a projected node assembles
        from a decoded list that holds only the selected columns."""
        var new_children = List[SchemaNode]()
        for ref c in self.children:
            new_children.append(c.with_remapped_leaves(mapping))
        var new_leaf = self.leaf_index
        if self.kind == NODE_LEAF:
            new_leaf = mapping[self.leaf_index]
        return SchemaNode(
            self.kind,
            self.field.copy(),
            new_children^,
            new_leaf,
            self.geom.copy(),
        )


struct SchemaMapping(Movable):
    """The bridge between an Arrow schema and its Parquet representation, built in
    either direction. It holds the Arrow `schema`, the flat Parquet
    `SchemaElement` list, the leaf-column descriptors (in column-chunk order), and
    the assembly `nodes` tree that ties nested Arrow structure back to the flat
    leaves. `from_parquet` parses a file's footer for reading; `from_arrow`
    converts a table's schema for writing. Covers flat columns, (nullable)
    structs, and nested lists.
    """

    var schema: Schema
    var elements: List[SchemaElement]
    var leaves: List[LeafColumn]
    var nodes: List[SchemaNode]

    def __init__(
        out self,
        var schema: Schema,
        var elements: List[SchemaElement],
        var leaves: List[LeafColumn],
        var nodes: List[SchemaNode],
    ):
        self.schema = schema^
        self.elements = elements^
        self.leaves = leaves^
        self.nodes = nodes^

    # -----------------------------------------------------------------------
    # Parquet metadata -> Arrow (read). A depth-first walk over the flat
    # `SchemaElement` list threads a `mut idx` cursor and appends to `leaves`.
    # -----------------------------------------------------------------------

    @staticmethod
    def from_parquet(meta: FileMetaData) raises -> SchemaMapping:
        var m = SchemaMapping(
            Schema(fields=List[dt.Field]()),
            meta.schema.copy(),
            List[LeafColumn](),
            List[SchemaNode](),
        )
        var idx = 1  # schema[0] is the root group
        var fields = List[dt.Field]()
        for _ in range(meta.schema[0].num_children):
            var node = m._parse_node(idx, 0, 0)
            fields.append(node.field.copy())
            m.nodes.append(node^)
        m.schema = Schema(fields=fields^)
        return m^

    @staticmethod
    def _time_unit(el: SchemaElement) -> dt.TimeUnit:
        """Arrow TimeUnit for a temporal leaf (logical unit, else converted)."""
        if el.logical_unit == 1:
            return dt.millisecond
        elif el.logical_unit == 2:
            return dt.microsecond
        elif el.logical_unit == 3:
            return dt.nanosecond
        elif (
            el.converted_type == ConvertedType.TIMESTAMP_MILLIS
            or el.converted_type == ConvertedType.TIME_MILLIS
        ):
            return dt.millisecond
        else:
            return dt.microsecond

    @staticmethod
    def _leaf_dtype(el: SchemaElement) raises -> dt.AnyDataType:
        """Arrow value type for a Parquet leaf `SchemaElement`."""
        var pt = el.type
        var ct = el.converted_type
        var lt = el.logical_type
        if pt == PhysicalType.BOOLEAN:
            return dt.bool_
        elif pt == PhysicalType.INT32:
            if ct == ConvertedType.DATE or lt == LogicalType.DATE:
                return dt.date32()
            elif ct == ConvertedType.TIME_MILLIS or lt == LogicalType.TIME:
                return dt.time32(dt.millisecond)
            elif ct == ConvertedType.INT_8:
                return dt.int8
            elif ct == ConvertedType.INT_16:
                return dt.int16
            elif ct == ConvertedType.UINT_8:
                return dt.uint8
            elif ct == ConvertedType.UINT_16:
                return dt.uint16
            elif ct == ConvertedType.UINT_32:
                return dt.uint32
            else:
                return dt.int32
        elif pt == PhysicalType.INT64:
            if (
                ct == ConvertedType.TIMESTAMP_MILLIS
                or ct == ConvertedType.TIMESTAMP_MICROS
                or lt == LogicalType.TIMESTAMP
            ):
                return dt.timestamp(
                    Self._time_unit(el),
                    "UTC" if el.logical_utc else String(""),
                )
            elif ct == ConvertedType.TIME_MICROS or lt == LogicalType.TIME:
                return dt.time64(Self._time_unit(el))
            elif ct == ConvertedType.UINT_64:
                return dt.uint64
            else:
                return dt.int64
        elif pt == PhysicalType.FLOAT:
            return dt.float32
        elif pt == PhysicalType.DOUBLE:
            return dt.float64
        elif pt == PhysicalType.BYTE_ARRAY:
            if ct == ConvertedType.UTF8 or lt == LogicalType.STRING:
                return dt.string
            else:
                return dt.binary
        else:
            raise Error("parquet: unsupported physical type " + String(pt.code))

    def _parse_node(
        mut self,
        mut idx: Int,
        def_base: Int,
        rep_base: Int,
        slot_def: Int = 0,
        under_optional: Bool = False,
    ) raises -> SchemaNode:
        """Consume the element at `idx` (advancing it) and return its Arrow node.

        `slot_def` is the definition level at/above which this element's value
        slot exists — bumped past each enclosing list's repeated group so leaves
        under nested lists read the right present/absent slots. `under_optional`
        marks that a nullable struct ancestor exists, so flat leaves must keep
        their def levels for the struct-null reconstruction."""
        var el = self.elements[idx].copy()
        idx += 1
        var rep = el.repetition_type
        var d = def_base + (1 if rep == Repetition.OPTIONAL else 0)
        var r = rep_base + (1 if rep == Repetition.REPEATED else 0)
        var nullable = rep != Repetition.REQUIRED

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
                    slot_def=slot_def,
                    carry_def=under_optional,
                )
            )
            return SchemaNode(
                NODE_LEAF,
                dt.Field(el.name, dtype^, nullable),
                List[SchemaNode](),
                li,
            )

        if (
            el.converted_type == ConvertedType.MAP
            or el.logical_type == LogicalType.MAP
        ):
            raise Error(
                "parquet: map columns not supported yet (column '"
                + el.name
                + "')"
            )

        if (
            el.converted_type == ConvertedType.LIST
            or el.logical_type == LogicalType.LIST
        ):
            # LIST = optional group(LIST) { repeated group { <element> } }. Skip
            # the repeated middle group (adds one def + one rep level) and parse
            # the element as this list's single child. The list holds an element
            # when the leaf def reaches `d + 1` (repeated group present); the list
            # itself is non-null at `d` (its own optional level).
            idx += 1
            var elem = self._parse_node(
                idx,
                d + 1,
                r + 1,
                slot_def=d + 1,
                under_optional=under_optional,
            )
            var item: dt.AnyDataType = dt.list_(elem.field.dtype.copy())
            var children = List[SchemaNode]()
            children.append(elem^)
            # rep_level = the repeated group's level (r + 1); slot_def =
            # `slot_def`, the innermost *enclosing list's* element floor (0 at
            # top / under a struct) — an optional struct being null still leaves a
            # row-slot, so the list gets a (null) entry there. With every
            # threshold on the node, the scan needs nothing from its parent, so
            # any nesting depth composes by recursion.
            return SchemaNode(
                NODE_LIST,
                dt.Field(el.name, item^, nullable),
                children^,
                -1,
                NodeGeom(
                    non_null_def=d,
                    optional=rep == Repetition.OPTIONAL,
                    rep_level=r + 1,
                    child_def=d + 1,
                    slot_def=slot_def,
                ),
            )

        # plain group -> Arrow struct. A nullable struct is reconstructed from its
        # leaves' def levels (below `d` -> struct null), so its descendants must
        # carry def levels.
        var child_nodes = List[SchemaNode]()
        var child_fields = List[dt.Field]()
        var child_optional = under_optional or nullable
        for _ in range(el.num_children):
            var cn = self._parse_node(
                idx, d, r, slot_def=slot_def, under_optional=child_optional
            )
            child_fields.append(cn.field.copy())
            child_nodes.append(cn^)
        var dtype = dt.struct_(child_fields^)
        return SchemaNode(
            NODE_STRUCT,
            dt.Field(el.name, dtype^, nullable),
            child_nodes^,
            -1,
            NodeGeom(
                non_null_def=d,
                optional=nullable,
                rep_level=r,
                slot_def=slot_def,
            ),
        )

    # -----------------------------------------------------------------------
    # Arrow -> Parquet metadata (write). A depth-first walk over Arrow fields
    # appends `SchemaElement`s and leaf descriptors. Structs are emitted as
    # required groups (struct-level nulls are a follow-up).
    # -----------------------------------------------------------------------

    @staticmethod
    def from_arrow(schema: Schema) raises -> SchemaMapping:
        var m = SchemaMapping(
            Schema(copy=schema),
            List[SchemaElement](),
            List[LeafColumn](),
            List[SchemaNode](),
        )
        var root = SchemaElement()
        root.name = "schema"
        root.num_children = len(schema.fields)
        m.elements.append(root^)
        for ref f in schema.fields:
            m.nodes.append(m._emit_field(f, 0))
        return m^

    @staticmethod
    def _physical(
        dtype: dt.AnyDataType,
    ) raises -> Tuple[PhysicalType, ConvertedType, LogicalType]:
        """`(physical, converted, logical)` for an Arrow leaf; NONE = absent."""
        comptime NO_CT = ConvertedType.NONE
        comptime NO_LT = LogicalType.NONE
        if dtype == dt.bool_:
            return (PhysicalType.BOOLEAN, NO_CT, NO_LT)
        elif dtype == dt.int8:
            return (PhysicalType.INT32, ConvertedType.INT_8, NO_LT)
        elif dtype == dt.int16:
            return (PhysicalType.INT32, ConvertedType.INT_16, NO_LT)
        elif dtype == dt.int32:
            return (PhysicalType.INT32, NO_CT, NO_LT)
        elif dtype == dt.uint8:
            return (PhysicalType.INT32, ConvertedType.UINT_8, NO_LT)
        elif dtype == dt.uint16:
            return (PhysicalType.INT32, ConvertedType.UINT_16, NO_LT)
        elif dtype == dt.uint32:
            return (PhysicalType.INT32, ConvertedType.UINT_32, NO_LT)
        elif dtype == dt.int64:
            return (PhysicalType.INT64, NO_CT, NO_LT)
        elif dtype == dt.uint64:
            return (PhysicalType.INT64, ConvertedType.UINT_64, NO_LT)
        elif dtype == dt.float32:
            return (PhysicalType.FLOAT, NO_CT, NO_LT)
        elif dtype == dt.float64:
            return (PhysicalType.DOUBLE, NO_CT, NO_LT)
        elif dtype.is_string():
            return (
                PhysicalType.BYTE_ARRAY,
                ConvertedType.UTF8,
                LogicalType.STRING,
            )
        elif dtype.is_binary():
            return (PhysicalType.BYTE_ARRAY, NO_CT, NO_LT)
        else:
            raise Error("parquet: cannot write Arrow type " + String(dtype))

    def _emit_field(
        mut self, field: dt.Field, def_base: Int
    ) raises -> SchemaNode:
        if field.dtype.is_struct():
            ref st = field.dtype.as_struct()
            var el = SchemaElement()
            el.name = field.name
            el.repetition_type = Repetition.REQUIRED
            el.num_children = len(st.fields)
            self.elements.append(el^)
            var child_nodes = List[SchemaNode]()
            for ref cf in st.fields:
                child_nodes.append(self._emit_field(cf, def_base))
            return SchemaNode(NODE_STRUCT, field.copy(), child_nodes^, -1)
        else:
            var phys, conv, logi = Self._physical(field.dtype)
            var el = SchemaElement()
            el.type = phys
            el.name = field.name
            el.repetition_type = (
                Repetition.OPTIONAL if field.nullable else Repetition.REQUIRED
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

    # -----------------------------------------------------------------------
    # Read plan
    # -----------------------------------------------------------------------

    def full(self) -> Projection:
        """Read plan for every column, in column-chunk order."""
        var decode_order = List[Int]()
        for i in range(len(self.leaves)):
            decode_order.append(i)
        return Projection(
            Schema(copy=self.schema), self.nodes.copy(), decode_order^
        )

    def project(self, columns: List[String]) raises -> Projection:
        """Read plan for the named top-level columns, in the given order. The
        assembly nodes are with_remapped_leaves onto a compact decoded list holding only the
        selected columns' leaves; unselected column chunks are never decoded."""
        var fields = List[dt.Field]()
        var nodes = List[SchemaNode]()
        var decode_order = List[Int]()
        var mapping = List[Int](length=len(self.leaves), fill=-1)
        for ci in range(len(columns)):
            var found = -1
            for ni in range(len(self.nodes)):
                if self.nodes[ni].field.name == columns[ci]:
                    found = ni
                    break
            if found == -1:
                raise Error("parquet: column not found: " + columns[ci])
            ref node = self.nodes[found]
            var node_leaves = List[Int]()
            node.collect_leaf_indices(node_leaves)
            for orig in node_leaves:
                mapping[orig] = len(decode_order)
                decode_order.append(orig)
            nodes.append(node.with_remapped_leaves(mapping))
            fields.append(node.field.copy())
        return Projection(Schema(fields=fields^), nodes^, decode_order^)


struct Projection(Movable):
    """A read plan: the output Arrow schema, the assembly nodes (leaf indices
    with_remapped_leaves to a compact decoded list), and `decode_order` — the original flat
    leaf/column-chunk indices to decode, in that compact order."""

    var schema: Schema
    var nodes: List[SchemaNode]
    var decode_order: List[Int]

    def __init__(
        out self,
        var schema: Schema,
        var nodes: List[SchemaNode],
        var decode_order: List[Int],
    ):
        self.schema = schema^
        self.nodes = nodes^
        self.decode_order = decode_order^


struct LeafColumn(Copyable, Movable):
    """A single Parquet leaf column and how it maps to an Arrow value type."""

    var name: String
    var dtype: dt.AnyDataType  # Arrow value type of the leaf
    var physical: PhysicalType
    var max_def: Int
    var max_rep: Int
    var nullable: Bool
    var slot_def: Int  # def level at/above which this leaf's value slot exists
    var carry_def: Bool  # keep def levels (leaf is under a nullable struct)

    def __init__(
        out self,
        var name: String,
        var dtype: dt.AnyDataType,
        physical: PhysicalType,
        max_def: Int,
        max_rep: Int,
        nullable: Bool,
        slot_def: Int = 0,
        carry_def: Bool = False,
    ):
        self.name = name^
        self.dtype = dtype^
        self.physical = physical
        self.max_def = max_def
        self.max_rep = max_rep
        self.nullable = nullable
        self.slot_def = slot_def
        self.carry_def = carry_def


# ---------------------------------------------------------------------------
# DecodedLeaf — the reader's per-column output, consumed by SchemaNode.assemble
# ---------------------------------------------------------------------------


struct DecodedLeaf(Movable):
    """One decoded leaf column. A flat leaf carries just its array; a repeated
    (list-element) leaf also carries the per-slot rep/def levels. All leaves
    under the same list share these levels, and the list geometry (which def
    levels mean present/empty/null) lives on the schema node, so this stays a
    plain data record."""

    var leveled: Bool
    var array: AnyArray  # flat column, or the list's element/child array
    var rep_levels: List[Int32]
    var def_levels: List[Int32]

    def __init__(
        out self,
        leveled: Bool,
        var array: AnyArray,
        var rep_levels: List[Int32],
        var def_levels: List[Int32],
    ):
        self.leveled = leveled
        self.array = array^
        self.rep_levels = rep_levels^
        self.def_levels = def_levels^

    @staticmethod
    def flat(var array: AnyArray) -> DecodedLeaf:
        return DecodedLeaf(False, array^, List[Int32](), List[Int32]())
