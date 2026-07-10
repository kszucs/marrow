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
comptime NODE_MAP: Int = 3


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
        elif self.kind == NODE_LIST or self.kind == NODE_MAP:
            # Assemble the element (leaf, struct, or another list) — one entry
            # per present element — then fold this list's own offsets over those
            # entries. A map is the same fold over its (key, value) entries
            # struct, producing a MapArray. The recursion composes to any depth
            # because each node's geom carries every threshold its scan needs.
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
        var mask_opt: Optional[BoolArray] = None
        if any_null:
            mask_opt = mask.finish()

        # Fold the offsets over the element via ListArray.from_arrays (which turns
        # the mask into a validity bitmap). A map is physically a list of its
        # (key, value) entries struct, so retag the result as a MapArray.
        var la = ListArray.from_arrays(offsets_arr, element^, mask_opt^)
        var out: AnyArray
        if self.field.dtype.is_map():
            out = la.to_map(self.field.dtype.as_map().keys_sorted)
        else:
            out = la^
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
        elif self.kind == NODE_LIST:
            # Descend into the list's flat child values — the innermost element
            # arrays hold the leaf values to write; the offsets/levels come from
            # the Dremel shred.
            self.children[0].collect_leaf_arrays(col.as_list().values(), leaf_arrays)
        elif self.kind == NODE_MAP:
            self.children[0].collect_leaf_arrays(col.as_map().values(), leaf_arrays)
        else:
            raise Error("parquet: unsupported schema node kind")

    def contains_repeated(self) -> Bool:
        """True if this subtree contains a list or map (a repeated group), so its
        column must be Dremel-shredded (rep/def levels) rather than written flat.
        """
        if self.kind == NODE_LIST or self.kind == NODE_MAP:
            return True
        for ref c in self.children:
            if c.contains_repeated():
                return True
        return False

    # -----------------------------------------------------------------------
    # Write-side Dremel striping (inverse of `assemble`): produce per-leaf
    # definition and repetition levels for a leveled column. `defs`/`reps` are
    # indexed by leaf index (in this node's subtree); only leaves with
    # max_rep>=1 accumulate rep levels.
    # -----------------------------------------------------------------------

    def shred_levels(
        self,
        col: AnyArray,
        meta: List[LeafColumn],
        mut defs: List[List[Int32]],
        mut reps: List[List[Int32]],
    ) raises:
        for i in range(col.length()):
            self._shred_elem(col, i, 0, meta, defs, reps)

    def _shred_elem(
        self,
        arr: AnyArray,
        i: Int,
        rep: Int,
        meta: List[LeafColumn],
        mut defs: List[List[Int32]],
        mut reps: List[List[Int32]],
    ) raises:
        if self.kind == NODE_LEAF:
            var li = self.leaf_index
            var present = arr.is_valid(i)
            defs[li].append(
                Int32(meta[li].max_def if present else meta[li].max_def - 1)
            )
            if meta[li].max_rep >= 1:
                reps[li].append(Int32(rep))
        elif self.kind == NODE_STRUCT:
            # Emitted REQUIRED on write, so the struct itself is always present;
            # each field is shredded at the same element with the same rep.
            ref sa = arr.as_struct()
            for c in range(len(self.children)):
                self.children[c]._shred_elem(
                    sa.children[c], i, rep, meta, defs, reps
                )
        else:  # NODE_LIST / NODE_MAP
            var valid: Bool
            var start: Int
            var end: Int
            var child_arr: AnyArray
            if self.kind == NODE_MAP:
                ref la = arr.as_map()
                valid = la.is_valid(i)
                start, end = la.child_range(i)
                child_arr = la.values().copy()
            else:
                ref la = arr.as_list()
                valid = la.is_valid(i)
                start, end = la.child_range(i)
                child_arr = la.values().copy()

            if self.geom.optional and not valid:
                # Null list/map: one absent marker for every leaf underneath.
                self._emit_absent(
                    self.geom.non_null_def - 1, rep, meta, defs, reps
                )
            elif start == end:
                # Present but empty: below the child element floor, no values.
                self._emit_absent(
                    self.geom.child_def - 1, rep, meta, defs, reps
                )
            else:
                for j in range(start, end):
                    var child_rep = rep if j == start else self.geom.rep_level
                    self.children[0]._shred_elem(
                        child_arr, j, child_rep, meta, defs, reps
                    )

    def _emit_absent(
        self,
        def_val: Int,
        rep: Int,
        meta: List[LeafColumn],
        mut defs: List[List[Int32]],
        mut reps: List[List[Int32]],
    ):
        """Emit a single absent/empty marker (one level entry) for every leaf in
        this subtree — a null or empty list/map contributes no child values."""
        var idxs = List[Int]()
        self.collect_leaf_indices(idxs)
        for li in idxs:
            defs[li].append(Int32(def_val))
            if meta[li].max_rep >= 1:
                reps[li].append(Int32(rep))

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


struct _LeafTypeRow(Copyable, Movable):
    """One row of the canonical Arrow<->Parquet leaf-type table: an Arrow value
    type and the `(physical, converted, logical)` triple it maps to. Both
    directions of `SchemaMapping` derive from the same rows so they cannot drift.
    The temporal types (date/time/timestamp) need unit + UTC disambiguation that
    does not invert cleanly and are special-cased outside this table."""

    var arrow: dt.AnyDataType
    var physical: PhysicalType
    var converted: ConvertedType
    var logical: LogicalType

    def __init__(
        out self,
        var arrow: dt.AnyDataType,
        physical: PhysicalType,
        converted: ConvertedType,
        logical: LogicalType,
    ):
        self.arrow = arrow^
        self.physical = physical
        self.converted = converted
        self.logical = logical


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
    def _leaf_type_rows() -> List[_LeafTypeRow]:
        """The single source of truth for the primitive / string / binary leaf
        types, as `(arrow, physical, converted, logical)` rows. `_physical`
        (Arrow -> Parquet) is a forward lookup by Arrow type; `_leaf_dtype`
        (Parquet -> Arrow) is the reverse lookup by physical + converted/logical.
        Each physical type has exactly one "default" row (converted and logical
        both NONE) that the reverse lookup falls back to. Temporal types are
        special-cased in `_leaf_dtype` / `_physical` and deliberately absent."""
        comptime NO_CT = ConvertedType.NONE
        comptime NO_LT = LogicalType.NONE
        var rows = List[_LeafTypeRow]()
        rows.append(_LeafTypeRow(dt.bool_, PhysicalType.BOOLEAN, NO_CT, NO_LT))
        rows.append(
            _LeafTypeRow(
                dt.int8, PhysicalType.INT32, ConvertedType.INT_8, NO_LT
            )
        )
        rows.append(
            _LeafTypeRow(
                dt.int16, PhysicalType.INT32, ConvertedType.INT_16, NO_LT
            )
        )
        rows.append(_LeafTypeRow(dt.int32, PhysicalType.INT32, NO_CT, NO_LT))
        rows.append(
            _LeafTypeRow(
                dt.uint8, PhysicalType.INT32, ConvertedType.UINT_8, NO_LT
            )
        )
        rows.append(
            _LeafTypeRow(
                dt.uint16, PhysicalType.INT32, ConvertedType.UINT_16, NO_LT
            )
        )
        rows.append(
            _LeafTypeRow(
                dt.uint32, PhysicalType.INT32, ConvertedType.UINT_32, NO_LT
            )
        )
        rows.append(_LeafTypeRow(dt.int64, PhysicalType.INT64, NO_CT, NO_LT))
        rows.append(
            _LeafTypeRow(
                dt.uint64, PhysicalType.INT64, ConvertedType.UINT_64, NO_LT
            )
        )
        rows.append(_LeafTypeRow(dt.float32, PhysicalType.FLOAT, NO_CT, NO_LT))
        rows.append(_LeafTypeRow(dt.float64, PhysicalType.DOUBLE, NO_CT, NO_LT))
        rows.append(
            _LeafTypeRow(
                dt.string,
                PhysicalType.BYTE_ARRAY,
                ConvertedType.UTF8,
                LogicalType.STRING,
            )
        )
        rows.append(
            _LeafTypeRow(dt.binary, PhysicalType.BYTE_ARRAY, NO_CT, NO_LT)
        )
        return rows^

    @staticmethod
    def _leaf_dtype(el: SchemaElement) raises -> dt.AnyDataType:
        """Arrow value type for a Parquet leaf `SchemaElement`."""
        var pt = el.type
        var ct = el.converted_type
        var lt = el.logical_type
        # Temporal leaves need unit + UTC disambiguation and array construction
        # that does not invert cleanly, so they are matched before the table.
        if pt == PhysicalType.INT32:
            if ct == ConvertedType.DATE or lt == LogicalType.DATE:
                return dt.date32()
            elif ct == ConvertedType.TIME_MILLIS or lt == LogicalType.TIME:
                return dt.time32(dt.millisecond)
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
        # Non-temporal bulk: match the physical type, then the specific
        # converted/logical annotation; falling back to that physical's default
        # row (both annotations NONE) when none applies.
        var rows = Self._leaf_type_rows()
        for ref row in rows:
            if row.physical == pt and (
                (row.converted != ConvertedType.NONE and row.converted == ct)
                or (row.logical != LogicalType.NONE and row.logical == lt)
            ):
                return row.arrow.copy()
        for ref row in rows:
            if (
                row.physical == pt
                and row.converted == ConvertedType.NONE
                and row.logical == LogicalType.NONE
            ):
                return row.arrow.copy()
        raise Error("parquet: unsupported physical type " + String(pt.code))

    @staticmethod
    def _group_element(
        name: String,
        repetition: Repetition,
        num_children: Int,
        converted: ConvertedType = ConvertedType.NONE,
        logical: LogicalType = LogicalType.NONE,
    ) -> SchemaElement:
        """A group `SchemaElement` (a non-leaf node) for the write path: sets the
        repetition, child count, and optional LIST/MAP annotation."""
        var el = SchemaElement()
        el.name = name
        el.repetition_type = repetition
        el.num_children = num_children
        el.converted_type = converted
        el.logical_type = logical
        return el^

    @staticmethod
    def _list_node(
        name: String,
        var elem: SchemaNode,
        d: Int,
        rep_base: Int,
        slot_def: Int,
        nullable: Bool,
    ) -> SchemaNode:
        """Assemble the `NODE_LIST` node from its (parsed/emitted) element node —
        shared by read (`_parse_node`) and write (`_emit_field`) so the list's
        Dremel geometry lives in one place. `d` is the list's own definition
        level; its repeated middle group sits at `rep_base + 1` / `d + 1`, so a
        slot holds an element at `d + 1` and the list itself is non-null at `d`."""
        var children = List[SchemaNode]()
        children.append(elem^)
        var item: dt.AnyDataType = dt.list_(children[0].field.dtype.copy())
        return SchemaNode(
            NODE_LIST,
            dt.Field(name, item^, nullable),
            children^,
            -1,
            NodeGeom(
                non_null_def=d,
                optional=nullable,
                rep_level=rep_base + 1,
                child_def=d + 1,
                slot_def=slot_def,
            ),
        )

    @staticmethod
    def _map_node(
        name: String,
        var key_node: SchemaNode,
        var val_node: SchemaNode,
        d: Int,
        rep_base: Int,
        slot_def: Int,
        nullable: Bool,
    ) -> SchemaNode:
        """Assemble the `NODE_MAP` node from its key and value child nodes —
        shared by read (`_parse_node`) and write (`_emit_field`) so the map's
        Dremel geometry lives in one place. `d` is the map's own definition
        level; its repeated `key_value` group sits at `rep_base + 1` / `d + 1`.
        The entries struct is non-nullable with a non-null "key"; the value keeps
        its nullability."""
        var key_type = key_node.field.dtype.copy()
        var value_type = val_node.field.dtype.copy()
        var value_nullable = val_node.field.nullable
        key_node.field = dt.Field("key", key_type.copy(), nullable=False)
        val_node.field = dt.Field(
            "value", value_type.copy(), nullable=value_nullable
        )
        var entries_fields = [key_node.field.copy(), val_node.field.copy()]
        var entries_children = List[SchemaNode]()
        entries_children.append(key_node^)
        entries_children.append(val_node^)
        var entries_node = SchemaNode(
            NODE_STRUCT,
            dt.Field("entries", dt.struct_(entries_fields^), nullable=False),
            entries_children^,
            -1,
            NodeGeom(
                non_null_def=d + 1,
                optional=False,
                rep_level=rep_base + 1,
                slot_def=d + 1,
            ),
        )
        var map_dtype: dt.AnyDataType = dt.MapType(
            key_type^, value_type^, value_nullable=value_nullable
        )
        var map_children = List[SchemaNode]()
        map_children.append(entries_node^)
        return SchemaNode(
            NODE_MAP,
            dt.Field(name, map_dtype^, nullable),
            map_children^,
            -1,
            NodeGeom(
                non_null_def=d,
                optional=nullable,
                rep_level=rep_base + 1,
                child_def=d + 1,
                slot_def=slot_def,
            ),
        )

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
            or el.converted_type == ConvertedType.MAP_KEY_VALUE
            or el.logical_type == LogicalType.MAP
        ):
            # MAP = optional group(MAP) { repeated group key_value { required key;
            # <value> } }. Physically a list of a 2-field entries struct, so it
            # reconstructs exactly like list<struct<key,value>> — the same Dremel
            # geometry — and only the final array type (MapArray) differs. Skip
            # the repeated key_value group and parse key + value at d+1 / r+1.
            var kv = self.elements[idx].copy()
            if kv.num_children != 2:
                raise Error(
                    "parquet: map 'key_value' group must have exactly a key and"
                    " a value (column '" + el.name + "')"
                )
            idx += 1
            var key_node = self._parse_node(
                idx, d + 1, r + 1, slot_def=d + 1, under_optional=under_optional
            )
            var val_node = self._parse_node(
                idx, d + 1, r + 1, slot_def=d + 1, under_optional=under_optional
            )
            return Self._map_node(
                el.name, key_node^, val_node^, d, r, slot_def, nullable
            )

        if (
            el.converted_type == ConvertedType.LIST
            or el.logical_type == LogicalType.LIST
        ):
            # LIST = optional group(LIST) { repeated group { <element> } }. Skip
            # the repeated middle group (adds one def + one rep level) and parse
            # the element as this list's single child.
            idx += 1
            var elem = self._parse_node(
                idx,
                d + 1,
                r + 1,
                slot_def=d + 1,
                under_optional=under_optional,
            )
            return Self._list_node(el.name, elem^, d, r, slot_def, nullable)

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
            m.nodes.append(m._emit_field(f, 0, 0, slot_def=0, under_optional=False))
        return m^

    @staticmethod
    def _physical(
        dtype: dt.AnyDataType,
    ) raises -> Tuple[PhysicalType, ConvertedType, LogicalType]:
        """`(physical, converted, logical)` for an Arrow leaf; NONE = absent."""
        for ref row in Self._leaf_type_rows():
            if row.arrow == dtype:
                return (row.physical, row.converted, row.logical)
        raise Error("parquet: cannot write Arrow type " + String(dtype))

    def _emit_field(
        mut self,
        field: dt.Field,
        def_base: Int,
        rep_base: Int,
        slot_def: Int = 0,
        under_optional: Bool = False,
    ) raises -> SchemaNode:
        """Emit the parquet `SchemaElement`s for an Arrow field and return its
        assembly node, threading the Dremel levels exactly as `_parse_node` does
        on read so a written file round-trips. Structs are emitted as REQUIRED
        (struct-level nulls on write are a follow-up); lists and maps add a
        repeated middle group (one def + one rep level)."""
        var nullable = field.nullable
        var d = def_base + (1 if nullable else 0)

        var group_rep = Repetition.OPTIONAL if nullable else Repetition.REQUIRED

        if field.dtype.is_list() or field.dtype.is_large_list():
            # LIST = <opt|req> group(LIST) { repeated group list { <element> } }.
            self.elements.append(
                Self._group_element(
                    field.name,
                    group_rep,
                    1,
                    ConvertedType.LIST,
                    LogicalType.LIST,
                )
            )
            self.elements.append(
                Self._group_element("list", Repetition.REPEATED, 1)
            )

            var elem_field = (
                field.dtype.as_list().value_field().copy() if field.dtype.is_list() else field.dtype.as_large_list().value_field().copy()
            )
            var elem = self._emit_field(
                elem_field,
                d + 1,
                rep_base + 1,
                slot_def=d + 1,
                under_optional=under_optional,
            )
            return Self._list_node(field.name, elem^, d, rep_base, slot_def, nullable)

        if field.dtype.is_map():
            # MAP = <opt|req> group(MAP) { repeated group key_value {
            #   required key; <value> } }.
            ref mt = field.dtype.as_map()
            self.elements.append(
                Self._group_element(
                    field.name, group_rep, 1, ConvertedType.MAP, LogicalType.MAP
                )
            )
            self.elements.append(
                Self._group_element("key_value", Repetition.REPEATED, 2)
            )

            var key_node = self._emit_field(
                mt.key_field(),
                d + 1,
                rep_base + 1,
                slot_def=d + 1,
                under_optional=under_optional,
            )
            var val_node = self._emit_field(
                mt.item_field(),
                d + 1,
                rep_base + 1,
                slot_def=d + 1,
                under_optional=under_optional,
            )
            return Self._map_node(
                field.name, key_node^, val_node^, d, rep_base, slot_def, nullable
            )

        if field.dtype.is_struct():
            # Emitted as a REQUIRED group (struct-level nulls on write TODO), so
            # the children inherit `def_base` unchanged.
            ref st = field.dtype.as_struct()
            self.elements.append(
                Self._group_element(
                    field.name, Repetition.REQUIRED, len(st.fields)
                )
            )
            var child_nodes = List[SchemaNode]()
            for ref cf in st.fields:
                child_nodes.append(
                    self._emit_field(
                        cf,
                        def_base,
                        rep_base,
                        slot_def=slot_def,
                        under_optional=under_optional,
                    )
                )
            return SchemaNode(
                NODE_STRUCT,
                field.copy(),
                child_nodes^,
                -1,
                NodeGeom(
                    non_null_def=def_base,
                    optional=False,
                    rep_level=rep_base,
                    slot_def=slot_def,
                ),
            )

        var phys, conv, logi = Self._physical(field.dtype)
        var el = SchemaElement()
        el.type = phys
        el.name = field.name
        el.repetition_type = (
            Repetition.OPTIONAL if nullable else Repetition.REQUIRED
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
                max_def=d,
                max_rep=rep_base,
                nullable=nullable,
                slot_def=slot_def,
                carry_def=under_optional,
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
