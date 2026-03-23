"""Join kernels for Arrow StructArrays.

Public API
----------
``hash_join``   — equijoin two StructArrays on positional key columns.
``HashJoin``    — hash join using DictHashTable; reusable across morsels.

Traits
------
``Join``        — abstract join algorithm (build + probe → IndexPairs).

Internal types
--------------
``IndexPairs``  — (left_indices, right_indices) result of a probe phase.

Supported join kinds (pass the JOIN_* constants from relations.mojo):
  JOIN_INNER  — only matched rows
  JOIN_LEFT   — all left + matched right (NULLs for non-matches)
  JOIN_RIGHT  — all right + matched left (NULLs for non-matches)
  JOIN_FULL   — all rows from both sides
  JOIN_SEMI   — left rows with at least one match (left columns only)
  JOIN_ANTI   — left rows with no match (left columns only)

Supported strictness:
  JOIN_ALL    — default: return all matching pairs (Cartesian for multi-match)
  JOIN_ANY    — return at most one matching right row per left row

Future join algorithms (implement the Join trait):
  RadixHashJoin   — partitioned hash join (DictHashTable + RadixPartitioner)
  SortMergeJoin   — sort both sides, two-pointer merge (no hash table)
"""

from std.gpu.host import DeviceContext

from ..arrays import PrimitiveArray, AnyArray, StructArray
from ..builders import PrimitiveBuilder
from ..dtypes import DataType, Field, int32, uint64, struct_
from .filter import take
from .hash_table import DictHashTable
from .hashing import hash_
from ..expr.relations import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_MARK,
    JOIN_ALL,
    JOIN_ANY,
)



# ---------------------------------------------------------------------------
# IndexPairs — parallel index arrays produced by the probe phase
# ---------------------------------------------------------------------------


struct IndexPairs(Movable):
    """Parallel (left_idx, right_idx) arrays produced by the probe phase.

    For LEFT/FULL joins: right_idx = -1 for unmatched left (build) rows.
    For RIGHT/FULL joins: left_idx = -1 for unmatched right (probe) rows.
    For SEMI/ANTI: right_idx is always -1 (only left columns in output).
    """

    var left_indices: PrimitiveArray[int32]
    var right_indices: PrimitiveArray[int32]

    def __init__(
        out self,
        var left_indices: PrimitiveArray[int32],
        var right_indices: PrimitiveArray[int32],
    ):
        self.left_indices = left_indices^
        self.right_indices = right_indices^


# ---------------------------------------------------------------------------
# Join trait — abstract join algorithm
# ---------------------------------------------------------------------------


trait Join(Movable):
    """Abstract join algorithm: build from left side, probe with right side.

    All join algorithms (hash, radix-hash, sort-merge) implement this.
    The trait is for static dispatch — operators use concrete types directly.
    Runtime algorithm selection uses if/elif at the call site, not type erasure.
    """

    def build(mut self, data: StructArray, key_indices: List[Int]) raises:
        """Build the join index from the left (build) side."""
        ...

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
    ) raises -> StructArray:
        """Probe with right (probe) side data.  Return assembled output."""
        ...

    def build_dtype(self) -> DataType:
        """DataType of the build side (for output schema construction)."""
        ...

    def num_build_rows(self) -> Int:
        """Number of build-side rows."""
        ...


# ---------------------------------------------------------------------------
# HashJoin — hash join using DictHashTable
# ---------------------------------------------------------------------------


struct HashJoin(Join):
    """Hash join using DictHashTable.

    Build phase: hash left-side key columns, insert rows into hash table.
    Probe phase: hash right-side key columns, look up in hash table,
    emit index pairs via comptime-specialised branchless loop.

    Stores the full build-side StructArray for output assembly via take().
    """

    var _table: DictHashTable[hash_]
    var _build_dtype: DataType
    var _left_data: Optional[StructArray]
    var _num_rows: Int

    def __init__(out self):
        self._table = DictHashTable[hash_]()
        self._build_dtype = DataType(code=0)
        self._left_data = None
        self._num_rows = 0

    def build(mut self, data: StructArray, key_indices: List[Int]) raises:
        self._build_dtype = data.dtype
        self._num_rows = data.length
        self._left_data = data.copy()
        # Re-init with capacity to avoid reallocs during insert loop.
        self._table = DictHashTable[hash_](capacity=self._num_rows)
        var keys = data.select(key_indices)
        var hashes = self._table.hash_keys(keys)
        for i in range(self._num_rows):
            _ = self._table.insert(UInt64(hashes.unsafe_get(i)), Int32(i))

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> StructArray:
        var pairs = self._dispatch_probe(data, key_indices, kind, strictness)
        return self._assemble(data, pairs, kind)

    def build_dtype(self) -> DataType:
        return self._build_dtype

    def num_build_rows(self) -> Int:
        return self._num_rows

    def output_dtype(self, probe: StructArray, kind: UInt8) -> DataType:
        """Build the output struct DataType for a join result."""
        var fields = List[Field]()
        for ref f in self._build_dtype.fields:
            fields.append(f.copy())

        if kind != JOIN_SEMI and kind != JOIN_ANTI:
            var left_names = List[String]()
            for ref f in self._build_dtype.fields:
                left_names.append(f.name)
            for ref f in probe.dtype.fields:
                var name = f.name
                var collides = False
                for ref ln in left_names:
                    if ln == name:
                        collides = True
                        break
                if collides:
                    name = name + "_right"
                fields.append(Field(name, f.dtype.copy()))

        return struct_(fields^)

    # --- internal probe dispatch ---

    def _dispatch_probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
    ) raises -> IndexPairs:
        """One-time runtime dispatch → comptime-specialised inner loop."""
        if kind == JOIN_INNER and strictness == JOIN_ALL:
            return self._probe[JOIN_INNER, JOIN_ALL](data, key_indices)
        if kind == JOIN_INNER and strictness == JOIN_ANY:
            return self._probe[JOIN_INNER, JOIN_ANY](data, key_indices)
        if kind == JOIN_LEFT and strictness == JOIN_ALL:
            return self._probe[JOIN_LEFT, JOIN_ALL](data, key_indices)
        if kind == JOIN_LEFT and strictness == JOIN_ANY:
            return self._probe[JOIN_LEFT, JOIN_ANY](data, key_indices)
        if kind == JOIN_RIGHT and strictness == JOIN_ALL:
            return self._probe[JOIN_RIGHT, JOIN_ALL](data, key_indices)
        if kind == JOIN_RIGHT and strictness == JOIN_ANY:
            return self._probe[JOIN_RIGHT, JOIN_ANY](data, key_indices)
        if kind == JOIN_FULL and strictness == JOIN_ALL:
            return self._probe[JOIN_FULL, JOIN_ALL](data, key_indices)
        if kind == JOIN_FULL and strictness == JOIN_ANY:
            return self._probe[JOIN_FULL, JOIN_ANY](data, key_indices)
        if kind == JOIN_SEMI:
            return self._probe[JOIN_SEMI, JOIN_ALL](data, key_indices)
        if kind == JOIN_ANTI:
            return self._probe[JOIN_ANTI, JOIN_ALL](data, key_indices)
        raise Error("hash_join: unsupported kind/strictness combination")

    def _probe[kind: UInt8, strictness: UInt8](
        self,
        data: StructArray,
        key_indices: List[Int],
    ) raises -> IndexPairs:
        """Branchless probe loop: all kind/strictness decisions are comptime."""
        var probe_keys = data.select(key_indices)
        var n = len(probe_keys)
        var probe_hashes = self._table.hash_keys(probe_keys)

        # Pre-allocate output builders to avoid reallocs in the hot loop.
        var est = n if n < self._num_rows else self._num_rows
        var left_indices = PrimitiveBuilder[int32](capacity=est)
        var right_indices = PrimitiveBuilder[int32](capacity=est)

        # Only allocate match-tracking arrays when the join kind needs them.
        # INNER+ALL never reads these — skip the 2×N allocation entirely.
        alias need_matched_build = (
            kind == JOIN_LEFT
            or kind == JOIN_FULL
            or kind == JOIN_SEMI
            or kind == JOIN_ANTI
            or strictness == JOIN_ANY
        )
        alias need_matched_probe = (
            kind == JOIN_RIGHT or kind == JOIN_FULL
        )
        var matched_build = List[Bool]()
        var matched_probe = List[Bool]()
        comptime if need_matched_build:
            matched_build = List[Bool](length=self._num_rows, fill=False)
        comptime if need_matched_probe:
            matched_probe = List[Bool](length=n, fill=False)

        for probe_row in range(n):
            var h = UInt64(probe_hashes.unsafe_get(probe_row))
            var bid = self._table.find(h)
            if bid == -1:
                comptime if kind == JOIN_RIGHT or kind == JOIN_FULL:
                    left_indices.unsafe_append(Scalar[int32.native](-1))
                    right_indices.unsafe_append(Scalar[int32.native](probe_row))
                continue

            # Walk the chain of build-side candidates.
            var entry = Int(self._table.bucket_head(bid))
            while entry != -1:
                var build_row = self._table.entry_row(entry)
                var br = Int(build_row)
                var next_entry = Int(self._table.entry_next(entry))

                comptime if kind == JOIN_SEMI:
                    matched_build[br] = True
                    break
                comptime if kind == JOIN_ANTI:
                    matched_build[br] = True
                    break

                comptime if kind != JOIN_SEMI and kind != JOIN_ANTI:
                    comptime if strictness == JOIN_ANY:
                        if matched_build[br]:
                            entry = next_entry
                            continue

                    left_indices.unsafe_append(Scalar[int32.native](build_row))
                    right_indices.unsafe_append(Scalar[int32.native](probe_row))

                    comptime if need_matched_build:
                        matched_build[br] = True
                    comptime if need_matched_probe:
                        matched_probe[probe_row] = True

                    comptime if strictness == JOIN_ANY:
                        break

                entry = next_entry

            comptime if kind == JOIN_RIGHT or kind == JOIN_FULL:
                if not matched_probe[probe_row]:
                    left_indices.unsafe_append(Scalar[int32.native](-1))
                    right_indices.unsafe_append(Scalar[int32.native](probe_row))

        comptime if kind == JOIN_LEFT or kind == JOIN_FULL:
            for i in range(self._num_rows):
                if not matched_build[i]:
                    left_indices.append(Scalar[int32.native](i))
                    right_indices.append(Scalar[int32.native](-1))

        comptime if kind == JOIN_SEMI:
            for i in range(self._num_rows):
                if matched_build[i]:
                    left_indices.append(Scalar[int32.native](i))
                    right_indices.append(Scalar[int32.native](-1))

        comptime if kind == JOIN_ANTI:
            for i in range(self._num_rows):
                if not matched_build[i]:
                    left_indices.append(Scalar[int32.native](i))
                    right_indices.append(Scalar[int32.native](-1))

        return IndexPairs(left_indices.finish(), right_indices.finish())

    def _assemble(
        self, right: StructArray, pairs: IndexPairs, kind: UInt8
    ) raises -> StructArray:
        """Gather left + right columns using index pairs."""
        ref left = self._left_data.value()
        var out_cols = List[AnyArray]()

        for c in range(len(left.children)):
            out_cols.append(take(left.children[c].copy(), pairs.left_indices))

        if kind != JOIN_SEMI and kind != JOIN_ANTI and kind != JOIN_MARK:
            for c in range(len(right.children)):
                out_cols.append(
                    take(right.children[c].copy(), pairs.right_indices)
                )

        var out_length = out_cols[0].length() if len(out_cols) > 0 else 0
        return StructArray(
            dtype=self.output_dtype(right, kind),
            length=out_length,
            nulls=0,
            offset=0,
            bitmap=None,
            children=out_cols^,
        )


# ---------------------------------------------------------------------------
# Future join algorithms (stubs — implement the Join trait)
# ---------------------------------------------------------------------------

# struct RadixHashJoin(Join):
#     """Radix-partitioned hash join.
#
#     Partitions both sides by hash prefix bits using RadixPartitioner,
#     then runs a standard hash join (DictHashTable) per partition.
#     Enables partition-parallel execution and better cache locality.
#
#     Uses the SAME DictHashTable as HashJoin — only the Partitioner differs.
#     """
#     var _partitioner: RadixPartitioner    # from hash_table.mojo
#     var _tables: List[DictHashTable]      # one per partition
#     var _build_dtype: DataType
#     var _left_data: Optional[StructArray]
#     var _num_rows: Int


# struct SortMergeJoin(Join):
#     """Sort-merge join.
#
#     Sorts both sides by key columns, then two-pointer linear merge.
#     O(N log N + M log M) time, zero hash table memory.
#
#     Does NOT use HashTable — proves the Join trait is not hash-specific.
#     """
#     var _sort_order: Optional[PrimitiveArray[int32]]
#     var _sorted_keys: Optional[StructArray]
#     var _build_dtype: DataType
#     var _left_data: Optional[StructArray]
#     var _num_rows: Int


# ---------------------------------------------------------------------------
# hash_join — top-level public API
# ---------------------------------------------------------------------------


def hash_join(
    left: StructArray,
    right: StructArray,
    left_on: List[Int],
    right_on: List[Int],
    kind: UInt8 = JOIN_INNER,
    strictness: UInt8 = JOIN_ALL,
    ctx: Optional[DeviceContext] = None,
) raises -> StructArray:
    """Equijoin two StructArrays on positional key column indices.

    The left side is always the build side; the right side is the probe side.

    Args:
        left: Build-side data as a StructArray (one child per column).
        right: Probe-side data as a StructArray (one child per column).
        left_on: Positional column indices in ``left`` to join on.
        right_on: Positional column indices in ``right`` to join on.
        kind: Join direction (JOIN_INNER, JOIN_LEFT, JOIN_RIGHT, JOIN_FULL,
              JOIN_SEMI, JOIN_ANTI).
        strictness: JOIN_ALL (default) or JOIN_ANY.
        ctx: Optional GPU device context (future GPU acceleration hook).

    Returns:
        Output StructArray:
        * INNER/LEFT/RIGHT/FULL: left columns + right columns.
        * SEMI/ANTI: left columns only.
    """
    if len(left_on) != len(right_on):
        raise Error("hash_join: len(left_on) != len(right_on)")

    # Algorithm selection — currently only HashJoin is implemented.
    # Future: if algorithm == JOIN_ALGO_SORT_MERGE: var j = SortMergeJoin() ...
    # Future: if algorithm == JOIN_ALGO_RADIX: var j = RadixHashJoin(bits=4) ...
    var join = HashJoin()
    join.build(left, left_on)
    return join.probe(right, right_on, kind, strictness)
