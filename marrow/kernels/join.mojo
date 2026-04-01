"""Join kernels for Arrow StructArrays.

Public API
----------
``hash_join``   — equijoin two StructArrays on positional key columns.
``HashJoin``    — hash join using SwissHashTable; reusable across morsels.

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
  RadixHashJoin   — partitioned hash join (SwissHashTable + RadixPartitioner)
  SortMergeJoin   — sort both sides, two-pointer merge (no hash table)
"""

from std.gpu.host import DeviceContext

from ..arrays import PrimitiveArray, AnyArray, StructArray
from ..builders import PrimitiveBuilder
from ..dtypes import ArrowType, Field, int32, uint64, UInt64Type, bool_ as bool_dt, struct_, null
from .boolean import and_
from .compare import equal
from .filter import take, filter_
from .hashtable import SwissHashTable
from .hashing import rapidhash
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


comptime IndexPairs = Tuple[PrimitiveArray[Int32Type], PrimitiveArray[Int32Type]]
"""Parallel (left_indices, right_indices) arrays from the probe phase."""


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

    def build_dtype(self) -> ArrowType:
        """DataType of the build side (for output schema construction)."""
        ...

    def num_left_rows(self) -> Int:
        """Number of build-side rows."""
        ...


# ---------------------------------------------------------------------------
# HashJoin — hash join using SwissHashTable
# ---------------------------------------------------------------------------


struct HashJoin[
    hasher: def(StructArray, Optional[DeviceContext]) raises -> PrimitiveArray[
        UInt64Type
    ] = rapidhash
](Join):
    """Hash join using SwissHashTable.

    Build phase: hash left-side key columns, insert rows into hash table.
    Probe phase: hash right-side key columns, look up in hash table,
    emit index pairs via comptime-specialised branchless loop.
    Equality post-pass: vectorized key comparison filters hash collisions.

    Stores the full build-side StructArray for output assembly via take().
    """

    var _table: SwissHashTable[Self.hasher]
    var _left_key_indices: List[Int]
    var _left_dtype: ArrowType
    var _left_data: Optional[StructArray]
    var _left_rows: Int

    def __init__(out self):
        self._table = SwissHashTable[Self.hasher]()
        self._left_key_indices = List[Int]()
        self._left_dtype = null
        self._left_data = None
        self._left_rows = 0

    def build(mut self, left: StructArray, left_key_indices: List[Int]) raises:
        self._left_dtype = left.dtype
        self._left_rows = left.length
        self._left_data = left.copy()
        self._left_key_indices = left_key_indices.copy()
        self._table.build(left.select(left_key_indices))

    def probe(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> StructArray:
        var left_keys = self._left_data.value().select(self._left_key_indices)
        var right_keys = right.select(right_key_indices)
        var pairs = self._table.probe(
            left_keys,
            right_keys,
            self._left_rows,
            single_match=strictness == JOIN_ANY,
        )
        var verified = (pairs[0].copy(), pairs[1].copy())
        # Emit unmatched rows for outer/semi/anti joins.
        var final = self._emit_unmatched(
            verified^, len(right), kind, strictness
        )
        return self._assemble(right, final, kind)

    def _emit_unmatched(
        self,
        var pairs: IndexPairs,
        right_rows: Int,
        kind: UInt8,
        strictness: UInt8,
    ) raises -> IndexPairs:
        """Phase 3: add unmatched rows for outer/semi/anti joins.

        Scans the verified pairs to determine which build/probe rows
        were matched, then appends unmatched rows as needed.
        INNER: returns pairs unchanged.
        SEMI: emits matched build rows only.
        ANTI: emits unmatched build rows only.
        LEFT/RIGHT/FULL: appends unmatched rows from the appropriate side.
        """
        if kind == JOIN_INNER:
            return pairs^

        # Compute which build/probe rows appear in the verified pairs.
        var matched_build = List[Bool](length=self._left_rows, fill=False)
        var matched_probe = List[Bool](length=right_rows, fill=False)
        var n_pairs = len(pairs[0])
        for i in range(n_pairs):
            var lid = Int(pairs[0].unsafe_get(i))
            var rid = Int(pairs[1].unsafe_get(i))
            if lid >= 0:
                matched_build[lid] = True
            if rid >= 0:
                matched_probe[rid] = True

        if kind == JOIN_SEMI:
            var lb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            var rb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            for i in range(self._left_rows):
                if matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return (lb.finish(), rb.finish())

        if kind == JOIN_ANTI:
            var lb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            var rb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return (lb.finish(), rb.finish())

        # LEFT / RIGHT / FULL: matched pairs + unmatched rows.
        var lb = PrimitiveBuilder[Int32Type](capacity=n_pairs + self._left_rows)
        var rb = PrimitiveBuilder[Int32Type](capacity=n_pairs + right_rows)
        for i in range(n_pairs):
            lb.append(pairs[0].unsafe_get(i))
            rb.append(pairs[1].unsafe_get(i))
        if kind == JOIN_LEFT or kind == JOIN_FULL:
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
        if kind == JOIN_RIGHT or kind == JOIN_FULL:
            for i in range(right_rows):
                if not matched_probe[i]:
                    lb.append_null()
                    rb.append(Scalar[int32.native](i))
        return (lb.finish(), rb.finish())

    def build_dtype(self) -> ArrowType:
        return self._left_dtype

    def num_left_rows(self) -> Int:
        return self._left_rows

    def output_dtype(self, probe: StructArray, kind: UInt8) -> ArrowType:
        """Build the output struct DataType for a join result."""
        var fields = List[Field]()
        for ref f in self._left_dtype.as_struct_type().fields:
            fields.append(f.copy())

        if kind != JOIN_SEMI and kind != JOIN_ANTI:
            var left_names = List[String]()
            for ref f in self._left_dtype.as_struct_type().fields:
                left_names.append(f.name)
            for ref f in probe.dtype.as_struct_type().fields:
                var name = f.name
                var collides = False
                for ref ln in left_names:
                    if ln == name:
                        collides = True
                        break
                if collides:
                    name = name + "_right"
                fields.append(Field(name, f.dtype))

        return struct_(fields^)

    def _assemble(
        self, right: StructArray, pairs: IndexPairs, kind: UInt8
    ) raises -> StructArray:
        """Gather left + right columns using index pairs."""
        ref left = self._left_data.value()
        var out_cols = List[AnyArray]()

        for c in range(len(left.children)):
            out_cols.append(take(left.children[c].copy(), pairs[0]))

        if kind != JOIN_SEMI and kind != JOIN_ANTI and kind != JOIN_MARK:
            for c in range(len(right.children)):
                out_cols.append(take(right.children[c].copy(), pairs[1]))

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
#     then runs a standard hash join (SwissHashTable) per partition.
#     Enables partition-parallel execution and better cache locality.
#
#     Uses the SAME SwissHashTable as HashJoin — only the Partitioner differs.
#     """
#     var _partitioner: RadixPartitioner    # from hash_table.mojo
#     var _tables: List[SwissHashTable]      # one per partition
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
#     var _sort_order: Optional[PrimitiveArray[Int32Type]]
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
