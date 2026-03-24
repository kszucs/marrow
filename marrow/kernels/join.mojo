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
from ..dtypes import DataType, Field, int32, uint64, bool_ as bool_dt, struct_
from .boolean import and_
from .compare import equal
from .filter import take, filter_
from .hash_table import SwissHashTable
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


struct HashJoin[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = rapidhash
](Join):
    """Hash join using SwissHashTable.

    Build phase: hash left-side key columns, insert rows into hash table.
    Probe phase: hash right-side key columns, look up in hash table,
    emit index pairs via comptime-specialised branchless loop.
    Equality post-pass: vectorized key comparison filters hash collisions.

    Stores the full build-side StructArray for output assembly via take().
    """

    var _table: SwissHashTable[Self.hash_fn]
    var _build_keys: Optional[StructArray]  # key columns for equality checks
    var _build_dtype: DataType
    var _left_data: Optional[StructArray]
    var _num_rows: Int

    def __init__(out self):
        self._table = SwissHashTable[Self.hash_fn]()
        self._build_keys = None
        self._build_dtype = DataType(code=0)
        self._left_data = None
        self._num_rows = 0

    def build(mut self, data: StructArray, key_indices: List[Int]) raises:
        self._build_dtype = data.dtype
        self._num_rows = data.length
        self._left_data = data.copy()
        var keys = data.select(key_indices)
        self._build_keys = keys.copy()
        var hashes = self._table.hash_keys(keys)
        self._table.build(hashes)

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> StructArray:
        var probe_keys = data.select(key_indices)
        var probe_hashes = self._table.hash_keys(probe_keys)
        # Phase 1: collect candidate pairs from hash matches (branchless).
        var pairs = self._dispatch_probe(probe_hashes, len(data), kind, strictness)
        # Phase 2: vectorized equality filter — removes hash collisions.
        var verified = self._verify_keys(probe_keys, pairs)
        # Phase 3: emit unmatched rows for outer/semi/anti joins.
        var final = self._emit_unmatched(
            verified^, len(data), kind, strictness
        )
        return self._assemble(data, final, kind)

    def _equal_column(
        self,
        build_col: AnyArray,
        probe_col: AnyArray,
        pairs: IndexPairs,
    ) raises -> PrimitiveArray[bool_dt]:
        """Compare one key column at matched indices."""
        var lk = take(build_col.copy(), pairs.left_indices)
        var rk = take(probe_col.copy(), pairs.right_indices)
        return equal(lk, rk).as_primitive[bool_dt]().copy()

    def _verify_keys(
        self, probe_keys: StructArray, pairs: IndexPairs
    ) raises -> IndexPairs:
        """Vectorized equality check on candidate index pairs.

        For each key column: gather build-side and probe-side values at
        the matched indices, run vectorized equal(), AND into a combined
        mask, then filter the pairs. This handles hash collisions correctly
        while keeping the probe loop branchless.

        For the common case (no collisions), all comparisons return True
        and filter is a no-op.
        """
        ref build_keys = self._build_keys.value()
        var n_keys = len(build_keys.children)

        var mask = self._equal_column(
            build_keys.children[0], probe_keys.children[0], pairs
        )

        for k in range(1, n_keys):
            var col_eq = self._equal_column(
                build_keys.children[k], probe_keys.children[k], pairs
            )
            mask = and_(mask.copy(), col_eq).copy()

        ref bool_mask = mask
        return IndexPairs(
            filter_[int32](pairs.left_indices, bool_mask),
            filter_[int32](pairs.right_indices, bool_mask),
        )

    def _emit_unmatched(
        self,
        var pairs: IndexPairs,
        n_probe: Int,
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
        var matched_build = List[Bool](length=self._num_rows, fill=False)
        var matched_probe = List[Bool](length=n_probe, fill=False)
        var n_pairs = len(pairs.left_indices)
        for i in range(n_pairs):
            var lid = Int(pairs.left_indices.unsafe_get(i))
            var rid = Int(pairs.right_indices.unsafe_get(i))
            if lid >= 0:
                matched_build[lid] = True
            if rid >= 0:
                matched_probe[rid] = True

        if kind == JOIN_SEMI:
            var lb = PrimitiveBuilder[int32](capacity=self._num_rows)
            var rb = PrimitiveBuilder[int32](capacity=self._num_rows)
            for i in range(self._num_rows):
                if matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return IndexPairs(lb.finish(), rb.finish())

        if kind == JOIN_ANTI:
            var lb = PrimitiveBuilder[int32](capacity=self._num_rows)
            var rb = PrimitiveBuilder[int32](capacity=self._num_rows)
            for i in range(self._num_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return IndexPairs(lb.finish(), rb.finish())

        # LEFT / RIGHT / FULL: matched pairs + unmatched rows.
        var lb = PrimitiveBuilder[int32](capacity=n_pairs + self._num_rows)
        var rb = PrimitiveBuilder[int32](capacity=n_pairs + n_probe)
        for i in range(n_pairs):
            lb.append(pairs.left_indices.unsafe_get(i))
            rb.append(pairs.right_indices.unsafe_get(i))
        if kind == JOIN_LEFT or kind == JOIN_FULL:
            for i in range(self._num_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
        if kind == JOIN_RIGHT or kind == JOIN_FULL:
            for i in range(n_probe):
                if not matched_probe[i]:
                    lb.append_null()
                    rb.append(Scalar[int32.native](i))
        return IndexPairs(lb.finish(), rb.finish())

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
        probe_hashes: PrimitiveArray[uint64],
        n: Int,
        kind: UInt8,
        strictness: UInt8,
    ) raises -> IndexPairs:
        """One-time runtime dispatch → comptime-specialised inner loop."""
        if kind == JOIN_INNER and strictness == JOIN_ALL:
            return self._probe[JOIN_INNER, JOIN_ALL](probe_hashes, n)
        if kind == JOIN_INNER and strictness == JOIN_ANY:
            return self._probe[JOIN_INNER, JOIN_ANY](probe_hashes, n)
        if kind == JOIN_LEFT and strictness == JOIN_ALL:
            return self._probe[JOIN_LEFT, JOIN_ALL](probe_hashes, n)
        if kind == JOIN_LEFT and strictness == JOIN_ANY:
            return self._probe[JOIN_LEFT, JOIN_ANY](probe_hashes, n)
        if kind == JOIN_RIGHT and strictness == JOIN_ALL:
            return self._probe[JOIN_RIGHT, JOIN_ALL](probe_hashes, n)
        if kind == JOIN_RIGHT and strictness == JOIN_ANY:
            return self._probe[JOIN_RIGHT, JOIN_ANY](probe_hashes, n)
        if kind == JOIN_FULL and strictness == JOIN_ALL:
            return self._probe[JOIN_FULL, JOIN_ALL](probe_hashes, n)
        if kind == JOIN_FULL and strictness == JOIN_ANY:
            return self._probe[JOIN_FULL, JOIN_ANY](probe_hashes, n)
        if kind == JOIN_SEMI:
            return self._probe[JOIN_SEMI, JOIN_ALL](probe_hashes, n)
        if kind == JOIN_ANTI:
            return self._probe[JOIN_ANTI, JOIN_ALL](probe_hashes, n)
        raise Error("hash_join: unsupported kind/strictness combination")

    def _probe[kind: UInt8, strictness: UInt8](
        self,
        probe_hashes: PrimitiveArray[uint64],
        n: Int,
    ) raises -> IndexPairs:
        """Phase 1: collect ALL candidate (build_row, probe_row) pairs from
        hash matches. No equality check, no match tracking — just pair
        collection. Equality filtering and unmatched-row emission happen
        in probe() after this returns."""
        var est = n if n < self._num_rows else self._num_rows
        var left_indices = PrimitiveBuilder[int32](capacity=est, zeroed=False)
        var right_indices = PrimitiveBuilder[int32](capacity=est, zeroed=False)

        for probe_row in range(n):
            var h = UInt64(probe_hashes.unsafe_get(probe_row))
            var bid = self._table.find(h)
            if bid == -1:
                continue

            var entry = Int(self._table.bucket_head(bid))
            while entry != -1:
                var build_row = self._table.entry_row(entry)
                var next_entry = Int(self._table.entry_next(entry))

                comptime if strictness == JOIN_ANY:
                    # For JOIN_ANY, emit only the first candidate per probe row.
                    left_indices.unsafe_append(Scalar[int32.native](build_row))
                    right_indices.unsafe_append(Scalar[int32.native](probe_row))
                    break

                left_indices.unsafe_append(Scalar[int32.native](build_row))
                right_indices.unsafe_append(Scalar[int32.native](probe_row))
                entry = next_entry

        return IndexPairs(
            left_indices.finish(shrink_to_fit=False),
            right_indices.finish(shrink_to_fit=False),
        )

    def _assemble(
        self, right: StructArray, pairs: IndexPairs, kind: UInt8
    ) raises -> StructArray:
        """Gather left + right columns using index pairs."""
        ref left = self._left_data.value()
        var out_cols = List[AnyArray]()

        for c in range(len(left.children)):
            out_cols.append(
                take(left.children[c].copy(), pairs.left_indices)
            )

        if kind != JOIN_SEMI and kind != JOIN_ANTI and kind != JOIN_MARK:
            for c in range(len(right.children)):
                out_cols.append(
                    take(
                        right.children[c].copy(), pairs.right_indices
                    )
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
