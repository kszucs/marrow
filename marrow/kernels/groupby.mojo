"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller through
     an ``AggKernel`` (``aggregate.mojo``).

The grouper is **aggregate-agnostic**: aggregates are ``AggKernel`` types, and
mapping a runtime function *name* onto one lives in the expression layer
(``marrow/expr``). ``Grouping`` is the placement axis an aggregate is
parameterised on — ``ScalarGrouping`` for one implicit slot, ``HashGrouping``
for a keyed one — and the operator that drives a query holds both a grouping
and its aggregates (``marrow/expr/physical.mojo``).
"""

from ..arrays import (
    StructArray,
    DynArray,
    UInt64Array,
    Int32Array,
)
from ..builders import DynBuilder, Int32Builder
from ..dtypes import DynType, Field, struct_
from .core import Groups
from .hashtable import SwissHashTable
from .hashing import RapidHashKernel
from .filter import Take
from ..utils import RapidHash64


# ---------------------------------------------------------------------------
# HashGrouper — keys-only hash grouping
# ---------------------------------------------------------------------------


struct HashGrouper(Movable):
    """Keys-only hash grouper (ClickHouse-style, ``SwissHashTable``-backed).

    ``consume_keys`` hashes a batch of key rows, returns their dense group ids,
    and appends newly-seen key rows to a per-column builder. Call it repeatedly
    to accumulate groups across batches. NULL keys are treated as equal (same
    group), matching SQL GROUP BY semantics (unlike join, where NULL != NULL).

    Aggregate state is owned by the caller, not the grouper — see
    ``AggregateOperator`` in ``marrow/expr/comptime/aggregates.mojo``, which
    pairs a ``Grouping`` with an ``AggKernel``.
    """

    var _table: SwissHashTable[RapidHash64]
    var _key_builders: List[DynBuilder]

    def __init__(out self):
        self._table = SwissHashTable[RapidHash64]()
        self._key_builders = List[DynBuilder]()

    def num_groups(self) -> Int:
        return self._table.num_keys()

    def consume_hashes(
        mut self, hashes: UInt64Array, grow_adaptively: Bool = True
    ) raises -> Tuple[Int32Array, Int32Array]:
        """Resolve a batch of key hashes to dense group ids, and report where
        each *newly created* group first appeared.

        The core both grouping paths share. Returns ``(group ids per row, the
        rows — indices into this batch — at which the new groups first showed
        up)``. What a caller does with those rows is what distinguishes them:
        ``consume_keys`` gathers the key values immediately, because it groups
        batch after batch; a one-shot partitioned grouping keeps them and
        gathers every partition's keys in a single pass at the end.

        Bucket ids are dense and assigned in row order, so first occurrences
        appear in increasing id order — one forward scan collects them all and
        stops as soon as the last new group is found (near-instant when the
        groups all appear early, which is the low-cardinality case)."""
        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(
            hashes, grow_adaptively=grow_adaptively
        )
        var num_now = self._table.num_keys()

        var first_rows = Int32Builder(capacity=num_now - prev, zeroed=False)
        var next_new = prev
        if num_now > prev:
            for i in range(len(bids)):
                if Int(bids.unsafe_get(i)) == next_new:
                    first_rows.unsafe_append(Int32(i))
                    next_new += 1
                    if next_new == num_now:
                        break
        return (bids^, first_rows.finish())

    def consume_keys(
        mut self, keys: StructArray, hashes: Optional[UInt64Array] = None
    ) raises -> Int32Array:
        """Hash keys and resolve group indices, materializing the unique key
        rows as it goes. Returns the per-row group ids.

        New keys get new (dense, contiguous) group ids; existing keys return
        their previous id. Safe to call across multiple batches — this is the
        incremental entry point, which is why it gathers each batch's new key
        rows straight away rather than deferring. Pass ``hashes`` when the
        caller already computed them to skip the re-hash."""
        var n = len(keys)
        if n == 0:
            var empty = Int32Builder(0)
            return empty.finish()

        var batch_hashes = (
            hashes.value().copy() if hashes else RapidHashKernel.apply(keys)
        )
        var grouped = self.consume_hashes(batch_hashes)
        if len(grouped[1]) > 0:
            self._register_new_groups(keys, grouped[1])
        return grouped[0].copy()

    @staticmethod
    def key_fields(keys: StructArray) -> List[Field]:
        """The key columns' fields, read off the keys struct's own dtype."""
        var fields = List[Field]()
        ref st = keys.dtype.as_struct()
        for k in range(len(st.fields)):
            fields.append(Field(st.fields[k].name, st.fields[k].dtype.copy()))
        return fields^

    def key_columns(mut self, key_fields: List[Field]) raises -> List[DynArray]:
        """The unique group-key columns (empty arrays when no groups yet).
        Finishes the per-column key builders — call once, at emit time."""
        var cols = List[DynArray]()
        for k in range(len(key_fields)):
            if len(self._key_builders) == 0:
                var empty = DynBuilder(key_fields[k].dtype)
                cols.append(empty.finish())
            else:
                cols.append(self._key_builders[k].finish())
        return cols^

    def _register_new_groups(
        mut self, keys: StructArray, rows: Int32Array
    ) raises:
        """Append the key rows for newly-created groups to the per-column
        builders. Gathers all new rows in one ``take`` + one bulk ``extend`` per
        column instead of a slice/extend per group."""
        if len(self._key_builders) == 0:
            for k in range(len(keys.children)):
                self._key_builders.append(DynBuilder(keys.children[k].dtype()))
        var gathered = Take.apply(keys, rows)
        for k in range(len(keys.children)):
            self._key_builders[k].extend(gathered.children[k])


# ---------------------------------------------------------------------------
# Grouping — the placement axis
# ---------------------------------------------------------------------------
trait Grouping(Deinitable, Movable):
    """Which slot does a row contribute to?

    The *strategy*; `Groups` is the assignment it produces. One of the four
    axes an aggregation composes from — algebra x input x placement x emission
    — and the one that decides whether a fold scatters at all.

    A trait rather than a flag, so window partitions and a sorted or radix
    placement arrive as **conformers** rather than as further branches inside
    the operator that drives them. Placement is a comptime type parameter of a
    fold, so the loop a placement implies is chosen when the plan is built, not
    per batch.

    Takes **already-evaluated key columns**, never a `RecordBatch`: `kernels`
    must not depend on the expression layer, and evaluating a key expression is
    the caller's job. That is also what lets one grouping serve every aggregate
    in a query — the keys are hashed once, not once per aggregate.
    """

    comptime scatters: Bool
    """Whether a fold must scatter into per-slot accumulators.

    False for a single implicit slot, which lets a fold accumulate in registers
    and reduce once at the end. Not a micro-optimisation: scattering at one
    group measured **14.6x** worse than the register fold, which is the whole
    reason `ScalarGrouping` is its own conformer rather than `HashGrouping`
    with one key.
    """

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Place this batch's rows, extending the grouping with any new slots.

        Ids are dense and stable across calls, so an accumulator that folded an
        earlier batch keeps its slots when a later one introduces new groups.
        """
        ...

    def num_groups(self) -> Int:
        """How many slots exist so far — the size of a per-slot accumulator."""
        ...

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        """One column per key field, one row per slot.

        Empty when there are no keys. Call once, at emit time — it finishes the
        key builders.
        """
        ...


struct ScalarGrouping(Grouping):
    """One slot for every row — `SELECT sum(x) FROM t`, with no `GROUP BY`.

    Holds nothing and allocates nothing per batch. In particular it does **not**
    build a zero vector to say "every row is group 0": a fold whose `scatters`
    is False never reads the ids, and materialising one `Int32` per row to
    communicate a constant is exactly the cost this conformer exists to avoid.
    """

    comptime scatters = False

    def __init__(out self):
        pass

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        return Groups.single(num_rows)

    def num_groups(self) -> Int:
        return 1

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        return List[DynArray]()


struct HashGrouping(Grouping):
    """Dense ids from a keys-only hash table, accumulated across batches.

    Wraps `HashGrouper`, which already owns the hashing, the dense-id
    assignment and the unique-key materialisation. This adds the `Grouping`
    surface over it so a fold can be parameterised on placement.
    """

    comptime scatters = True

    var _grouper: HashGrouper

    def __init__(out self):
        self._grouper = HashGrouper()

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Hash the key columns and resolve dense ids.

        The field names built here are positional and deliberately arbitrary —
        `HashGrouper` keys on the *values*, and the caller supplies the real
        fields at `key_columns` time, where they reach the output schema.
        """
        var fields = List[Field](capacity=len(keys))
        for i in range(len(keys)):
            fields.append(Field("k" + String(i), keys[i].dtype()))
        var st = StructArray(
            dtype=struct_(fields^),
            length=num_rows,
            nulls=0,
            offset=0,
            bitmap=None,
            children=keys.copy(),
        )
        var ids = self._grouper.consume_keys(st)
        return Groups(ids^, self._grouper.num_groups())

    def num_groups(self) -> Int:
        return self._grouper.num_groups()

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        return self._grouper.key_columns(fields)

