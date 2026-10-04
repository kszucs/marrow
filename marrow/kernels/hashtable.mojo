# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Hash indexes — 64-bit hash to dense id.

Two types, one job at two scales:

- ``SwissHashTable`` — one open-addressing table, filled on one thread.
- ``HashIndex`` — the same mapping over one table or, once a batch is large
  and distinct enough, ``2**6`` radix-partitioned tables filled in parallel,
  with every id it has handed out kept across the switch.

**Neither knows what a key is.** They map hashes to dense ids, and two keys
that share a hash share an id unless the caller says otherwise: ``find_one``
takes the caller's equality and ``insert_new`` hands out an id without
looking, one hash at a time, which is how a caller keeps two colliding keys
apart. Deciding what "the same key" means is the key identity layer's
(``hashing.mojo``), and holding keys is ``DictionaryEncoder``'s
(``dictionary.mojo``).
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.math import ceildiv, exp
from std.memory import pack_bits
from std.sys import get_defined_int, size_of


from ..arrays import Int32Array, UInt64Array
from ..buffers import Buffer
from ..dtypes import int32, uint64
from ..errors import InternalError, InvalidError
from ..execution import ExecContext
from .partition import RadixPartitioner
from ..utils import RapidHash64


# ---------------------------------------------------------------------------
# SwissHashTable
# ---------------------------------------------------------------------------


comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes per group (matches Mojo Dict / abseil)."""

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""

comptime _PIPE_DEPTH: Int = 16
"""Number of probes pipelined in a single batch (prefetch window)."""


@fieldwise_init
struct Placement(Copyable, Movable):
    """Where one insert placed a batch of hashes: ``ids[i]`` is the id of hash
    ``i``, and ``firsts[j]`` the position of the hash that introduced the
    ``j``-th new id, in id order — what a consumer storing one value per id
    stores from.

    A named pair rather than ``Tuple[Int32Array, Int32Array]`` for the reason
    ``JoinIndex`` and ``Groups`` are: the two arrays differ in length and in
    meaning, and ``placed[0]`` / ``placed[1]`` said neither.
    """

    var ids: Int32Array
    """The id of every hash, in batch order."""
    var firsts: Int32Array
    """The position of each new id's first hash, in id order."""


struct SwissHashTable(Copyable, Movable, Sized):
    """Swiss Table from 64-bit hash to dense id (abseil / hashbrown layout).

    Each id is handed out once, in order — 0, 1, 2, ... — and owns one slot and
    one stored hash. ``insert_hashes`` / ``find_hashes`` are the batch paths
    and treat equal hashes as one id. ``find_one`` walks every id holding a
    hash and lets the caller's equality pick, and ``insert_new`` hands out an
    id without looking, which is how a caller keeps two keys that collide
    apart.

    Storage::

        ctrl    [ 0xFF | 0xFF | h2=0x3A | 0xFF | h2=0x1B | ... ]  capacity + 16 bytes
        slots   [  --  |  --  |   id=0  |  --  |   id=1  | ... ]  capacity x 4 bytes
        hashes  [ hash of id 0, hash of id 1, ... ]               one per id

    A control byte is ``0xFF`` (empty) or a 7-bit fingerprint of the hash
    (``_h2``). A probe loads 16 control bytes at once and SIMD-matches the
    fingerprint, so one comparison checks 16 slots; a fingerprint match is
    confirmed against the stored hash. Batch paths prefetch the control group
    ``_PIPE_DEPTH`` hashes ahead.
    """

    var _ctrl: Buffer[mut=True]
    """Control bytes: one per slot, plus ``_GROUP_WIDTH`` mirrored at the end
    so a SIMD load never runs past the array."""

    var _slots: Buffer[mut=True]
    """The id stored at each slot (Int32)."""

    var _hashes: Buffer[mut=True]
    """The hash of each id, indexed by id."""

    var _capacity: Int
    """Total number of slots (always a power of 2)."""

    var _mask: Int
    """``_capacity - 1``, for ``hash & _mask``."""

    var _len: Int
    """Ids handed out — also the number of occupied slots."""

    var _max_len: Int
    """Grow past this many ids: ``_capacity * 7 / 8``."""

    def __init__(out self, capacity: Int = 0):
        """An empty table sized for ``capacity`` ids without growing."""
        var cap = Int(next_power_of_two(max(capacity * 2, _GROUP_WIDTH)))
        self._capacity = cap
        self._mask = cap - 1
        self._len = 0
        self._max_len = cap * 7 // 8
        self._ctrl = Buffer.alloc_filled(cap + _GROUP_WIDTH, fill=_CTRL_EMPTY)
        self._slots = Buffer.alloc_uninit[DType.int32](cap)
        # One entry per slot: ids never exceed `_max_len < _capacity`.
        self._hashes = Buffer.alloc_uninit[DType.uint64](cap)

    def __len__(self) -> Int:
        """How many ids have been handed out."""
        return self._len

    # --- probing ------------------------------------------------------------

    @staticmethod
    @always_inline
    def _h2(h: UInt64) -> UInt8:
        """A 7-bit fingerprint from bits ``48..54`` of ``h``, a value in
        ``0x00..0x7F`` — disjoint from ``_CTRL_EMPTY (0xFF)``.

        **Not the top bits, because those are spent.** `RadixPartitioner`
        routes on the top ``num_bits``, so inside one partition every hash
        shares them; a fingerprint taken from the top seven had one bit of
        entropy in a 64-way split and matched half of every occupied slot.
        Bits ``48..54`` stay clear of any partition count up to 2^9 and of the
        slot index (``h & mask``) for any capacity under 2^48 — slot,
        fingerprint and partition read disjoint bits.
        """
        return UInt8((h >> 48) & 0x7F)

    @always_inline
    def prefetch(self, h: UInt64):
        """Start loading the control group ``h`` probes first — issued a few
        hashes ahead, it hides the cache miss a lookup would otherwise wait
        on."""
        self._ctrl.view[DType.uint8]().prefetch_at(Int(h & UInt64(self._mask)))

    @always_inline
    def _id_at(self, slot: Int) -> Int:
        return Int(self._slots.unsafe_get[DType.int32](slot))

    @always_inline
    def _hash_of(self, id: Int) -> UInt64:
        return self._hashes.unsafe_get[DType.uint64](id)

    @always_inline
    def _matches(self, pos: Int, h: UInt64) -> UInt16:
        """Slots of the control group at ``pos`` whose fingerprint is ``h``'s.
        """
        var group = self._ctrl.view[DType.uint8]().load[_GROUP_WIDTH](pos)
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](Self._h2(h))))

    @always_inline
    def _empties(self, pos: Int) -> UInt16:
        """Empty slots of the control group at ``pos``."""
        var group = self._ctrl.view[DType.uint8]().load[_GROUP_WIDTH](pos)
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_EMPTY)))

    @always_inline
    def find(self, h: UInt64) -> Int:
        """The first id holding ``h``, or ``-1``."""
        var pos = Int(h & UInt64(self._mask))
        while True:
            var hits = self._matches(pos, h)
            while hits != 0:
                var id = self._id_at(
                    (pos + count_trailing_zeros(Int(hits))) & self._mask
                )
                if self._hash_of(id) == h:
                    return id
                hits &= hits - 1
            if self._empties(pos) != 0:
                return -1
            pos = (pos + _GROUP_WIDTH) & self._mask

    @always_inline
    def find_one[
        Eq: def(Int) raises -> Bool
    ](self, h: UInt64, eq: Eq) raises -> Int:
        """The first id holding ``h`` that ``eq`` accepts, or ``-1`` — every
        id holding ``h`` is offered, so ``eq`` can tell colliding keys apart.
        """
        var pos = Int(h & UInt64(self._mask))
        while True:
            var hits = self._matches(pos, h)
            while hits != 0:
                var id = self._id_at(
                    (pos + count_trailing_zeros(Int(hits))) & self._mask
                )
                if self._hash_of(id) == h and eq(id):
                    return id
                hits &= hits - 1
            if self._empties(pos) != 0:
                return -1
            pos = (pos + _GROUP_WIDTH) & self._mask

    @always_inline
    def _empty_slot(self, h: UInt64) -> Int:
        """The first empty slot on ``h``'s probe sequence."""
        var pos = Int(h & UInt64(self._mask))
        while True:
            var empties = self._empties(pos)
            if empties != 0:
                return (pos + count_trailing_zeros(Int(empties))) & self._mask
            pos = (pos + _GROUP_WIDTH) & self._mask

    @always_inline
    def insert_new(mut self, h: UInt64) raises -> Int:
        """Hand out the next id for ``h`` without looking for one it already
        has, growing first if the table is at its load limit — for a hash the
        caller knows is new (``insert_hashes``), or whose holders ``find_one``
        just turned down. Also how ids already known to be distinct are
        re-placed, where a lookup would be wasted work and wrong, since two of
        them may share a hash."""
        if self._len >= self._max_len:
            self.reserve(self._capacity)
        var slot = self._empty_slot(h)
        var h2 = Self._h2(h)
        self._ctrl.unsafe_set[DType.uint8](slot, h2)
        if slot < _GROUP_WIDTH:
            self._ctrl.unsafe_set[DType.uint8](self._capacity + slot, h2)
        var id = self._len
        self._slots.unsafe_set[DType.int32](slot, Int32(id))
        self._hashes.unsafe_set[DType.uint64](id, h)
        self._len += 1
        return id

    # --- capacity -----------------------------------------------------------

    def reserve(mut self, n: Int) raises:
        """Make room for ``n`` ids without growing again.

        Growing re-places every id by its stored hash; ids and hashes stay
        where they are.
        """
        var needed = Int(next_power_of_two(max(n * 2, _GROUP_WIDTH)))
        if needed <= self._capacity:
            return
        var old_capacity = self._capacity
        var old_ctrl = self._ctrl^
        var old_slots = self._slots^
        self._capacity = needed
        self._mask = needed - 1
        self._max_len = needed * 7 // 8
        self._ctrl = Buffer.alloc_filled(
            needed + _GROUP_WIDTH, fill=_CTRL_EMPTY
        )
        self._slots = Buffer.alloc_uninit[DType.int32](needed)
        for slot in range(old_capacity):
            if old_ctrl.unsafe_get[DType.uint8](slot) != _CTRL_EMPTY:
                var id = Int(old_slots.unsafe_get[DType.int32](slot))
                var h = self._hash_of(id)
                var to = self._empty_slot(h)
                var h2 = Self._h2(h)
                self._ctrl.unsafe_set[DType.uint8](to, h2)
                if to < _GROUP_WIDTH:
                    self._ctrl.unsafe_set[DType.uint8](needed + to, h2)
                self._slots.unsafe_set[DType.int32](to, Int32(id))
        if needed * size_of[UInt64]() > len(self._hashes):
            self._hashes.resize[DType.uint64](needed)

    # --- batch --------------------------------------------------------------

    def insert_hashes(mut self, hashes: UInt64Array) raises -> Placement:
        """The id of every hash, handing out a new one to each hash not seen
        before. Equal hashes get one id, and new ids come in row order, so
        ``firsts`` increases. Grows as it goes; ``reserve`` first when the
        number of distinct hashes is known."""
        var n = len(hashes)
        # Plain buffers, not builders: a builder's append also sets a validity
        # bit, a read-modify-write of one byte per eight rows that chains each
        # row to the one before it — 25% of this loop at 1,000 ids.
        var ids = Buffer.alloc_uninit[int32.native](max(n, 1))
        var firsts = Buffer.alloc_uninit[int32.native](max(n, 1))
        var created = 0
        for i in range(min(_PIPE_DEPTH, n)):
            self.prefetch(UInt64(hashes.unsafe_get(i)))
        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))
            if i + _PIPE_DEPTH < n:
                self.prefetch(UInt64(hashes.unsafe_get(i + _PIPE_DEPTH)))
            var id = self.find(h)
            if id == -1:
                id = self.insert_new(h)
                firsts.unsafe_set[int32.native](created, Int32(i))
                created += 1
            ids.unsafe_set[int32.native](i, Int32(id))
        return Placement(
            ids=Int32Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=ids^.to_immutable(),
            ),
            firsts=Int32Array(
                length=created,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=firsts^.to_immutable(),
            ),
        )

    def find_hashes(self, hashes: UInt64Array) raises -> Int32Array:
        """The id of every hash, or ``-1`` where the table has none."""
        var n = len(hashes)
        var ids = Buffer.alloc_uninit[int32.native](max(n, 1))
        for i in range(min(_PIPE_DEPTH, n)):
            self.prefetch(UInt64(hashes.unsafe_get(i)))
        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))
            if i + _PIPE_DEPTH < n:
                self.prefetch(UInt64(hashes.unsafe_get(i + _PIPE_DEPTH)))
            ids.unsafe_set[int32.native](i, Int32(self.find(h)))
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=ids^.to_immutable()
        )

    def hashes(self) raises -> UInt64Array:
        """The hash of every id, in id order."""
        var n = self._len
        var buf = Buffer.alloc_uninit[DType.uint64](max(n, 1))
        buf.extend(self._hashes.view[DType.uint64](0, n), 0, n)
        return UInt64Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )


# ---------------------------------------------------------------------------
# HashIndex
# ---------------------------------------------------------------------------


comptime _RADIX_MIN_ROWS: Int = get_defined_int[
    "MARROW_GROUPBY_RADIX_MIN_ROWS", 50_000
]()
"""Batch size below which placement stays serial.

Below it the radix pass, the 64 tables and the two extra row-order passes cost
more than the probe loop they replace. Measured by the calibration sweep in
`tests/bench_groupby.mojo` under `ExecContext.auto()` on an M4 Max, int32 keys
at half-distinct — an insert-heavy shape, where radix has the most to win — as
serial time over radix time:

    rows    15k   30k   40k   50k   60k   90k   125k   250k
    int32  0.60  0.77  1.04  0.97  1.26     —   2.98   2.09
    string    —  0.79     —     —  0.95  1.12   1.31   1.27

The int32 edge sits between 40k and 60k and 50k is its middle. String keys
cross later, near 75k, for a reason not yet profiled. One constant serves both,
and costs a string batch at most ~5% between the two edges. Overridable with
`-D MARROW_GROUPBY_RADIX_MIN_ROWS=N`, which is how the sweep forces either
path.
"""

comptime _RADIX_BITS: Int = 6
"""64 partitions, chosen so a partition's table tends to stay in L2.

**Raising it does not buy load balance, and was measured.** A quarter to a
third of thread-time on this path is spent parked in `semaphore_wait_trap`, and
the obvious reading — 64 work items split across 16 threads is too coarse — is
wrong. At 10M rows / 5M groups, profiling every thread and taking the minimum
of two benchmark runs:

    bits   partitions   idle share   work samples   par8 min
      6            64        31.7%           3799    22.40 ms
      7           128        29.6%           4248    24.18 ms
      8           256        35.1%           4513    28.24 ms

Idle barely moves while *work* grows 12–19%. The extra cost is the scatter,
which went 493 -> 760 samples at 256 buckets — past 64 output streams the write
cursors stop fitting, which is the classic radix fan-out limit. Spinning
longer at each barrier instead of sleeping does not help either (measured
through MAX's `MODULAR_THREAD_BUSY_WAIT_US`; marrow's own pool exposes
`MARROW_THREADS_SPIN_NS`): 200 is a wash and 2000 costs 50%.

**Fusing the phases does not work either, and the reason is the id space.** A
worker cannot carry its partition from insert straight through to write-back:
`base[i]` is a prefix sum over every *preceding* partition's new-id count,
known only once their inserts are done. Removing that dependency means fixed id
blocks, which makes the ids sparse — and they are dense because a consumer
allocates one slot per id. The idle is a cost of dense ids."""

comptime _SAMPLE_ROWS: Int = 4096
"""How many rows the cardinality probe draws. Small enough that its table stays
in L1 and the probe costs tens of microseconds on a million rows."""

comptime _RADIX_MIN_DISTINCT: Int = get_defined_int[
    "MARROW_GROUPBY_RADIX_MIN_GROUPS", 30_000
]()
"""How many distinct hashes a batch must hold before radix placement pays.

**Row count alone is the wrong question.** Radix reads the rows three extra
times — histogram, scatter, and the id write-back — where the serial path
probes once. At 1M rows and 1,000 distinct hashes the serial table never leaves
L1; at 500,000 it does not fit in cache, the probe dominates, and splitting it
64 ways is what the partitioning is for.

Measured by the calibration sweep, at 1M rows, as serial time over radix time:

    distinct  2k    5k   10k   20k   25k   30k   40k   50k   75k   100k   150k
    int32   0.81  0.88  0.92  0.98  0.97  1.03  1.08  1.10     —   1.28      —
    string  0.98  0.94     —  0.95     —     —     —  0.97  1.04   1.07   1.24

The int32 edge is 30,000; strings cross later, near 65,000. The probe cannot
count distinct hashes directly — see `HashIndex._looks_high_cardinality` — so
the gate compares its sample against `HashIndex._MIN_SAMPLE_DISTINCT`.
Overridable with
`-D MARROW_GROUPBY_RADIX_MIN_GROUPS=N`; `0` opens the gate for every batch.
"""


struct HashIndex(Movable, Sized):
    """Hash to dense id, on one table or on 64 radix-partitioned ones — the
    placement a consumer of dense ids needs at any size.

    **Serial until a batch earns radix, then radix for good.** Every batch on
    the serial table asks whether it is large (``_RADIX_MIN_ROWS``) and
    distinct (``_RADIX_MIN_DISTINCT``) enough; the first that is moves every
    id already handed out onto partitioned tables (``_migrate``), keeping its
    number. From then on each batch is split by the **top bits of its hashes**
    and each partition's table is filled by its own worker. One id numbering
    spans all of them, through ``_local_to_global``, so a consumer never learns
    which placement ran.

    Placing by hash is what lets a group-by run in parallel with no merge: a
    key's rows all land in one partition, so every aggregate still sees each of
    its rows once, in one slot — see ``groupby.mojo``.

    **Ids are dense, stable across batches, and new ones come in first-seen
    order on the serial path** — partition-major on the radix path, which is a
    renumbering nothing observes.
    """

    var _ctx: ExecContext
    """How placement executes — held whole, never reduced to a worker count."""

    var _table: SwissHashTable
    """The serial table — the only one, until ``_migrate`` empties it."""

    var _parts: List[SwissHashTable]
    """The radix tables, one per partition; empty until ``_migrate``.
    Persistent: partition ``i`` always routes to table ``i``, so an id keeps
    its partition-local number in every later batch."""

    var _local_to_global: List[List[Int32]]
    """``_local_to_global[p][local]`` is the id of partition ``p``'s local id —
    dense and append-only, so partitions number independently while the index
    hands out one numbering."""

    var _len: Int
    """Ids handed out, on either path."""

    var _reserved: Int
    """Ids the next ``insert`` was told to expect (``reserve``), applied to
    whichever tables that insert places on — so a reservation made before the
    placement is decided never sizes a table that is then abandoned."""

    def __init__(out self, var ctx: ExecContext = ExecContext()):
        self._ctx = ctx^
        self._table = SwissHashTable()
        self._parts = List[SwissHashTable]()
        self._local_to_global = List[List[Int32]]()
        self._len = 0
        self._reserved = 0

    def __len__(self) -> Int:
        return self._len

    def is_partitioned(self) -> Bool:
        """Whether ids live on the radix-partitioned tables. Read off
        ``_parts``: only ``_migrate`` creates them."""
        return len(self._parts) > 0

    def _partitioner(self) -> RadixPartitioner:
        """How this index splits a batch — the same bits for every batch, so
        partition ``i`` always routes to table ``i``."""
        return RadixPartitioner(num_bits=_RADIX_BITS, ctx=self._ctx.copy())

    # --- the radix gate -----------------------------------------------------

    comptime _MIN_SAMPLE_DISTINCT: Int = Self._expected_distinct(
        _RADIX_MIN_DISTINCT, _SAMPLE_ROWS
    )
    """`_RADIX_MIN_DISTINCT` in the sample's units: 3,828 of 4,096 at 30,000.

    A batch 10% either side of the gate lands ~25 draws away, so the edge is
    soft by about ±15%; the sweep puts the cost of misplacing a batch in that
    band under 5%. A gate much past 100,000 needs a larger sample."""

    @staticmethod
    def _expected_distinct(distinct: Int, draws: Int) -> Int:
        """Distinct values expected among `draws` uniform draws, with
        replacement, from `distinct` equally frequent values:
        `d (1 - e^(-draws/d))`. Non-raising so it can run at comptime."""
        if distinct <= 0:
            return 0
        return Int(
            Float64(distinct) * (1.0 - exp(-Float64(draws) / Float64(distinct)))
        )

    @staticmethod
    def _looks_high_cardinality(hashes: UInt64Array) raises -> Bool:
        """Does this batch hold at least ``_RADIX_MIN_DISTINCT`` distinct
        hashes?

        Draws ``_SAMPLE_ROWS`` rows **at pseudo-random positions, with
        replacement**, and counts the distinct hashes among them, which makes
        the answer a function of the key *distribution* alone, whatever the
        row order. Positions come from wyhash's PRNG step, reduced to `[0, n)`
        with a multiply-high: an arithmetic stride aliases with keys generated
        as a linear function of the row index, and once misread 5,000 groups
        as 0.24 distinct. ``hashes`` must be non-empty.
        """
        var n = UInt64(len(hashes))
        var sample = Buffer.alloc_uninit[uint64.native](_SAMPLE_ROWS)
        var seed = UInt64(0)
        for i in range(_SAMPLE_ROWS):
            seed += 0x2D358DCCAA6C78A5
            var r = RapidHash64.mix(seed, seed ^ 0x8BB84B93962EACC9)
            sample.unsafe_set[uint64.native](
                i, hashes.unsafe_get(Int(RapidHash64.mum(r, n)[1]))
            )
        var table = SwissHashTable(capacity=_SAMPLE_ROWS)
        _ = table.insert_hashes(
            UInt64Array(
                length=_SAMPLE_ROWS,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=sample^.to_immutable(),
            )
        )
        return len(table) >= Self._MIN_SAMPLE_DISTINCT

    # --- batch --------------------------------------------------------------

    def reserve(mut self, n: Int):
        """Expect ``n`` new ids from the next ``insert``, so its tables are
        sized once instead of grown — a join's build side, whose rows are
        usually its keys. A group-by, which cannot know, grows on demand."""
        self._reserved = n

    def insert(mut self, hashes: UInt64Array) raises -> Placement:
        """The id of every hash, handing out new ones. ``firsts[j]`` is the
        position of the first hash given id ``len_before + j``; on the serial
        table they increase, on the radix tables they come partition by
        partition."""
        if len(hashes) == 0:
            return Placement(
                ids=Int32Array.empty(int32), firsts=Int32Array.empty(int32)
            )
        if (
            not self.is_partitioned()
            and self._ctx.worth_parallel(len(hashes), _RADIX_MIN_ROWS)
            and Self._looks_high_cardinality(hashes)
        ):
            self._migrate()
        var placed: Placement
        if self.is_partitioned():
            placed = self._insert_radix(hashes)
        else:
            placed = self._insert_serial(hashes)
        self._reserved = 0
        return placed^

    def find(self, hashes: UInt64Array) raises -> Int32Array:
        """The id of every hash, or ``-1`` where there is none.

        On partitioned tables, a batch too small to share out is looked up in
        place — each hash in its own partition's table — because splitting it
        costs more than the lookups: a join probing in 8192-row morsels paid
        1.8x for a 64-way split over a 16-way one (1M build rows, 8 workers).
        A large batch is split, so each worker searches one table.
        """
        if not self.is_partitioned():
            return self._table.find_hashes(hashes)
        if not self._ctx.worth_parallel(len(hashes), _RADIX_MIN_ROWS):
            return self._find_in_place(hashes)

        def find_partition(
            i: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises {imm} -> Int32Array:
            return self._parts[i].find_hashes(part_hashes)

        var split = self._partitioner().map_partitions[Int32Array](
            hashes.copy(), find_partition
        )
        ref routed = split[0]
        ref found = split[1]
        var n = len(hashes)
        var buf = Buffer.alloc_uninit[int32.native](n)
        var out = buf.view[int32.native](0, n)

        def scatter(i: Int) {mut out, imm}:
            ref rows = routed[i].row_indices
            ref local = found[i]
            ref l2g = self._local_to_global[i]
            for j in range(len(rows)):
                var id = Int(local.unsafe_get(j))
                out.store[1](
                    Int(rows.unsafe_get(j)), Int32(-1) if id < 0 else l2g[id]
                )

        self._ctx.run(len(routed), scatter)
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )

    def _find_in_place(self, hashes: UInt64Array) raises -> Int32Array:
        """``find`` on the calling thread, each hash looked up in its own
        partition's table and prefetched ``_PIPE_DEPTH`` hashes ahead, as the
        serial table does."""
        var n = len(hashes)
        var ids = Buffer.alloc_uninit[int32.native](max(n, 1))
        for i in range(min(_PIPE_DEPTH, n)):
            var h = UInt64(hashes.unsafe_get(i))
            self._parts[RadixPartitioner.partition_of(h, _RADIX_BITS)].prefetch(
                h
            )
        for i in range(n):
            if i + _PIPE_DEPTH < n:
                var ahead = UInt64(hashes.unsafe_get(i + _PIPE_DEPTH))
                self._parts[
                    RadixPartitioner.partition_of(ahead, _RADIX_BITS)
                ].prefetch(ahead)
            var h = UInt64(hashes.unsafe_get(i))
            var p = RadixPartitioner.partition_of(h, _RADIX_BITS)
            var local = self._parts[p].find(h)
            ids.unsafe_set[int32.native](
                i, Int32(-1) if local < 0 else self._local_to_global[p][local]
            )
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=ids^.to_immutable()
        )

    # --- one hash, the caller's equality ------------------------------------

    def find_one[
        Eq: def(Int) raises -> Bool
    ](self, h: UInt64, eq: Eq) raises -> Int:
        """The id holding ``h`` that ``eq`` accepts, or ``-1``. ``eq`` is asked
        about ids, never about partition-local numbers."""
        if not self.is_partitioned():
            return self._table.find_one(h, eq)
        var p = RadixPartitioner.partition_of(h, _RADIX_BITS)

        def local_eq(local: Int) raises {imm} -> Bool:
            return eq(Int(self._local_to_global[p][local]))

        var local = self._parts[p].find_one(h, local_eq)
        return -1 if local < 0 else Int(self._local_to_global[p][local])

    def insert_new(mut self, h: UInt64) raises -> Int:
        """A new id for ``h`` without looking for one it already has — for a
        hash whose holders ``find_one`` turned down. Reachable by
        ``find_one`` from then on."""
        var id = self._len
        self._check_id_space(id + 1)
        if self.is_partitioned():
            var p = RadixPartitioner.partition_of(h, _RADIX_BITS)
            _ = self._parts[p].insert_new(h)
            self._local_to_global[p].append(Int32(id))
        else:
            _ = self._table.insert_new(h)
        self._len += 1
        return id

    # --- placement ----------------------------------------------------------

    @staticmethod
    def _check_id_space(n: Int) raises:
        """Ids are Int32 all the way into an accumulator's scatter; past 2^31
        they would truncate negative while the count stayed right."""
        if n > Int(Int32.MAX):
            raise InvalidError(
                t"hash index: {n} ids exceeds the Int32 id space"
            )

    def _insert_serial(mut self, hashes: UInt64Array) raises -> Placement:
        """Placement through one table: its ids are the index's ids."""
        if self._reserved > 0:
            self._table.reserve(len(self._table) + self._reserved)
        var placed = self._table.insert_hashes(hashes)
        self._check_id_space(len(self._table))
        self._len = len(self._table)
        return placed^

    def _migrate(mut self) raises:
        """Move every id onto 64 partitioned tables, keeping its number.

        A move and not a flag flip: flipping with ids in ``_table`` would
        strand them, and a hash seen before would get a second id.
        ``_table.hashes()`` lists one hash per id in id order; partitioning it
        routes each to the table its later rows will probe, and re-inserting
        with ``insert_new`` — never a lookup, since two ids may share a hash —
        hands out local ids in that order, so appending the ids is the whole
        mapping. Committed last, so a raise leaves the index serial.
        """
        if self.is_partitioned():
            raise InternalError("HashIndex._migrate: already partitioned")
        var p = 1 << _RADIX_BITS
        var parts = List[SwissHashTable](capacity=p)
        var l2g = List[List[Int32]](capacity=p)
        for _ in range(p):
            parts.append(SwissHashTable())
            l2g.append(List[Int32]())
        var existing = self._table.hashes()
        if len(existing) > 0:
            var routed = self._partitioner().partition(existing^)
            for i in range(p):
                ref ids = routed[i].row_indices
                ref part_hashes = routed[i].hashes
                if len(ids) > 0:
                    parts[i].reserve(len(ids))
                    l2g[i].reserve(len(ids))
                    for k in range(len(ids)):
                        _ = parts[i].insert_new(
                            UInt64(part_hashes.unsafe_get(k))
                        )
                        l2g[i].append(ids.unsafe_get(k))
        self._parts = parts^
        self._local_to_global = l2g^
        self._table = SwissHashTable()

    def _insert_radix(mut self, hashes: UInt64Array) raises -> Placement:
        """Placement across the 64 tables, with nothing O(rows) and serial.

        1. One radix pass, then one worker per partition inserting into its own
           table and finding the first row of each id it creates.
        2. A prefix sum over 64 new-id counts lays out the id space: partition
           ``i`` takes the contiguous block starting at ``base[i]``.
        3. One worker per partition records its new ids, writes their first
           rows into the shared block, and scatters every row's id back.

        An earlier version numbered ids by one serial scan over rows, which
        cost more than the parallel insert saved (1M rows, 1,000 ids: 2.7 ms
        serial against 6.4 ms on 8 workers).
        """
        var p = 1 << _RADIX_BITS

        # Every partition's map must be as long as its table: step 3 appends
        # onto it. A worker that raised mid-batch would leave a table ahead of
        # its map, and every later id in that partition wrong — so a mismatch
        # is refused rather than continued.
        var prev = List[Int](capacity=p)
        for i in range(p):
            var here = len(self._parts[i])
            if len(self._local_to_global[i]) != here:
                raise InternalError(
                    t"hash index: partition {i} holds {here} ids against "
                    t"{len(self._local_to_global[i])} mapped; a previous "
                    t"insert failed partway"
                )
            prev.append(here)

        def insert_partition(
            i: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises {mut self, imm} -> Placement:
            var before = len(self._parts[i])
            # Sized once rather than doubled from `_GROUP_WIDTH` — every
            # doubling re-places every id (13.3% of this path in a profile).
            # A reservation is shared out by rows; without one, half the
            # partition's rows: the gate says only "at least
            # `_RADIX_MIN_DISTINCT`", and a high guess costs one power of two.
            var expect = len(part_hashes) // 2
            if self._reserved > 0:
                expect = ceildiv(self._reserved * len(part_hashes), len(hashes))
            self._parts[i].reserve(before + expect)
            var placed = self._parts[i].insert_hashes(part_hashes)
            # The table reports positions within the partition; the batch's
            # rows are what a consumer stores from.
            ref local_firsts = placed.firsts
            var firsts = Buffer.alloc_uninit[int32.native](
                max(len(local_firsts), 1)
            )
            for k in range(len(local_firsts)):
                firsts.unsafe_set[int32.native](
                    k, rows.unsafe_get(Int(local_firsts.unsafe_get(k)))
                )
            return Placement(
                ids=placed.ids.copy(),
                firsts=Int32Array(
                    length=len(local_firsts),
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    buffer=firsts^.to_immutable(),
                ),
            )

        var split = self._partitioner().map_partitions[Placement](
            hashes.copy(), insert_partition
        )
        ref routed = split[0]
        ref per_part = split[1]

        var start = self._len
        var base = List[Int](capacity=p)
        var running = start
        for i in range(p):
            base.append(running)
            running += len(self._parts[i]) - prev[i]
        self._check_id_space(running)
        var new_total = running - start
        self._len = running

        var n = len(hashes)
        var id_buf = Buffer.alloc_uninit[int32.native](n)
        var id_view = id_buf.view[int32.native](0, n)
        var first_cap = max(new_total, 1)
        var first_buf = Buffer.alloc_uninit[int32.native](first_cap)
        var first_view = first_buf.view[int32.native](0, first_cap)
        # Taken before the closure, which borrows `self` mutably.
        var ctx = self._ctx.copy()

        def finish_partition(i: Int) {mut self, imm}:
            var pk = prev[i]
            var b = base[i]
            var fo = b - start
            # Built locally and spliced in once: the 64 inner `List` headers
            # share cache lines, and an append per new id would bounce them.
            ref l2g = self._local_to_global[i]
            ref firsts = per_part[i].firsts
            var mine = List[Int32](capacity=len(firsts))
            for k in range(len(firsts)):
                mine.append(Int32(b + k))
                first_view.store[1](fo + k, firsts.unsafe_get(k))
            l2g.extend(mine^)
            ref rows = routed[i].row_indices
            ref local = per_part[i].ids
            for j in range(len(rows)):
                # A local id from this batch maps by arithmetic; only one
                # carried in from an earlier batch needs the map.
                var lid = Int(local.unsafe_get(j))
                var id = Int32(b + lid - pk) if lid >= pk else l2g[lid]
                id_view.store[1](Int(rows.unsafe_get(j)), id)

        ctx.run(p, finish_partition)
        return Placement(
            ids=Int32Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=id_buf^.to_immutable(),
            ),
            firsts=Int32Array(
                length=new_total,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=first_buf^.to_immutable(),
            ),
        )
