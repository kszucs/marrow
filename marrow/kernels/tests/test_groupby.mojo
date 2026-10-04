# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for group-by placement through `DictionaryEncoder` — its two
placement paths, and its exactness under hash collisions.

The radix path only engages at `_RADIX_MIN_ROWS` (50k) rows and
`_RADIX_MIN_DISTINCT` (30k) groups with a context that resolves to more than one
worker, so **every case here that means to test it has to be that big**. A 5-row parallel context takes the serial path
and asserts nothing about the code it was written for; that is why the sizes
below look gratuitous and are not.

The property under test is that the two paths produce **the same partition of
rows into groups**, and the same key row per group — *not* the same ids. The
radix path numbers partition-major, so its ids are a renumbering of the serial
path's. `_bijection` is what makes that precise: it pairs the two numberings
row by row and fails if one path ever splits a group the other merged, or
merges one the other split. That is the whole of what `GROUP BY` promises, and
it is what lets every aggregate stay untouched — a fold still sees every row of
its group in one accumulator, so `mean`, the Welford variance triple and
`count_distinct` never learn that placement was parallel.

An earlier version of this file asserted id-for-id equality, and the
implementation paid for it with an O(rows) *serial* numbering pass that cost
more than the parallel insert saved. Do not reinstate that assertion without
re-reading `HashIndex._insert_radix`.

**Placement tests name `RapidHash64`**, never the build's `KeyHash`: whether
the radix path engages depends on how many distinct hashes the cardinality
sample sees, so under `-D MARROW_HASH_BITS` (`pixi run test_collisions`) they
would be testing the truncation rather than the placement. Exactness under
collisions is the `test_collisions_*` cases', which choose their hasher.
"""

from std.math import nan
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import (
    DictionaryArray,
    DynArray,
    Int32Array,
    Int64Array,
    StringViewArray,
    StructArray,
)
from ...buffers import Bitmap
from ...builders import (
    Decimal128Builder,
    Float64Builder,
    Int8Builder,
    Int32Builder,
    LargeStringBuilder,
    StringBuilder,
    Int64Builder,
    ListBuilder,
    StructBuilder,
    array,
)
from ...dtypes import (
    Decimal128Type,
    DynType,
    Field,
    Int8Type,
    Float64Type,
    Int16Type,
    Int32Type,
    Int64Type,
    LargeStringType,
    StringType,
    decimal128,
    dictionary,
    field,
    int8,
    int16,
    int32,
    int64,
    string,
    struct_,
)
from ...kernels.filter import take
from ...kernels.cast import decode_dictionary
from ...kernels.hashing import HashKernel, NULL_HASH_SENTINEL
from ...kernels.hashtable import SwissHashTable
from ...utils import Fmix64, Hasher, RapidHash64, TruncatedHash64
from ...execution import ExecContext
from ...kernels.aggregate import (
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    MeanFold,
    SumFold,
)
from ...kernels.dictionary import DictionaryEncoder
from ...kernels.groupby import Groups


def _encoder[
    H: Hasher = RapidHash64
](keys: List[DynArray], var ctx: ExecContext) -> DictionaryEncoder[H]:
    """An encoder over columns of `keys`' types."""
    var types = List[DynType]()
    for ref k in keys:
        types.append(k.dtype())
    return DictionaryEncoder[H](types^, ctx^)


def _int32_encoder(var ctx: ExecContext) -> DictionaryEncoder[RapidHash64]:
    var types = List[DynType]()
    types.append(int32)
    return DictionaryEncoder[RapidHash64](types^, ctx^)


def _groups[
    H: Hasher
](mut encoder: DictionaryEncoder[H], keys: List[DynArray]) raises -> Groups:
    """One batch placed: its codes, and how many exist after it."""
    var placed = encoder.encode(keys)
    return Groups(placed.ids.copy(), len(encoder))


comptime _BIG: Int = 100_000
"""Comfortably over the 50k `_RADIX_MIN_ROWS` threshold."""


def _int_keys(n: Int, card: Int) raises -> DynArray:
    """`n` int32 keys over `card` distinct values, deliberately not in order.

    The stride is coprime with `card`, so first appearances are interleaved
    rather than blocked — a numbering that only agreed for sorted input would
    still pass a blocked pattern.
    """
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32((i * 7919) % card))
    return b.finish()


def _sorted_keys(n: Int, card: Int) raises -> DynArray:
    """`n` int32 keys over `card` distinct values, in runs: the blocked layout
    `_int_keys` avoids, for the cases where the layout is the point."""
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32(i * card // n))
    return b.finish()


def _payload(n: Int) raises -> DynArray:
    var b = Int32Builder(capacity=n)
    for i in range(n):
        b.append(Int32(i % 1000))
    return b.finish()


def _bijection(
    left: Int32Array, right: Int32Array, num_groups: Int
) raises -> List[Int]:
    """Pair two group numberings of the same rows; fail if they disagree.

    Returns `fwd`, where `fwd[right_id] == left_id`. The first row carrying a
    given `right_id` fixes its partner, and every later row must agree in both
    directions — so a group that one path split and the other did not fails
    here, and so does the reverse.
    """
    assert_equal(len(left), len(right))
    var fwd = List[Int](length=num_groups, fill=-1)
    var rev = List[Int](length=num_groups, fill=-1)
    for i in range(len(left)):
        var l = Int(left.unsafe_get(i))
        var r = Int(right.unsafe_get(i))
        if fwd[r] == -1 and rev[l] == -1:
            fwd[r] = l
            rev[l] = r
        assert_equal(fwd[r], l)
        assert_equal(rev[l], r)
    return fwd^


def _any_id_differs(left: Int32Array, right: Int32Array) -> Bool:
    """Whether two numberings differ anywhere — how a test tells that the radix
    path actually ran, since it numbers partition-major and the serial path
    numbers by first appearance."""
    for i in range(len(left)):
        if left.unsafe_get(i) != right.unsafe_get(i):
            return True
    return False


struct _Placed(Movable):
    """What one grouping run produced — all of it, so equality is total."""

    var ids: Int32Array
    var num_groups: Int
    var keys: List[DynArray]

    def __init__(
        out self, var ids: Int32Array, num_groups: Int, var keys: List[DynArray]
    ):
        self.ids = ids^
        self.num_groups = num_groups
        self.keys = keys^


def _place(var key: DynArray, n: Int, var ctx: ExecContext) raises -> _Placed:
    """Group one batch of a single key column under `ctx`."""
    var cols = List[DynArray]()
    cols.append(key^)
    var g = _encoder(cols, ctx^)
    var groups = _groups(g, cols)
    var fields = List[Field]()
    fields.append(Field("k", int32))
    return _Placed(groups.ids.copy(), groups.num_groups, g.values())


def _assert_same_placement(var a: _Placed, var b: _Placed) raises:
    """Serial and radix placement agree on everything a fold can observe: the
    same grouping of rows, and the same key value stored for each group."""
    assert_equal(a.num_groups, b.num_groups)
    var fwd = _bijection(a.ids, b.ids, a.num_groups)
    assert_equal(len(a.keys), len(b.keys))
    for c in range(len(a.keys)):
        ref ak = a.keys[c].as_int32()
        ref bk = b.keys[c].as_int32()
        assert_equal(len(ak), a.num_groups)
        assert_equal(len(bk), b.num_groups)
        for q in range(b.num_groups):
            assert_equal(bk[q].value(), ak[fwd[q]].value())


# ---------------------------------------------------------------------------
# Placement equivalence — the core contract
# ---------------------------------------------------------------------------


def test_low_cardinality_stays_serial_under_a_parallel_context() raises:
    """1,000 groups over 100k rows: far under `_RADIX_MIN_DISTINCT`, so the
    cardinality gate keeps placement on the single-table path even though the
    row count and the worker count would both allow radix.

    Asserted through the *numbering*, which is the only externally visible
    difference between the paths: radix is partition-major, so had it run these
    ids would be a renumbering rather than identical.
    """
    var serial = _place(_int_keys(_BIG, 1_000), _BIG, ExecContext.serial())
    var par = _place(_int_keys(_BIG, 1_000), _BIG, ExecContext.parallel(4))
    assert_equal(serial.num_groups, 1_000)
    assert_true(serial.ids == par.ids)
    _assert_same_placement(serial^, par^)


def test_cardinality_gate_ignores_row_order() raises:
    """The gate answers the same for sorted and interleaved keys: 10,000 groups
    stay serial either way, and 50,000 take radix either way.

    The probe used to walk an arithmetic stride, and that made the answer a
    function of the layout: on these 100k rows it sampled 10,000 *sorted*
    groups as all-distinct and sent them to radix, while the same 10,000
    groups interleaved sampled at half that and stayed serial. Random draws
    see a key distribution rather than a layout — both layouts sample within
    ten values of the expected 3,360, against a gate at 3,828.
    """
    var ctx = ExecContext.parallel(4)
    var small = List[DynArray]()
    small.append(_int_keys(_BIG, 10_000))
    small.append(_sorted_keys(_BIG, 10_000))
    for i in range(len(small)):
        var serial = _place(small[i].copy(), _BIG, ExecContext.serial())
        var par = _place(small[i].copy(), _BIG, ctx.copy())
        assert_equal(serial.num_groups, 10_000)
        assert_true(serial.ids == par.ids)

    var large = List[DynArray]()
    large.append(_int_keys(_BIG, 50_000))
    large.append(_sorted_keys(_BIG, 50_000))
    for i in range(len(large)):
        var serial = _place(large[i].copy(), _BIG, ExecContext.serial())
        var par = _place(large[i].copy(), _BIG, ctx.copy())
        assert_equal(serial.num_groups, 50_000)
        assert_true(_any_id_differs(serial.ids, par.ids))
        _assert_same_placement(serial^, par^)


def test_radix_placement_matches_serial_high_cardinality() raises:
    """50k groups over 100k rows — the insert-heavy case radix exists for."""
    var serial = _place(_int_keys(_BIG, 50_000), _BIG, ExecContext.serial())
    var par = _place(_int_keys(_BIG, 50_000), _BIG, ExecContext.parallel(4))
    assert_equal(serial.num_groups, 50_000)
    # Radix really ran: partition-major numbering cannot coincide with
    # first-appearance order across 50,000 groups.
    assert_true(_any_id_differs(serial.ids, par.ids))
    _assert_same_placement(serial^, par^)


def test_radix_placement_matches_serial_all_distinct() raises:
    """Every row its own group — maximum table growth per partition."""
    var serial = _place(_int_keys(_BIG, _BIG), _BIG, ExecContext.serial())
    var par = _place(_int_keys(_BIG, _BIG), _BIG, ExecContext.parallel(8))
    assert_equal(serial.num_groups, _BIG)
    _assert_same_placement(serial^, par^)


def test_radix_placement_matches_serial_on_auto_context() raises:
    """`ExecContext.auto()` is what `execute()` passes by default, so this is
    the configuration a real query actually takes."""
    var serial = _place(_int_keys(_BIG, 50_000), _BIG, ExecContext.serial())
    var par = _place(_int_keys(_BIG, 50_000), _BIG, ExecContext.auto())
    _assert_same_placement(serial^, par^)


def test_radix_placement_matches_serial_with_string_keys() raises:
    """Strings hash through a different kernel path than fixed-width keys, and
    the grouper hashes with the caller's context now rather than serially."""
    var sb = StringBuilder(_BIG)
    var sb2 = StringBuilder(_BIG)
    for i in range(_BIG):
        var s = String("key-") + String((i * 7919) % 60_000)
        sb.append(s)
        sb2.append(s)

    var a = List[DynArray]()
    a.append(sb.finish())
    var ga = _encoder(a, ExecContext.serial())
    var serial_groups = _groups(ga, a)

    var b = List[DynArray]()
    b.append(sb2.finish())
    var gb = _encoder(b, ExecContext.parallel(4))
    var par_groups = _groups(gb, b)

    assert_equal(serial_groups.num_groups, 60_000)
    assert_equal(par_groups.num_groups, 60_000)
    _ = _bijection(serial_groups.ids, par_groups.ids, 60_000)


def test_radix_placement_matches_serial_with_two_keys() raises:
    """Two key columns hash into one row hash; partitioning must not change
    which rows are considered equal."""
    var ka = Int32Builder(capacity=_BIG)
    var kb = Int32Builder(capacity=_BIG)
    var ka2 = Int32Builder(capacity=_BIG)
    var kb2 = Int32Builder(capacity=_BIG)
    for i in range(_BIG):
        var x = Int32((i * 7919) % 997)
        var y = Int32((i * 104_729) % 53)
        ka.append(x)
        kb.append(y)
        ka2.append(x)
        kb2.append(y)

    var s = List[DynArray]()
    s.append(ka.finish())
    s.append(kb.finish())
    var gs = _encoder(s, ExecContext.serial())
    var sg = _groups(gs, s)

    var p = List[DynArray]()
    p.append(ka2.finish())
    p.append(kb2.finish())
    var gp = _encoder(p, ExecContext.parallel(4))
    var pg = _groups(gp, p)

    assert_equal(sg.num_groups, pg.num_groups)
    _ = _bijection(sg.ids, pg.ids, sg.num_groups)


def test_radix_placement_handles_null_keys() raises:
    """NULL keys group together under SQL `GROUP BY`, and must keep doing so
    when the null rows are spread across partitions."""
    var a = Int32Builder(capacity=_BIG)
    var b = Int32Builder(capacity=_BIG)
    for i in range(_BIG):
        if i % 97 == 0:
            a.append_null()
            b.append_null()
        else:
            a.append(Int32(i % 60_000))
            b.append(Int32(i % 60_000))

    var sc = List[DynArray]()
    sc.append(a.finish())
    var gs = _int32_encoder(ExecContext.serial())
    var sg = _groups(gs, sc)

    var pc = List[DynArray]()
    pc.append(b.finish())
    var gp = _int32_encoder(ExecContext.parallel(4))
    var pg = _groups(gp, pc)

    assert_equal(sg.num_groups, pg.num_groups)
    _ = _bijection(sg.ids, pg.ids, sg.num_groups)


# ---------------------------------------------------------------------------
# Multi-batch — ids must stay stable, which is what lets a fold keep its slots
# ---------------------------------------------------------------------------


def test_radix_ids_are_stable_across_batches() raises:
    """A key seen in batch 1 keeps its id in batch 2.

    This is the invariant the per-partition tables have to be *persistent* for:
    a fresh set of tables per batch would renumber every key and silently
    corrupt any accumulator that had already folded batch 1. Here the two
    batches carry identical values, so the ids must come back identical —
    exactly, not merely up to renumbering.
    """
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_BIG, 50_000))
    var g1 = _groups(g, first)
    assert_equal(g1.num_groups, 50_000)

    # Same key values again — no group is new, so the count must not move.
    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = _groups(g, second)
    assert_equal(g2.num_groups, 50_000)
    assert_true(g1.ids == g2.ids)

    var fields = List[Field]()
    fields.append(Field("k", int32))
    var cols = g.values()
    assert_equal(len(cols[0]), 50_000)


def test_radix_second_batch_extends_the_grouping() raises:
    """New keys in a later batch append, they do not renumber."""
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_BIG, 60_000))
    var g1 = _groups(g, first)
    assert_equal(g1.num_groups, 60_000)

    var b = Int32Builder(capacity=_BIG)
    for i in range(_BIG):
        b.append(Int32(60_000 + ((i * 7919) % 30_000)))
    var second = List[DynArray]()
    second.append(b.finish())
    var g2 = _groups(g, second)
    assert_equal(g2.num_groups, 90_000)
    # Every key in the second batch is new, so all of its ids land past the
    # block the first batch already claimed.
    for i in range(0, _BIG, 997):
        assert_true(Int(g2.ids[i].value()) >= 60_000)


def test_below_threshold_stays_serial_under_a_parallel_context() raises:
    """A small batch takes the single-table path even when threads are forced —
    `worth_parallel` reads a forced count as a budget, not an instruction."""
    var small = 1_000
    var serial = _place(_int_keys(small, 37), small, ExecContext.serial())
    var par = _place(_int_keys(small, 37), small, ExecContext.parallel(4))
    assert_equal(serial.num_groups, 37)
    # Both took the serial path, so this is exact and not merely a bijection.
    assert_true(serial.ids == par.ids)
    _assert_same_placement(serial^, par^)


def test_empty_batch_under_a_parallel_context() raises:
    """A zero-row batch must not latch a path or invent a group."""
    var g = _int32_encoder(ExecContext.parallel(4))
    var empty = List[DynArray]()
    var b = Int32Builder(0)
    empty.append(b.finish())
    var got = _groups(g, empty)
    assert_equal(got.num_groups, 0)
    assert_equal(len(got.ids), 0)


# ---------------------------------------------------------------------------
# Folds over the parallel placement — no aggregate state is merged, so these
# are checks that the ids really do reach the accumulator intact.
# ---------------------------------------------------------------------------


def test_grouped_sum_agrees_between_paths() raises:
    """`sum` over radix placement equals `sum` over serial placement, group for
    group under the renumbering.

    Integer addition, so this is exact — and the rows of a group are folded in
    ascending row order on both paths, since the scatter writes by original row
    index. No reassociation is involved even for a float column.
    """
    var sk = List[DynArray]()
    sk.append(_int_keys(_BIG, 50_000))
    var gs = _int32_encoder(ExecContext.serial())
    var sgroups = _groups(gs, sk)
    var ssum = Fold[SumFold, Int32Type].grouped(
        sgroups, _payload(_BIG).as_int32().copy()
    )

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, 50_000))
    var gp = _int32_encoder(ExecContext.parallel(4))
    var pgroups = _groups(gp, pk)
    var psum = Fold[SumFold, Int32Type].grouped(
        pgroups, _payload(_BIG).as_int32().copy()
    )

    assert_equal(len(ssum), 50_000)
    assert_equal(len(psum), 50_000)
    var fwd = _bijection(sgroups.ids, pgroups.ids, 50_000)
    for q in range(50_000):
        assert_equal(psum[q].value(), ssum[fwd[q]].value())


def test_grouped_mean_agrees_between_paths() raises:
    """`mean` keeps `(sum, count)` and divides at finish. Radix placement never
    splits a group across accumulators, so there are no partial means to
    combine — the divisor is the group's whole count on both paths."""
    var sk = List[DynArray]()
    sk.append(_int_keys(_BIG, 60_000))
    var gs = _int32_encoder(ExecContext.serial())
    var sgroups = _groups(gs, sk)
    var smean = Fold[MeanFold, Int32Type].grouped(
        sgroups, _payload(_BIG).as_int32().copy()
    )

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, 60_000))
    var gp = _int32_encoder(ExecContext.parallel(4))
    var pgroups = _groups(gp, pk)
    var pmean = Fold[MeanFold, Int32Type].grouped(
        pgroups, _payload(_BIG).as_int32().copy()
    )

    assert_equal(len(smean), 60_000)
    var fwd = _bijection(sgroups.ids, pgroups.ids, 60_000)
    for q in range(60_000):
        assert_equal(pmean[q].value(), smean[fwd[q]].value())


def _in[A: AggKernel](column: DynArray) raises -> A.InArray:
    """An erased column narrowed to whatever the kernel under test eats — the
    same helper `test_agg_kernels.mojo` uses."""
    return A.InArray(column.to_data())


def _int64_payload(n: Int) raises -> DynArray:
    var b = Int64Builder(capacity=n)
    for i in range(n):
        b.append(Int64(i % 500))
    return b.finish()


def _grouped_pair(card: Int) raises -> Tuple[Groups, Groups]:
    """The same key column placed twice — once serial, once radix.

    `card` must clear the distinctness gate or the second grouping quietly
    takes the serial path and the comparison tests nothing.
    """
    var sk = List[DynArray]()
    sk.append(_int_keys(_BIG, card))
    var gs = _int32_encoder(ExecContext.serial())
    var sgroups = _groups(gs, sk)

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, card))
    var gp = _int32_encoder(ExecContext.parallel(4))
    var pgroups = _groups(gp, pk)

    assert_equal(sgroups.num_groups, pgroups.num_groups)
    assert_true(_any_id_differs(sgroups.ids, pgroups.ids))
    return (sgroups^, pgroups^)


def test_grouped_variance_agrees_between_paths() raises:
    """Sample variance over radix placement equals sample variance over serial
    placement, group for group.

    **This is the case that would be silently wrong under the design this one
    was chosen over.** `Dispersion` keeps Welford's `(n, mean, M2)` triple per
    slot, and combining two partial triples needs the Chan/Golub/LeVeque
    correction term — `M2_a + M2_b` is not the union's `M2`, and averaging two
    partial means is not the union's mean. Radix placement never creates a
    partial: a group lives in exactly one partition and therefore in exactly
    one accumulator, so the triple is only ever updated, never combined.

    Equality is exact, not approximate. Both paths visit a group's rows in
    ascending row order, so the Welford recurrence sees the identical sequence
    and no reassociation occurs.
    """
    var pair = _grouped_pair(60_000)
    var payload = _payload(_BIG)
    var sv = Dispersion[1, False, Int32Type].grouped(
        pair[0], _in[Dispersion[1, False, Int32Type]](payload.copy())
    )
    var pv = Dispersion[1, False, Int32Type].grouped(
        pair[1], _in[Dispersion[1, False, Int32Type]](payload.copy())
    )
    assert_equal(len(sv), 60_000)
    var fwd = _bijection(pair[0].ids, pair[1].ids, 60_000)
    var checked = 0
    for q in range(60_000):
        assert_equal(pv.is_valid(q), sv.is_valid(fwd[q]))
        if pv.is_valid(q):
            assert_equal(pv[q].value(), sv[fwd[q]].value())
            checked += 1
    assert_true(checked > 0)


def test_grouped_stddev_agrees_between_paths() raises:
    """`root=True` takes the square root of the same triple, so it inherits the
    argument above; asserted separately because it is a distinct
    instantiation."""
    var pair = _grouped_pair(60_000)
    var payload = _payload(_BIG)
    var ss = Dispersion[0, True, Int32Type].grouped(
        pair[0], _in[Dispersion[0, True, Int32Type]](payload.copy())
    )
    var ps = Dispersion[0, True, Int32Type].grouped(
        pair[1], _in[Dispersion[0, True, Int32Type]](payload.copy())
    )
    var fwd = _bijection(pair[0].ids, pair[1].ids, 60_000)
    for q in range(60_000):
        assert_equal(ps.is_valid(q), ss.is_valid(fwd[q]))
        if ps.is_valid(q):
            assert_equal(ps[q].value(), ss[fwd[q]].value())


def test_grouped_count_distinct_agrees_between_paths() raises:
    """Exact `count_distinct` is the fold with **no correct merge at all** — its
    state is one hash table over `(group, value)` pairs, so two thread-local
    tables would carry incompatible bucket numbering and double-count any value
    both threads saw. Radix placement never splits a group, so the question
    never arises."""
    var pair = _grouped_pair(60_000)
    var payload = _int64_payload(_BIG)
    var sd = DistinctCount[Int64Array].grouped(
        pair[0], _in[DistinctCount[Int64Array]](payload.copy())
    )
    var pd = DistinctCount[Int64Array].grouped(
        pair[1], _in[DistinctCount[Int64Array]](payload.copy())
    )
    assert_equal(len(sd), 60_000)
    var fwd = _bijection(pair[0].ids, pair[1].ids, 60_000)
    for q in range(60_000):
        assert_equal(pd[q].value(), sd[fwd[q]].value())


# ---------------------------------------------------------------------------
# Crossing from serial to radix mid-stream. The placement choice used to be
# made from the first non-empty batch alone, so one small morsel pinned a whole
# query to the serial path no matter what followed it.
# ---------------------------------------------------------------------------


comptime _SMALL: Int = 1_000
"""Under the 50k threshold, so a batch this size cannot qualify on its own."""

comptime _SMALL_CARD: Int = 37
"""Keys 0..36 — every one of them reappears in the large batch below."""


def _two_batch(
    var ctx: ExecContext,
) raises -> Tuple[Int32Array, Int32Array, Int]:
    """Push a below-threshold batch and then a qualifying one through a single
    grouper. Returns both batches' ids and the final group count."""
    var g = _int32_encoder(ctx^)

    var first = List[DynArray]()
    first.append(_int_keys(_SMALL, _SMALL_CARD))
    var g1 = _groups(g, first)

    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = _groups(g, second)
    return (g1.ids.copy(), g2.ids.copy(), g2.num_groups)


def test_small_first_batch_does_not_pin_the_grouper_to_serial() raises:
    """A 1,000-row opener must not cost the 100,000-row batch behind it its
    placement. The tell is that the second batch's ids stop matching a
    forced-serial run's: radix numbers partition-major, so if the numbering
    still agrees row for row, the migration never happened."""
    var par = _two_batch(ExecContext.parallel(4))
    var ser = _two_batch(ExecContext.serial())

    assert_equal(par[2], 50_000)
    assert_equal(ser[2], 50_000)
    assert_true(_any_id_differs(par[1], ser[1]))


def test_migration_agrees_with_serial_on_both_batches() raises:
    """Crossing mid-stream still groups the rows the way one serial grouper
    would have, in both batches — one bijection covering the pair, so a key
    that changed id between them would fail here."""
    var par = _two_batch(ExecContext.parallel(4))
    var ser = _two_batch(ExecContext.serial())

    _ = _bijection(ser[0], par[0], par[2])
    _ = _bijection(ser[1], par[1], par[2])


def test_migration_preserves_ids_issued_before_the_switch() raises:
    """The invariant the migration exists for: a key numbered by the serial
    table keeps that number once placement moves to the partitioned tables.

    Any accumulator the caller has already folded batch 1 into is indexed by
    those ids, so a key that came back renumbered would split its own group
    across two slots — with no error anywhere.
    """
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_SMALL, _SMALL_CARD))
    var g1 = _groups(g, first)
    assert_equal(g1.num_groups, _SMALL_CARD)

    # What id did the serial table give each key value?
    var id_of = List[Int](length=_SMALL_CARD, fill=-1)
    for i in range(_SMALL):
        id_of[(i * 7919) % _SMALL_CARD] = Int(g1.ids.unsafe_get(i))

    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = _groups(g, second)
    assert_equal(g2.num_groups, 50_000)

    var seen = 0
    for i in range(_BIG):
        var key = (i * 7919) % 50_000
        if key < _SMALL_CARD:
            assert_equal(Int(g2.ids.unsafe_get(i)), id_of[key])
            seen += 1
    # The carried-over keys have to actually occur, or the loop proves nothing.
    assert_true(seen > 0)


def test_migration_keeps_one_key_row_per_group() raises:
    """The key columns still carry one row per group, in global id order,
    across the switch — the migration moves ids, not key storage."""
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_SMALL, _SMALL_CARD))
    _ = _groups(g, first)

    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = _groups(g, second)

    var fields = List[Field]()
    fields.append(Field("k", int32))
    var cols = g.values()
    assert_equal(len(cols[0]), 50_000)

    # Every key value 0..49,999 appears exactly once.
    ref k = cols[0].as_int32()
    var count = List[Int](length=50_000, fill=0)
    for i in range(g2.num_groups):
        count[Int(k[i].value())] += 1
    for v in range(50_000):
        assert_equal(count[v], 1)


def test_empty_batch_after_migration() raises:
    """A zero-row batch once the grouper is already on the radix path.

    `consume_keys` returns before the placement test, so this exercises the
    early-out with 64 live tables behind it rather than with none — and must
    leave the group count where the qualifying batch left it.
    """
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_BIG, 50_000))
    var g1 = _groups(g, first)
    assert_equal(g1.num_groups, 50_000)

    var none = List[DynArray]()
    var b = Int32Builder(0)
    none.append(b.finish())
    var g2 = _groups(g, none)
    assert_equal(len(g2.ids), 0)
    assert_equal(g2.num_groups, 50_000)


def test_three_batches_after_migration_keep_their_ids() raises:
    """Small, then large (which migrates), then large again.

    Two batches only ever exercise the migration's *seeded* ids against one
    following batch. A third distinguishes ids that survived the migration from
    ids issued after it: both must still resolve, and the first batch's keys
    must answer with the same number all three times.
    """
    var g = _int32_encoder(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_SMALL, _SMALL_CARD))
    var g1 = _groups(g, first)

    var id_of = List[Int](length=_SMALL_CARD, fill=-1)
    for i in range(_SMALL):
        id_of[(i * 7919) % _SMALL_CARD] = Int(g1.ids.unsafe_get(i))

    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = _groups(g, second)
    assert_equal(g2.num_groups, 50_000)

    # A third batch over the same key space adds nothing new, so the count must
    # hold and every id must match what the second batch handed out.
    var third = List[DynArray]()
    third.append(_int_keys(_BIG, 50_000))
    var g3 = _groups(g, third)
    assert_equal(g3.num_groups, 50_000)
    assert_true(g2.ids == g3.ids)

    # And the pre-migration keys still carry the ids the serial table gave them.
    for i in range(_BIG):
        var key = (i * 7919) % 50_000
        if key < _SMALL_CARD:
            assert_equal(Int(g3.ids.unsafe_get(i)), id_of[key])


# ---------------------------------------------------------------------------
# Exactness under hash collisions
#
# Real rapidhash collisions cannot be produced on demand, so these run the
# grouper under `TruncatedHash64[bits]`, which has only `2**bits` distinct
# digests: at `bits = 0` every key shares one hash. A grouping that resolved
# keys by hash alone collapses to a handful of groups; an exact one must answer
# exactly what it answers under the full hash.
# ---------------------------------------------------------------------------


def _group[
    H: Hasher
](batches: List[List[DynArray]], var ctx: ExecContext) raises -> _Placed:
    """Group every batch through one `DictionaryEncoder[H]`, concatenating the ids.
    """
    var g = _encoder[H](batches[0], ctx^)
    var ids = Int32Builder()
    for ref batch in batches:
        var groups = _groups(g, batch)
        for i in range(len(groups.ids)):
            ids.append(groups.ids.unsafe_get(i))
    var fields = List[Field]()
    for ref col in batches[0]:
        fields.append(Field("k", col.dtype()))
    return _Placed(ids.finish(), len(g), g.values())


def _assert_same_grouping(var exact: _Placed, var other: _Placed) raises:
    """The same partition of rows, and the same key row stored per group.

    Keys are compared through their rendering after a `take` into the other
    numbering: `DynArray.__eq__` is structural, and NaN never equals itself.
    """
    assert_equal(other.num_groups, exact.num_groups)
    var fwd = _bijection(exact.ids, other.ids, exact.num_groups)
    var order = Int32Builder(capacity=len(fwd))
    for q in range(len(fwd)):
        order.append(Int32(fwd[q]))
    var idx = order.finish()
    for c in range(len(exact.keys)):
        assert_equal(
            String(_decoded(other.keys[c])),
            String(take(_decoded(exact.keys[c]), idx.copy())),
        )


def _decoded(col: DynArray) raises -> DynArray:
    """A dictionary key column by value: a `take` permutes its indices, not
    its entries, so two equal groupings render differently undecoded."""
    if col.dtype().is_dictionary():
        return decode_dictionary(col.as_dictionary())
    return col.copy()


def _assert_exact(batches: List[List[DynArray]]) raises:
    """Every row lands where the full hash puts it, however much collides."""
    var ctx = ExecContext.serial()
    _assert_same_grouping(
        _group[RapidHash64](batches, ctx.copy()),
        _group[TruncatedHash64[0]](batches, ctx.copy()),
    )
    _assert_same_grouping(
        _group[RapidHash64](batches, ctx.copy()),
        _group[TruncatedHash64[2]](batches, ctx.copy()),
    )


def _one(var col: DynArray) -> List[List[DynArray]]:
    var batch = List[DynArray]()
    batch.append(col^)
    var batches = List[List[DynArray]]()
    batches.append(batch^)
    return batches^


def test_collisions_int64_keys() raises:
    _assert_exact(
        _one(array[Int64Type]([1, 2, None, 1, 3, 2, None, 0, -1], int64))
    )


def test_collisions_float_keys() raises:
    """NaN is one group and `-0.0` is `0.0`, as under the full hash."""
    var n = nan[DType.float64]()
    var b = Float64Builder()
    for v in [1.5, n, -0.0, 0.0, n, 2.5, 1.5]:
        b.append(v)
    b.append_null()
    b.append(0.0)
    b.append_null()
    _assert_exact(_one(b.finish()))


def test_collisions_bool_keys() raises:
    _assert_exact(_one(array([True, False, None, True, False, None])))


def test_collisions_string_keys() raises:
    """The empty string and NULL are different groups."""
    _assert_exact(_one(array(["a", "", None, "b", "a", "", None, "ab", "ba"])))


def test_collisions_string_view_keys() raises:
    """The view layout: inline and out-of-line values sharing a four-byte
    prefix, the empty string and NULL, all colliding."""
    var values = List[Optional[String]]()
    for v in [
        "abcd",
        "",
        "abcd-inline",
        "abcd-a key longer than twelve",
        "abcd",
        "abcd-a key longer than twelvf",
        "abcd-a key longer than twelve",
        "",
    ]:
        values.append(String(v))
    values.append(None)
    values.append(String("abcd-inline"))
    values.append(None)
    _assert_exact(_one(StringViewArray.from_values(values)))


def test_collisions_two_key_columns() raises:
    var batch = List[DynArray]()
    batch.append(array[Int16Type]([1, 1, 2, 2, 1, None, None, 1], int16))
    batch.append(array(["x", "y", "x", "y", "x", "x", None, "y"]))
    var batches = List[List[DynArray]]()
    batches.append(batch^)
    _assert_exact(batches)


def test_collisions_list_keys() raises:
    var lb = ListBuilder(Int32Builder(), capacity=6)
    var child_any = lb.values()
    ref child = child_any.as_int32()
    child.append(1)
    child.append(2)
    lb.append_valid()  # [1, 2]
    child.append(1)
    lb.append_valid()  # [1]
    lb.append_null()  # null
    lb.append_valid()  # []
    child.append(1)
    child.append(2)
    lb.append_valid()  # [1, 2]
    child.append(2)
    child.append(1)
    lb.append_valid()  # [2, 1]
    _assert_exact(_one(lb.finish().to_dyn()))


def test_collisions_struct_keys() raises:
    var sb = StructBuilder([field("a", int32), field("b", string)])
    var a = [1, 1, 2, 1]
    var b = ["x", "y", "x", "x"]
    for i in range(len(a)):
        sb.field_builder(0).as_int32().append(Int32(a[i]))
        sb.field_builder(1).as_string().append(b[i])
        sb.append_valid()
    _assert_exact(_one(sb.finish().to_dyn()))


def test_collisions_across_batches() raises:
    """A key colliding with one from an earlier batch still gets its own group,
    and keeps it when it reappears in a third."""
    var batches = List[List[DynArray]]()
    var chunks: List[List[Optional[Int]]] = [
        [1, 2, 3],
        [4, 1, 5],
        [5, 4, 3, 2, 1, 6],
    ]
    for ref values in chunks:
        var batch = List[DynArray]()
        batch.append(array[Int64Type](values, int64))
        batches.append(batch^)
    _assert_exact(batches)


def test_collisions_on_the_radix_path() raises:
    """20 bits leave ~1M digests for 50,000 keys: about 1,200 colliding pairs,
    yet few enough that the cardinality gate still picks the radix path. A
    narrower truncation never reaches it — at 16 bits the 4,096-row sample
    already sees too few distinct hashes."""
    var batches = _one(_int_keys(_BIG, 50_000))
    var distinct_hashes = SwissHashTable()
    _ = distinct_hashes.insert_hashes(
        HashKernel[TruncatedHash64[20]].dispatch(batches[0][0])
    )
    assert_true(len(distinct_hashes) < 50_000)

    var exact = _group[RapidHash64](batches, ExecContext.serial())
    var radix = _group[TruncatedHash64[20]](batches, ExecContext.parallel(4))
    assert_true(_any_id_differs(exact.ids, radix.ids))
    _assert_same_grouping(exact^, radix^)


def _keys_as(values: Int32Array, strings: Bool) raises -> DynArray:
    """The same keys as int32, or as the strings `"key-<n>"`."""
    if not strings:
        return values.copy()
    var b = StringBuilder(len(values))
    for i in range(len(values)):
        b.append(String("key-") + String(values.unsafe_get(i)))
    return b.finish()


def _migrating_batches(strings: Bool) raises -> List[List[DynArray]]:
    """A small first batch that stays serial, then two large ones that move
    the grouper to the radix path and back over the same keys.

    The first batch is keys 0..4,999: under 20 bits about a dozen of those
    pairs share a hash, so the serial table already holds colliding groups
    when the migration re-routes them."""
    var first = Int32Builder(capacity=5_000)
    for i in range(5_000):
        first.append(Int32(i))
    var batches = List[List[DynArray]]()
    var b1 = List[DynArray]()
    b1.append(_keys_as(first.finish(), strings))
    batches.append(b1^)
    for _ in range(2):
        var big = List[DynArray]()
        big.append(_keys_as(_int_keys(_BIG, 50_000).as_int32().copy(), strings))
        batches.append(big^)
    return batches^


def _assert_exact_across_migration(strings: Bool) raises:
    var batches = _migrating_batches(strings)
    var exact = _group[RapidHash64](batches, ExecContext.serial())
    var migrated = _group[TruncatedHash64[20]](batches, ExecContext.parallel(4))
    assert_true(_any_id_differs(exact.ids, migrated.ids))
    _assert_same_grouping(exact^, migrated^)


def test_collisions_across_a_migration_word_keys() raises:
    _assert_exact_across_migration(strings=False)


def test_collisions_across_a_migration_column_keys() raises:
    _assert_exact_across_migration(strings=True)


def test_collisions_on_the_radix_path_column_keys() raises:
    var batches = _one(
        _keys_as(_int_keys(_BIG, 50_000).as_int32().copy(), strings=True)
    )
    var exact = _group[RapidHash64](batches, ExecContext.serial())
    var radix = _group[TruncatedHash64[20]](batches, ExecContext.parallel(4))
    assert_true(_any_id_differs(exact.ids, radix.ids))
    _assert_same_grouping(exact^, radix^)


def _null_hash_batches(null_first: Bool) raises -> List[List[DynArray]]:
    """NULL and the one int64 value whose hash is `NULL_HASH_SENTINEL`, in
    either order, then again in a second batch: four groups.

    A word key's hash is injective, so exactly one value shares NULL's — the
    value built here, by inverting `Fmix64.mix`."""
    var w = NULL_HASH_SENTINEL
    w ^= w >> 33
    w *= 0x9CB4B2F8129337DB
    w ^= w >> 33
    w *= 0x4F74430C22A54005
    w ^= w >> 33
    assert_equal(Fmix64.mix[1](w)[0], NULL_HASH_SENTINEL)
    var v = Int(Int64(w))
    var first: List[Optional[Int]] = [None, v, 1] if null_first else [
        v,
        None,
        1,
    ]
    var batches = List[List[DynArray]]()
    var b1 = List[DynArray]()
    b1.append(array[Int64Type](first, int64))
    batches.append(b1^)
    var b2 = List[DynArray]()
    b2.append(array[Int64Type]([v, None, 2, v], int64))
    batches.append(b2^)
    return batches^


def test_word_key_sharing_the_null_hash_is_not_null() raises:
    """The value sharing NULL's hash stays apart from NULL in either order, on
    both paths, and across batches."""
    for null_first in [True, False]:
        var batches = _null_hash_batches(null_first)
        for ctx in [ExecContext.serial(), ExecContext.parallel(4)]:
            var placed = _group[RapidHash64](batches, ctx.copy())
            assert_equal(placed.num_groups, 4)


def test_collisions_dictionary_keys() raises:
    """Groups by decoded value across batches whose dictionaries differ — "b"
    sits at index 1 in one and 0 in the other — and emits the key column in
    its declared dictionary type."""
    var i1: List[Optional[Int]] = [0, 1, None, 0]
    var i2: List[Optional[Int]] = [1, 0, 0]
    var batches = List[List[DynArray]]()
    var b1 = List[DynArray]()
    b1.append(
        DictionaryArray.from_arrays(
            array[Int32Type](i1, int32), array(["a", "b"])
        )
    )
    batches.append(b1^)
    var b2 = List[DynArray]()
    b2.append(
        DictionaryArray.from_arrays(
            array[Int32Type](i2, int32), array(["c", "b"])
        )
    )
    batches.append(b2^)
    _assert_exact(batches)

    var placed = _group[RapidHash64](batches, ExecContext.serial())
    assert_equal(placed.num_groups, 4)
    assert_true(placed.keys[0].dtype().is_dictionary())
    assert_equal(
        String(_decoded(placed.keys[0])), String(array(["a", "b", None, "c"]))
    )


def test_collisions_long_strings_sharing_a_prefix() raises:
    """Keys longer than eight bytes that agree on their first eight: the
    group header cannot tell them apart, so the stored bytes past it must."""
    _assert_exact(
        _one(
            array(
                [
                    "abcdefgh-1",
                    "abcdefgh-2",
                    "abcdefgh-1",
                    "abcdefgh-10",
                    "abcdefgh",
                    "abcdefgh-2",
                    "abcdefgh-10",
                    "abcdefghijklmnopqrstuvwxyz-1",
                    "abcdefghijklmnopqrstuvwxyz-2",
                    "abcdefghijklmnopqrstuvwxyz-1",
                ]
            )
        )
    )


# ---------------------------------------------------------------------------
# Dictionary-encoded keys at the edges of their layout
# ---------------------------------------------------------------------------


def _strings_with(prefix: String, n: Int) raises -> DynArray:
    var b = StringBuilder(n)
    for i in range(n):
        b.append(prefix + String(i))
    return b.finish()


def test_dictionary_keys_beyond_their_index_type_raise() raises:
    """Two batches of 100 distinct `dictionary<int8, string>` keys each: the
    200 groups cannot be numbered by int8 indices, so emitting the key column
    in its declared type must fail rather than wrap."""
    var indices = Int8Builder(100)
    for i in range(100):
        indices.append(Int8(i))
    var codes = indices.finish()
    var batches = List[List[DynArray]]()
    for prefix in ["a-", "b-"]:
        var batch = List[DynArray]()
        batch.append(
            DictionaryArray.from_arrays(
                codes.copy().to_dyn(), _strings_with(String(prefix), 100)
            )
        )
        batches.append(batch^)
    var encoder = _encoder[RapidHash64](batches[0], ExecContext.serial())
    for ref batch in batches:
        _ = _groups(encoder, batch)
    assert_equal(len(encoder), 200)
    with assert_raises(contains="int8"):
        _ = encoder.values()


def _struct_key(
    var a: List[Optional[Int]],
    var indices: List[Optional[Int]],
    var entries: List[Optional[String]],
) raises -> List[DynArray]:
    """One `struct<a: int32, d: dictionary<int32, string>>` key column."""
    var children = List[DynArray]()
    children.append(array[Int32Type](a, int32))
    children.append(
        DictionaryArray.from_arrays(
            array[Int32Type](indices, int32), array(entries)
        )
    )
    var fields = List[Field]()
    fields.append(field("a", int32))
    fields.append(field("d", dictionary(int32, string)))
    var batch = List[DynArray]()
    batch.append(StructArray.from_arrays(children^, fields))
    return batch^


def test_struct_keys_with_a_dictionary_child_across_merged_chunks() raises:
    """The stored keys of two batches merge into one chunk, and each batch
    brought its own dictionary: the merged child must still hold the values
    its rows had, so a third batch finds the groups it repeats."""
    var b1 = _struct_key([1, 2], [0, 1], ["x", "y"])
    var b2 = _struct_key([3, 4], [0, 1], ["z", "w"])
    var b3 = _struct_key([1, 4, 2], [1, 0, 2], ["w", "x", "y"])
    var encoder = _encoder[RapidHash64](b1, ExecContext.serial())
    _ = _groups(encoder, b1)
    _ = _groups(encoder, b2)
    var third = _groups(encoder, b3)
    assert_equal(len(encoder), 4)
    assert_true(third.ids == array([0, 3, 1], int32))
    var keys = encoder.values()
    ref key = keys[0].as_struct()
    assert_true(key.field("a").as_int32() == array([1, 2, 3, 4], int32))
    assert_equal(
        String(decode_dictionary(key.field("d").as_dictionary())),
        String(array(["x", "y", "z", "w"])),
    )


def test_null_struct_keys_are_one_group_whatever_their_fields_hold() raises:
    """Rows 1 and 2 are NULL structs over different fields: one key, so one
    group, and the valid row another."""
    var children = List[DynArray]()
    children.append(array([1, 2, 3], int32))
    var batch = List[DynArray]()
    batch.append(
        StructArray(
            dtype=struct_(Field("a", int32)),
            length=3,
            nulls=2,
            offset=0,
            bitmap=Bitmap([True, False, False]).to_immutable(),
            children=children^,
        )
    )
    var encoder = _encoder[RapidHash64](batch, ExecContext.serial())
    var groups = _groups(encoder, batch)
    assert_equal(groups.num_groups, 2)
    assert_true(groups.ids == array([0, 1, 1], int32))
