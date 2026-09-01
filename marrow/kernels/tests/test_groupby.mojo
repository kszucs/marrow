"""Tests for `HashGrouper` — and specifically for its two placement paths.

The radix path only engages at `_PARALLEL_GROUPBY_MIN_ROWS` (60k) rows with a
context that resolves to more than one worker, so **every case here that means
to test it has to be that big**. A 5-row parallel context takes the serial path
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
re-reading `_consume_keys_radix`.
"""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray, Int32Array, Int64Array, StructArray
from ...builders import Int32Builder, StringBuilder, Int64Builder
from ...dtypes import Field, Int32Type, int32, int64, string
from ...execution import ExecContext
from ...kernels.aggregate import (
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    MeanFold,
    SumFold,
)
from ...kernels.groupby import HashGrouping
from ...kernels.groups import Groups


comptime _BIG: Int = 100_000
"""Comfortably over the 60k radix threshold."""


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
    var g = HashGrouping(ctx^)
    var groups = g.assign(cols, n)
    var fields = List[Field]()
    fields.append(Field("k", int32))
    return _Placed(groups.ids.copy(), groups.num_groups, g.key_columns(fields))


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
    """1,000 groups over 100k rows: the sample comes back ~24% distinct, so the
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
        var s = String("key-") + String((i * 7919) % 40_000)
        sb.append(s)
        sb2.append(s)

    var a = List[DynArray]()
    a.append(sb.finish())
    var ga = HashGrouping(ExecContext.serial())
    var serial_groups = ga.assign(a, _BIG)

    var b = List[DynArray]()
    b.append(sb2.finish())
    var gb = HashGrouping(ExecContext.parallel(4))
    var par_groups = gb.assign(b, _BIG)

    assert_equal(serial_groups.num_groups, 40_000)
    assert_equal(par_groups.num_groups, 40_000)
    _ = _bijection(serial_groups.ids, par_groups.ids, 40_000)


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
    var gs = HashGrouping(ExecContext.serial())
    var sg = gs.assign(s, _BIG)

    var p = List[DynArray]()
    p.append(ka2.finish())
    p.append(kb2.finish())
    var gp = HashGrouping(ExecContext.parallel(4))
    var pg = gp.assign(p, _BIG)

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
            a.append(Int32(i % 40_000))
            b.append(Int32(i % 40_000))

    var sc = List[DynArray]()
    sc.append(a.finish())
    var gs = HashGrouping(ExecContext.serial())
    var sg = gs.assign(sc, _BIG)

    var pc = List[DynArray]()
    pc.append(b.finish())
    var gp = HashGrouping(ExecContext.parallel(4))
    var pg = gp.assign(pc, _BIG)

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
    var g = HashGrouping(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_BIG, 50_000))
    var g1 = g.assign(first, _BIG)
    assert_equal(g1.num_groups, 50_000)

    # Same key values again — no group is new, so the count must not move.
    var second = List[DynArray]()
    second.append(_int_keys(_BIG, 50_000))
    var g2 = g.assign(second, _BIG)
    assert_equal(g2.num_groups, 50_000)
    assert_true(g1.ids == g2.ids)

    var fields = List[Field]()
    fields.append(Field("k", int32))
    var cols = g.key_columns(fields)
    assert_equal(len(cols[0]), 50_000)


def test_radix_second_batch_extends_the_grouping() raises:
    """New keys in a later batch append, they do not renumber."""
    var g = HashGrouping(ExecContext.parallel(4))

    var first = List[DynArray]()
    first.append(_int_keys(_BIG, 40_000))
    var g1 = g.assign(first, _BIG)
    assert_equal(g1.num_groups, 40_000)

    var b = Int32Builder(capacity=_BIG)
    for i in range(_BIG):
        b.append(Int32(40_000 + ((i * 7919) % 30_000)))
    var second = List[DynArray]()
    second.append(b.finish())
    var g2 = g.assign(second, _BIG)
    assert_equal(g2.num_groups, 70_000)
    # Every key in the second batch is new, so all of its ids land past the
    # block the first batch already claimed.
    for i in range(0, _BIG, 997):
        assert_true(Int(g2.ids[i].value()) >= 40_000)


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
    var g = HashGrouping(ExecContext.parallel(4))
    var empty = List[DynArray]()
    var b = Int32Builder(0)
    empty.append(b.finish())
    var got = g.assign(empty, 0)
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
    var gs = HashGrouping(ExecContext.serial())
    var sgroups = gs.assign(sk, _BIG)
    var ssum = Fold[SumFold, Int32Type].grouped(
        sgroups, _payload(_BIG).as_int32().copy()
    )

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, 50_000))
    var gp = HashGrouping(ExecContext.parallel(4))
    var pgroups = gp.assign(pk, _BIG)
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
    sk.append(_int_keys(_BIG, 40_000))
    var gs = HashGrouping(ExecContext.serial())
    var sgroups = gs.assign(sk, _BIG)
    var smean = Fold[MeanFold, Int32Type].grouped(
        sgroups, _payload(_BIG).as_int32().copy()
    )

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, 40_000))
    var gp = HashGrouping(ExecContext.parallel(4))
    var pgroups = gp.assign(pk, _BIG)
    var pmean = Fold[MeanFold, Int32Type].grouped(
        pgroups, _payload(_BIG).as_int32().copy()
    )

    assert_equal(len(smean), 40_000)
    var fwd = _bijection(sgroups.ids, pgroups.ids, 40_000)
    for q in range(40_000):
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
    var gs = HashGrouping(ExecContext.serial())
    var sgroups = gs.assign(sk, _BIG)

    var pk = List[DynArray]()
    pk.append(_int_keys(_BIG, card))
    var gp = HashGrouping(ExecContext.parallel(4))
    var pgroups = gp.assign(pk, _BIG)

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
    var pair = _grouped_pair(40_000)
    var payload = _payload(_BIG)
    var sv = Dispersion[1, False, Int32Type].grouped(
        pair[0], _in[Dispersion[1, False, Int32Type]](payload.copy())
    )
    var pv = Dispersion[1, False, Int32Type].grouped(
        pair[1], _in[Dispersion[1, False, Int32Type]](payload.copy())
    )
    assert_equal(len(sv), 40_000)
    var fwd = _bijection(pair[0].ids, pair[1].ids, 40_000)
    var checked = 0
    for q in range(40_000):
        assert_equal(pv.is_valid(q), sv.is_valid(fwd[q]))
        if pv.is_valid(q):
            assert_equal(pv[q].value(), sv[fwd[q]].value())
            checked += 1
    assert_true(checked > 0)


def test_grouped_stddev_agrees_between_paths() raises:
    """`root=True` takes the square root of the same triple, so it inherits the
    argument above; asserted separately because it is a distinct
    instantiation."""
    var pair = _grouped_pair(40_000)
    var payload = _payload(_BIG)
    var ss = Dispersion[0, True, Int32Type].grouped(
        pair[0], _in[Dispersion[0, True, Int32Type]](payload.copy())
    )
    var ps = Dispersion[0, True, Int32Type].grouped(
        pair[1], _in[Dispersion[0, True, Int32Type]](payload.copy())
    )
    var fwd = _bijection(pair[0].ids, pair[1].ids, 40_000)
    for q in range(40_000):
        assert_equal(ps.is_valid(q), ss.is_valid(fwd[q]))
        if ps.is_valid(q):
            assert_equal(ps[q].value(), ss[fwd[q]].value())


def test_grouped_count_distinct_agrees_between_paths() raises:
    """Exact `count_distinct` is the fold with **no correct merge at all** — its
    state is one hash table over `(group, value)` pairs, so two thread-local
    tables would carry incompatible bucket numbering and double-count any value
    both threads saw. Radix placement never splits a group, so the question
    never arises."""
    var pair = _grouped_pair(40_000)
    var payload = _int64_payload(_BIG)
    var sd = DistinctCount[True, Int64Array].grouped(
        pair[0], _in[DistinctCount[True, Int64Array]](payload.copy())
    )
    var pd = DistinctCount[True, Int64Array].grouped(
        pair[1], _in[DistinctCount[True, Int64Array]](payload.copy())
    )
    assert_equal(len(sd), 40_000)
    var fwd = _bijection(pair[0].ids, pair[1].ids, 40_000)
    for q in range(40_000):
        assert_equal(pd[q].value(), sd[fwd[q]].value())
