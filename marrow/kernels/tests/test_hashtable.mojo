# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hash indexes: `SwissHashTable` and `HashIndex` map 64-bit
hashes to dense ids, and know nothing of keys."""

from std.testing import assert_equal, assert_true

from ...arrays import Int32Array, UInt64Array
from ...builders import UInt64Builder
from ...execution import ExecContext
from ...kernels.hashtable import HashIndex, SwissHashTable
from ...utils import Fmix64


def _hashes(*values: Int) raises -> UInt64Array:
    var b = UInt64Builder(capacity=len(values))
    for i in range(len(values)):
        b.append(UInt64(values[i]))
    return b.finish()


def _spread(n: Int, distinct: Int, salt: Int = 0) raises -> UInt64Array:
    """`n` well-spread hashes over `distinct` values — the top bits matter to
    radix placement, so small integers will not do."""
    var b = UInt64Builder(capacity=n)
    for i in range(n):
        b.append(Fmix64.mix[1](UInt64(((i * 7919) % distinct) + salt))[0])
    return b.finish()


def _ids(ids: Int32Array) -> List[Int]:
    var out = List[Int]()
    for i in range(len(ids)):
        out.append(Int(ids.unsafe_get(i)))
    return out^


# ---------------------------------------------------------------------------
# SwissHashTable
# ---------------------------------------------------------------------------


def test_insert_hashes_hands_out_ids_in_order() raises:
    """Equal hashes share an id; new ones are numbered in first-seen order,
    and each new id's first position is reported in id order."""
    var t = SwissHashTable()
    var placed = t.insert_hashes(_hashes(100, 200, 100, 300, 200))
    assert_true(_ids(placed.ids) == [0, 1, 0, 2, 1])
    assert_true(_ids(placed.firsts) == [0, 1, 3])
    assert_equal(len(t), 3)


def test_insert_hashes_empty() raises:
    var t = SwissHashTable()
    var placed = t.insert_hashes(_hashes())
    assert_equal(len(placed.ids), 0)
    assert_equal(len(placed.firsts), 0)
    assert_equal(len(t), 0)


def test_insert_hashes_across_calls() raises:
    """An id is stable across calls."""
    var t = SwissHashTable()
    _ = t.insert_hashes(_hashes(10, 20))
    var placed = t.insert_hashes(_hashes(20, 30))
    assert_true(_ids(placed.ids) == [1, 2])
    assert_true(_ids(placed.firsts) == [1])


def test_insert_hashes_with_high_bits() raises:
    """Hashes with bit 63 set must not produce the EMPTY control byte —
    regression for a signed shift in the fingerprint."""
    var t = SwissHashTable()
    var h = _spread(1000, 1000)
    _ = t.insert_hashes(h)
    assert_equal(len(t), 1000)
    var found = t.find_hashes(h)
    for i in range(1000):
        assert_true(found.unsafe_get(i) >= 0)


def test_insert_hashes_grows_and_keeps_ids() raises:
    """Growing re-places every id by its stored hash; no id moves."""
    var t = SwissHashTable()
    var n = 100_000
    var h = _spread(n, n)
    var first = t.insert_hashes(h)
    assert_equal(len(t), n)
    assert_true(first.ids == t.find_hashes(h))


def test_find_hashes_absent_is_minus_one() raises:
    var t = SwissHashTable()
    _ = t.insert_hashes(_hashes(10, 20, 30))
    assert_true(_ids(t.find_hashes(_hashes(20, 40, 10))) == [1, -1, 0])
    assert_equal(len(t), 3)


def test_insert_new_keeps_equal_hashes_apart() raises:
    """Rebuilding from ids already distinct must not merge two that share a
    hash."""
    var t = SwissHashTable()
    assert_equal(t.insert_new(7), 0)
    assert_equal(t.insert_new(7), 1)
    assert_equal(t.insert_new(9), 2)
    assert_equal(len(t), 3)
    var h = t.hashes()
    assert_true(h[0].value() == 7 and h[1].value() == 7 and h[2].value() == 9)


def test_find_one_and_insert_new_separate_keys_sharing_a_hash() raises:
    """Every key hashes to 7; the caller's equality alone tells them apart,
    and a key seen before is found again past the ids it does not match."""
    var keys: List[Int] = [10, 20, 10, 30, 20]
    var t = SwissHashTable()
    var owner = List[Int]()  # id -> key
    var got = List[Int]()
    for i in range(len(keys)):
        var k = keys[i]

        def same(id: Int) raises {imm} -> Bool:
            return owner[id] == k

        var id = t.find_one(7, same)
        if id < 0:
            id = t.insert_new(7)
            owner.append(k)
        got.append(id)
    assert_true(got == [0, 1, 0, 2, 1])
    assert_equal(len(t), 3)


def test_find_one_never_inserts() raises:
    var t = SwissHashTable()
    _ = t.insert_new(7)
    _ = t.insert_new(7)

    def second(id: Int) raises {imm} -> Bool:
        return id == 1

    def neither(id: Int) raises {imm} -> Bool:
        return False

    assert_equal(t.find_one(7, second), 1)
    assert_equal(t.find_one(7, neither), -1)
    assert_equal(t.find_one(8, second), -1)
    assert_equal(len(t), 2)


# ---------------------------------------------------------------------------
# HashIndex
# ---------------------------------------------------------------------------


def test_hash_index_reports_first_rows_in_id_order() raises:
    var index = HashIndex(ExecContext.serial())
    var placed = index.insert(_hashes(5, 6, 5, 7, 6))
    assert_true(_ids(placed.ids) == [0, 1, 0, 2, 1])
    assert_true(_ids(placed.firsts) == [0, 1, 3])
    placed = index.insert(_hashes(7, 8))
    assert_true(_ids(placed.ids) == [2, 3])
    assert_true(_ids(placed.firsts) == [1])


def test_hash_index_keeps_ids_across_the_move_to_radix() raises:
    """A small batch stays serial; a large distinct one moves every id onto
    the partitioned tables, and every hash seen before keeps its id."""
    var index = HashIndex(ExecContext.parallel(4))
    var small = _spread(1_000, 500)
    var before = index.insert(small).ids.copy()
    var n = 100_000
    var big = _spread(n, 60_000)
    var placed = index.insert(big)
    # The first batch's 500 values are among the second batch's 60,000.
    assert_equal(len(index), 60_000)
    assert_true(index.find(small) == before)
    assert_true(index.find(big) == placed.ids)


def test_hash_index_radix_find_and_one() raises:
    var index = HashIndex(ExecContext.parallel(4))
    var n = 100_000
    var big = _spread(n, 60_000)
    _ = index.insert(big)
    var absent = _spread(10, 10, salt=1_000_000)
    for id in _ids(index.find(absent)):
        assert_equal(id, -1)

    def any_id(id: Int) raises {imm} -> Bool:
        return True

    var h = UInt64(big.unsafe_get(0))
    assert_equal(index.find_one(h, any_id), Int(index.find(big).unsafe_get(0)))

    def none(id: Int) raises {imm} -> Bool:
        return False

    assert_equal(index.find_one(h, none), -1)
    assert_equal(index.insert_new(h), 60_000)
    assert_equal(len(index), 60_001)

    def only_new(id: Int) raises {imm} -> Bool:
        return id == 60_000

    # A new id lands in its hash's partition and is found there again.
    assert_equal(index.find_one(h, only_new), 60_000)
    assert_equal(index.find_one(h, none), -1)
