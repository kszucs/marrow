# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hash indexes: `SwissHashTable` and `HashIndex` map 64-bit
hashes to dense ids, and know nothing of keys."""

from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DictionaryArray, DynArray, Int32Array, UInt64Array
from ...builders import Int64Builder, UInt64Builder, array
from ...dtypes import DynType, Int32Type, dictionary, int32, int64, string
from ...execution import ExecContext
from ...kernels.dictionary import DictionaryEncoder
from ...kernels.hashtable import (
    HashIndex,
    find_hashes,
    insert_hashes,
)
from ...utils import SwissHashTable
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
    var placed = insert_hashes(t, _hashes(100, 200, 100, 300, 200))
    assert_true(_ids(placed.ids) == [0, 1, 0, 2, 1])
    assert_true(_ids(placed.firsts) == [0, 1, 3])
    assert_equal(len(t), 3)


def test_insert_hashes_empty() raises:
    var t = SwissHashTable()
    var placed = insert_hashes(t, _hashes())
    assert_equal(len(placed.ids), 0)
    assert_equal(len(placed.firsts), 0)
    assert_equal(len(t), 0)


def test_insert_hashes_across_calls() raises:
    """An id is stable across calls."""
    var t = SwissHashTable()
    _ = insert_hashes(t, _hashes(10, 20))
    var placed = insert_hashes(t, _hashes(20, 30))
    assert_true(_ids(placed.ids) == [1, 2])
    assert_true(_ids(placed.firsts) == [1])


def test_insert_hashes_with_high_bits() raises:
    """Hashes with bit 63 set must not produce the EMPTY control byte —
    regression for a signed shift in the fingerprint."""
    var t = SwissHashTable()
    var h = _spread(1000, 1000)
    _ = insert_hashes(t, h)
    assert_equal(len(t), 1000)
    var found = find_hashes(t, h)
    for i in range(1000):
        assert_true(found.unsafe_get(i) >= 0)


def test_insert_hashes_grows_and_keeps_ids() raises:
    """Growing re-places every id by its stored hash; no id moves."""
    var t = SwissHashTable()
    var n = 100_000
    var h = _spread(n, n)
    var first = insert_hashes(t, h)
    assert_equal(len(t), n)
    assert_true(first.ids == find_hashes(t, h))


def test_find_hashes_absent_is_minus_one() raises:
    var t = SwissHashTable()
    _ = insert_hashes(t, _hashes(10, 20, 30))
    assert_true(_ids(find_hashes(t, _hashes(20, 40, 10))) == [1, -1, 0])
    assert_equal(len(t), 3)


def test_insert_new_keeps_equal_hashes_apart() raises:
    """Rebuilding from ids already distinct must not merge two that share a
    hash."""
    var t = SwissHashTable()
    assert_equal(t.insert_new(7), 0)
    assert_equal(t.insert_new(7), 1)
    assert_equal(t.insert_new(9), 2)
    assert_equal(len(t), 3)
    assert_true(t.hash_of(0) == 7 and t.hash_of(1) == 7 and t.hash_of(2) == 9)


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
    assert_true(index.find(small, ExecContext.parallel(4)) == before)
    assert_true(index.find(big, ExecContext.parallel(4)) == placed.ids)


def test_hash_index_radix_find_and_one() raises:
    var index = HashIndex(ExecContext.parallel(4))
    var n = 100_000
    var big = _spread(n, 60_000)
    _ = index.insert(big)
    var absent = _spread(10, 10, salt=1_000_000)
    for id in _ids(index.find(absent, ExecContext.parallel(4))):
        assert_equal(id, -1)

    def any_id(id: Int) raises {imm} -> Bool:
        return True

    var h = UInt64(big.unsafe_get(0))
    assert_equal(
        index.find_one(h, any_id),
        Int(index.find(big, ExecContext.parallel(4)).unsafe_get(0)),
    )

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


def test_hash_index_finds_on_the_callers_context() raises:
    """A batch big enough to split on the context the ids were placed with
    finds the same ids when the lookup is asked to stay on one thread."""
    var index = HashIndex(ExecContext.parallel(4))
    var big = _spread(100_000, 60_000)
    var placed = index.insert(big)
    assert_true(index.is_partitioned())
    assert_true(index.find(big, ExecContext.serial()) == placed.ids)
    assert_true(index.find(big, ExecContext.parallel(4)) == placed.ids)


def test_encoder_extends_its_keys_unindexed() raises:
    """Keys appended without codes follow the codes' keys in `values`, and
    no lookup finds them."""
    var types: List[DynType] = [int64]
    var encoder = DictionaryEncoder(types^)
    var first: List[DynArray] = [array([1, 2, 1], int64)]
    _ = encoder.encode(first)
    var later: List[DynArray] = [array([5, 6], int64)]
    encoder.extend_unindexed(later^)
    assert_equal(len(encoder), 2)
    var values = encoder.values()
    assert_true(values[0].as_int64() == array([1, 2, 5, 6], int64))
    var probe: List[DynArray] = [array([5], int64)]
    assert_true(
        encoder.lookup(probe, ExecContext.serial()) == array([-1], int32)
    )
    # Many appended runs come back whole, in order.
    for i in range(40):
        var run: List[DynArray] = [array([100 + i], int64)]
        encoder.extend_unindexed(run^)
    var all = encoder.values()[0].as_int64().copy()
    assert_equal(len(all), 44)
    assert_equal(Int(all[43].value()), 139)


def test_encoder_refuses_unindexed_keys_of_another_type() raises:
    var types: List[DynType] = [int64]
    var encoder = DictionaryEncoder(types^)
    var wrong: List[DynArray] = [array([1, 2], int32)]
    with assert_raises(contains="key column 0"):
        encoder.extend_unindexed(wrong^)


def test_encoder_lookup_on_the_callers_context() raises:
    """A radix-built encoder looked up on a serial context answers what it
    answers on the context it was built with."""
    var types: List[DynType] = [int64]
    var encoder = DictionaryEncoder(types^, ExecContext.parallel(4))
    var b = Int64Builder(150_000)
    for i in range(150_000):
        b.append(Int64((i * 7919) % 120_000))
    var keys: List[DynArray] = [b.finish()]
    var placed = encoder.encode(keys)
    assert_true(encoder.is_partitioned())
    assert_true(encoder.lookup(keys, ExecContext.serial()) == placed.ids)
    assert_true(encoder.lookup(keys, ExecContext.parallel(4)) == placed.ids)


def test_estimated_distinct_inverts_the_sample() raises:
    """From how many distinct values a uniform sample saw, the domain it was
    drawn from: within 5% where the sample can tell, the row count where it
    saw every draw distinct."""
    assert_equal(HashIndex.estimated_distinct(1_000_000, 4096, 4096), 1_000_000)
    assert_equal(HashIndex.estimated_distinct(500, 4096, 500), 500)
    var domains: List[Int] = [1_000, 10_000, 50_000]
    for d in domains:
        var seen = HashIndex.expected_distinct(d, 4096)
        var est = HashIndex.estimated_distinct(1_000_000, 4096, seen)
        assert_true(abs(est - d) <= d // 20, String(d, " -> ", est))


def test_encoder_refuses_unindexed_dictionary_keys() raises:
    """Unindexed keys are kept as plain values, so a dictionary-typed key
    column is refused with an error."""
    var types: List[DynType] = [dictionary(int32, string)]
    var encoder = DictionaryEncoder(types^)
    var indices: List[Int32] = [0, 1]
    var entries: List[Optional[String]] = [String("a"), String("b")]
    var keys: List[DynArray] = [
        DictionaryArray.from_arrays(
            array[Int32Type](indices, int32), array(entries)
        ).to_dyn()
    ]
    with assert_raises(contains="dictionary"):
        encoder.extend_unindexed(keys^)
