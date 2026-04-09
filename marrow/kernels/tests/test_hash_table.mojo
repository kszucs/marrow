"""Tests for SwissHashTable: insert, build, probe, num_keys."""

from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray, AnyArray, StructArray
from marrow.buffers import Bitmap
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, uint64, struct_, Field, Int32Type, UInt64Type
from marrow.kernels.hashtable import SwissHashTable
from marrow.kernels.hashing import rapidhash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keys(*values: Int) raises -> StructArray:
    """Build a single-column StructArray from uint64 values."""
    var b = PrimitiveBuilder[UInt64Type](capacity=len(values))
    for i in range(len(values)):
        b.append(Scalar[uint64.native](values[i]))
    var children = List[AnyArray]()
    children.append(b.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", uint64)),
        length=len(values),
        nulls=0,
        offset=0,
        bitmap=Optional[Bitmap[]](None),
        children=children^,
    )


def _keys_range(n: Int, offset: Int = 0) raises -> StructArray:
    """Build a single-column StructArray with n sequential uint64 values."""
    var b = PrimitiveBuilder[UInt64Type](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i + offset))
    var children = List[AnyArray]()
    children.append(b.finish().to_any())
    return StructArray(
        dtype=struct_(Field("k", uint64)),
        length=n,
        nulls=0,
        offset=0,
        bitmap=Optional[Bitmap[]](None),
        children=children^,
    )


# ---------------------------------------------------------------------------
# insert tests
# ---------------------------------------------------------------------------


def test_insert_empty() raises:
    """Inserting zero keys returns an empty array."""
    var t = SwissHashTable[rapidhash]()
    var bids = t.insert(_keys_range(0))
    assert_equal(len(bids), 0)
    assert_equal(t.num_keys(), 0)


def test_insert_unique() raises:
    """Each unique key gets a sequential bucket ID."""
    var t = SwissHashTable[rapidhash]()
    var bids = t.insert(_keys(100, 200, 300))
    assert_equal(len(bids), 3)
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids.unsafe_get(0)), 0)
    assert_equal(Int(bids.unsafe_get(1)), 1)
    assert_equal(Int(bids.unsafe_get(2)), 2)


def test_insert_duplicates() raises:
    """Duplicate keys return their existing bucket ID."""
    var t = SwissHashTable[rapidhash]()
    var bids = t.insert(_keys(100, 200, 100, 300, 200))
    assert_equal(len(bids), 5)
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids.unsafe_get(0)), 0)  # 100 → new bid 0
    assert_equal(Int(bids.unsafe_get(1)), 1)  # 200 → new bid 1
    assert_equal(Int(bids.unsafe_get(2)), 0)  # 100 → existing bid 0
    assert_equal(Int(bids.unsafe_get(3)), 2)  # 300 → new bid 2
    assert_equal(Int(bids.unsafe_get(4)), 1)  # 200 → existing bid 1


def test_insert_all_same() raises:
    """All identical keys get the same bucket ID."""
    var t = SwissHashTable[rapidhash]()
    var bids = t.insert(_keys(42, 42, 42, 42))
    assert_equal(t.num_keys(), 1)
    for i in range(4):
        assert_equal(Int(bids.unsafe_get(i)), 0)


def test_insert_incremental() raises:
    """Multiple insert calls accumulate buckets."""
    var t = SwissHashTable[rapidhash]()
    var bids1 = t.insert(_keys(10, 20))
    assert_equal(t.num_keys(), 2)

    var bids2 = t.insert(_keys(20, 30))
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids2.unsafe_get(0)), 1)  # 20 → existing bid 1
    assert_equal(Int(bids2.unsafe_get(1)), 2)  # 30 → new bid 2


def test_insert_high_bit_keys() raises:
    """Keys whose hashes have bit 63 set must not produce H2 = 0xFF.

    Regression: a signed right-shift in _h2() produced 0xFF for hashes
    with the sign bit set, colliding with the EMPTY sentinel.
    """
    # Use keys that produce high-bit hashes. With rapidhash, any key
    # can produce a high-bit hash; we use a range and verify all are found.
    var keys = _keys_range(1000)
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var pairs = t.probe(keys, keys, 1000)
    assert_equal(len(pairs[0]), 1000)


# ---------------------------------------------------------------------------
# build + probe tests
# ---------------------------------------------------------------------------


def test_probe_empty_table() raises:
    """Probing an empty table returns no matches."""
    var build = _keys_range(0)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys(100, 200), num_build_rows=0)
    assert_equal(len(pairs[0]), 0)
    assert_equal(len(pairs[1]), 0)


def test_probe_empty_keys() raises:
    """Probing with zero keys returns no matches."""
    var build = _keys(100, 200)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys_range(0), num_build_rows=2)
    assert_equal(len(pairs[0]), 0)
    assert_equal(len(pairs[1]), 0)


def test_probe_all_match() raises:
    """All probe keys match build keys (1:1 join)."""
    var keys = _keys(10, 20, 30)
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var pairs = t.probe(keys, keys, num_build_rows=3)

    assert_equal(len(pairs[0]), 3)
    for i in range(3):
        assert_equal(Int(pairs[0].unsafe_get(i)), i)
        assert_equal(Int(pairs[1].unsafe_get(i)), i)


def test_probe_no_match() raises:
    """No probe keys match build keys."""
    var build = _keys(10, 20, 30)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys(40, 50, 60), num_build_rows=3)
    assert_equal(len(pairs[0]), 0)


def test_probe_partial_match() raises:
    """Some probe keys match, some don't."""
    var build = _keys(10, 20, 30)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys(20, 40, 10), num_build_rows=3)

    # 20 matches build row 1, 10 matches build row 0. 40 is skipped.
    assert_equal(len(pairs[0]), 2)


def test_probe_duplicate_build_keys() raises:
    """Duplicate build-side keys produce multiple matches per probe row."""
    var build = _keys(10, 20, 10)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys(10), num_build_rows=3)

    # Probe key 10 matches bucket 0, which has build rows 0 and 2.
    assert_equal(len(pairs[0]), 2)
    assert_equal(Int(pairs[1].unsafe_get(0)), 0)
    assert_equal(Int(pairs[1].unsafe_get(1)), 0)


def test_probe_single_match() raises:
    """With single_match=True, emit at most one match per probe row."""
    var build = _keys(10, 20, 10)
    var t = SwissHashTable[rapidhash]()
    t.build(build)
    var pairs = t.probe(build, _keys(10), num_build_rows=3, single_match=True)

    assert_equal(len(pairs[0]), 1)


# ---------------------------------------------------------------------------
# Scale tests
# ---------------------------------------------------------------------------


def test_insert_large() raises:
    """Insert 100K unique keys."""
    var n = 100_000
    var keys = _keys_range(n)
    var t = SwissHashTable[rapidhash]()
    var bids = t.insert(keys)
    assert_equal(t.num_keys(), n)
    assert_equal(len(bids), n)
    assert_equal(Int(bids.unsafe_get(0)), 0)
    assert_equal(Int(bids.unsafe_get(n - 1)), n - 1)


def test_build_probe_large() raises:
    """Build and probe with 100K rows, all matching."""
    var n = 100_000
    var keys = _keys_range(n)
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var pairs = t.probe(keys, keys, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


def test_build_probe_1m() raises:
    """Build and probe with 1M rows."""
    var n = 1_000_000
    var keys = _keys_range(n)
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var pairs = t.probe(keys, keys, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


def test_build_probe_10m() raises:
    """Build and probe with 10M rows."""
    var n = 10_000_000
    var keys = _keys_range(n)
    var t = SwissHashTable[rapidhash]()
    t.build(keys)
    var pairs = t.probe(keys, keys, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


# ---------------------------------------------------------------------------
# num_keys tests
# ---------------------------------------------------------------------------


def test_num_keys_empty() raises:
    var t = SwissHashTable[rapidhash]()
    assert_equal(t.num_keys(), 0)


def test_num_keys_after_insert() raises:
    var t = SwissHashTable[rapidhash]()
    _ = t.insert(_keys(1, 2, 3, 2, 1))
    assert_equal(t.num_keys(), 3)


def main() raises:
    TestSuite.run[__functions_in_module()]()
