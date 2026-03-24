"""Tests for SwissHashTable: insert, build, probe, num_keys."""

from std.testing import assert_equal, assert_true, TestSuite

from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, uint64
from marrow.kernels.hash_table import SwissHashTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hashes(*values: Int) raises -> PrimitiveArray[uint64]:
    """Build a PrimitiveArray[uint64] from values."""
    var b = PrimitiveBuilder[uint64](capacity=len(values))
    for i in range(len(values)):
        b.append(Scalar[uint64.native](values[i]))
    return b.finish()


def _hashes_range(n: Int, offset: Int = 0) raises -> PrimitiveArray[uint64]:
    """Build a PrimitiveArray[uint64] with n sequential values."""
    var b = PrimitiveBuilder[uint64](capacity=n)
    for i in range(n):
        b.append(Scalar[uint64.native](i + offset))
    return b.finish()


# ---------------------------------------------------------------------------
# insert tests
# ---------------------------------------------------------------------------


def test_insert_empty() raises:
    """Inserting zero hashes returns an empty array."""
    var t = SwissHashTable[uint64]()
    var bids = t.insert(_hashes_range(0))
    assert_equal(len(bids), 0)
    assert_equal(t.num_keys(), 0)


def test_insert_unique() raises:
    """Each unique hash gets a sequential bucket ID."""
    var t = SwissHashTable[uint64]()
    var bids = t.insert(_hashes(100, 200, 300))
    assert_equal(len(bids), 3)
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids.unsafe_get(0)), 0)
    assert_equal(Int(bids.unsafe_get(1)), 1)
    assert_equal(Int(bids.unsafe_get(2)), 2)


def test_insert_duplicates() raises:
    """Duplicate hashes return their existing bucket ID."""
    var t = SwissHashTable[uint64]()
    var bids = t.insert(_hashes(100, 200, 100, 300, 200))
    assert_equal(len(bids), 5)
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids.unsafe_get(0)), 0)  # 100 → new bid 0
    assert_equal(Int(bids.unsafe_get(1)), 1)  # 200 → new bid 1
    assert_equal(Int(bids.unsafe_get(2)), 0)  # 100 → existing bid 0
    assert_equal(Int(bids.unsafe_get(3)), 2)  # 300 → new bid 2
    assert_equal(Int(bids.unsafe_get(4)), 1)  # 200 → existing bid 1


def test_insert_all_same() raises:
    """All identical hashes get the same bucket ID."""
    var t = SwissHashTable[uint64]()
    var bids = t.insert(_hashes(42, 42, 42, 42))
    assert_equal(t.num_keys(), 1)
    for i in range(4):
        assert_equal(Int(bids.unsafe_get(i)), 0)


def test_insert_incremental() raises:
    """Multiple insert calls accumulate buckets."""
    var t = SwissHashTable[uint64]()
    var bids1 = t.insert(_hashes(10, 20))
    assert_equal(t.num_keys(), 2)

    var bids2 = t.insert(_hashes(20, 30))
    assert_equal(t.num_keys(), 3)
    assert_equal(Int(bids2.unsafe_get(0)), 1)  # 20 → existing bid 1
    assert_equal(Int(bids2.unsafe_get(1)), 2)  # 30 → new bid 2


def test_insert_h2_collision() raises:
    """Hashes with the same top 7 bits (H2 collision) get separate buckets.

    Two hashes that differ only in the lower bits will have the same H2
    fingerprint. The table must verify the full hash to distinguish them.
    """
    # Construct two hashes with the same top 7 bits but different values.
    # For uint64, H2 = h >> 57. So h1 and h2 share the top 7 bits if
    # they differ only in the lower 57 bits.
    var h1 = UInt64(1) << 57  # H2 = 1
    var h2 = (UInt64(1) << 57) | 1  # H2 = 1 (same), but different hash

    var hashes = PrimitiveBuilder[uint64](capacity=2)
    hashes.append(Scalar[uint64.native](h1))
    hashes.append(Scalar[uint64.native](h2))
    var arr = hashes.finish()

    var t = SwissHashTable[uint64]()
    var bids = t.insert(arr)
    assert_equal(t.num_keys(), 2)
    assert_equal(Int(bids.unsafe_get(0)), 0)
    assert_equal(Int(bids.unsafe_get(1)), 1)


def test_insert_high_bit_hashes() raises:
    """Hashes with bit 63 set must not produce H2 = 0xFF (CTRL_EMPTY).

    Regression: a signed right-shift in _h2() produced 0xFF for hashes
    with the sign bit set, colliding with the EMPTY sentinel. This caused
    _find_slot to treat occupied slots as empty, losing rows.
    """
    # Build hashes where the top 7 bits are all 1s (H2 would be 0x7F
    # with a correct unsigned shift, but was 0xFF with a signed shift).
    var b = PrimitiveBuilder[uint64](capacity=3)
    b.append(Scalar[uint64.native](UInt64(0xFF) << 57 | 1))
    b.append(Scalar[uint64.native](UInt64(0xFF) << 57 | 2))
    b.append(Scalar[uint64.native](UInt64(0xFF) << 57 | 3))
    var hashes = b.finish()

    var t = SwissHashTable[uint64]()
    t.build(hashes)

    # All 3 must be found during probe.
    var pairs = t.probe(hashes, 3)
    assert_equal(len(pairs[0]), 3)

    # Verify each hash is findable individually.
    for i in range(3):
        var bid = t._find_slot(hashes.unsafe_get(i))
        assert_true(bid >= 0, "hash with high bits set must be found")


# ---------------------------------------------------------------------------
# build + probe tests
# ---------------------------------------------------------------------------


def test_probe_empty_table() raises:
    """Probing an empty table returns no matches."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes_range(0))
    var pairs = t.probe(_hashes(100, 200), num_build_rows=0)
    assert_equal(len(pairs[0]), 0)
    assert_equal(len(pairs[1]), 0)


def test_probe_empty_hashes() raises:
    """Probing with zero hashes returns no matches."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes(100, 200))
    var pairs = t.probe(_hashes_range(0), num_build_rows=2)
    assert_equal(len(pairs[0]), 0)
    assert_equal(len(pairs[1]), 0)


def test_probe_all_match() raises:
    """All probe hashes match build hashes (1:1 join)."""
    var hashes = _hashes(10, 20, 30)
    var t = SwissHashTable[uint64]()
    t.build(hashes)
    var pairs = t.probe(hashes, num_build_rows=3)

    assert_equal(len(pairs[0]), 3)
    # Each probe row should match exactly one build row.
    for i in range(3):
        assert_equal(Int(pairs[0].unsafe_get(i)), i)
        assert_equal(Int(pairs[1].unsafe_get(i)), i)


def test_probe_no_match() raises:
    """No probe hashes match build hashes."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes(10, 20, 30))
    var pairs = t.probe(_hashes(40, 50, 60), num_build_rows=3)
    assert_equal(len(pairs[0]), 0)


def test_probe_partial_match() raises:
    """Some probe hashes match, some don't."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes(10, 20, 30))
    var pairs = t.probe(_hashes(20, 40, 10), num_build_rows=3)

    # 20 matches build row 1, 10 matches build row 0. 40 is skipped.
    assert_equal(len(pairs[0]), 2)


def test_probe_duplicate_build_keys() raises:
    """Duplicate build-side hashes produce multiple matches per probe row."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes(10, 20, 10))  # bid 0 has rows [0, 2]
    var pairs = t.probe(_hashes(10), num_build_rows=3)

    # Probe hash 10 matches bucket 0, which has build rows 0 and 2.
    assert_equal(len(pairs[0]), 2)
    # Both matches are for probe_row 0.
    assert_equal(Int(pairs[1].unsafe_get(0)), 0)
    assert_equal(Int(pairs[1].unsafe_get(1)), 0)


def test_probe_single_match() raises:
    """With single_match=True, emit at most one match per probe row."""
    var t = SwissHashTable[uint64]()
    t.build(_hashes(10, 20, 10))  # bid 0 has rows [0, 2]
    var pairs = t.probe(_hashes(10), num_build_rows=3, single_match=True)

    # Only one match even though bucket 0 has 2 rows.
    assert_equal(len(pairs[0]), 1)


# ---------------------------------------------------------------------------
# Scale tests
# ---------------------------------------------------------------------------


def test_insert_large() raises:
    """Insert 100K unique hashes."""
    var n = 100_000
    var hashes = _hashes_range(n)
    var t = SwissHashTable[uint64]()
    var bids = t.insert(hashes)
    assert_equal(t.num_keys(), n)
    assert_equal(len(bids), n)
    # Verify first and last bucket IDs.
    assert_equal(Int(bids.unsafe_get(0)), 0)
    assert_equal(Int(bids.unsafe_get(n - 1)), n - 1)


def test_build_probe_large() raises:
    """Build and probe with 100K rows, all matching."""
    var n = 100_000
    var hashes = _hashes_range(n)
    var t = SwissHashTable[uint64]()
    t.build(hashes)
    var pairs = t.probe(hashes, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


def test_build_probe_1m() raises:
    """Build and probe with 1M rows."""
    var n = 1_000_000
    var hashes = _hashes_range(n)
    var t = SwissHashTable[uint64]()
    t.build(hashes)
    var pairs = t.probe(hashes, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


def test_build_probe_10m() raises:
    """Build and probe with 10M rows."""
    var n = 10_000_000
    var hashes = _hashes_range(n)
    var t = SwissHashTable[uint64]()
    t.build(hashes)
    var pairs = t.probe(hashes, num_build_rows=n)
    assert_equal(len(pairs[0]), n)


# ---------------------------------------------------------------------------
# num_keys tests
# ---------------------------------------------------------------------------


def test_num_keys_empty() raises:
    var t = SwissHashTable[uint64]()
    assert_equal(t.num_keys(), 0)


def test_num_keys_after_insert() raises:
    var t = SwissHashTable[uint64]()
    _ = t.insert(_hashes(1, 2, 3, 2, 1))
    assert_equal(t.num_keys(), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
