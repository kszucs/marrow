from std.testing import assert_equal, assert_true

from ...arrays import DynArray
from ...builders import array, nulls, Int64Builder, StringBuilder
from ...dtypes import int32, int64, string
from ...kernels.distinct import count_distinct, approx_count_distinct
from ...execution import ExecContext


def test_count_distinct_basic() raises:
    var a: DynArray = array([1, 2, 2, 3, 3, 3], int32)
    assert_equal(count_distinct(a).value(), 3)


def test_count_distinct_all_unique() raises:
    var a: DynArray = array([10, 20, 30, 40], int64)
    assert_equal(count_distinct(a).value(), 4)


def test_count_distinct_all_same() raises:
    var a: DynArray = array([7, 7, 7, 7, 7], int32)
    assert_equal(count_distinct(a).value(), 1)


def test_count_distinct_excludes_nulls() raises:
    """Nulls do not count as a distinct value (SQL COUNT(DISTINCT))."""
    var b = Int64Builder(6)
    b.append(1)
    b.append(2)
    b.append_null()
    b.append(2)
    b.append_null()
    b.append(3)
    assert_equal(count_distinct(b.finish()).value(), 3)


def test_count_distinct_empty() raises:
    var a: DynArray = array(int32)
    assert_equal(count_distinct(a).value(), 0)


def test_count_distinct_all_null() raises:
    var a: DynArray = nulls(5, int64)
    assert_equal(count_distinct(a).value(), 0)


def test_count_distinct_strings() raises:
    var b = StringBuilder()
    b.append("apple")
    b.append("banana")
    b.append("apple")
    b.append("cherry")
    b.append("banana")
    assert_equal(count_distinct(b.finish()).value(), 3)


def _distinct_int64(n: Int, distinct: Int) raises -> DynArray:
    """`n` rows drawn from `distinct` distinct values (i % distinct)."""
    var b = Int64Builder(n)
    for i in range(n):
        b.append(Int64(i % distinct))
    return b.finish()


def test_approx_small_is_exact_regime() raises:
    """Linear counting keeps small cardinalities essentially exact."""
    var a = _distinct_int64(1000, 50)
    assert_equal(approx_count_distinct(a).value(), 50)


def test_approx_large_within_tolerance() raises:
    """~0.65% standard error → within 2% on 100k distinct values."""
    var true_distinct = 100_000
    var a = _distinct_int64(true_distinct, true_distinct)
    var est = Int(approx_count_distinct(a).value())
    var err = Float64(abs(est - true_distinct)) / Float64(true_distinct)
    assert_true(err < 0.02)


def test_approx_excludes_nulls() raises:
    # Rows 0..249 hold values 0..49 (5x each → 50 distinct); rows 250..299 are
    # null. A contiguous null tail can't alias any live value, so the true
    # non-null distinct count is exactly 50.
    var b = Int64Builder(300)
    for i in range(300):
        if i >= 250:
            b.append_null()
        else:
            b.append(Int64(i % 50))
    var est = Int(approx_count_distinct(b.finish()).value())
    assert_true(abs(est - 50) <= 1)


def test_count_distinct_parallel_matches_serial() raises:
    """Above the parallel threshold, radix-on-value (Σ per-partition distinct,
    no merge) agrees exactly with the serial hash-set dedup. 300k rows with a
    known 5000 distinct values, plus a null tail."""
    var b = Int64Builder(300_000)
    for i in range(300_000):
        if i < 3000:
            b.append_null()
        else:
            b.append(Int64(i % 5000))
    var a: DynArray = b.finish()
    assert_equal(count_distinct(a, ExecContext.serial()).value(), 5000)
    assert_equal(count_distinct(a, ExecContext.parallel(4)).value(), 5000)
