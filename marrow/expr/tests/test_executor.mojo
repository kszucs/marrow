"""Tests for expression execution: correctness, morsel boundaries, edge cases."""

from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import int64, float64, bool_ as bool_dt
from marrow.expr import AnyValue, col, lit, DISPATCH_CPU, execute
from marrow.builders import arange
from marrow.expr.executor import ExecutionContext


# ---------------------------------------------------------------------------
# Sequential fallback (morsel_size > array length)
# ---------------------------------------------------------------------------


def test_sequential_fallback() raises:
    """When morsel_size > array length, uses single-thread path."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([10, 20, 30, 40, 50])
    var expr = col(0) + col(1)

    var result = execute(expr, [Array(a), Array(b)], morsel_size=1_000_000)
    var sequential = execute(expr, [Array(a), Array(b)])
    assert_equal(result == sequential, True)


# ---------------------------------------------------------------------------
# Parallel path — output matches sequential
# ---------------------------------------------------------------------------


def test_parallel_matches_sequential_add() raises:
    """Parallel execute produces the same result as sequential for add."""
    var n = 1000
    var a = arange[int64](0, n)
    var b = arange[int64](0, n)
    var expr = col(0) + col(1)

    var sequential = execute(expr, [Array(a), Array(b)])
    var parallel = execute(
        expr, [Array(a), Array(b)], morsel_size=64, num_cpu_workers=4
    )

    assert_equal(parallel == sequential, True)


def test_parallel_matches_sequential_mul() raises:
    """Parallel execute produces the same result as sequential for mul."""
    var n = 500
    var a = arange[int64](1, n + 1)
    var b = arange[int64](0, n)
    var expr = col(0) * col(1)

    var sequential = execute(expr, [Array(a), Array(b)])
    var parallel = execute(
        expr, [Array(a), Array(b)], morsel_size=50, num_cpu_workers=2
    )

    assert_equal(parallel == sequential, True)


# ---------------------------------------------------------------------------
# Chunk boundary correctness
# ---------------------------------------------------------------------------


def test_chunk_boundary_values() raises:
    """Values at chunk boundaries (first/last element of each morsel) are correct.
    """
    var a = arange[int64](0, 128)
    var expr = col(0) + lit[int64](1)

    var result = execute(expr, [Array(a)], morsel_size=32, num_cpu_workers=2)
    var expected = execute(expr, [Array(a)])

    assert_equal(result == expected, True)


# ---------------------------------------------------------------------------
# Non-SIMD-aligned lengths
# ---------------------------------------------------------------------------


def test_non_aligned_length_parallel() raises:
    """Parallel execute handles lengths not divisible by morsel_size."""
    var a = arange[int64](0, 100)
    var expr = -col(0)

    var result = execute(expr, [Array(a)], morsel_size=32, num_cpu_workers=2)
    var expected = execute(expr, [Array(a)])

    assert_equal(result == expected, True)


# ---------------------------------------------------------------------------
# Single-element arrays
# ---------------------------------------------------------------------------


def test_single_element_parallel() raises:
    """Single-element array falls through to sequential path gracefully."""
    var a = array[int64]([42])
    var b = array[int64]([8])
    var result = execute(col(0) + col(1), [Array(a), Array(b)])
    assert_equal(result == Array(array[int64]([50])), True)


# ---------------------------------------------------------------------------
# Predicate execution
# ---------------------------------------------------------------------------


def test_execute_pred_parallel() raises:
    """Parallel predicate produces same result as sequential."""
    var n = 200
    var a = arange[int64](0, n)
    var b = arange[int64](0, n)
    var expr = col(0) < col(1)

    var sequential = execute(expr, [Array(a), Array(b)])
    var parallel = execute(
        expr, [Array(a), Array(b)], morsel_size=40, num_cpu_workers=3
    )

    assert_equal(parallel == sequential, True)


# ---------------------------------------------------------------------------
# Chained expression
# ---------------------------------------------------------------------------


def test_parallel_chained_expression() raises:
    """Parallel executor handles chained expressions correctly."""
    var a = arange[int64](0, 256)
    var b = arange[int64](1, 257)

    # (a + b) * (a - b)
    var expr = (col(0) + col(1)) * (col(0) - col(1))
    var sequential = execute(expr, [Array(a), Array(b)])
    var parallel = execute(
        expr, [Array(a), Array(b)], morsel_size=64, num_cpu_workers=2
    )

    assert_equal(parallel == sequential, True)


# ---------------------------------------------------------------------------
# DISPATCH_CPU hint bypasses GPU auto-dispatch
# ---------------------------------------------------------------------------


def test_dispatch_cpu_hint() raises:
    """DISPATCH_CPU hint keeps execution on CPU even if ctx has device."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([5, 4, 3, 2, 1])

    var result = execute(
        (col(0) + col(1)).with_dispatch(DISPATCH_CPU), [Array(a), Array(b)]
    )
    assert_equal(result == Array(array[int64]([6, 6, 6, 6, 6])), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
