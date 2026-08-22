"""`col` and `lit` — one surface, two lanes.

The claim under test is that the overloads are not two APIs: given the same
column, the typed and untyped spellings must produce the same values. What
differs is only what the caller had to know, and when it is resolved.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import Int64Type, int64
from ...scalars import DynScalar, Int64Scalar
from ...tabular import RecordBatch, record_batch
from ..builders import col, lit
from ..core import Shape, into_array


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_col_with_dtype_takes_the_comptime_lane() raises:
    """A dtype in hand fuses: the result is a `Column[T]`, not erased."""
    var c = col("a", int64)
    assert_equal(c.name(), "a")
    assert_true(c.shape == Shape.columnar)
    assert_true(type_of(c).Type == Int64Type)


def test_col_without_dtype_takes_the_runtime_lane() raises:
    var c = col("a")
    assert_equal(c.name(), "a")
    assert_true(c.shape == Shape.columnar)


def test_both_col_overloads_evaluate_alike() raises:
    """The two lanes are different machinery, not different answers."""
    var b = _batch()
    var typed = into_array(col("b", int64).evaluate(b), b.num_rows())
    var erased = into_array(col("b").evaluate(b), b.num_rows())
    assert_true(typed == erased)
    assert_true(typed == b.column("b"))


def test_lit_with_dtype_stays_scalar() raises:
    """The comptime literal does not materialise — that is what `Shape.scalar`
    buys, and it is the difference from the runtime lane below."""
    var v = lit(7, int64)
    assert_true(v.shape == Shape.scalar)
    var d = v.evaluate(_batch())
    assert_true(d.isa[DynScalar]())


def test_lit_without_dtype_broadcasts() raises:
    """The runtime lane promised `Shape.columnar`, so it pays for it.

    This is the case the test fixtures got wrong before `literal` existed: they
    wired a literal to the column evaluator, which reads `Payload[String]` and
    would have failed the moment anything evaluated it.
    """
    var b = _batch()
    var v = lit(DynScalar(Int64Scalar(7)))
    assert_true(v.shape == Shape.columnar)
    var arr = into_array(v.evaluate(b), b.num_rows())
    assert_equal(len(arr), 4)
    assert_true(arr == array([7, 7, 7, 7], int64))


def test_lit_names_itself_by_value() raises:
    """A literal's `name()` is its value, so a projection of one is not
    anonymous — `SELECT 1` has a column called `1`."""
    assert_equal(lit(1, int64).name(), "1")
