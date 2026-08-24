"""`col` and `lit` — one surface, two lanes.

The claim under test is that the overloads are not two APIs: given the same
column, the typed and untyped spellings must produce the same values. What
differs is only what the caller had to know, and when it is resolved.
"""

from std.testing import assert_equal, assert_true

from ...builders import Date32Builder, array
from ...dtypes import Date32Type, DynType, Int64Type, bool_, date32, int64
from ...scalars import DynScalar, Int64Scalar
from ...tabular import RecordBatch, record_batch
from ..builders import col, lit
from ..logical import DynValue, Shape
from ..physical import Morsel


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
    var typed = col("b", int64).evaluate(b).to_array(b.num_rows())
    var erased = col("b").evaluate(b).to_array(b.num_rows())
    assert_true(typed == erased)
    assert_true(typed == b.column("b"))


def test_lit_with_dtype_stays_scalar() raises:
    """The comptime literal does not materialise — that is what `Shape.scalar`
    buys, and it is the difference from the runtime lane below."""
    var v = lit(7, int64)
    assert_true(v.shape == Shape.scalar)
    var d = v.evaluate(_batch())
    assert_true(d.is_scalar())


def test_lit_without_dtype_broadcasts() raises:
    """The runtime lane promised `Shape.columnar`, so it pays for it.

    This is the case the test fixtures got wrong before `literal` existed: they
    wired a literal to the column evaluator, which reads `Payload[String]` and
    would have failed the moment anything evaluated it.
    """
    var b = _batch()
    var v = lit(DynScalar(Int64Scalar(7)))
    assert_true(v.shape == Shape.columnar)
    var arr = v.evaluate(b).to_array(b.num_rows())
    assert_equal(len(arr), 4)
    assert_true(arr == array([7, 7, 7, 7], int64))


def test_lit_names_itself_by_value() raises:
    """A literal's `name()` is its value, so a projection of one is not
    anonymous — `SELECT 1` has a column called `1`."""
    assert_equal(lit(1, int64).name(), "1")


def test_col_with_bool_dtype_takes_the_comptime_lane() raises:
    """`BoolColumn` existed but had no `col` overload, so a fused bool column
    could only be reached by naming the node type directly."""
    var b = record_batch([array([True, False, True]).copy()], names=["flag"])
    var v = col("flag", bool_)
    var op = DynValue(v).to_operator(False)
    var got = op.push(Morsel.ungrouped(b.copy())).value().to_array(3)
    assert_true(got.as_bool()[0].value())
    assert_true(not got.as_bool()[1].value())


def test_col_with_temporal_dtype_takes_the_comptime_lane() raises:
    """A temporal column is readable by the fused lane at all — which it was
    not before `PrimitiveValue` split the machinery from the domain.

    Its dtype comes from the bound column rather than from `Self.T()`, because
    a temporal type is not `Defaultable`: a timestamp carries a unit and a
    timezone, and neither can be conjured.
    """
    var d = Date32Builder(date32(), 3)
    d.append(Int32(19000))
    d.append(Int32(19001))
    d.append(Int32(19002))
    var b = record_batch([d.finish().to_dyn()], names=["d"])
    var v = col("d", date32())
    var op = DynValue(v).to_operator(False)
    var got = op.push(Morsel.ungrouped(b.copy())).value().to_array(3)
    assert_true(got.dtype() == DynType(date32()))
    assert_true(got.as_primitive[Date32Type]()[0].value() == 19000)


def test_a_temporal_column_is_not_arithmetic() raises:
    """`date + date` is meaningless and must not compile.

    Not assertable in Mojo — it is a *compile* error, so the guard is that
    `Add` binds on `NumericValue` while `TemporalColumn` conforms only to
    `TemporalValue`. Recorded here so the intent survives; verified by probe.
    """
    assert_true(True)
