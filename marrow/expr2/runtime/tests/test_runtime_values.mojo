"""The runtime lane on its own terms.

`RuntimeValue` is one struct holding children, a payload and a function
pointer, so its correctness is mostly about **structure**: what it reports about
itself before anything evaluates it. Those answers are what the optimizer reads,
and a wrong one is silent — a plan that narrows the wrong columns still runs.
"""

from std.testing import assert_equal, assert_true

from ....arrays import DynArray
from ....builders import array
from ....dtypes import int64
from ....scalars import DynScalar, Int64Scalar
from ....tabular import RecordBatch, record_batch
from ...core import Shape
from ..values import Payload, RuntimeValue, column, literal


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def _unevaluated(
    kids: List[DynArray], p: Payload, b: RecordBatch
) raises -> DynArray:
    """A stand-in `EvalFn` for nodes built to be analysed, never run."""
    raise Error("_unevaluated: this node exists for structural assertions only")


def test_runtime_shape_is_always_columnar() raises:
    """The lane materialises unconditionally, so it answers truthfully rather
    than aspirationally — `Datum.to_array` never has to broadcast its result."""
    assert_true(RuntimeValue.shape == Shape.columnar)
    assert_true(column("a").shape == Shape.columnar)
    assert_true(literal(DynScalar(Int64Scalar(1))).shape == Shape.columnar)


def test_runtime_columns_are_deduped_in_first_seen_order() raises:
    """Projection pushdown reads this list; a repeat would narrow twice and a
    reorder would build a schema in the wrong order."""
    var v = RuntimeValue(
        "add",
        _unevaluated,
        column("b"),
        RuntimeValue("add", _unevaluated, column("a"), column("b")),
    )
    var cols = v.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "b")
    assert_equal(cols[1], "a")


def test_runtime_dtype_agrees_with_evaluation() raises:
    """The lane that has to look itself up in a schema must agree with what it
    then produces."""
    var b = _batch()
    var v = column("b")
    var produced = v.evaluate(b).to_array(b.num_rows()).dtype()
    assert_true(v.dtype(b.schema) == produced)


def test_runtime_column_reads_the_named_column() raises:
    var b = _batch()
    var got = column("b").evaluate(b).to_array(b.num_rows())
    assert_true(got == b.column("b"))


def test_runtime_literal_broadcasts_to_the_batch_length() raises:
    """A literal owes a full column here, unlike the comptime lane's, which
    stays a scalar until something asks."""
    var b = _batch()
    var got = (
        literal(DynScalar(Int64Scalar(7))).evaluate(b).to_array(b.num_rows())
    )
    assert_equal(len(got), 4)
    assert_true(got == array([7, 7, 7, 7], int64))


def test_runtime_a_subtree_copies_in_constant_time() raises:
    """Children sit behind `ArcPointer`, so copying a plan shares rather than
    clones. Equality of what the copy reports is the observable half."""
    var v = RuntimeValue("add", _unevaluated, column("a"), column("b"))
    var w = v.copy()
    assert_equal(len(w.columns()), 2)
    assert_equal(w.columns()[0], "a")
    assert_equal(w.name(), v.name())
