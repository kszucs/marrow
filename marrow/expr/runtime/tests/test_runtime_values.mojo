"""The runtime lane on its own terms.

`RuntimeValue` is one struct holding children, a payload and a function
pointer, so its correctness is mostly about **structure**: what it reports about
itself before anything evaluates it. Those answers are what the optimizer reads,
and a wrong one is silent — a plan that narrows the wrong columns still runs.
"""

from std.testing import assert_equal, assert_true

from ....arrays import StructArray, DynArray
from ...params import Bindings
from ....builders import array
from ....dtypes import int64
from ....scalars import DynScalar, Int32Scalar, Int64Scalar, StringScalar
from ....tabular import RecordBatch, record_batch
from ...logical import Shape
from ....builders import StringBuilder
from ....dtypes import string
from ..values import (
    Payload,
    RuntimeValue,
    and_,
    column,
    eq,
    ge,
    gt,
    le,
    literal,
    lt,
    ne,
    not_,
    or_,
    xor,
)


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


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
        column("b"),
        RuntimeValue("add", column("a"), column("b")),
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
    var produced = (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .dtype()
    )
    assert_true(v.dtype(b.schema) == produced)


def test_runtime_column_reads_the_named_column() raises:
    var b = _batch()
    var got = (
        column("b")
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
    )
    assert_true(got == b.column("b"))


def test_runtime_literal_broadcasts_to_the_batch_length() raises:
    """A literal owes a full column here, unlike the comptime lane's, which
    stays a scalar until something asks."""
    var b = _batch()
    var got = (
        literal(DynScalar(Int64Scalar(7)))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
    )
    assert_equal(len(got), 4)
    assert_true(got == array([7, 7, 7, 7], int64))


def test_runtime_a_subtree_copies_in_constant_time() raises:
    """Children sit behind `ArcPointer`, so copying a plan shares rather than
    clones. Equality of what the copy reports is the observable half."""
    var v = RuntimeValue("add", column("a"), column("b"))
    var w = v.copy()
    assert_equal(len(w.columns()), 2)
    assert_equal(w.columns()[0], "a")
    assert_equal(w.name(), v.name())


# ---------------------------------------------------------------------------
# Comparisons and boolean connectives
# ---------------------------------------------------------------------------


def _bits(v: RuntimeValue) raises -> String:
    """Render a predicate's result as `t`/`f`/`?` per row, over `_batch()`."""
    var b = _batch()
    var got = (
        v.evaluate(b.to_struct_array(), Bindings()).to_array(4).as_bool().copy()
    )
    var out = String()
    for i in range(len(got)):
        if got.is_null(i):
            out += "?"
        else:
            out += "t" if got[i].value() else "f"
    return out^


def _lit(i: Int) -> RuntimeValue:
    return literal(DynScalar(Int64Scalar(Int64(i))))


def test_runtime_all_six_comparisons() raises:
    """`a` is [1, 2, null, 4]; compared against 2.

    All six, not just the ordering pair -- `eq` is what statistics pruning and
    bloom filters key on, and the lane had none of them.
    """
    # a = [1, 2, null, 4]; the null sits at index 2.
    assert_equal(_bits(eq(column("a"), _lit(2))), "ft?f")
    assert_equal(_bits(ne(column("a"), _lit(2))), "tf?t")
    assert_equal(_bits(lt(column("a"), _lit(2))), "tf?f")
    assert_equal(_bits(le(column("a"), _lit(2))), "tt?f")
    assert_equal(_bits(gt(column("a"), _lit(2))), "ff?t")
    assert_equal(_bits(ge(column("a"), _lit(2))), "ft?t")


def test_runtime_comparison_between_two_columns() raises:
    """`a` [1,2,null,4] vs `b` [10,20,30,40] -- null propagates from either."""
    assert_equal(_bits(lt(column("a"), column("b"))), "tt?t")


def test_runtime_boolean_connectives_are_three_valued() raises:
    """Kleene, matching the fused lane: `null AND false` is false, not null."""
    var t = gt(column("b"), _lit(0))  # tttt
    var n = gt(column("a"), _lit(2))  # ff?t
    assert_equal(_bits(and_(t.copy(), n.copy())), "ff?t")
    assert_equal(_bits(or_(t.copy(), n.copy())), "tttt")
    assert_equal(_bits(xor(t.copy(), n.copy())), "tt?f")
    assert_equal(_bits(not_(n.copy())), "tt?f")


def test_runtime_comparison_promotes_mixed_widths() raises:
    """An int64 column against an int32 literal compares rather than raising."""
    var lit32 = literal(DynScalar(Int32Scalar(Int32(2))))
    assert_equal(_bits(gt(column("a"), lit32^)), "ff?t")


def test_runtime_string_comparison_uses_the_string_kernel() raises:
    """The same operator, dispatched on the runtime dtype."""
    var sb = StringBuilder(4)
    sb.append("apple")
    sb.append("pear")
    sb.append_null()
    sb.append("quince")
    var b = record_batch([sb.finish().to_dyn()], names=["s"])
    var pred = lt(column("s"), literal(DynScalar(StringScalar("pear"))))
    var got = (
        pred.evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # "apple" < "pear"
    assert_true(not got[1].value())  # "pear" == "pear"
    assert_true(got.is_null(2))
    assert_true(not got[3].value())  # "quince" > "pear"


def test_runtime_predicate_reports_its_columns() raises:
    """What the optimizer reads before anything evaluates."""
    var p = and_(gt(column("a"), _lit(1)), lt(column("b"), _lit(99)))
    var cols = String()
    for ref c in p.columns():
        cols += c
        cols += ","
    assert_equal(cols, "a,b,")
