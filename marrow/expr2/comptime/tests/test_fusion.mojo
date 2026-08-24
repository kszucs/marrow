"""Fusion, and the validity it must not lose.

A fused subtree compiles to one loop with no dispatch inside it, which is the
lane's whole reason to exist. The risk that buys is **validity**: `lane`
produces data bits only, so a null that the loop happily computes over has to be
recorded separately or it silently becomes a value. Every test here pairs a
fused result with the nulls it should have kept.
"""

from std.testing import assert_equal, assert_false, assert_true

from ....builders import array
from ....dtypes import DynType, Int64Type, int64
from ....tabular import RecordBatch, record_batch
from ....scalars import DynScalar
from ...logical import DynValue, Shape
from ...physical import Morsel
from ..leaves import Column, Literal
from ..numeric import Add, Gt, Mul, Sub


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_dtype_agrees_with_evaluation_comptime() raises:
    """The comptime lane answers from `Type` without touching the batch.

    That is the whole reason `dtype` exists rather than probing — so the two
    answers being equal is the property, not an implementation detail.
    """
    var b = _batch()
    var v = DynValue(Column[Int64Type]("a"))
    # A value is reached only through `to_operator` now: it is a stateless
    # description, and running it is the operator's job.
    var op = v.to_operator(False)
    var produced = (
        op.push(Morsel.ungrouped(b.copy()))
        .value()
        .to_array(b.num_rows())
        .dtype()
    )
    assert_true(v.dtype(b.schema) == produced)
    assert_true(v.dtype(b.schema) == DynType(int64))


def test_validity_matches_the_bound_column() raises:
    """`lane` produces data bits only, so validity is a separate contract.

    Reading it from the `Bound` rather than the batch is what stops the second
    pass `expr/` needed; the property is that it still reports the same nulls.
    """
    var b = _batch()
    var c = Column[Int64Type]("a")
    var bound = c.bind(b)
    var v = c.validity(bound)

    assert_true(v)
    # `a` is [1, 2, None, 4] — exactly one null, at index 2.
    assert_false(v.value()[2])
    assert_true(v.value()[0])


def test_a_literal_is_never_null() raises:
    var b = _batch()
    var l = Literal[Int64Type](7)
    assert_false(Bool(l.validity(l.bind(b))))


# ---------------------------------------------------------------------------
# Fusion: the lane computes data bits, validity records what they mean


def test_a_comparison_over_a_null_is_null_not_false() raises:
    """The defect class this whole validity contract exists to prevent.

    A SIMD lane compares whatever is in the slot, so the *data* bit for a null
    row is whatever the payload happened to be — usually zero, which reads as
    `False`. Only the validity bitmap records that the bit is meaningless.
    Reading the data bit without it is exactly what made NULL join keys match
    each other in `expr/`.
    """
    var b = _batch()  # a = [1, 2, None, 4]
    var pred = Gt(Column[Int64Type]("a"), Literal[Int64Type](2))
    var arr = pred.evaluate(b).to_array(b.num_rows())
    ref out = arr.as_bool()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    # The rows that are not null answer the comparison.
    assert_false(out[0].value())  # 1 > 2
    assert_false(out[1].value())  # 2 > 2
    assert_true(out[3].value())  # 4 > 2


def test_arithmetic_propagates_nulls_through_fusion() raises:
    """Null in, null out — across a fused subtree, not just one node."""
    var b = _batch()
    var sum = Add(Column[Int64Type]("a"), Column[Int64Type]("b"))
    var arr = sum.evaluate(b).to_array(b.num_rows())
    ref out = arr.as_int64()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    assert_equal(out[0].value(), 11)
    assert_equal(out[3].value(), 44)


def test_a_fused_subtree_is_one_expression() raises:
    """`(a + b) > 10` fuses three nodes; the null still propagates to the top.

    The point is not the arithmetic — it is that `bind` descends the whole
    subtree once and validity survives two levels, which is what a single
    fused pass has to preserve.
    """
    var b = _batch()
    var pred = Gt(
        Add(Column[Int64Type]("a"), Column[Int64Type]("b")),
        Literal[Int64Type](10),
    )
    var arr = pred.evaluate(b).to_array(b.num_rows())
    ref out = arr.as_bool()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    assert_true(out[0].value())  # 1 + 10 = 11 > 10
    assert_true(out[3].value())  # 4 + 40 = 44 > 10


def test_a_literal_only_expression_stays_scalar() raises:
    """`Shape.scalar` must survive composition, or every constant folds into a
    column before it is used."""
    var b = _batch()
    var both = Add(Literal[Int64Type](1), Literal[Int64Type](2))
    assert_true(both.shape == Shape.scalar)
    assert_true(both.evaluate(b).is_scalar())


# ---------------------------------------------------------------------------
# End to end: a dynamic plan holding a fused predicate


def test_sub_and_mul_fuse_like_add() raises:
    """`Sub` had zero test references and `Mul` only appeared inside an
    aggregate — found by auditing public names against the test corpus.

    They share `NumericBinary`, so this guards the alias wiring rather than the
    arithmetic: a mis-parameterised alias would compute the wrong operation
    while still type-checking.
    """
    var b = _batch()
    var op = DynValue(
        Sub(Column[Int64Type]("b"), Column[Int64Type]("a"))
    ).to_operator(False)
    var got = op.push(Morsel.ungrouped(b.copy())).value().to_array(4)
    # b = [10, 20, 30, 40], a = [1, 2, None, 4]
    assert_true(got.as_int64()[0].value() == 9)
    assert_true(got.as_int64().is_null(2))

    var m = DynValue(
        Mul(Column[Int64Type]("a"), Literal[Int64Type](3))
    ).to_operator(False)
    var prod = m.push(Morsel.ungrouped(b.copy())).value().to_array(4)
    assert_true(prod.as_int64()[1].value() == 6)
