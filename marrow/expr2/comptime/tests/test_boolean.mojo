"""Three-valued logic, against the full truth table.

`AND` and `OR` are the two operators whose nulls are **not** null-in-null-out,
so a structural validity rule silently gets them wrong in exactly four of nine
cases. The full 3x3 is spelled out rather than sampled, because those four are
the whole point: `FALSE AND NULL` is `FALSE`, not `NULL`.
"""

from std.testing import assert_equal, assert_true

from ....arrays import BoolArray
from ....builders import BoolBuilder, array
from ....dtypes import Int64Type, int64
from ....tabular import RecordBatch, record_batch
from ...core import into_array
from ..boolean import And, Not, Or, Xor
from ..core import ComptimeValue
from ..leaves import BoolColumn, Column, Literal
from ..numeric import Gt, Lt


def _flags() raises -> RecordBatch:
    """Every combination of TRUE / FALSE / NULL, in both operands."""
    return record_batch(
        [
            _bools([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            _bools([1, 0, -1, 1, 0, -1, 1, 0, -1]),
        ],
        names=["p", "q"],
    )


def _bools(codes: List[Int]) raises -> BoolArray:
    """`1` TRUE, `0` FALSE, `-1` NULL."""
    var b = BoolBuilder(capacity=len(codes))
    for ref c in codes:
        if c < 0:
            b.append_null()
        else:
            b.append(c == 1)
    return b.finish()


def _eval(v: Some[ComptimeValue], b: RecordBatch) raises -> BoolArray:
    return into_array(v.evaluate(b), b.num_rows()).as_bool().copy()


def test_and_follows_the_kleene_truth_table() raises:
    """`FALSE AND NULL` is FALSE — a known-false operand decides the result,
    so three of these are valid where a null-in-null-out rule says null."""
    var b = _flags()
    var got = _eval(And(BoolColumn("p"), BoolColumn("q")), b)
    assert_true(got == _bools([1, 0, -1, 0, 0, 0, -1, 0, -1]))


def test_or_follows_the_kleene_truth_table() raises:
    """`TRUE OR NULL` is TRUE — the dual: a known-true operand decides."""
    var b = _flags()
    var got = _eval(Or(BoolColumn("p"), BoolColumn("q")), b)
    assert_true(got == _bools([1, 1, 1, 1, 0, -1, 1, -1, -1]))


def test_not_is_null_preserving() raises:
    """Negation *is* structural — `NOT NULL` is NULL, unlike AND and OR."""
    var b = _flags()
    var got = _eval(Not(BoolColumn("p")), b)
    assert_true(got == _bools([0, 0, 0, 1, 1, 1, -1, -1, -1]))


def test_xor_propagates_nulls() raises:
    """No value of one operand decides an XOR, so it stays null-in-null-out."""
    var b = _flags()
    var got = _eval(Xor(BoolColumn("p"), BoolColumn("q")), b)
    assert_equal(got.null_count(), 5)


def test_and_takes_comparisons_as_operands() raises:
    """`a > 1 AND b < 40` — the shape a predicate actually has.

    Its operands are `NumericCompare`s, which belong to the *fusing* family.
    This node binds its operands on `ComptimeValue`, so the two families mix
    without either knowing about the other.
    """
    var b = record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )
    var v = And(
        Gt(Column[Int64Type]("a"), Literal[Int64Type](1)),
        Lt(Column[Int64Type]("b"), Literal[Int64Type](40)),
    )
    assert_true(_eval(v, b) == _bools([0, 1, 1, 0]))


def test_and_nests_within_itself() raises:
    """An `AND` is a valid operand of an `AND`, which only works because the
    operand bound is the base trait and not the fusing family."""
    var b = _flags()
    var v = And(And(BoolColumn("p"), BoolColumn("q")), BoolColumn("p"))
    assert_true(_eval(v, b) == _bools([1, 0, -1, 0, 0, 0, -1, 0, -1]))


def test_a_non_boolean_operand_is_rejected_by_name() raises:
    """The bound is `ComptimeValue`, so an `int64` operand type-checks. It has
    to be caught at evaluation, and the message has to say what it got."""
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    var v = And(BoolColumn("a"), BoolColumn("a"))
    var raised = False
    try:
        _ = _eval(v, b)
    except e:
        raised = True
        assert_true("expected bool operand" in String(e))
    assert_true(raised)
