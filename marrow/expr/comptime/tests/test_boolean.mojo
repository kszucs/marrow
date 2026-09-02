"""Three-valued logic, against the full truth table.

`AND` and `OR` are the two operators whose nulls are **not** null-in-null-out,
so a structural validity rule silently gets them wrong in exactly four of nine
cases. The full 3x3 is spelled out rather than sampled, because those four are
the whole point: `FALSE AND NULL` is `FALSE`, not `NULL`.
"""

from std.testing import assert_equal, assert_true

from ...builders import col, is_in, lit
from ....arrays import BoolArray
from ...bindings import Bindings
from ....builders import BoolBuilder, StringBuilder, array
from ....dtypes import Int64Type, int32, int64, string
from ....tabular import RecordBatch, record_batch
from ..core import ComptimeValue
from ..boolean import And, IsIn, Not, Or, Xor

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
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_bool()
        .copy()
    )


def test_and_follows_the_kleene_truth_table() raises:
    """`FALSE AND NULL` is FALSE — a known-false operand decides the result,
    so three of these are valid where a null-in-null-out rule says null."""
    var b = _flags()
    var got = _eval((BoolColumn("p") & BoolColumn("q")), b)
    assert_true(got == _bools([1, 0, -1, 0, 0, 0, -1, 0, -1]))


def test_or_follows_the_kleene_truth_table() raises:
    """`TRUE OR NULL` is TRUE — the dual: a known-true operand decides."""
    var b = _flags()
    var got = _eval((BoolColumn("p") | BoolColumn("q")), b)
    assert_true(got == _bools([1, 1, 1, 1, 0, -1, 1, -1, -1]))


def test_not_is_null_preserving() raises:
    """Negation *is* structural — `NOT NULL` is NULL, unlike AND and OR."""
    var b = _flags()
    var got = _eval(Not(BoolColumn("p")), b)
    assert_true(got == _bools([0, 0, 0, 1, 1, 1, -1, -1, -1]))


def test_xor_propagates_nulls() raises:
    """No value of one operand decides an XOR, so it stays null-in-null-out."""
    var b = _flags()
    var got = _eval((BoolColumn("p") ^ BoolColumn("q")), b)
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
    var v = (col("a", int64) > lit(1, int64)) & (
        col("b", int64) < lit(40, int64)
    )
    assert_true(_eval(v, b) == _bools([0, 1, 1, 0]))


def test_and_nests_within_itself() raises:
    """An `AND` is a valid operand of an `AND`, which only works because the
    operand bound is the base trait and not the fusing family."""
    var b = _flags()
    var v = (BoolColumn("p") & BoolColumn("q")) & BoolColumn("p")
    assert_true(_eval(v, b) == _bools([1, 0, -1, 0, 0, 0, -1, 0, -1]))


def test_a_non_boolean_operand_is_rejected_by_name() raises:
    """The bound is `ComptimeValue`, so an `int64` operand type-checks. It has
    to be caught at evaluation, and the message has to say what it got."""
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    var v = BoolColumn("a") & BoolColumn("a")
    var raised = False
    try:
        _ = _eval(v, b)
    except e:
        raised = True
        assert_true("expected bool operand" in String(e))
    assert_true(raised)


# ---------------------------------------------------------------------------
# IsIn — membership against a constant set
# ---------------------------------------------------------------------------
def test_is_in_marks_the_members_and_never_answers_null() raises:
    """`IsInKernel`'s null rule, read through the node.

    The output is always valid — a null probes as TRUE only when the set holds
    a null, and FALSE otherwise — so a null row here answers FALSE rather than
    NULL. `ColumnBound` reports the kernel's bitmap, which is why the rule is
    stated once and not restated on the node.
    """
    var b = record_batch([array([1, 2, None, 4], int64).copy()], names=["a"])
    var got = _eval(is_in(col("a", int64), array([2, 4], int64).to_dyn()), b)
    assert_equal(got.null_count(), 0)
    assert_true(got == _bools([0, 1, 0, 1]))


def test_is_in_takes_a_null_in_the_set_as_a_match() raises:
    """The other half of `null_matching_behavior="match"`: with a null in the
    set, a null value *is* a member."""
    var b = record_batch([array([1, 2, None, 4], int64).copy()], names=["a"])
    var got = _eval(is_in(col("a", int64), array([2, None], int64).to_dyn()), b)
    assert_true(got == _bools([0, 1, 1, 0]))


def test_is_in_accepts_a_string_operand() raises:
    """The operand is bound on `ComptimeValue`, not on a family — membership
    is decided on the 64-bit hash, so one node serves every type
    `RapidHashKernel` supports."""
    var sb = StringBuilder(3)
    sb.append("apple")
    sb.append("pear")
    sb.append("plum")
    var b = record_batch([sb.finish().to_dyn()], names=["s"])
    var wanted = StringBuilder(2)
    wanted.append("plum")
    wanted.append("apple")
    var got = _eval(is_in(col("s", string), wanted.finish().to_dyn()), b)
    assert_true(got == _bools([1, 0, 1]))


def test_is_in_is_a_breaker_whose_operand_still_fuses() raises:
    """`is_in(a + 1, {3, 5})` — the membership test runs over a materialised
    column, but `a + 1` under it is still one fused loop. That is what a
    breaker costs and all it costs."""
    var b = record_batch([array([1, 2, 3, 4], int64).copy()], names=["a"])
    var v = is_in(
        col("a", int64) + lit(1, int64), array([3, 5], int64).to_dyn()
    )
    assert_true(_eval(v, b) == _bools([0, 1, 0, 1]))


def test_is_in_composes_with_the_kleene_connectives() raises:
    """A `BoolValue` is an ordinary operand of `AND`, so `IsIn` needed no
    special case to appear in a predicate."""
    var b = record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )
    var v = is_in(col("a", int64), array([2, 3, 4], int64).to_dyn()) & (
        col("b", int64) < lit(40, int64)
    )
    assert_true(_eval(v, b) == _bools([0, 1, 1, 0]))


def test_is_in_rejects_a_set_of_another_dtype() raises:
    """`bind` goes through `IsInKernel.dispatch`, so the same-dtype check the
    kernel owns is the one the node reports."""
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    var raised = False
    try:
        _ = _eval(is_in(col("a", int64), array([1, 2], int32).to_dyn()), b)
    except e:
        raised = True
        assert_true("int32" in String(e))
    assert_true(raised)


def test_is_in_prints_its_operand_and_its_set() raises:
    """`Unnamed`, so it has no name of its own — what it can say is what it
    computes over."""
    var v = is_in(col("a", int64), array([2, 4], int64).to_dyn())
    assert_equal(v.name(), "")
    assert_equal(v.columns()[0], "a")
    assert_true("is_in(col(a)" in String(v))
