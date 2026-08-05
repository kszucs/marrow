"""Regression pins for `StringPredicate.prepare` routing a scalar (literal)
right operand through `StringPredicateKernel.apply_scalar` instead of
splatting it into an n-row array first."""

from std.testing import assert_true, assert_equal

from ...builders import StringBuilder
from ...dtypes import string, int32
from ...tabular import record_batch
from ...expr.values import col, lit, into_array


def test_like_with_literal_matches_like_with_column() raises:
    """A constant pattern and a column of that same constant must agree.

    The literal route now skips materialising the pattern per row; this is what
    proves that is only a performance difference.
    """
    var sb = StringBuilder(capacity=4)
    sb.append("hello")
    sb.append("help")
    sb.append_null()
    sb.append("world")

    var pb = StringBuilder(capacity=4)
    for _ in range(4):
        pb.append("hel%")

    var batch = record_batch(
        [sb.finish().to_dyn(), pb.finish().to_dyn()], names=["s", "p"]
    )

    var by_literal = col("s", string).like(lit("hel%")).execute(batch)
    var by_column = col("s", string).like(col("p", string)).execute(batch)
    assert_true(
        into_array(by_literal, 4).as_bool() == into_array(by_column, 4).as_bool()
    )


def test_string_eq_with_literal_matches_eq_with_column() raises:
    """The same agreement for the default `apply_scalar` body, which the other
    five predicates inherit. `WHERE url = 'x'` is this shape."""
    var sb = StringBuilder(capacity=4)
    sb.append("x")
    sb.append_null()
    sb.append("y")
    sb.append("x")

    var pb = StringBuilder(capacity=4)
    for _ in range(4):
        pb.append("x")

    var batch = record_batch(
        [sb.finish().to_dyn(), pb.finish().to_dyn()], names=["s", "p"]
    )

    var by_literal = (col("s", string) == lit("x")).execute(batch)
    var by_column = (col("s", string) == col("p", string)).execute(batch)
    assert_true(
        into_array(by_literal, 4).as_bool() == into_array(by_column, 4).as_bool()
    )


def test_like_literal_preserves_null_rows() raises:
    """A null input row stays null, not False. The array path gets this from
    `Bitmap.intersect`; the scalar path must carry the left operand's validity
    itself."""
    var sb = StringBuilder(capacity=3)
    sb.append("hello")
    sb.append_null()
    sb.append("nope")
    var batch = record_batch([sb.finish().to_dyn()], names=["s"])

    var out = into_array(
        col("s", string).like(lit("hel%")).execute(batch), 3
    ).as_bool().copy()
    assert_true(out[0].value())
    assert_true(out.is_null(1))
    assert_true(not out[2].value())


def test_b27_string_length_agrees_fused_and_alone() raises:
    """`s.len()` and `s.len() + 0` must agree.

    B27: the fused form measured 25x faster than the bare one, which is only
    possible if they are not doing the same work. If it is computing something
    else, this catches it.
    """
    var sb = StringBuilder(4)
    sb.append("a")
    sb.append("bb")
    sb.append("ccc")
    sb.append("")
    var batch = record_batch([sb.finish().to_dyn()], names=["s"])

    var alone_arr = into_array(
        col("s", string).length().execute(batch), 4
    )
    var fused_arr = into_array(
        (col("s", string).length() + lit(0, int32)).execute(batch), 4
    )
    ref alone = alone_arr.as_int32()
    ref fused = fused_arr.as_int32()

    for i in range(4):
        assert_equal(Int(alone[i].value()), Int(fused[i].value()))
    assert_equal(Int(alone[0].value()), 1)
    assert_equal(Int(alone[1].value()), 2)
    assert_equal(Int(alone[2].value()), 3)
    assert_equal(Int(alone[3].value()), 0)


def test_b27_two_breakers_read_distinct_slots() raises:
    """Two breaker stages in one expression must read their own slots.

    `Context.get_ref` hands out a *borrow* into the slot's `Datum` rather than a
    copy (B27). Borrows are where aliasing goes wrong, so this pins the case with
    two live stages: if both lanes read the same slot, or a borrow outlived its
    slot, the sum would not be double the length.
    """
    var sb = StringBuilder(3)
    sb.append("a")
    sb.append("bb")
    sb.append("cccc")
    var batch = record_batch([sb.finish().to_dyn()], names=["s"])

    var doubled_arr = into_array(
        (col("s", string).length() + col("s", string).length()).execute(batch),
        3,
    )
    ref doubled = doubled_arr.as_int32()
    assert_equal(Int(doubled[0].value()), 2)
    assert_equal(Int(doubled[1].value()), 4)
    assert_equal(Int(doubled[2].value()), 8)


def test_b27_breaker_result_survives_the_fused_lane() raises:
    """A borrowed slot must stay valid for the whole fused pass.

    The array lives in the `Context`, which outlives the lane loop -- but a
    borrow that the compiler decided to shorten would read freed memory, and at
    these lengths that shows up as wrong values rather than a crash. 4096 rows
    is past any small-buffer or single-chunk special case.
    """
    var n = 4096
    var sb = StringBuilder(n)
    for i in range(n):
        sb.append(String("x", i))  # length varies with i
    var batch = record_batch([sb.finish().to_dyn()], names=["s"])

    var got_arr = into_array(
        (col("s", string).length() + lit(1, int32)).execute(batch), n
    )
    ref got = got_arr.as_int32()
    for i in range(n):
        var want = String("x", i).byte_length() + 1
        assert_equal(Int(got[i].value()), want)
