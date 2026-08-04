"""Regression pins for `StringPredicate.prepare` routing a scalar (literal)
right operand through `StringPredicateKernel.apply_scalar` instead of
splatting it into an n-row array first."""

from std.testing import assert_true

from ...builders import StringBuilder
from ...dtypes import string
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
