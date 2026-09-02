"""Arithmetic beyond `+ - *`: the unary family, the float-returning family,
and the two value predicates.

Every kernel exercised here had been in `kernels/numeric.mojo` and
`kernels/boolean.mojo` since long before this lane existed, and none of them
was reachable from an expression — `NumericValue` carried three operators and
six comparisons and nothing else. What these cases pin is therefore the
*wiring*: which node each method builds, and which of the three output rules it
follows.

The three rules are the whole design, and the file is organised around them:

- **`NumericUnary` keeps the operand's dtype** — `abs(int32)` is `int32`.
- **`FloatUnary` / `FloatBinary` answer `float64` whatever went in** — `sqrt`
  and `/` are not closed over the integers, so preserving the dtype would
  truncate rather than widen.
- **`ValuePredicate` answers `bool` and propagates nulls** — `is_nan(NULL)` is
  NULL, which is the entire difference from `is_null`.
"""

from std.math import inf, nan
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from ...builders import col, lit, maximum, minimum
from ...bindings import Bindings
from ....arrays import BoolArray, Float64Array, Int64Array
from ....builders import array
from ....dtypes import float64, int32, int64
from ....tabular import RecordBatch, record_batch
from ..core import BoolValue, NumericValue


def _ints() raises -> RecordBatch:
    """A negative, a zero, a positive and a null, so `abs`/`sign`/`neg` each
    have every case they can distinguish."""
    var n: List[Optional[Int]] = [-9, 0, 4, None]
    var d: List[Optional[Int]] = [3, 3, 3, 3]
    return record_batch(
        [array(n, int64).to_dyn(), array(d, int64).to_dyn()], names=["n", "d"]
    )


def _floats() raises -> RecordBatch:
    var x: List[Optional[Float64]] = [-1.5, 2.25, 9.0, None]
    var y: List[Optional[Float64]] = [2.0, 4.0, 1.0, 2.0]
    return record_batch(
        [array(x, float64).to_dyn(), array(y, float64).to_dyn()],
        names=["x", "y"],
    )


def _as_i64(v: Some[NumericValue], b: RecordBatch) raises -> Int64Array:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_int64()
        .copy()
    )


def _as_f64(v: Some[NumericValue], b: RecordBatch) raises -> Float64Array:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_float64()
        .copy()
    )


def _as_bool(v: Some[BoolValue], b: RecordBatch) raises -> BoolArray:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_bool()
        .copy()
    )


# ---------------------------------------------------------------------------
# NumericUnary — the operand's dtype survives
# ---------------------------------------------------------------------------


def test_unary_arithmetic_keeps_the_operand_dtype() raises:
    """`-`, `abs` and `sign` over `int64` answer `int64`.

    Asserted by comparing against an `Int64Array` rather than by reading the
    dtype: structural equality already includes the dtype, so an accidental
    widening to `float64` fails here rather than needing its own case.
    """
    var b = _ints()
    var negated: List[Optional[Int]] = [9, 0, -4, None]
    assert_true(_as_i64(-col("n", int64), b) == array(negated, int64))
    var absolute: List[Optional[Int]] = [9, 0, 4, None]
    assert_true(_as_i64(col("n", int64).abs(), b) == array(absolute, int64))
    var signs: List[Optional[Int]] = [-1, 0, 1, None]
    assert_true(_as_i64(col("n", int64).sign(), b) == array(signs, int64))


def test_unary_rounding_differs_on_negatives() raises:
    """`floor`, `ceil`, `round` and `trunc` agree on 2.25 and disagree on
    -1.5, which is why all four are asserted against the same column: a
    mis-wiring that pointed `trunc` at `FloorKernel` survives any test that
    only ever looks at a positive value."""
    var b = _floats()
    var floored: List[Optional[Float64]] = [-2.0, 2.0, 9.0, None]
    assert_true(
        _as_f64(col("x", float64).floor(), b) == array(floored, float64)
    )
    var ceiled: List[Optional[Float64]] = [-1.0, 3.0, 9.0, None]
    assert_true(_as_f64(col("x", float64).ceil(), b) == array(ceiled, float64))
    var truncated: List[Optional[Float64]] = [-1.0, 2.0, 9.0, None]
    assert_true(
        _as_f64(col("x", float64).trunc(), b) == array(truncated, float64)
    )
    var rounded: List[Optional[Float64]] = [-2.0, 2.0, 9.0, None]
    assert_true(
        _as_f64(col("x", float64).round(), b) == array(rounded, float64)
    )


def test_unary_arithmetic_forwards_validity() raises:
    """One operand means nothing to intersect: the result is null exactly
    where the input was, and none of these kernels manufactures a null."""
    var b = _ints()
    var got = _as_i64(col("n", int64).abs(), b)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(3))


# ---------------------------------------------------------------------------
# FloatUnary / FloatBinary — float64 whatever went in
# ---------------------------------------------------------------------------


def test_true_division_of_integers_is_not_integer_division() raises:
    """`-9 / 3` is -3.0 and `4 / 3` is 1.333…, not 1.

    This is the divergence from PyArrow worth stating: `pc.divide` on two
    integer arrays returns an integer. marrow follows Python so that `/`, `//`
    and `%` agree with each other — `a == (a // b) * b + a % b`.
    """
    var b = _ints()
    var got = _as_f64(col("n", int64) / col("d", int64), b)
    assert_almost_equal(got[0].value(), Float64(-3.0))
    assert_almost_equal(got[2].value(), Float64(4.0 / 3.0))
    assert_true(got.is_null(3))


def test_floordiv_and_mod_take_the_sign_of_the_divisor() raises:
    """`-9 // 3` is -3 and `-9 % 3` is 0; the interesting row is 4, where
    floored and truncated division still agree. The identity
    `a == (a // b) * b + a % b` is what these two are for, and it holds only
    under one convention — SQL and arrow-rs pick the other."""
    var b = _ints()
    var quotients: List[Optional[Int]] = [-3, 0, 1, None]
    assert_true(
        _as_i64(col("n", int64) // col("d", int64), b)
        == array(quotients, int64)
    )
    var remainders: List[Optional[Int]] = [0, 0, 1, None]
    assert_true(
        _as_i64(col("n", int64) % col("d", int64), b)
        == array(remainders, int64)
    )


def test_pow_and_the_float_unaries_answer_float64() raises:
    """`sqrt(9)` is 3.0 and `exp(0)`/`ln(1)` are the two inputs whose result
    is exact under IEEE 754 — the same restriction `golden/cases/math_exp_zero`
    and `math_ln_one` impose, because neither function is correctly rounded and
    a general input would make the case flake."""
    var b = _floats()
    var squared: List[Optional[Float64]] = [2.25, 5.0625, 81.0, None]
    assert_true(
        _as_f64(col("x", float64) ** lit(2.0, float64), b)
        == array(squared, float64)
    )
    assert_almost_equal(
        _as_f64(col("x", float64).sqrt(), b)[2].value(), Float64(3.0)
    )
    # `y` is [2, 4, 1, 2]: `ln(1)` is exactly 0.0 and `exp(x - x)` is exactly
    # 1.0. Both are the only inputs either function answers exactly.
    assert_almost_equal(
        _as_f64(col("y", float64).ln(), b)[2].value(), Float64(0.0)
    )
    var ones = _as_f64((col("y", float64) - col("y", float64)).exp(), b)
    assert_almost_equal(ones[0].value(), Float64(1.0))
    assert_almost_equal(ones[1].value(), Float64(1.0))


def test_float_returning_ops_widen_an_integer_operand() raises:
    """`sqrt` over `int64` is `float64`, not a truncated `int64`.

    That is what separates `FloatUnary` from `NumericUnary`: `abs` is closed
    over the integers and keeps its dtype, `sqrt` is not and cannot.
    """
    var b = _ints()
    var got = _as_f64(col("n", int64).abs().sqrt(), b)
    assert_almost_equal(got[0].value(), Float64(3.0))
    assert_almost_equal(got[2].value(), Float64(2.0))


# ---------------------------------------------------------------------------
# ValuePredicate — is_nan / is_inf
# ---------------------------------------------------------------------------


def test_is_nan_and_is_inf_are_null_on_a_null() raises:
    """The rule that makes these different from `is_null`: `is_null(NULL)` is
    TRUE, `is_nan(NULL)` is NULL. A null is not a NaN.

    The kernel propagates the operand's validity and `ColumnBound` reads it
    back, so this asserts the wiring reached the right kernel rather than
    re-deriving the rule.
    """
    var vals: List[Optional[Float64]] = [1.0, None, 3.0]
    var b = record_batch([array(vals, float64).to_dyn()], names=["x"])
    var nan = _as_bool(col("x", float64).is_nan(), b)
    assert_equal(nan.null_count(), 1)
    assert_true(nan.is_null(1))
    assert_true(not nan[0].value())
    var inf = _as_bool(col("x", float64).is_inf(), b)
    assert_equal(inf.null_count(), 1)
    assert_true(not inf[2].value())


def test_is_nan_finds_a_nan_and_is_inf_finds_an_infinity() raises:
    """The values are put into the column, not produced by dividing by zero.

    That is worth stating, because the obvious construction does not work:
    `DivKernel.core` substitutes 1 for **any** zero divisor — `a /
    b.eq(0).select(1, b)` — to dodge SIGFPE on integers, and the guard is not
    conditioned on the dtype. So `0.0 / 0.0` evaluates to 0.0 here and
    `1.0 / 0.0` to 1.0, where IEEE 754 says NaN and +inf. Whether marrow's `/`
    should produce an infinity on a float column is a question about
    `DivKernel`, not about these nodes, so this case sidesteps it and the
    golden corpus's `math_div_float64` avoids it by using a divisor column with
    no zero.
    """
    var vals: List[Optional[Float64]] = [
        nan[DType.float64](),
        inf[DType.float64](),
        -inf[DType.float64](),
        3.0,
    ]
    var b = record_batch([array(vals, float64).to_dyn()], names=["x"])
    var is_nan = _as_bool(col("x", float64).is_nan(), b)
    assert_true(is_nan[0].value())
    assert_true(not is_nan[1].value())
    assert_true(not is_nan[3].value())
    var is_inf = _as_bool(col("x", float64).is_inf(), b)
    assert_true(not is_inf[0].value())
    # Both signs, which is what "is an infinity" means.
    assert_true(is_inf[1].value() and is_inf[2].value())
    assert_true(not is_inf[3].value())


def test_arithmetic_nodes_still_fuse_into_one_pass() raises:
    """A unary node's `Bound` is its operand's, so `abs(a - b) * 2` binds the
    two columns once and the whole tree is one loop — the property the lane
    exists for, asserted here for the nodes this change added."""
    var b = _ints()
    var got = _as_i64(
        (col("n", int64) - col("d", int64)).abs() * lit(2, int64), b
    )
    var expected: List[Optional[Int]] = [24, 6, 2, None]
    assert_true(got == array(expected, int64))


# ---------------------------------------------------------------------------
# The rest of the two families — kernels that had no verb until 2026-08-31
# ---------------------------------------------------------------------------


def test_the_rest_of_the_float_unary_family_is_reachable() raises:
    """Six `UnaryFloatKernel`s that `NumericValue` named none of.

    Exactness is chosen the way the `sqrt`/`exp`/`ln` case above chooses it.
    `y` is [2, 4, 1, 2], so `exp2` and `log2` land on powers of two and are
    exact; `y - y` is zero, the one input at which `log1p`, `sin` and `cos`
    are.
    """
    var b = _floats()
    var powers: List[Optional[Float64]] = [4.0, 16.0, 2.0, 4.0]
    assert_true(_as_f64(col("y", float64).exp2(), b) == array(powers, float64))
    var logs: List[Optional[Float64]] = [1.0, 2.0, 0.0, 1.0]
    assert_true(_as_f64(col("y", float64).log2(), b) == array(logs, float64))
    # At `y[1] == 4.0`, not at `y[2] == 1.0`: every logarithm is 0 at 1, so a
    # `Log10 = FloatUnary[Log2Kernel]` mis-wiring survives an assertion there.
    assert_almost_equal(
        _as_f64(col("y", float64).log10(), b)[1].value(),
        Float64(0.6020599913279624),
    )
    var zero = col("y", float64) - col("y", float64)
    assert_almost_equal(_as_f64(zero.log1p(), b)[0].value(), Float64(0.0))
    assert_almost_equal(_as_f64(zero.sin(), b)[0].value(), Float64(0.0))
    assert_almost_equal(_as_f64(zero.cos(), b)[0].value(), Float64(1.0))


def test_the_new_float_unaries_propagate_a_null() raises:
    """`FloatUnary` forwards its operand's validity, so a null stays a null
    rather than turning into a NaN — the rule `sqrt` already followed."""
    var b = _floats()
    assert_true(_as_f64(col("x", float64).sin(), b).is_null(3))
    assert_true(_as_f64(col("x", float64).exp2(), b).is_null(3))


def test_minimum_and_maximum_choose_between_two_operands() raises:
    """The row-wise extrema, not the aggregates: these pick between two
    operands on each row where `MIN(x)` folds a whole column.

    `NumericBinary` nodes, so they inherit its rules rather than restating
    them — including the null rule the case below is about.
    """
    var b = _ints()
    var smaller: List[Optional[Int]] = [-9, 0, 3, None]
    assert_true(
        _as_i64(minimum(col("n", int64), col("d", int64)), b)
        == array(smaller, int64)
    )
    var larger: List[Optional[Int]] = [3, 3, 4, None]
    assert_true(
        _as_i64(maximum(col("n", int64), col("d", int64)), b)
        == array(larger, int64)
    )


def test_minimum_and_maximum_are_null_in_null_out_not_sql_least() raises:
    """The reason the verbs are not called `least` / `greatest`.

    `n` is null on row 3 and `d` is 3 there. SQL's `LEAST(NULL, 3)` is **3**
    — it skips nulls — and so does `pc.min_element_wise` at its
    `skip_nulls=True` default. `MinKernel` intersects validity like every
    other `BinaryNumericKernel`, so this answers NULL. That is a legitimate
    operation under PyArrow's `skip_nulls=False`, and a wrong answer under
    SQL's name, which is why the SQL name is withheld until a kernel skips.
    """
    var b = _ints()
    assert_true(
        _as_i64(minimum(col("n", int64), col("d", int64)), b).is_null(3)
    )


def test_minimum_and_maximum_promote_a_narrower_operand() raises:
    """`promote` is `NumericBinary`'s, unchanged: an `int32` literal against
    an `int64` column gives an `int64` result, so `_as_i64` can read it."""
    var b = _ints()
    var capped: List[Optional[Int]] = [-9, 0, 2, None]
    assert_true(
        _as_i64(minimum(col("n", int64), lit(2, int32)), b)
        == array(capped, int64)
    )
