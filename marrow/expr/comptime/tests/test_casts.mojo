"""Casts: the three that fuse and the two that break.

The node constructors are exercised directly here. The *spelling* a caller
uses is `x.cast(float64, safe=False)` — a method whose target comes from the
argument's type, so `golden/cases/cast_*.mojo` is one text in both lanes —
and `test_cast_method_builds_the_same_node` holds the two together.

The interesting half is validity. A fusing cast is null exactly where its
operand is; a parsing cast is null where its operand is *not*, and
`test_cast_string_to_int_null_survives_a_fused_parent` is the case that
distinguishes them.
"""

from std.testing import assert_equal, assert_true

from ...builders import col, lit
from ...bindings import Bindings
from ....builders import Int64Builder, array
from ....arrays import Int64Array, StringArray
from ....dtypes import (
    BoolType,
    Float64Type,
    Int32Type,
    Int64Type,
    StringType,
    bool_,
    float64,
    int64,
    string,
)
from ....tabular import RecordBatch, record_batch
from ..casts import BoolToNum, NumToBool, NumToString, NumericCast, StringToNum


def _ints() raises -> Int64Array:
    """`array()` has no nullable overload, so a column with a null is built
    through the builder."""
    var b = Int64Builder(4)
    b.append(0)
    b.append(7)
    b.append_null()
    b.append(-3)
    return b.finish()


def _strings() raises -> StringArray:
    """`"x"` is unparseable and `None` is already null — the two ways a parsed
    column acquires a null, which the cases below keep apart."""
    var values: List[Optional[String]] = ["12", "x", None, "-5"]
    return array(values)


def _batch() raises -> RecordBatch:
    # `array()`'s bool overload takes `List[Optional[Bool]]` and no dtype —
    # booleans are bit-packed, so there is no `PrimitiveArray[bool_]` to ask for.
    var flags: List[Optional[Bool]] = [True, False, True, False]
    return record_batch(
        [
            _ints().to_dyn(),
            _strings().to_dyn(),
            array([1.9, -1.9, 2.5, 0.0], float64).copy(),
            array(flags).to_dyn(),
        ],
        names=["i", "s", "f", "b"],
    )


# ---------------------------------------------------------------------------
# The three that fuse
# ---------------------------------------------------------------------------
def test_cast_int_to_float_widens() raises:
    var b = _batch()
    var got = (
        NumericCast[Float64Type](col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_float64()
        .copy()
    )
    assert_equal(got[0].value(), 0.0)
    assert_equal(got[1].value(), 7.0)
    assert_equal(got[3].value(), -3.0)


def test_cast_float_to_int_truncates_toward_zero() raises:
    """Truncation, not rounding — `1.9 -> 1` and `-1.9 -> -1`. The kernel owns
    the rule; this pins it so a cast node cannot quietly change it."""
    var b = _batch()
    var got = (
        NumericCast[Int64Type](col("f", float64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int64()
        .copy()
    )
    assert_equal(got[0].value(), 1)
    assert_equal(got[1].value(), -1)
    assert_equal(got[2].value(), 2)


def test_cast_narrows_to_int32() raises:
    var b = _batch()
    var got = (
        NumericCast[Int32Type](col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int32()
        .copy()
    )
    assert_equal(got[1].value(), 7)
    assert_equal(got[3].value(), -3)


def test_cast_is_null_exactly_where_its_operand_is() raises:
    """Structural validity: a cast changes what a value is, never whether
    there is one."""
    var b = _batch()
    var got = (
        NumericCast[Float64Type](col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.is_null(2))
    assert_true(not got.is_null(0))
    assert_equal(got.null_count(), 1)


def test_cast_fuses_into_arithmetic() raises:
    """The point of a fusing cast: no intermediate column. Observable only as
    the right answer at the right type, which is what this pins."""
    var b = _batch()
    var got = (
        (NumericCast[Float64Type](col("i", int64)) + lit(0.5, float64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_float64()
        .copy()
    )
    assert_equal(got[1].value(), 7.5)
    assert_true(got.is_null(2))


def test_cast_int_to_bool_is_nonzero() raises:
    var b = _batch()
    var got = (
        NumToBool(col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(not got[0].value())  # 0
    assert_true(got[1].value())  # 7
    assert_true(got.is_null(2))
    assert_true(got[3].value())  # -3 is nonzero, so True


def test_cast_bool_to_int() raises:
    var b = _batch()
    var got = (
        BoolToNum[Int64Type](col("b", bool_))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int64()
        .copy()
    )
    assert_true(got == array([1, 0, 1, 0], int64))


# ---------------------------------------------------------------------------
# The two that break
# ---------------------------------------------------------------------------
def test_cast_string_to_int_parses() raises:
    var b = _batch()
    var got = (
        StringToNum[Int64Type](col("s", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int64()
        .copy()
    )
    assert_equal(got[0].value(), 12)
    assert_equal(got[3].value(), -5)


def test_cast_string_to_int_nulls_what_will_not_parse() raises:
    """`"x"` is a null the *input does not have* — the one place a cast's
    validity is not its operand's, and the reason this node cannot inherit
    structural validity."""
    var b = _batch()
    var got = (
        StringToNum[Int64Type](col("s", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.is_null(1))  # "x" did not parse
    assert_true(got.is_null(2))  # the input was already null
    assert_equal(got.null_count(), 2)


def test_cast_string_to_int_null_survives_a_fused_parent() raises:
    """The regression this port exists to not reintroduce.

    The previous expression layer answered validity from the *batch* as well
    as from the state, and its batch-side answer re-ran the parse; inheriting
    the all-valid default instead made `to_int(s) + 1` yield 0 where it should
    be null. Here there is one source — the bound — so a fused parent reading
    `validity(bound)` sees the parse failure.
    """
    var b = _batch()
    var got = (
        (StringToNum[Int64Type](col("s", string)) + lit(1, int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.is_null(1), "'x' + 1 must be null, not 1")
    assert_equal(got.as_int64()[0].value(), 13)
    assert_equal(got.as_int64()[3].value(), -4)


def test_cast_int_to_string_formats() raises:
    var b = _batch()
    var got = (
        NumToString[StringType](col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_string()
        .copy()
    )
    assert_equal(String(got[0].value()), "0")
    assert_equal(String(got[1].value()), "7")
    assert_equal(String(got[3].value()), "-3")


def test_cast_int_to_string_preserves_nulls() raises:
    var b = _batch()
    var got = (
        NumToString[StringType](col("i", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.is_null(2))
    assert_equal(got.null_count(), 1)


# ---------------------------------------------------------------------------
# The method spelling
# ---------------------------------------------------------------------------


def test_cast_method_builds_the_same_node() raises:
    """`x.cast(t)` must build the node the constructor builds.

    The method exists so a golden case is one text in both lanes; if it built
    anything else the corpus would be testing a different query in each."""
    var direct = NumericCast[Float64Type](col("i", int64))
    var method = col("i", int64).cast(float64, safe=False)
    assert_equal(String(direct), String(method))

    var to_text = NumToString[StringType](col("i", int64))
    assert_equal(
        String(to_text), String(col("i", int64).cast(string, safe=False))
    )

    var parsed = StringToNum[Int64Type](col("s", string))
    assert_equal(
        String(parsed), String(col("s", string).cast(int64, safe=False))
    )

    var from_bool = BoolToNum[Int64Type](col("b", bool_))
    assert_equal(
        String(from_bool), String(col("b", bool_).cast(int64, safe=False))
    )

    var to_bool = NumToBool(col("i", int64))
    assert_equal(
        String(to_bool), String(col("i", int64).cast(bool_, safe=False))
    )


def test_cast_method_computes() raises:
    var batch = _batch()
    var out = (
        col("i", int64)
        .cast(float64, safe=False)
        .evaluate(batch.to_struct_array(), Bindings())
        .to_array(batch.num_rows())
    )
    assert_equal(len(out), 4)


def test_cast_refuses_a_checked_cast() raises:
    """`safe=True` has no fused implementation, so it is refused rather than
    silently answering the wrapping conversion — a caller who asked for a
    checked cast and got a wrapping one has a wrong answer, not a slow one."""
    var raised = False
    try:
        _ = col("i", int64).cast(float64, safe=True)
    except e:
        raised = True
        assert_true("no checked cast" in String(e))
    assert_true(raised, "safe=True must raise in the comptime lane")
