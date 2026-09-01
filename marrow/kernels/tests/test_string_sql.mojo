"""Edge cases for the SQL string function surface (`StringOperands` kernels).

The golden corpus asks these functions against its `text` fixture, six rows
chosen for *shape*: a separator, no separator, the empty string, a null,
surrounding padding, and a multi-byte character. What it cannot reach is the
boundary behaviour around each function's **arguments** — an empty needle, a
negative index, a window past the end, an empty fill — and that is where these
functions actually go wrong. Those cases are here.

The arguments are columns now, so a case that used to be unwritable is here
too: a **null argument**, which DuckDB answers with NULL in every position.
"""

from std.testing import assert_equal, assert_true

from ...arrays import Int64Array, StringArray
from ...builders import array, Int64Builder, StringBuilder
from ...dtypes import int64
from ...kernels.string import (
    AsciiKernel,
    CharLengthKernel,
    LPadKernel,
    LeftKernel,
    PositionKernel,
    RPadKernel,
    RepeatKernel,
    ReplaceKernel,
    RightKernel,
    SplitPartKernel,
    StringOperands,
    SubstrKernel,
    TrimCharsKernel,
)


def _null_middle() raises -> StringArray:
    """['ab', null, 'cd'] — the null row every kernel here must preserve."""
    var b = StringBuilder(capacity=3)
    b.append("ab")
    b.append_null()
    b.append("cd")
    return b.finish()


def _ops(
    n: Int,
    var text: String = String(),
    var alt: String = String(),
    start: Int = 0,
    count: Int = 0,
) raises -> StringOperands[]:
    """Constant arguments broadcast across `n` rows.

    What the expression layer produces from a literal operand — `Datum` splats
    a scalar into a column and hands the kernel that — spelled once here so the
    boundary cases below read as they did when the arguments *were* constants.
    Every slot is filled: a kernel that does not read one cannot tell.
    """
    var tb = StringBuilder(capacity=n)
    var ab = StringBuilder(capacity=n)
    var sb = Int64Builder(capacity=n)
    var cb = Int64Builder(capacity=n)
    for _ in range(n):
        tb.append(text)
        ab.append(alt)
        sb.append(Int64(start))
        cb.append(Int64(count))
    var ops = StringOperands()
    ops.text = tb.finish()
    ops.alt = ab.finish()
    ops.start = sb.finish()
    ops.count = cb.finish()
    return ops^


def _nulls(n: Int) raises -> Int64Array:
    """`n` null int64s — one argument column that is null everywhere."""
    var b = Int64Builder(capacity=n)
    for _ in range(n):
        b.append_null()
    return b.finish()


# --- substr ----------------------------------------------------------------


def test_substr_start_below_one_is_absorbed_by_the_count() raises:
    """`substr('abc', 0, 2)` is `'a'`, not `'ab'`: the window ends at
    `start + count`, so a start before the string eats part of the length."""
    var a = array(["abc", "abc"])
    assert_true(
        SubstrKernel.apply(a, _ops(len(a), start=0, count=2))
        == array(["a", "a"])
    )
    assert_true(
        SubstrKernel.apply(a, _ops(len(a), start=-1, count=3))
        == array(["a", "a"])
    )


def test_substr_past_the_end_clamps_and_zero_length_is_empty() raises:
    var a = array(["abc", "abc", "abc"])
    assert_true(
        SubstrKernel.apply(a, _ops(len(a), start=2, count=99))
        == array(["bc", "bc", "bc"])
    )
    assert_true(
        SubstrKernel.apply(a, _ops(len(a), start=9, count=3))
        == array(["", "", ""])
    )
    assert_true(
        SubstrKernel.apply(a, _ops(len(a), start=2, count=0))
        == array(["", "", ""])
    )


def test_substr_counts_characters_not_bytes() raises:
    # `é` is two bytes; one character from index 2 must come back whole.
    assert_true(
        SubstrKernel.apply(array(["héllo"]), _ops(1, start=2, count=1))
        == array(["é"])
    )


def test_substr_preserves_nulls() raises:
    var got = SubstrKernel.apply(_null_middle(), _ops(3, start=1, count=1))
    assert_true(got.is_null(1))
    assert_equal(String(got[0].value()), "a")


# --- left / right ----------------------------------------------------------


def test_left_and_right_negative_count_trims_the_other_end() raises:
    """DuckDB's reading: `left(s, -n)` is "all but the last n"."""
    var a = array(["abcde"])
    assert_true(LeftKernel.apply(a, _ops(len(a), count=-2)) == array(["abc"]))
    assert_true(RightKernel.apply(a, _ops(len(a), count=-2)) == array(["cde"]))


def test_left_and_right_clamp_past_the_end_and_at_zero() raises:
    var a = array(["abc"])
    assert_true(LeftKernel.apply(a, _ops(len(a), count=99)) == array(["abc"]))
    assert_true(RightKernel.apply(a, _ops(len(a), count=99)) == array(["abc"]))
    assert_true(LeftKernel.apply(a, _ops(len(a), count=0)) == array([""]))
    assert_true(RightKernel.apply(a, _ops(len(a), count=0)) == array([""]))


# --- lpad / rpad -----------------------------------------------------------


def test_pad_truncates_when_already_longer() raises:
    """Padding is also truncation, and both ends keep the *first* `count`
    characters — `rpad` does not truncate from the left."""
    var a = array(["abcdef"])
    assert_true(
        LPadKernel.apply(a, _ops(len(a), text="*", count=3)) == array(["abc"])
    )
    assert_true(
        RPadKernel.apply(a, _ops(len(a), text="*", count=3)) == array(["abc"])
    )


def test_pad_cycles_a_multi_character_fill() raises:
    var a = array(["x"])
    assert_true(
        LPadKernel.apply(a, _ops(len(a), text="ab", count=5))
        == array(["ababx"])
    )
    assert_true(
        RPadKernel.apply(a, _ops(len(a), text="ab", count=5))
        == array(["xabab"])
    )


def test_pad_with_an_empty_fill_cannot_pad() raises:
    """The loop-forever case: an empty fill leaves the input alone rather than
    never terminating."""
    assert_true(
        LPadKernel.apply(array(["ab"]), _ops(1, text="", count=8))
        == array(["ab"])
    )


def test_pad_counts_characters() raises:
    # 'héllo' is 5 characters and 6 bytes: padding to 6 adds exactly one.
    assert_true(
        LPadKernel.apply(array(["héllo"]), _ops(1, text="*", count=6))
        == array(["*héllo"])
    )


# --- replace ---------------------------------------------------------------


def test_replace_is_global_and_leaves_non_matches_alone() raises:
    assert_true(
        ReplaceKernel.apply(array(["a,b,c", "xyz"]), _ops(2, text=",", alt=";"))
        == array(["a;b;c", "xyz"])
    )


def test_replace_with_an_empty_pattern_is_the_identity() raises:
    """Arrow C++'s `replace_substring` loops forever on an empty pattern;
    DuckDB returns the input unchanged, and so does this."""
    assert_true(
        ReplaceKernel.apply(array(["abc"]), _ops(1, text="", alt="X"))
        == array(["abc"])
    )


def test_replace_with_an_empty_replacement_deletes() raises:
    assert_true(
        ReplaceKernel.apply(array(["a,b,c"]), _ops(1, text=",", alt=""))
        == array(["abc"])
    )


def test_replace_does_not_rescan_its_own_output() raises:
    """Replacing `a` with `aa` must terminate, doubling each `a` exactly
    once."""
    assert_true(
        ReplaceKernel.apply(array(["aba"]), _ops(1, text="a", alt="aa"))
        == array(["aabaa"])
    )


# --- split_part ------------------------------------------------------------


def test_split_part_out_of_range_is_the_empty_string() raises:
    """Empty, not null — DuckDB's choice, and the one this is written to."""
    var a = array(["a,b", "a,b"])
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=9))
        == array(["", ""])
    )
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=0))
        == array(["", ""])
    )


def test_split_part_without_the_separator_present() raises:
    var a = array(["abc", "abc"])
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=1))
        == array(["abc", "abc"])
    )
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=2))
        == array(["", ""])
    )


def test_split_part_empty_separator_yields_one_field() raises:
    var a = array(["abc", "abc"])
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text="", count=1))
        == array(["abc", "abc"])
    )
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text="", count=2))
        == array(["", ""])
    )


def test_split_part_adjacent_separators_give_an_empty_field() raises:
    var a = array(["a,,c", "a,,c"])
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=2))
        == array(["", ""])
    )
    assert_true(
        SplitPartKernel.apply(a, _ops(len(a), text=",", count=3))
        == array(["c", "c"])
    )


# --- trim(chars) -----------------------------------------------------------


def test_trim_takes_a_character_set_not_a_substring() raises:
    """`trim('xyzzyx', 'xy')` strips each member independently, so the inner
    `zz` survives and the trailing `x` goes."""
    assert_true(
        TrimCharsKernel.apply(array(["xyzzyx"]), _ops(1, text="xy"))
        == array(["zz"])
    )


def test_trim_is_case_sensitive_and_can_consume_everything() raises:
    assert_true(
        TrimCharsKernel.apply(array(["a,b,c", "  Ab  "]), _ops(2, text=" Ab"))
        == array(["a,b,c", ""])
    )


def test_trim_with_an_empty_set_strips_nothing() raises:
    assert_true(
        TrimCharsKernel.apply(array(["  ab  "]), _ops(1, text=""))
        == array(["  ab  "])
    )


def test_trim_multi_byte_set_member() raises:
    """A multi-byte member matches a whole character, never half of one."""
    assert_true(
        TrimCharsKernel.apply(array(["ééabéé"]), _ops(1, text="é"))
        == array(["ab"])
    )


# --- position / char_length / ascii ----------------------------------------


def test_position_is_one_based_and_zero_when_absent() raises:
    assert_true(
        PositionKernel.apply(array(["a,b", "abc"]), _ops(2, text=","))
        == array([2, 0], int64)
    )


def test_position_of_an_empty_needle_is_one() raises:
    """The classic trap: an empty needle is found at the start, so the SQL
    answer is 1 rather than 0."""
    assert_true(
        PositionKernel.apply(array(["abc"]), _ops(1, text=""))
        == array([1], int64)
    )


def test_position_counts_characters_not_bytes() raises:
    # 'héllo,' — the comma is the 6th character but the 7th byte.
    assert_true(
        PositionKernel.apply(array(["héllo,x"]), _ops(1, text=","))
        == array([6], int64)
    )


def test_char_length_counts_characters() raises:
    assert_true(
        CharLengthKernel.apply(array(["héllo wörld", ""]), _ops(2))
        == array([11, 0], int64)
    )


def test_char_length_of_a_null_is_null_not_zero() raises:
    var got = CharLengthKernel.apply(_null_middle(), _ops(3))
    assert_true(got.is_null(1))
    assert_equal(got[0].value(), Int64(2))


def test_ascii_decodes_utf8_and_answers_zero_for_empty() raises:
    # 233 for `é`, not the lead byte 0xC3 — the one case that must decode.
    assert_true(
        AsciiKernel.apply(array(["abc", "", "éa"]), _ops(3))
        == array([97, 0, 233], int64)
    )


def test_ascii_of_a_null_is_null() raises:
    var got = AsciiKernel.apply(_null_middle(), _ops(3))
    assert_true(got.is_null(1))


# --- repeat ----------------------------------------------------------------


def test_repeat_zero_and_negative_both_give_the_empty_string() raises:
    assert_true(
        RepeatKernel.apply(array(["ab", "ab", "ab"]), array([2, 0, -3], int64))
        == array(["abab", "", ""])
    )


def test_repeat_propagates_a_null_string() raises:
    """The null *count* side is asserted by the golden case, whose last row
    holds a null `n`."""
    var b = StringBuilder(capacity=2)
    b.append_null()
    b.append("ab")
    var got = RepeatKernel.apply(b.finish(), array([2, 2], int64))
    assert_true(got.is_null(0))
    assert_equal(String(got[1].value()), "abab")


# --- column arguments ------------------------------------------------------
#
# The half of the surface `StringArgs`-as-configuration made unrepresentable.
# An argument is a column now, so it can differ per row and it can be null.


def test_substr_reads_a_different_window_per_row() raises:
    """One `substr` call, three windows — what a constant cannot express."""
    var ops = StringOperands()
    ops.start = array([1, 2, 3], int64)
    ops.count = array([2, 3, 1], int64)
    assert_true(
        SubstrKernel.apply(array(["abcde", "abcde", "abcde"]), ops)
        == array(["ab", "bcd", "c"])
    )


def test_split_part_reads_a_different_separator_per_row() raises:
    var ops = StringOperands()
    ops.text = array([",", ";"])
    ops.count = array([2, 2], int64)
    assert_true(
        SplitPartKernel.apply(array(["a,b", "a;b"]), ops) == array(["b", "b"])
    )


def test_a_null_argument_makes_the_row_null() raises:
    """DuckDB answers NULL for `substring('abc', NULL, 2)` and for every other
    argument position; so does this, and the input string is untouched."""
    var ops = StringOperands()
    ops.start = _nulls(2)
    ops.count = array([2, 2], int64)
    var got = SubstrKernel.apply(array(["abc", "abc"]), ops)
    assert_true(got.is_null(0))
    assert_true(got.is_null(1))
    assert_equal(got.null_count(), 2)


def test_a_null_argument_nulls_only_its_own_row() raises:
    var b = Int64Builder(capacity=3)
    b.append(Int64(1))
    b.append_null()
    b.append(Int64(3))
    var ops = StringOperands()
    ops.count = b.finish()
    var got = LeftKernel.apply(array(["abc", "abc", "abc"]), ops)
    assert_equal(String(got[0].value()), "a")
    assert_true(got.is_null(1))
    assert_equal(String(got[2].value()), "abc")


def test_a_null_argument_makes_a_measure_null() raises:
    """`position(NULL IN 'abc')` is NULL, not 0 — 0 is the answer for "not
    found", which is a different fact."""
    var b = StringBuilder(capacity=2)
    b.append(",")
    b.append_null()
    var ops = StringOperands()
    ops.text = b.finish()
    var got = PositionKernel.apply(array(["a,b", "a,b"]), ops)
    assert_equal(got[0].value(), Int64(2))
    assert_true(got.is_null(1))
    assert_equal(got.null_count(), 1)
