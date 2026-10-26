# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Edge cases for the SQL string function surface.

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


def _ints(n: Int, value: Int) raises -> Int64Array:
    """A constant integer argument broadcast across `n` rows, which is what
    the expression layer produces from a literal operand."""
    var b = Int64Builder(capacity=n)
    for _ in range(n):
        b.append(Int64(value))
    return b.finish()


def _texts(n: Int, value: String) raises -> StringArray:
    """A constant string argument broadcast across `n` rows."""
    var b = StringBuilder(capacity=n)
    for _ in range(n):
        b.append(value)
    return b.finish()


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
        SubstrKernel.apply(a, _ints(len(a), 0), _ints(len(a), 2))
        == array(["a", "a"])
    )
    assert_true(
        SubstrKernel.apply(a, _ints(len(a), -1), _ints(len(a), 3))
        == array(["a", "a"])
    )


def test_substr_past_the_end_clamps_and_zero_length_is_empty() raises:
    var a = array(["abc", "abc", "abc"])
    assert_true(
        SubstrKernel.apply(a, _ints(len(a), 2), _ints(len(a), 99))
        == array(["bc", "bc", "bc"])
    )
    assert_true(
        SubstrKernel.apply(a, _ints(len(a), 9), _ints(len(a), 3))
        == array(["", "", ""])
    )
    assert_true(
        SubstrKernel.apply(a, _ints(len(a), 2), _ints(len(a), 0))
        == array(["", "", ""])
    )


def test_substr_counts_characters_not_bytes() raises:
    # `é` is two bytes; one character from index 2 must come back whole.
    assert_true(
        SubstrKernel.apply(array(["héllo"]), _ints(1, 2), _ints(1, 1))
        == array(["é"])
    )


def test_substr_preserves_nulls() raises:
    var got = SubstrKernel.apply(_null_middle(), _ints(3, 1), _ints(3, 1))
    assert_true(got.is_null(1))
    assert_equal(String(got[0].value()), "a")


# --- left / right ----------------------------------------------------------


def test_left_and_right_negative_count_trims_the_other_end() raises:
    """DuckDB's reading: `left(s, -n)` is "all but the last n"."""
    var a = array(["abcde"])
    assert_true(LeftKernel.apply(a, _ints(len(a), -2)) == array(["abc"]))
    assert_true(RightKernel.apply(a, _ints(len(a), -2)) == array(["cde"]))


def test_left_and_right_clamp_past_the_end_and_at_zero() raises:
    var a = array(["abc"])
    assert_true(LeftKernel.apply(a, _ints(len(a), 99)) == array(["abc"]))
    assert_true(RightKernel.apply(a, _ints(len(a), 99)) == array(["abc"]))
    assert_true(LeftKernel.apply(a, _ints(len(a), 0)) == array([""]))
    assert_true(RightKernel.apply(a, _ints(len(a), 0)) == array([""]))


# --- lpad / rpad -----------------------------------------------------------


def test_pad_truncates_when_already_longer() raises:
    """Padding is also truncation, and both ends keep the *first* `count`
    characters — `rpad` does not truncate from the left."""
    var a = array(["abcdef"])
    assert_true(
        LPadKernel.apply(a, _ints(len(a), 3), _texts(len(a), "*"))
        == array(["abc"])
    )
    assert_true(
        RPadKernel.apply(a, _ints(len(a), 3), _texts(len(a), "*"))
        == array(["abc"])
    )


def test_pad_cycles_a_multi_character_fill() raises:
    var a = array(["x"])
    assert_true(
        LPadKernel.apply(a, _ints(len(a), 5), _texts(len(a), "ab"))
        == array(["ababx"])
    )
    assert_true(
        RPadKernel.apply(a, _ints(len(a), 5), _texts(len(a), "ab"))
        == array(["xabab"])
    )


def test_pad_with_an_empty_fill_cannot_pad() raises:
    """The loop-forever case: an empty fill leaves the input alone rather than
    never terminating."""
    assert_true(
        LPadKernel.apply(array(["ab"]), _ints(1, 8), _texts(1, ""))
        == array(["ab"])
    )


def test_pad_counts_characters() raises:
    # 'héllo' is 5 characters and 6 bytes: padding to 6 adds exactly one.
    assert_true(
        LPadKernel.apply(array(["héllo"]), _ints(1, 6), _texts(1, "*"))
        == array(["*héllo"])
    )


# --- replace ---------------------------------------------------------------


def test_replace_is_global_and_leaves_non_matches_alone() raises:
    assert_true(
        ReplaceKernel.apply(
            array(["a,b,c", "xyz"]), _texts(2, ","), _texts(2, ";")
        )
        == array(["a;b;c", "xyz"])
    )


def test_replace_with_an_empty_pattern_is_the_identity() raises:
    """Arrow C++'s `replace_substring` loops forever on an empty pattern;
    DuckDB returns the input unchanged, and so does this."""
    assert_true(
        ReplaceKernel.apply(array(["abc"]), _texts(1, ""), _texts(1, "X"))
        == array(["abc"])
    )


def test_replace_with_an_empty_replacement_deletes() raises:
    assert_true(
        ReplaceKernel.apply(array(["a,b,c"]), _texts(1, ","), _texts(1, ""))
        == array(["abc"])
    )


def test_replace_does_not_rescan_its_own_output() raises:
    """Replacing `a` with `aa` must terminate, doubling each `a` exactly
    once."""
    assert_true(
        ReplaceKernel.apply(array(["aba"]), _texts(1, "a"), _texts(1, "aa"))
        == array(["aabaa"])
    )


# --- split_part ------------------------------------------------------------


def test_split_part_out_of_range_is_the_empty_string() raises:
    """Empty, not null — DuckDB's choice, and the one this is written to."""
    var a = array(["a,b", "a,b"])
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 9))
        == array(["", ""])
    )
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 0))
        == array(["", ""])
    )


def test_split_part_without_the_separator_present() raises:
    var a = array(["abc", "abc"])
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 1))
        == array(["abc", "abc"])
    )
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 2))
        == array(["", ""])
    )


def test_split_part_empty_separator_yields_one_field() raises:
    var a = array(["abc", "abc"])
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ""), _ints(len(a), 1))
        == array(["abc", "abc"])
    )
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ""), _ints(len(a), 2))
        == array(["", ""])
    )


def test_split_part_adjacent_separators_give_an_empty_field() raises:
    var a = array(["a,,c", "a,,c"])
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 2))
        == array(["", ""])
    )
    assert_true(
        SplitPartKernel.apply(a, _texts(len(a), ","), _ints(len(a), 3))
        == array(["c", "c"])
    )


# --- trim(chars) -----------------------------------------------------------


def test_trim_takes_a_character_set_not_a_substring() raises:
    """`trim('xyzzyx', 'xy')` strips each member independently, so the inner
    `zz` survives and the trailing `x` goes."""
    assert_true(
        TrimCharsKernel.apply(array(["xyzzyx"]), _texts(1, "xy"))
        == array(["zz"])
    )


def test_trim_is_case_sensitive_and_can_consume_everything() raises:
    assert_true(
        TrimCharsKernel.apply(array(["a,b,c", "  Ab  "]), _texts(2, " Ab"))
        == array(["a,b,c", ""])
    )


def test_trim_with_an_empty_set_strips_nothing() raises:
    assert_true(
        TrimCharsKernel.apply(array(["  ab  "]), _texts(1, ""))
        == array(["  ab  "])
    )


def test_trim_multi_byte_set_member() raises:
    """A multi-byte member matches a whole character, never half of one."""
    assert_true(
        TrimCharsKernel.apply(array(["ééabéé"]), _texts(1, "é"))
        == array(["ab"])
    )


# --- position / char_length / ascii ----------------------------------------


def test_position_is_one_based_and_zero_when_absent() raises:
    assert_true(
        PositionKernel.apply(array(["a,b", "abc"]), _texts(2, ","))
        == array([2, 0], int64)
    )


def test_position_of_an_empty_needle_is_one() raises:
    """The classic trap: an empty needle is found at the start, so the SQL
    answer is 1 rather than 0."""
    assert_true(
        PositionKernel.apply(array(["abc"]), _texts(1, "")) == array([1], int64)
    )


def test_position_counts_characters_not_bytes() raises:
    # 'héllo,' — the comma is the 6th character but the 7th byte.
    assert_true(
        PositionKernel.apply(array(["héllo,x"]), _texts(1, ","))
        == array([6], int64)
    )


def test_char_length_counts_characters() raises:
    assert_true(
        CharLengthKernel.apply(array(["héllo wörld", ""]))
        == array([11, 0], int64)
    )


def test_char_length_of_a_null_is_null_not_zero() raises:
    var got = CharLengthKernel.apply(_null_middle())
    assert_true(got.is_null(1))
    assert_equal(got[0].value(), Int64(2))


def test_ascii_decodes_utf8_and_answers_zero_for_empty() raises:
    # 233 for `é`, not the lead byte 0xC3 — the one case that must decode.
    assert_true(
        AsciiKernel.apply(array(["abc", "", "éa"]))
        == array([97, 0, 233], int64)
    )


def test_ascii_of_a_null_is_null() raises:
    var got = AsciiKernel.apply(_null_middle())
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
# The half of the surface the constants made unrepresentable. An argument is a
# column now, so it can differ per row and it can be null.


def test_substr_reads_a_different_window_per_row() raises:
    """One `substr` call, three windows — what a constant cannot express."""
    assert_true(
        SubstrKernel.apply(
            array(["abcde", "abcde", "abcde"]),
            array([1, 2, 3], int64),
            array([2, 3, 1], int64),
        )
        == array(["ab", "bcd", "c"])
    )


def test_split_part_reads_a_different_separator_per_row() raises:
    assert_true(
        SplitPartKernel.apply(
            array(["a,b", "a;b"]), array([",", ";"]), array([2, 2], int64)
        )
        == array(["b", "b"])
    )


def test_a_null_argument_makes_the_row_null() raises:
    """DuckDB answers NULL for `substring('abc', NULL, 2)` and for every other
    argument position; so does this, and the input string is untouched."""
    var got = SubstrKernel.apply(
        array(["abc", "abc"]), _nulls(2), array([2, 2], int64)
    )
    assert_true(got.is_null(0))
    assert_true(got.is_null(1))
    assert_equal(got.null_count(), 2)


def test_a_null_argument_nulls_only_its_own_row() raises:
    var b = Int64Builder(capacity=3)
    b.append(Int64(1))
    b.append_null()
    b.append(Int64(3))
    var got = LeftKernel.apply(array(["abc", "abc", "abc"]), b.finish())
    assert_equal(String(got[0].value()), "a")
    assert_true(got.is_null(1))
    assert_equal(String(got[2].value()), "abc")


def test_a_null_argument_makes_a_measure_null() raises:
    """`position(NULL IN 'abc')` is NULL, not 0 — 0 is the answer for "not
    found", which is a different fact."""
    var b = StringBuilder(capacity=2)
    b.append(",")
    b.append_null()
    var got = PositionKernel.apply(array(["a,b", "a,b"]), b.finish())
    assert_equal(got[0].value(), Int64(2))
    assert_true(got.is_null(1))
    assert_equal(got.null_count(), 1)


# --- string_view: through the offsets layout and back ----------------------

from ...arrays import StringViewArray
from ...dtypes import string_view


def test_substr_string_view() raises:
    var a: StringViewArray = ["abcdef", "a value longer than twelve"]
    var out = SubstrKernel.dispatch(a^.to_dyn(), _ints(2, 2), _ints(2, 3))
    assert_true(out.dtype() == string_view)
    ref v = out.as_string_view()
    assert_equal(v[0].value(), "bcd")
    assert_equal(v[1].value(), " va")


def test_char_length_string_view() raises:
    var a: StringViewArray = ["héllo", "a value longer than twelve"]
    var out = CharLengthKernel.dispatch(a^.to_dyn())
    assert_equal(out.as_int64()[0].value(), 5)
    assert_equal(out.as_int64()[1].value(), 26)


# --- hex and base64: the edges the golden corpus cannot spell --------------

from std.testing import assert_raises

from ...arrays import BinaryArray
from ...builders import BinaryBuilder, LargeBinaryBuilder
from ...dtypes import binary, large_binary, large_string, string
from ...kernels.string import (
    FromBase64Kernel,
    FromHexKernel,
    ToBase64Kernel,
    ToHexKernel,
)


def _blob(bytes: List[Byte]) -> StringSlice[origin_of(bytes)]:
    return StringSlice(unsafe_from_utf8=Span(bytes))


def test_to_hex_reads_binary_and_keeps_nulls() raises:
    var b = BinaryBuilder(capacity=3)
    var raw: List[Byte] = [0x00, 0xFF, 0x0A]
    b.append(_blob(raw))
    b.append_null()
    b.append("")
    var out = ToHexKernel.dispatch(b.finish().to_dyn())
    assert_true(out.dtype() == string)
    var want = StringBuilder(capacity=3)
    want.append("00FF0A")
    want.append_null()
    want.append("")
    assert_true(out.as_string() == want.finish())


def test_to_hex_of_large_binary_is_large_string() raises:
    var b = LargeBinaryBuilder(capacity=1)
    b.append("ab")
    var out = ToHexKernel.dispatch(b.finish().to_dyn())
    assert_true(out.dtype() == large_string)
    assert_equal(out.as_large_string()[0].value(), "6162")


def test_from_hex_takes_either_case_and_pads_an_odd_count() raises:
    """DuckDB reads `'abc'` as `0x0abc`: an odd count is led by a `0`."""
    var a = StringBuilder(capacity=4)
    for text in ["00ff", "C3a9", "abc"]:
        a.append(text)
    a.append_null()
    var out = FromHexKernel.dispatch(a.finish().to_dyn())
    assert_true(out.dtype() == binary)
    var b = BinaryBuilder(capacity=4)
    var r0: List[Byte] = [0x00, 0xFF]
    var r1: List[Byte] = [0xC3, 0xA9]
    var r2: List[Byte] = [0x0A, 0xBC]
    b.append(_blob(r0))
    b.append(_blob(r1))
    b.append(_blob(r2))
    b.append_null()
    var want: BinaryArray = b.finish()
    assert_true(out.as_binary() == want)


def test_from_hex_names_the_row_it_cannot_decode() raises:
    var a: StringArray = ["00", "0g"]
    with assert_raises(contains="row 1"):
        _ = FromHexKernel.dispatch(a^.to_dyn())


def test_from_hex_wants_text_not_bytes() raises:
    var b = BinaryBuilder(capacity=1)
    b.append("00")
    with assert_raises(contains="expected a string array"):
        _ = FromHexKernel.dispatch(b.finish().to_dyn())


def test_base64_pads_as_rfc_4648_does() raises:
    """RFC 4648 section 10's test vectors, and back."""
    var a: StringArray = ["", "f", "fo", "foo", "foob", "fooba", "foobar"]
    var out = ToBase64Kernel.dispatch(a^.to_dyn())
    var want: StringArray = [
        "",
        "Zg==",
        "Zm8=",
        "Zm9v",
        "Zm9vYg==",
        "Zm9vYmE=",
        "Zm9vYmFy",
    ]
    assert_true(out.as_string() == want)
    var back = FromBase64Kernel.dispatch(out)
    assert_true(back.dtype() == binary)
    assert_equal(String(back.as_binary()[6].value()), "foobar")


def test_from_base64_rejects_a_truncated_quantum() raises:
    var a: StringArray = ["Zm9v", "Zm9"]
    with assert_raises(contains="row 1"):
        _ = FromBase64Kernel.dispatch(a^.to_dyn())


def test_to_hex_string_view() raises:
    var a: StringViewArray = ["é", "a value longer than twelve"]
    var out = ToHexKernel.dispatch(a^.to_dyn())
    assert_equal(out.as_string()[0].value(), "C3A9")
