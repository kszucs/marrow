"""Unit tests for the string compute kernels (marrow.kernels.string)."""

from std.testing import assert_equal, assert_true, assert_false

from marrow.testing import TestSuite
from marrow.arrays import AnyArray, StringArray, BoolArray
from marrow.builders import array, StringBuilder
from marrow.dtypes import string, int32

from marrow.kernels.string import (
    LengthKernel,
    UpperKernel,
    LowerKernel,
    ReverseKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    CapitalizeKernel,
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    LikeKernel,
    ILikeKernel,
)


def _broadcast(pat: String, n: Int) raises -> StringArray:
    """A length-``n`` StringArray with every element equal to ``pat``."""
    var b = StringBuilder(capacity=n)
    for _ in range(n):
        b.append(pat)
    return b.finish()


def _like(s: StringArray, pat: String) raises -> BoolArray:
    return LikeKernel.apply(s, _broadcast(pat, len(s)))


def _ilike(s: StringArray, pat: String) raises -> BoolArray:
    return ILikeKernel.apply(s, _broadcast(pat, len(s)))


def _with_null() raises -> StringArray:
    """['ab', null, 'cde']."""
    var b = StringBuilder(capacity=3)
    b.append("ab")
    b.append_null()
    b.append("cde")
    return b.finish()


# --- length ----------------------------------------------------------------


def test_length_basic() raises:
    var a = array(["a", "bb", "ccc", ""])
    assert_true(LengthKernel.apply(a) == array([1, 2, 3, 0], int32))


def test_length_sliced() raises:
    var full = array(["aa", "b", "ccc", "dddd", "e"])
    var a = full.slice(1, 3)
    assert_true(LengthKernel.apply(a) == array([1, 3, 4], int32))


def test_length_propagates_nulls() raises:
    # matches pc.utf8_length: null input -> null length (not 0)
    var r = LengthKernel.apply(_with_null())  # ['ab', null, 'cde']
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_equal(r[0].value(), 2)
    assert_equal(r[2].value(), 3)


def test_length_sliced_with_nulls() raises:
    # slicing past the null keeps the validity offset correct
    var full = _with_null()  # ['ab', null, 'cde']
    var a = full.slice(1, 2)  # [null, 'cde']
    var r = LengthKernel.apply(a)
    assert_equal(r.null_count(), 1)
    assert_false(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_equal(r[1].value(), 3)


# --- unary string -> string ------------------------------------------------


def test_upper() raises:
    var a = array(["Hello", "wORLd", ""])
    assert_true(UpperKernel.apply(a) == array(["HELLO", "WORLD", ""]))


def test_lower() raises:
    var a = array(["Hello", "wORLd"])
    assert_true(LowerKernel.apply(a) == array(["hello", "world"]))


def test_reverse() raises:
    var a = array(["abc", "", "xy"])
    assert_true(ReverseKernel.apply(a) == array(["cba", "", "yx"]))


def test_strip_family() raises:
    var a = array(["  x ", "\ty\n"])
    assert_true(StripKernel.apply(a) == array(["x", "y"]))
    var b = array(["  x ", " y "])
    assert_true(LStripKernel.apply(b) == array(["x ", "y "]))
    var c = array(["  x ", " y "])
    assert_true(RStripKernel.apply(c) == array(["  x", " y"]))


def test_capitalize() raises:
    var a = array(["hello", "WORLD", "aBc", ""])
    assert_true(
        CapitalizeKernel.apply(a) == array(["Hello", "World", "Abc", ""])
    )


def test_unary_preserves_nulls() raises:
    # null in -> null out, valid values transformed
    var r = UpperKernel.apply(_with_null())
    var expected = StringBuilder(capacity=3)
    expected.append("AB")
    expected.append_null()
    expected.append("CDE")
    assert_true(r == expected.finish())


# --- binary predicates -----------------------------------------------------


def test_startswith() raises:
    var s = array(["apple", "apricot", "banana"])
    var p = array(["ap", "ap", "ap"])
    assert_true(StartsWithKernel.apply(s, p) == array([True, True, False]))


def test_endswith() raises:
    var s = array(["cat.txt", "dog.csv", "note.txt"])
    var p = array([".txt", ".txt", ".txt"])
    assert_true(EndsWithKernel.apply(s, p) == array([True, False, True]))


def test_contains() raises:
    var s = array(["hello", "world", "help"])
    var p = array(["ell", "ell", "ell"])
    assert_true(ContainsKernel.apply(s, p) == array([True, False, False]))


def test_predicate_propagates_nulls() raises:
    # left null at index 1 -> result null (bit 0), not just false
    var left = _with_null()  # ['ab', null, 'cde']
    var right = array(["a", "x", "zz"])
    var r = StartsWithKernel.apply(left, right)
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))


# --- LIKE / ILIKE ----------------------------------------------------------


def test_like_contains() raises:
    # ClickBench-style `URL LIKE '%google%'`
    var s = array(
        ["google.com", "www.google.com/x", "GOOGLE", "abc", "", "a%b"]
    )
    assert_true(
        _like(s, "%google%") == array([True, True, False, False, False, False])
    )


def test_like_single_char() raises:
    var s = array(["a_b", "axb", "ab", "axxb"])
    # '_' matches exactly one character; the literal underscore in 'a_b' is one
    assert_true(_like(s, "a_b") == array([True, True, False, False]))


def test_like_any_run() raises:
    var s = array(["ab", "axb", "axxb", "ba"])
    assert_true(_like(s, "a%b") == array([True, True, True, False]))


def test_like_empty_pattern() raises:
    var s = array(["", "a"])
    assert_true(_like(s, "") == array([True, False]))


def test_like_percent_only() raises:
    var s = array(["", "a", "abc"])
    assert_true(_like(s, "%") == array([True, True, True]))


def test_like_anchored() raises:
    var s = array(["apple", "pineapple", "app"])
    assert_true(_like(s, "app%") == array([True, False, True]))
    assert_true(_like(s, "%ple") == array([True, True, False]))


def test_like_escape() raises:
    # '\%' matches a literal percent; '\_' a literal underscore
    var s = array(["a%b", "aXb", "a_b", "axb"])
    assert_true(_like(s, "a\\%b") == array([True, False, False, False]))
    assert_true(_like(s, "a\\_b") == array([False, False, True, False]))


def test_like_escape_backslash() raises:
    # '\\' matches a single literal backslash
    var s = array(["a\\b", "aXb", "ab"])
    assert_true(_like(s, "a\\\\b") == array([True, False, False]))


def test_ilike_case_insensitive() raises:
    var s = array(["google.com", "GOOGLE.COM", "Google", "yahoo"])
    assert_true(_ilike(s, "%google%") == array([True, True, True, False]))


def test_ilike_single_and_run() raises:
    var s = array(["ABC", "aXc", "AbbC"])
    assert_true(_ilike(s, "a_c") == array([True, True, False]))
    assert_true(_ilike(s, "a%c") == array([True, True, True]))


def test_like_propagates_nulls() raises:
    var s = _with_null()  # ['ab', null, 'cde']
    var r = LikeKernel.apply(s, _broadcast("%b%", 3))
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_true(r[0].value())  # 'ab' contains 'b'
    assert_false(r[2].value())  # 'cde' does not


def main() raises:
    TestSuite.run[__functions_in_module()]()
