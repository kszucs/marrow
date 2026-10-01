# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The regex kernels against DuckDB's semantics.

The golden `string_regexp_*` cases ask one pattern of the `text` fixture each;
these cover what those cannot: a pattern that varies per row, a null argument,
an offset slice, `large_string`, the group and replacement edge cases, and the
one known-wrong answer the engine gives.
"""

from std.testing import assert_true

from ...builders import array, Int64Builder, LargeStringBuilder, StringBuilder
from ...kernels.regex import (
    RegexpExtractKernel,
    RegexpMatchesKernel,
    RegexpReplaceKernel,
)
from ...kernels.string import StringOperands


def _ops(
    n: Int,
    var text: String = String(),
    var alt: String = String(),
    count: Int = 0,
) raises -> StringOperands[]:
    """Constant arguments broadcast across `n` rows, as `Datum` splats a
    literal."""
    var tb = StringBuilder(capacity=n)
    var ab = StringBuilder(capacity=n)
    var cb = Int64Builder(capacity=n)
    for _ in range(n):
        tb.append(text)
        ab.append(alt)
        cb.append(Int64(count))
    var ops = StringOperands()
    ops.text = tb.finish()
    ops.alt = ab.finish()
    ops.count = cb.finish()
    return ops^


# --- regexp_matches --------------------------------------------------------


def test_regexp_matches_is_a_partial_match() raises:
    var s = array(["a,b", "xa,", "", None, "abc"])
    assert_true(
        RegexpMatchesKernel.apply_scalar(s, "a,")
        == array([True, True, False, None, False])
    )
    assert_true(
        RegexpMatchesKernel.apply_scalar(s, "^a,")
        == array([True, False, False, None, False])
    )


def test_regexp_matches_empty_pattern_matches_everything() raises:
    assert_true(
        RegexpMatchesKernel.apply_scalar(array(["", "x"]), "")
        == array([True, True])
    )


def test_regexp_matches_reads_a_pattern_per_row() raises:
    var s = array(["abc", "abc", "abc", "abc"])
    var p = array(["^a", "^a", "c$", None])
    assert_true(
        RegexpMatchesKernel.apply(s, p) == array([True, True, True, None])
    )


def test_regexp_matches_on_a_slice() raises:
    var s = array(["zz", "ab", None, "b"]).slice(1, 3)
    assert_true(
        RegexpMatchesKernel.apply_scalar(s, "b") == array([True, None, True])
    )


def test_regexp_matches_large_string() raises:
    var b = LargeStringBuilder(capacity=2)
    b.append("foo123")
    b.append("bar")
    assert_true(
        RegexpMatchesKernel.apply_scalar(b.finish(), "[0-9]+")
        == array([True, False])
    )


def test_regexp_matches_malformed_pattern_raises() raises:
    var raised = False
    try:
        _ = RegexpMatchesKernel.apply_scalar(array(["a"]), "(")
    except:
        raised = True
    assert_true(raised)


# --- regexp_extract --------------------------------------------------------


def test_regexp_extract_group() raises:
    var s = array(["a,b,c", "xyz", "", None, "12-345"])
    assert_true(
        RegexpExtractKernel.apply(s, _ops(5, text="([a-z]+)", count=1))
        == array(["a", "xyz", "", None, ""])
    )
    assert_true(
        RegexpExtractKernel.apply(s, _ops(5, text="(\\d+)-(\\d+)", count=2))
        == array(["", "", "", None, "345"])
    )


def test_regexp_extract_group_zero_is_the_whole_match() raises:
    assert_true(
        RegexpExtractKernel.apply(
            array(["x 12-345 y", "foobar"]), _ops(2, text="\\d+-\\d+|foobar")
        )
        == array(["12-345", "foobar"])
    )


def test_regexp_extract_missing_group_is_empty() raises:
    assert_true(
        RegexpExtractKernel.apply(array(["abc"]), _ops(1, text="(b)", count=5))
        == array([""])
    )


# --- regexp_replace --------------------------------------------------------


def test_regexp_replace_first_match_only() raises:
    var s = array(["a,b,c", "xyz", "", None, "héllo wörld"])
    assert_true(
        RegexpReplaceKernel.apply(s, _ops(5, text="[aeiou]", alt="_"))
        == array(["_,b,c", "xyz", "", None, "héll_ wörld"])
    )


def test_regexp_replace_group_references() raises:
    assert_true(
        RegexpReplaceKernel.apply(
            array(["x 12-345 y"]), _ops(1, text="(\\d+)-(\\d+)", alt="\\2-\\1")
        )
        == array(["x 345-12 y"])
    )
    assert_true(
        RegexpReplaceKernel.apply(
            array(["abc"]), _ops(1, text="b", alt="[\\0|\\\\]")
        )
        == array(["a[b|\\]c"])
    )


def test_regexp_replace_empty_match_inserts() raises:
    assert_true(
        RegexpReplaceKernel.apply(array(["abc"]), _ops(1, text="x*", alt="_"))
        == array(["_abc"])
    )


def test_regexp_replace_null_argument_is_null() raises:
    var repl = StringBuilder(capacity=2)
    repl.append("_")
    repl.append_null()
    var ops = _ops(2, text="a")
    ops.alt = repl.finish()
    assert_true(
        RegexpReplaceKernel.apply(array(["a", "a"]), ops) == array(["_", None])
    )


def test_regexp_replace_optional_group_known_bug() raises:
    """Pins the upstream mojo-regex bug: the capture path never enters an
    optional group, so the match it reports starts after `foo`.

    DuckDB, RE2 and CPython answer `[bar]`. When a fixed engine lands this
    fails, and the expectation flips to the right answer.
    """
    assert_true(
        RegexpReplaceKernel.apply(
            array(["foobar"]), _ops(1, text="(?:foo)?(bar)", alt="[\\1]")
        )
        == array(["foo[bar]"])
    )
