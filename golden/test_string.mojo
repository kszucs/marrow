"""Golden cases — string kernels, the AOT lane.

No regex: there is no regex engine in the tree (backlog M2.6).
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import string
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import (
    EndsWith,
    ILike,
    Like,
    Lower,
    StartsWith,
    StringLength,
    Strip,
    Upper,
)


def _words() raises -> DynRelation:
    return in_memory_table(_fixture("words"))


def test_golden_string_upper() raises:
    _check(
        "test_golden_string_upper",
        _words().project(["u"], [Upper(col("s", string))]),
    )


def test_golden_string_lower() raises:
    _check(
        "test_golden_string_lower",
        _words().project(["l"], [Lower(col("s", string))]),
    )


def test_golden_string_strip() raises:
    _check(
        "test_golden_string_strip",
        _words().project(["p"], [Strip(col("s", string))]),
    )


def test_golden_string_length_counts_bytes() raises:
    _check(
        "test_golden_string_length_counts_bytes",
        _words().project(["n"], [StringLength(col("s", string))]),
    )


def test_golden_string_starts_with() raises:
    _check(
        "test_golden_string_starts_with",
        _words().project(["b"], [StartsWith(col("s", string), lit("H"))]),
    )


def test_golden_string_like_is_case_sensitive() raises:
    _check(
        "test_golden_string_like_is_case_sensitive",
        _words().project(["b"], [Like(col("s", string), lit("h%"))]),
    )


def test_golden_string_ilike_is_not() raises:
    _check(
        "test_golden_string_ilike_is_not",
        _words().project(["b"], [ILike(col("s", string), lit("h%"))]),
    )


def test_golden_string_filter_on_like() raises:
    _check(
        "test_golden_string_filter_on_like",
        _words().filter(Like(col("s", string), lit("%o%"))),
    )
