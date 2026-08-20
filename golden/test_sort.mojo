"""Golden cases — sort depth, the AOT lane."""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import int64, string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation, in_memory_table


def _basic() raises -> DynRelation:
    return in_memory_table(_fixture("basic"))


def test_golden_sort_string_desc_nulls_last() raises:
    _check(
        "test_golden_sort_string_desc_nulls_last",
        _basic().sort([col("k", string)], [False], nulls_first=False),
    )


def test_golden_sort_mixed_directions() raises:
    _check(
        "test_golden_sort_mixed_directions",
        _basic().sort(
            [col("k", string), col("v", int64)],
            [True, False],
        ),
    )


def test_golden_sort_top_k() raises:
    _check(
        "test_golden_sort_top_k",
        _basic().sort([col("v", int64)], [False], nulls_first=False).limit(2),
    )


def test_golden_sort_all_null_key() raises:
    _check(
        "test_golden_sort_all_null_key",
        in_memory_table(_fixture("nulls")).sort(
            [col("a", int64), col("b", int64)], [True, True]
        ),
    )
