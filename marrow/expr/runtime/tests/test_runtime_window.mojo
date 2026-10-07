# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window aggregates in the runtime lane."""

# Apart from `expr/tests/test_window.mojo`, and run by CI as a unit of its own:
# a runtime aggregate links the whole name x dtype ladder, so these few cases
# cost more to compile than every other expression test together.

from std.testing import assert_true

from ....builders import array
from ....dtypes import int64
from ....tabular import record_batch
from ...builders import col, table


def test_a_runtime_aggregate_takes_the_one_pass_path() raises:
    """The runtime lane resolves its kernel by name at execution, `sum` to a
    fold and `count` to `ValidCount`'s prefix count -- over two partitions,
    each holding a null."""
    var batch = record_batch(
        [
            array(["a", "a", "a", "a", "b", "b", "b"]).copy(),
            array([0, 1, 2, 3, 4, 5, 6], int64).copy(),
            array([1, None, 3, 6, 10, 20, None], int64).copy(),
        ],
        names=["k", "o", "v"],
    )
    var plan = table(batch^).with_columns(
        ["s", "n"],
        [
            col("v").sum().over(partition_by=[col("k")], order_by=[col("o")]),
            col("v")
            .count()
            .over(partition_by=[col("k")], order_by=[col("o")], rows=(-1, 0)),
        ],
    )
    var out = plan.execute()
    assert_true(
        out.column("s").as_int64() == array([1, 1, 4, 10, 10, 30, 30], int64)
    )
    assert_true(
        out.column("n").as_int64() == array([1, 1, 1, 2, 1, 2, 1], int64)
    )


def test_a_runtime_windowed_filter_must_be_boolean() raises:
    """A window aggregate lowers through `to_evaluator`, not `to_operator`,
    so it must reject a non-boolean `FILTER` there too, or narrowing the
    predicate to a `BoolArray` aborts the process."""
    var b = record_batch([array([1, 2, 3], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["s"],
        [col("v").sum().filter(col("v")).over(order_by=[col("v")])],
    )
    var raised = False
    try:
        _ = plan.execute()
    except:
        raised = True
    assert_true(raised)
