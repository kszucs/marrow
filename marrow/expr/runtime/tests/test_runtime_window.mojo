# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window aggregates in the runtime lane."""

# Apart from `expr/tests/test_window.mojo`, and run by CI as a unit of its own:
# a runtime aggregate links the whole name x dtype ladder, so these few cases
# cost more to compile than every other expression test together.

from std.testing import assert_raises, assert_true

from ....arrays import RunEndEncodedArray
from ....builders import array
from ....dtypes import int32, int64
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


def test_a_runtime_rows_offset_past_every_partition_is_its_edge() raises:
    """`Int.MIN`, `Int.MAX` and `2^62` reach past any partition, so each is
    the partition's edge, in the second partition as in the first."""
    var batch = record_batch(
        [
            array(["a", "a", "b", "b", "b"]).copy(),
            array([0, 1, 2, 3, 4], int64).copy(),
            array([1, 2, 10, None, 30], int64).copy(),
        ],
        names=["k", "o", "v"],
    )
    var plan = table(batch^).with_columns(
        ["up", "all", "down", "n"],
        [
            col("v")
            .sum()
            .over(
                partition_by=[col("k")],
                order_by=[col("o")],
                rows=(Int.MIN, 0),
            ),
            col("v")
            .sum()
            .over(
                partition_by=[col("k")],
                order_by=[col("o")],
                rows=(-(1 << 62), 1 << 62),
            ),
            col("v")
            .sum()
            .over(
                partition_by=[col("k")],
                order_by=[col("o")],
                rows=(0, Int.MAX),
            ),
            col("v")
            .count()
            .over(
                partition_by=[col("k")],
                order_by=[col("o")],
                rows=(Int.MIN, 0),
            ),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("up").as_int64() == array([1, 3, 10, 10, 40], int64))
    assert_true(
        out.column("all").as_int64() == array([3, 3, 40, 40, 40], int64)
    )
    assert_true(
        out.column("down").as_int64() == array([3, 2, 40, 30, 30], int64)
    )
    assert_true(out.column("n").as_int64() == array([1, 2, 1, 1, 2], int64))


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


def test_a_runtime_window_aggregate_refuses_run_end_encoded() raises:
    """A window aggregate evaluates its operand to a column and folds it, so a
    run-end encoded column's absent bitmap would read as "no nulls"; the
    catalog refuses the dtype first."""
    var b = record_batch(
        [
            array([0, 1, 2], int64).copy(),
            RunEndEncodedArray.from_arrays(
                array([2, 3], int32), array([1, None], int64)
            ).to_dyn(),
        ],
        names=["o", "v"],
    )
    with assert_raises(contains="TypeError"):
        _ = (
            table(b^)
            .with_columns(["n"], [col("v").count().over(order_by=[col("o")])])
            .execute()
        )
