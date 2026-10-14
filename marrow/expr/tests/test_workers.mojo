# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parallel execution against serial, plan by plan.

Every plan here runs once through `Driver.serial()` and then through
`Driver.dynamic()` at several worker counts and morsel sizes — down to one row
per morsel, so every morsel boundary a plan could mishandle is crossed. Where a
plan defines an order — a filter's, a limit's, a sort's — the rows must come
back in it; where it defines none — an aggregate's, a join's — the same rows
must come back in any order.
"""

from std.os.path import join
from std.testing import assert_equal, assert_true

from ...arrays import DictionaryArray, StructArray
from ...builders import (
    Float64Builder,
    Int32Builder,
    Int64Builder,
    StringBuilder,
)
from ...dtypes import float64, int64, string
from ...engine import Driver
from ...execution import ExecContext
from ...io import FileSink
from ...formats.ipc import write_ipc_file
from ...formats.parquet.writer import Compression, FileWriter
from ...kernels.join import (
    BUILD_LEFT,
    BUILD_RIGHT,
    JOIN_ALL,
    JOIN_ANTI,
    JOIN_ANY,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    JoinBuildSide,
    JoinKind,
)
from ...tabular import RecordBatch, Table, record_batch
from ...utils.stats import STATS_ENABLED, counter
from ...utils.testing import (
    ScratchDir,
    assert_values_equal,
    canonical_rows,
)
from ..builders import col, count_star, lit, row_number, scan, scan_ipc, table
from ..logical import DynRelation
from ..optimizer import AllRules


def _data(n: Int, seed: Int) raises -> RecordBatch:
    """`n` rows: a key `k` of five values and some nulls, an integer `v` and
    a float `f` of exact quarters with nulls of their own, and a string
    `s`."""
    var k = Int64Builder(n)
    var v = Int64Builder(n)
    var f = Float64Builder(n)
    var s = StringBuilder()
    for i in range(n):
        if (i + seed) % 11 == 0:
            k.append_null()
        else:
            k.append(Int64((i * 7 + seed) % 5))
        if (i + seed) % 7 == 0:
            v.append_null()
            f.append_null()
        else:
            var x = (i * 13 + seed) % 17 - 8
            v.append(Int64(x))
            f.append(Float64(x) / 4)
        s.append(String("s", (i * 3 + seed) % 4))
    return record_batch(
        [k.finish(), v.finish(), f.finish(), s.finish()],
        names=["k", "v", "f", "s"],
    )


def _run(plan: DynRelation, var driver: Driver) raises -> StructArray:
    return driver.collect(plan.lower(driver), plan.schema())


def _assert_same(got: StructArray, want: StructArray, at: String) raises:
    """The same rows: compared by value, column by column, since a result may
    sit at an offset into its buffers where the other does not."""
    assert_true(got.dtype == want.dtype, "schema of " + at)
    assert_equal(len(got), len(want), "rows of " + at)
    for i in range(len(want.dtype.as_struct().fields)):
        assert_values_equal(got.field(i), want.field(i), at)


def _check(plan: DynRelation, ordered: Bool) raises:
    var want = _run(plan, Driver.serial(ExecContext.serial()))
    if not ordered:
        want = canonical_rows(want)
    var morsels: List[Int] = [1, 2, 3, 1000]
    var workers: List[Int] = [2, 4, 7]
    for m in morsels:
        for w in workers:
            var got = _run(
                plan, Driver.dynamic(ExecContext.parallel(w), morsel_rows=m)
            )
            if not ordered:
                got = canonical_rows(got)
            _assert_same(
                got,
                want,
                String(plan, " at ", w, " workers, ", m, "-row morsels"),
            )


def _sizes() -> List[Int]:
    return [0, 1, 37]


def test_workers_keep_the_order_of_a_filter_and_project() raises:
    for n in _sizes():
        _check(
            table(_data(n, 1))
            .filter(col("v", int64) > lit(0, int64))
            .select("k", "v", "s"),
            ordered=True,
        )


def test_workers_take_the_first_rows_of_a_limit() raises:
    for n in _sizes():
        _check(table(_data(n, 1)).limit(5, offset=3), ordered=True)
        _check(
            table(_data(n, 2))
            .filter(col("v", int64) > lit(-3, int64))
            .limit(4),
            ordered=True,
        )


def test_workers_sort_ties_in_input_order() raises:
    for n in _sizes():
        _check(
            table(_data(n, 1)).sort_by([col("k", int64)], [True]),
            ordered=True,
        )
        _check(
            table(_data(n, 1)).sort_by(
                [col("k", int64), col("v", int64)], [False, True]
            ),
            ordered=True,
        )


def test_workers_cut_a_top_n_per_morsel() raises:
    for n in _sizes():
        var plan = (
            table(_data(n, 3))
            .sort_by([col("k", int64)], [True])
            .limit(6)
            .optimize[AllRules]()
        )
        _check(plan, ordered=True)


def test_workers_combine_grouped_aggregates() raises:
    for n in _sizes():
        var plan = table(_data(n, 1)).aggregate(
            [
                col("v", int64).sum().alias("sum"),
                col("v", int64).count().alias("count"),
                col("v", int64).min().alias("min"),
                col("v", int64).max().alias("max"),
                col("f", float64).mean().alias("mean"),
                col("v", int64).count_distinct().alias("distinct"),
                col("v", int64).approx_count_distinct().alias("approx"),
                count_star().alias("rows"),
                col("v").sum().alias("runtime_sum"),
                col("s").min().alias("first_s"),
                col("s").count_distinct().alias("distinct_s"),
            ],
            [col("k", int64)],
        )
        _check(plan, ordered=False)


def _distinct_keys(n: Int) raises -> RecordBatch:
    """`n` rows whose key `k` never repeats, and a value `v`."""
    var k = Int64Builder(n)
    var v = Int64Builder(n)
    for i in range(n):
        k.append(Int64(i * 7_919))
        v.append(Int64(i % 13))
    return record_batch([k.finish(), v.finish()], names=["k", "v"])


def _partial_inserts(plan: DynRelation, morsel_rows: Int) raises -> Int:
    """How many rows the partial stage of `plan`'s aggregate encoded, on four
    workers, under `-D MARROW_STATS=true`."""
    var driver = Driver.analyze(ExecContext.parallel(4), morsel_rows)
    var report = driver.inspect(plan)
    return counter(report.phases[0].counters, "hash.inserts")


def test_workers_pass_keys_through_a_table_that_never_hits() raises:
    """A worker whose table keeps taking new keys stops encoding them and
    hands its rows on as they are, for the combine to group: the answer is
    the same, and the partial stage encodes far fewer rows than it reads."""
    var plan = table(_distinct_keys(40_000)).aggregate(
        [
            col("v", int64).sum().alias("s"),
            count_star().alias("n"),
            col("v", int64).min().alias("lo"),
            col("v", int64).max().alias("hi"),
            col("v", int64).mean().alias("mean"),
            col("v", int64).count_distinct().alias("distinct"),
            col("v").sum().alias("runtime_sum"),
        ],
        [col("k", int64)],
    )
    _check(plan, ordered=False)
    comptime if STATS_ENABLED:
        # Without `count_distinct`, whose own encoder counts too.
        var counted = table(_distinct_keys(40_000)).aggregate(
            [col("v", int64).sum().alias("s"), count_star().alias("n")],
            [col("k", int64)],
        )
        var inserts = _partial_inserts(counted, 1_000)
        assert_true(inserts < 32_000, String(inserts, " rows encoded"))


def test_workers_keep_grouping_a_filtered_trickle() raises:
    """A filter that keeps few rows hands a worker morsels of a handful of
    rows, each a new key at first; that is too little to judge a table by, so
    every row is still grouped where it is read."""
    var n = 40_000
    var g = Int64Builder(n)
    var f = Int64Builder(n)
    for i in range(n):
        g.append(Int64(i % 16))
        f.append(Int64(i % 101))
    var batch = record_batch([g.finish(), f.finish()], names=["g", "f"])
    var plan = (
        table(batch^)
        .filter(col("f", int64) == lit(0, int64))
        .aggregate([count_star().alias("n")], [col("g", int64)])
    )
    _check(plan, ordered=False)
    comptime if STATS_ENABLED:
        assert_equal(_partial_inserts(plan, 1_000), n // 101 + 1)


def test_workers_never_pass_dictionary_keys() raises:
    """Dictionary-encoded keys are grouped where they are read even when
    they never repeat: their values are not plain keys to hand on."""
    var n = 6_000
    var codes = Int32Builder(n)
    var words = StringBuilder()
    for i in range(n):
        codes.append(Int32(i))
        words.append(String("w", i))
    var key = DictionaryArray.from_arrays(codes.finish(), words.finish())
    var v = Int64Builder(n)
    for i in range(n):
        v.append(Int64(i % 7))
    var batch = record_batch([key^.to_dyn(), v.finish()], names=["k", "v"])
    # The sums alone: the harness compares plain columns.
    var plan = (
        table(batch^)
        .aggregate([col("v", int64).sum().alias("s")], [col("k")])
        .select("s")
    )
    _check(plan, ordered=False)


def test_workers_combine_aggregates_over_two_keys() raises:
    for n in _sizes():
        var plan = table(_data(n, 2)).aggregate(
            [col("v", int64).sum().alias("sum"), count_star().alias("rows")],
            [col("k", int64), col("s")],
        )
        _check(plan, ordered=False)


def test_workers_combine_a_whole_table_aggregate() raises:
    for n in _sizes():
        var plan = table(_data(n, 1)).aggregate(
            [
                col("v", int64).sum().alias("sum"),
                col("v", int64).min().alias("min"),
                col("f", float64).mean().alias("mean"),
                col("v", int64).count_distinct().alias("distinct"),
                col("v", int64).approx_count_distinct().alias("approx"),
                count_star().alias("rows"),
                col("s").max().alias("last_s"),
            ]
        )
        _check(plan, ordered=True)


def test_workers_keep_a_having_filter_after_the_combine() raises:
    var plan = (
        table(_data(37, 4))
        .aggregate([count_star().alias("rows")], [col("k", int64)])
        .filter(col("rows", int64) > lit(7, int64))
    )
    _check(plan, ordered=False)


def test_workers_distinct() raises:
    for n in _sizes():
        _check(table(_data(n, 5)).select("k", "s").distinct(), ordered=False)


def _join(
    kind: JoinKind, side: JoinBuildSide, strictness: UInt8, n: Int
) raises -> DynRelation:
    return table(_data(n, 1)).join(
        table(_data(n // 2 + 1, 2)), [0], [0], kind, side, strictness
    )


def test_workers_join_every_kind_from_either_side() raises:
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    var sides: List[JoinBuildSide] = [BUILD_LEFT, BUILD_RIGHT]
    for n in _sizes():
        for ref kind in kinds:
            for side in sides:
                _check(_join(kind, side, JOIN_ALL, n), ordered=False)


def _stage_labels(plan: DynRelation) raises -> String:
    """Every stage label of `plan`'s analyzed run, one per line."""
    var driver = Driver.analyze(ExecContext.parallel(4), 5)
    var report = driver.inspect(plan)
    var out = String()
    for ref p in report.phases:
        for ref s in p.stages:
            out += s + "\n"
    return out^


def test_workers_build_the_smaller_join_side() raises:
    """A commuting join builds whichever side has fewer rows, whatever side
    it was written to build: an aggregate's few groups rather than the table
    it is joined with, and a three-row table rather than a 37-row one."""
    var dim = table(_data(200, 2)).aggregate(
        [count_star().alias("n")], [col("k", int64)]
    )
    var plan = table(_data(200, 1)).select("k", "v").join(dim^, [0], [0])
    _check(plan, ordered=False)
    assert_true("built on the right" in _stage_labels(plan))
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref kind in kinds:
        var small = table(_data(37, 1)).join(
            table(_data(3, 2)), [0], [0], kind, BUILD_LEFT
        )
        _check(small, ordered=False)
        assert_true("built on the right" in _stage_labels(small))


def test_workers_keep_the_build_side_over_a_growing_probe() raises:
    """A probe side holding a join has no row bound, so the side written to
    be built is built."""
    var inner = (
        table(_data(40, 2))
        .select("k", "v")
        .join(table(_data(40, 3)).select("k"), [0], [0])
    )
    var plan = table(_data(200, 1)).select("k").join(inner^, [0], [0])
    _check(plan, ordered=False)
    assert_true("built on the right" not in _stage_labels(plan))


def test_workers_keep_the_build_side_of_a_join_any() raises:
    """Which row a `JOIN_ANY` keeps depends on the side built, so it is never
    swapped."""
    var plan = table(_data(37, 1)).join(
        table(_data(3, 2)), [0], [0], JOIN_INNER, BUILD_LEFT, JOIN_ANY
    )
    _check(plan, ordered=False)
    assert_true("built on the right" not in _stage_labels(plan))


def test_workers_join_any_keeps_the_serial_match() raises:
    var kinds: List[JoinKind] = [JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ANTI]
    for ref kind in kinds:
        _check(_join(kind, BUILD_LEFT, JOIN_ANY, 37), ordered=False)
        _check(_join(kind, BUILD_RIGHT, JOIN_ANY, 37), ordered=False)


def test_workers_join_a_chain_and_aggregate_it() raises:
    var plan = (
        table(_data(37, 1))
        .select("k", "v")
        .join(table(_data(19, 2)).select("k", "s"), [0], [0], JOIN_LEFT)
        .join(
            table(_data(11, 3)).select("k", "f"),
            [0],
            [0],
            JOIN_FULL,
            rname="{name}_third",
        )
    )
    _check(plan, ordered=False)
    _check(
        plan.aggregate([count_star().alias("rows")], [col("s")]),
        ordered=False,
    )


def test_workers_union_and_set_operations() raises:
    var left = table(_data(37, 1)).select("k", "s")
    var right = table(_data(23, 2)).select("k", "s")
    _check(left.union_all(right.copy()), ordered=True)
    _check(left.intersect(right.copy()), ordered=False)
    _check(left.except_(right.copy()), ordered=False)
    _check(left.except_all(right.copy()), ordered=False)


def test_workers_number_rows_over_a_window() raises:
    var plan = table(_data(37, 1)).with_columns(
        ["rn"], [row_number().over(order_by=[col("v", int64)])]
    )
    _check(plan, ordered=True)


def test_workers_scan_parquet_a_row_group_per_morsel() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.parquet")
        var data = _data(37, 1)
        var writer = FileWriter(
            FileSink(path), data.schema, Compression.SNAPPY, row_group_size=5
        )
        writer.write(Table.from_batches(data.schema, [data.copy()]))
        writer.close()
        var rows = scan(path, data.schema)
        _check(rows, ordered=True)
        _check(
            rows.filter(col("v", int64) > lit(2, int64)).optimize[AllRules](),
            ordered=True,
        )
        _check(
            rows.aggregate([count_star().alias("rows")], [col("k", int64)]),
            ordered=False,
        )


def test_workers_scan_ipc_a_batch_per_morsel() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.arrow")
        var data = _data(37, 2)
        write_ipc_file(
            path, [data.slice(0, 10), data.slice(10, 20), data.slice(30, 7)]
        )
        _check(scan_ipc(path, data.schema), ordered=True)
