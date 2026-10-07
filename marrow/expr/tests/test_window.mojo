# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window functions — the shapes the golden corpus cannot reach.

`golden/cases/window_*.mojo` checks the seven semantics differentially against
DuckDB, on one seven-row fixture. What it cannot vary is the *shape* of the
input, and every question below is a shape question: a partition of one row, a
partition where every row ties, an ordering whose key is null more than once,
and an input that carries no rows at all.

The recurring failure these guard is a **boundary read off the end of a
partition**. Every window function is a function of two boundaries, so a
one-row partition and an all-ties partition are the two degenerate cases where
`partition_start`, `peer_start` and `peer_end` collapse onto each other — and
where an off-by-one produces a plausible number rather than a crash.
"""

from std.math import nan
from std.python import Python
from std.testing import assert_almost_equal, assert_true
from std.os.path import join

from ...utils.testing import ScratchDir
from ...builders import array, nulls
from ...dtypes import float64, int64, string
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue, Over, WindowFunction
from ..optimizer import AllRules
from ..builders import (
    col,
    count_star,
    cume_dist,
    dense_rank,
    lit,
    ntile,
    percent_rank,
    rank,
    row_number,
    scan,
    table,
)


# ---------------------------------------------------------------------------
# Row order
# ---------------------------------------------------------------------------
def test_window_leaves_its_input_in_input_order() raises:
    """The sort a window runs is internal — it must not reorder the batch.

    `with_columns` means `SELECT *, f() OVER ()`, so the rows come back as they
    went in and the computed column is scattered to sit beside the row it
    describes. Every golden case sorts afterwards and so could not tell the
    difference; this is the only place the claim is checked.
    """
    var b = record_batch([array([30, 10, 20], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rn"], [row_number().over(order_by=[col("a", int64)])]
    )
    var out = plan.execute()
    assert_true(out.column("a").as_int64() == array([30, 10, 20], int64))
    assert_true(out.column("rn").as_int64() == array([3, 1, 2], int64))


# ---------------------------------------------------------------------------
# Degenerate partitions
# ---------------------------------------------------------------------------
def test_window_over_a_single_row() raises:
    """One row is its own partition, its own peer group, and both frame edges.

    So every ranking function answers 1, both offsets fall off the edge and
    answer null, and both frame edges name the row itself.
    """
    var b = record_batch([array([7], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rn", "rk", "dr", "lg", "ld", "fv", "lv"],
        [
            row_number().over(order_by=[col("a", int64)]),
            rank().over(order_by=[col("a", int64)]),
            dense_rank().over(order_by=[col("a", int64)]),
            col("a", int64).lag().over(order_by=[col("a", int64)]),
            col("a", int64).lead().over(order_by=[col("a", int64)]),
            col("a", int64).first_value().over(order_by=[col("a", int64)]),
            col("a", int64).last_value().over(order_by=[col("a", int64)]),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rn").as_int64() == array([1], int64))
    assert_true(out.column("rk").as_int64() == array([1], int64))
    assert_true(out.column("dr").as_int64() == array([1], int64))
    assert_true(out.column("lg").as_int64() == nulls(1, int64))
    assert_true(out.column("ld").as_int64() == nulls(1, int64))
    assert_true(out.column("fv").as_int64() == array([7], int64))
    assert_true(out.column("lv").as_int64() == array([7], int64))


def test_window_over_no_rows_at_all() raises:
    """An empty input produces an empty column, not a missing one.

    The window column has to exist in the output even when nothing was
    computed for it, or the batch disagrees with the schema `Window` declared —
    which everything above reads by index, so it corrupts rather than raises.
    """
    var b = record_batch([array([5], int64).copy()], names=["a"])
    var plan = (
        table(b^)
        .filter(col("a", int64) > col("a", int64))
        .with_columns(["rn"], [row_number().over(order_by=[col("a", int64)])])
    )
    var out = plan.execute()
    assert_true(out.num_rows() == 0)
    assert_true(out.schema.get_field_index("rn") == 1)


# ---------------------------------------------------------------------------
# Ties — the whole difference between the three ranking functions
# ---------------------------------------------------------------------------
def test_all_rows_tied_collapses_rank_but_not_row_number() raises:
    """The degenerate tie: one peer group spanning the whole partition.

    `rank` and `dense_rank` both answer 1 everywhere because there is one peer
    group; `row_number` still counts, because it is the one function ties do
    not reach. A `rank` that read the row's own position instead of its peer
    group's would answer 1,2,3 here and look perfectly reasonable.
    """
    var b = record_batch([array([4, 4, 4], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rn", "rk", "dr"],
        [
            row_number().over(order_by=[col("a", int64)]),
            rank().over(order_by=[col("a", int64)]),
            dense_rank().over(order_by=[col("a", int64)]),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rn").as_int64() == array([1, 2, 3], int64))
    assert_true(out.column("rk").as_int64() == array([1, 1, 1], int64))
    assert_true(out.column("dr").as_int64() == array([1, 1, 1], int64))


def test_rank_leaves_the_gap_dense_rank_closes() raises:
    """The two differ only after a tie, so a run of three is where they part.

    `rank` names the peer group by its first position and therefore skips to 5;
    `dense_rank` names it by its ordinal and goes to 3. Implementing either as
    the other is the single most likely way to get this wrong.
    """
    var b = record_batch([array([1, 2, 2, 2, 3], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rk", "dr"],
        [
            rank().over(order_by=[col("a", int64)]),
            dense_rank().over(order_by=[col("a", int64)]),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rk").as_int64() == array([1, 2, 2, 2, 5], int64))
    assert_true(out.column("dr").as_int64() == array([1, 2, 2, 2, 3], int64))


# ---------------------------------------------------------------------------
# Nulls in the ordering key
# ---------------------------------------------------------------------------
def test_two_nulls_in_the_order_key_are_peers() raises:
    """`ORDER BY` compares with `IS NOT DISTINCT FROM`, so nulls tie.

    This is the case that separates a correct boundary test from one that
    reads `equal`'s output directly: `equal(null, null)` is *null*, and taking
    that for "not equal" gives each null its own peer group — `rank` would
    answer 1,2,3 here instead of 1,1,3.
    """
    var b = record_batch([array([None, None, 5], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rk", "dr"],
        [
            rank().over(order_by=[col("a", int64)]),
            dense_rank().over(order_by=[col("a", int64)]),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rk").as_int64() == array([1, 1, 3], int64))
    assert_true(out.column("dr").as_int64() == array([1, 1, 2], int64))


def test_two_nans_in_the_order_key_are_peers() raises:
    """The NaN half of the same rule the case above states for null.

    `ORDER BY` compares with `IS NOT DISTINCT FROM`, so the two NaNs are one
    peer group and both rank 3. marrow answered 3, 4 when the sort placed them
    adjacent and `equal` then said they differ, because it is IEEE and the sort
    is not. `WindowExtents.of_sorted` asks `KeyCompare`, under which NaN is
    NaN, for that reason.

    The nulls go last here, where Arrow's placement — NaN beside the nulls —
    and DuckDB's — NaN above every number — agree, so DuckDB 1.5.5's answer
    (measured 2026-09-22) still holds. With the nulls first the two differ:
    marrow sorts as Arrow does and ranks the NaNs 1.

    `dense_rank` is asserted alongside because it is a separate kernel, not
    because it is more sensitive — in this shape the two answer identically,
    and a split peer group would move both.

    A mixed-sign pair is asserted alongside, because it takes *both* halves to
    work: a NaN-safe comparison alone left `-nan` and `+nan` at opposite ends of
    the partition, where `WindowExtents.of_sorted` — which compares adjacent
    rows — never handed them to it. The sort setting every NaN aside together
    is what brings them together.
    """
    var q = nan[DType.float64]()
    var b = record_batch([array([q, 1.0, q, 2.0], float64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["rk", "dr"],
        [
            rank().over(order_by=[col("a", float64)], nulls_first=False),
            dense_rank().over(order_by=[col("a", float64)], nulls_first=False),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rk").as_int64() == array([3, 1, 3, 2], int64))
    assert_true(out.column("dr").as_int64() == array([3, 1, 3, 2], int64))

    # the same shape with the NaNs' signs opposed answers identically
    var mixed = record_batch(
        [array([q, 1.0, -q, 2.0], float64).copy()], names=["a"]
    )
    var out2 = (
        table(mixed^)
        .with_columns(
            ["rk"],
            [rank().over(order_by=[col("a", float64)], nulls_first=False)],
        )
        .execute()
    )
    assert_true(out2.column("rk").as_int64() == array([3, 1, 3, 2], int64))


def test_lag_cannot_tell_a_missing_row_from_a_null_one() raises:
    """`lag` reads the neighbouring *row*, whatever that row holds.

    Two nulls sort first, so the second row's predecessor is a null value — a
    null meaning "the value there was null" — while the first row's is a
    missing row. Both reach the output as null and nothing distinguishes them,
    which is what SQL specifies rather than a limitation here.
    """
    var b = record_batch([array([None, None, 5], int64).copy()], names=["a"])
    var plan = table(b^).with_columns(
        ["lg"], [col("a", int64).lag().over(order_by=[col("a", int64)])]
    )
    var out = plan.execute()
    assert_true(out.column("lg").as_int64() == nulls(3, int64))


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------
def test_row_number_restarts_at_every_partition() raises:
    """A partition boundary resets the count — including for singletons.

    `c` is a partition of one, so its `row_number` is 1 rather than a
    continuation of `b`'s. A boundary scan that carried the running start
    across the change would number it 4 here.
    """
    var b = record_batch(
        [
            array(["a", "a", "b", "c"]).copy(),
            array([2, 1, 5, 9], int64).copy(),
        ],
        names=["k", "v"],
    )
    var plan = table(b^).with_columns(
        ["rn"],
        [
            row_number().over(
                partition_by=[col("k", string)], order_by=[col("v", int64)]
            )
        ],
    )
    var out = plan.execute()
    # Input order is a(2), a(1), b(5), c(9); within `a` the row holding 1 comes
    # first, so the row holding 2 is numbered 2 — in its input position.
    assert_true(out.column("rn").as_int64() == array([2, 1, 1, 1], int64))


def test_a_null_partition_key_is_one_partition() raises:
    """Nulls group together under `PARTITION BY`, exactly as under `GROUP BY`.

    Two null keys form one partition of two rows, not two partitions of one,
    so `row_number` reaches 2. This is the partition-side twin of the peer test
    above and fails the same way if null-versus-null reads as distinct.
    """
    var b = record_batch(
        [
            array([Optional[String](None), None, "z"]).copy(),
            array([1, 2, 3], int64).copy(),
        ],
        names=["k", "v"],
    )
    var plan = table(b^).with_columns(
        ["rn"],
        [
            row_number().over(
                partition_by=[col("k", string)], order_by=[col("v", int64)]
            )
        ],
    )
    var out = plan.execute()
    assert_true(out.column("rn").as_int64() == array([1, 2, 1], int64))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def test_a_rows_frame_clamps_at_the_partition_edge() raises:
    """`ROWS 1 PRECEDING` cannot reach into the previous partition.

    The first row of each partition has no predecessor *within it*, so its
    frame is one row. Without the clamp the second partition's first row would
    sum in the last row of the first, which is a wrong answer that looks like a
    plausible running total.
    """
    var b = record_batch(
        [
            array(["a", "a", "b", "b"]).copy(),
            array([1, 2, 10, 20], int64).copy(),
        ],
        names=["k", "v"],
    )
    var plan = table(b^).with_columns(
        ["s"],
        [
            col("v", int64)
            .sum()
            .over(
                partition_by=[col("k", string)],
                order_by=[col("v", int64)],
                rows=(-1, 0),
            )
        ],
    )
    var out = plan.execute()
    assert_true(out.column("s").as_int64() == array([1, 3, 10, 30], int64))


def test_the_default_frame_runs_to_the_peer_group_not_the_row() raises:
    """`RANGE` counts peers, so tied rows share a frame and share an answer.

    Two rows tied at 2 both see `{1, 2, 2}` and both answer 5. A `ROWS` reading
    of the same default would answer 3 and 5 — the divergence
    `window_explicit_rows_frame` exists to pin down, checked here on the tie
    that makes the two differ.
    """
    var b = record_batch([array([1, 2, 2], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["s"], [col("v", int64).sum().over(order_by=[col("v", int64)])]
    )
    var out = plan.execute()
    assert_true(out.column("s").as_int64() == array([1, 5, 5], int64))


def test_a_frame_that_spans_its_partition_repeats_for_every_row() raises:
    """Every row's frame is the whole partition, so every row shares one
    answer — and the partition boundary is what has to interrupt it.

    Two partitions of equal size, so a reuse that compared only the frame's
    *length*, or never noticed the bounds had moved, would carry the first
    partition's answer into the second. `sum` meets that edge in
    `Windowable.over`, where the second partition restarts the running fold;
    `variance` has no `over` and meets it in `_per_frame`'s reuse of equal
    frames.
    """
    var b = record_batch(
        [
            array(["a", "a", "b", "b"]).copy(),
            array([1, 2, 10, 20], int64).copy(),
        ],
        names=["k", "v"],
    )
    var plan = table(b^).with_columns(
        ["s", "var"],
        [
            col("v", int64)
            .sum()
            .over(
                partition_by=[col("k", string)],
                order_by=[col("v", int64)],
                rows=(-1000, 1000),
            ),
            col("v", int64)
            .variance()
            .over(
                partition_by=[col("k", string)],
                order_by=[col("v", int64)],
                rows=(-1000, 1000),
            ),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("s").as_int64() == array([3, 3, 30, 30], int64))
    assert_true(
        out.column("var").as_float64()
        == array([0.25, 0.25, 25.0, 25.0], float64)
    )


def test_a_windowed_count_star_reads_no_column() raises:
    """`COUNT(*)` names no column, so the batch its frames slice is narrowed
    to no columns at all — and must still carry each frame's row count."""
    var b = record_batch(
        [
            array(["a", "a", "b", "c"]).copy(),
            array([2, 1, 5, 9], int64).copy(),
        ],
        names=["k", "v"],
    )
    var plan = table(b^).with_columns(
        ["n"],
        [
            count_star().over(
                partition_by=[col("k", string)],
                order_by=[col("v", int64)],
                rows=(-1000, 1000),
            )
        ],
    )
    assert_true(
        plan.execute().column("n").as_int64() == array([2, 2, 1, 1], int64)
    )


def test_a_sum_over_an_all_null_frame_is_null() raises:
    """The window aggregate inherits the kernel's null rule rather than
    restating it.

    `SUM` skips nulls and answers null when it saw none, so the row whose frame
    is `{null}` is null while the row whose frame is `{null, 4}` is 4. Getting
    this from `SumFold` rather than from an accumulator written here is the
    reason the frame is folded with the aggregate's own `FoldKernel`.
    """
    var b = record_batch([array([None, 4], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["s"], [col("v", int64).sum().over(order_by=[col("v", int64)])]
    )
    var out = plan.execute()
    assert_true(out.column("s").as_int64() == array([None, 4], int64))


# ---------------------------------------------------------------------------
# What the surface refuses
# ---------------------------------------------------------------------------
def test_a_window_value_is_refused_where_a_value_per_row_is_needed() raises:
    """A window value has no answer until its partition is read, so every
    position evaluated once per row refuses it at plan time — a filter
    (`QUALIFY` is a filter over a column `with_columns` added), a projection,
    a sort key, a grouping key, an aggregate, and another window's key.

    `col("v", int64).over(...)` is not here: a per-row value has no `.over`,
    so that mistake does not compile.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["v"])
    var rn: DynValue = row_number().over(order_by=[col("v", int64)])
    var attempts = 0
    var refused = 0
    for position in range(6):
        attempts += 1
        try:
            if position == 0:
                _ = table(b.copy()).filter(rn.copy())
            elif position == 1:
                _ = table(b.copy()).project(["x"], [rn.copy()])
            elif position == 2:
                _ = table(b.copy()).sort_by([rn.copy()], [True])
            elif position == 3:
                _ = table(b.copy()).aggregate(
                    [col("v", int64).sum()], [rn.copy()]
                )
            elif position == 4:
                _ = table(b.copy()).aggregate([rn.copy()])
            else:
                _ = rank().over(order_by=[rn.copy()])
        except e:
            assert_true("window function" in String(e), String(e))
            refused += 1
    assert_true(refused == attempts)


def test_a_window_column_cannot_shadow_an_existing_one() raises:
    """`Window` appends, so a repeated name would duplicate rather than replace.

    A per-row value replaces in place because a `Project` names every output
    column anyway; a window value raises instead, since a duplicated name
    makes every later read by name ambiguous.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["v"])
    var raised = False
    try:
        _ = table(b^).with_columns(
            ["v"], [row_number().over(order_by=[col("v", int64)])]
        )
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# The pushdown boundary
# ---------------------------------------------------------------------------
def test_a_filter_above_a_window_does_not_prune_the_window_s_input() raises:
    """A predicate above a window must not reach the scan beneath it.

    `PushFilterIntoScan` moves a filter's predicate onto a `ParquetScan` so it
    can skip row groups, descending through `Filter` and nothing else. A window function reads its whole partition, so a predicate
    that got past one would have `row_number()` count a smaller population,
    silently.

    **Run through `optimize[AllRules]()`, not `execute()`.** Pruning is a
    rewrite now, so an unoptimized plan skips nothing and this case would pass
    without proving anything at all. Optimizing is what puts the rule in a
    position to get it wrong.

    The file holds `a` in `[0, 100)` across four disjoint row groups, so
    `a > 74` can prove three of the four away. With the pushdown stopped, the
    surviving rows keep the row numbers they had in the full ordering --
    76..100. If it descends, the window sees only the last group and numbers
    it 1..25 instead: a plausible answer, and the reason a wrong-population
    bug like this is invisible without an assertion on the *values*.
    """
    with ScratchDir() as dir:
        var path = join(dir, "marrow_window_pushdown.parquet")
        var pa = Python.import_module("pyarrow")
        var pq = Python.import_module("pyarrow.parquet")
        var a = Python.list()
        for i in range(100):
            a.append(i)
        pq.write_table(
            pa.table(Python.dict(a=pa.array(a))),
            path,
            row_group_size=25,
            compression="none",
        )

        # The file is written by pyarrow because marrow's writer does not expose
        # `row_group_size`, and disjoint row groups are the whole point here. Its
        # schema comes from a matching batch rather than being spelled out.
        var proto = record_batch([array([0], int64).copy()], names=["a"])
        var plan = (
            scan(path, proto.schema.copy())
            .with_columns(
                ["rn"], [row_number().over(order_by=[col("a", int64)])]
            )
            .filter(col("a", int64) > lit(74, int64))
        )
        var optimized = plan.optimize[AllRules]()
        assert_true(
            "pruned by" not in String(optimized),
            "a pruner reached the scan through a window: " + String(optimized),
        )
        var out = optimized.execute()

        assert_true(out.num_rows() == 25, "expected the 25 rows above 74")
        ref rn = out.column("rn").as_int64()
        assert_true(
            Int(rn[0].value()) == 76,
            "row_number restarted -- a pruner reached the scan: got "
            + String(rn[0].value()),
        )
        assert_true(Int(rn[24].value()) == 100, String(rn[24].value()))


def test_an_empty_frame_takes_the_aggregate_s_identity_not_null() raises:
    """`COUNT` over no rows is 0; `MIN` over no rows is NULL.

    A `ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING` frame is empty at the
    partition's first row, and which value that row gets is the *aggregate's*
    decision, not the frame loop's. `BufferedAggregateOperator.drain` already
    answers both correctly for an input that produced no morsel; this pins
    that the window path reaches it rather than short-circuiting to null,
    which is what it used to do -- reporting NULL where DuckDB reports 0 for
    every partition's first row.
    """
    var b = record_batch([array([1, 2, 3], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["c", "m"],
        [
            col("v", int64)
            .count()
            .over(order_by=[col("v", int64)], rows=(-3, -1)),
            col("v", int64)
            .min()
            .over(order_by=[col("v", int64)], rows=(-3, -1)),
        ],
    )
    var out = plan.execute()

    # Compared with `__eq__`, not element by element: a null `PrimitiveScalar`
    # stores `NativeScalar(0)` (`scalars.mojo`), so `c[0].value()` is `0`
    # whether the count is a real zero or a null -- an element assertion here
    # passes under the very bug it is meant to catch. `__eq__` compares
    # `null_count` and the validity bitmap, so it tells 0 from NULL.
    ref c = out.column("c").as_int64()
    assert_true(c == array([0, 1, 2], int64), "count of an empty frame is 0")
    assert_true(c.is_valid(0), "the count must be valid, not a null reading 0")

    var expected_min: List[Optional[Int]] = [None, 1, 1]
    ref m = out.column("m").as_int64()
    assert_true(
        m == array(expected_min, int64), "min of an empty frame is null"
    )


def test_a_frame_wholly_past_its_partition_is_empty() raises:
    """`ROWS BETWEEN 5 FOLLOWING AND 10 FOLLOWING` on a three-row input starts
    past the last row for every row, so every frame is empty — the aggregate
    and the edge gather both read it, and neither may slice or index past the
    batch."""
    var b = record_batch([array([1, 2, 3], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["c", "l"],
        [
            col("v", int64)
            .count()
            .over(order_by=[col("v", int64)], rows=(5, 10)),
            col("v", int64)
            .last_value()
            .over(order_by=[col("v", int64)], rows=(5, 10)),
        ],
    )
    var out = plan.execute()
    assert_true(out.column("c").as_int64() == array([0, 0, 0], int64))
    var expected: List[Optional[Int]] = [None, None, None]
    assert_true(out.column("l").as_int64() == array(expected, int64))


# ---------------------------------------------------------------------------
# The distribution functions
# ---------------------------------------------------------------------------
def test_percent_rank_and_cume_dist_over_a_tie() raises:
    """Both are built on peer groups, and the tie is where they differ.

    `v = [10, 20, 20, 40]`. `rank` is 1, 2, 2, 4 so `percent_rank` is
    `(rank-1)/3` = 0, 1/3, 1/3, 1. `cume_dist` counts through the peer group
    over 4 rows: 1/4, 3/4, 3/4, 1 — the tied pair reaches 3/4 because both
    rows are at or before the group's end, which is what makes it *not*
    `rank/n`.
    """
    var b = record_batch([array([10, 20, 20, 40], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["p", "c"],
        [
            percent_rank().over(order_by=[col("v", int64)]),
            cume_dist().over(order_by=[col("v", int64)]),
        ],
    )
    var out = plan.execute()
    ref p = out.column("p").as_float64()
    assert_almost_equal(Float64(p[0].value()), 0.0)
    assert_almost_equal(Float64(p[1].value()), 1.0 / 3.0)
    assert_almost_equal(Float64(p[2].value()), 1.0 / 3.0)
    assert_almost_equal(Float64(p[3].value()), 1.0)

    ref c = out.column("c").as_float64()
    assert_almost_equal(Float64(c[0].value()), 0.25)
    assert_almost_equal(Float64(c[1].value()), 0.75)
    assert_almost_equal(Float64(c[2].value()), 0.75)
    assert_almost_equal(Float64(c[3].value()), 1.0)


def test_percent_rank_of_a_one_row_partition_is_zero() raises:
    """The degenerate case: `(rank-1)/(rows-1)` divides by zero unless the
    definition names it, and SQL names it 0."""
    var b = record_batch([array([7], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["p"], [percent_rank().over(order_by=[col("v", int64)])]
    )
    ref p = plan.execute().column("p").as_float64()
    assert_almost_equal(Float64(p[0].value()), 0.0)


def test_ntile_gives_the_remainder_to_the_earliest_buckets() raises:
    """5 rows in 2 buckets is 3 then 2, not 2 then 3.

    Every SQL engine front-loads the remainder, and getting it backwards is a
    plausible-looking answer rather than a failure — so the sizes are asserted,
    not just the range.
    """
    var b = record_batch([array([1, 2, 3, 4, 5], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["t"], [ntile(2).over(order_by=[col("v", int64)])]
    )
    ref t = plan.execute().column("t").as_int64()
    assert_true(t == array([1, 1, 1, 2, 2], int64))


def test_ntile_with_more_buckets_than_rows() raises:
    """Each row is its own bucket and the surplus buckets stay empty."""
    var b = record_batch([array([1, 2], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["t"], [ntile(5).over(order_by=[col("v", int64)])]
    )
    ref t = plan.execute().column("t").as_int64()
    assert_true(t == array([1, 2], int64))


def test_nth_value_is_null_past_the_end_of_the_frame() raises:
    """The default frame runs to the peer group, so `n` reaches only as far as
    the frame has rows — row 0's frame holds one row, so `nth_value(v, 2)` is
    null there and 20 from row 1 on."""
    var b = record_batch([array([10, 20, 30], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["n2"], [col("v", int64).nth_value(2).over(order_by=[col("v", int64)])]
    )
    ref n2 = plan.execute().column("n2").as_int64()
    var expected: List[Optional[Int]] = [None, 20, 20]
    assert_true(n2 == array(expected, int64))


# ---------------------------------------------------------------------------
# One-pass window aggregates — `Windowable.over` against hand-computed frames
# ---------------------------------------------------------------------------
def _partitioned_with_nulls() raises -> RecordBatch:
    """Two partitions in a distinct order `o`, each holding a null, so every
    frame shape meets a null and a partition edge."""
    return record_batch(
        [
            array(["a", "a", "a", "a", "b", "b", "b"]).copy(),
            array([0, 1, 2, 3, 4, 5, 6], int64).copy(),
            array([1, None, 3, 6, 10, 20, None], int64).copy(),
        ],
        names=["k", "o", "v"],
    )


def _sliding[F: WindowFunction](function: F) raises -> Over[F]:
    """`function OVER (PARTITION BY k ORDER BY o ROWS 1 PRECEDING)`."""
    return function.over(
        partition_by=[col("k", string)],
        order_by=[col("o", int64)],
        rows=(-1, 0),
    )


def test_a_sliding_frame_folds_mean_min_and_max() raises:
    """`ROWS 1 PRECEDING` moves its start every row, so every frame is a
    range query rather than an extension of the previous one."""
    var plan = table(_partitioned_with_nulls()).with_columns(
        ["mean", "min", "max"],
        [
            _sliding(col("v", int64).mean()),
            _sliding(col("v", int64).min()),
            _sliding(col("v", int64).max()),
        ],
    )
    var out = plan.execute()
    assert_true(
        out.column("mean").as_float64()
        == array([1.0, 1.0, 3.0, 4.5, 10.0, 15.0, 20.0], float64)
    )
    assert_true(
        out.column("min").as_int64() == array([1, 1, 3, 3, 10, 10, 20], int64)
    )
    assert_true(
        out.column("max").as_int64() == array([1, 1, 3, 6, 10, 20, 20], int64)
    )


def test_a_cumulative_frame_restarts_at_each_partition() raises:
    """The default frame extends one running fold row by row; the partition
    edge is where its start moves, and the running total must not leak."""
    var plan = table(_partitioned_with_nulls()).with_columns(
        ["s"],
        [
            col("v", int64)
            .sum()
            .over(partition_by=[col("k", string)], order_by=[col("o", int64)])
        ],
    )
    assert_true(
        plan.execute().column("s").as_int64()
        == array([1, 1, 4, 10, 10, 30, 30], int64)
    )


def test_a_filtered_aggregate_skips_rows_its_filter_rejects() raises:
    """`SUM(v) FILTER (WHERE v > 2) OVER (ORDER BY v)`: a rejected row folds
    like a null, so a frame of rejected rows only is null, not 0."""
    var b = record_batch([array([1, 2, 3, 4], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["s"],
        [
            col("v", int64)
            .sum()
            .filter(col("v", int64) > lit(2, int64))
            .over(order_by=[col("v", int64)])
        ],
    )
    assert_true(
        plan.execute().column("s").as_int64()
        == array([None, None, 3, 7], int64)
    )


def test_an_aggregate_without_over_is_evaluated_per_frame() raises:
    """`variance` has no `Windowable.over`, so it is evaluated frame by frame
    through its operator — the fallback, pinned so it stays reachable."""
    var b = record_batch([array([1.0, 2.0, 3.0], float64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["var"],
        [col("v", float64).variance().over(order_by=[col("v", float64)])],
    )
    ref got = plan.execute().column("var").as_float64()
    assert_almost_equal(got[0].value(), 0.0)
    assert_almost_equal(got[1].value(), 0.25)
    assert_almost_equal(got[2].value(), 2.0 / 3.0)


def test_a_windowed_filter_must_be_boolean() raises:
    """A window aggregate lowers through `to_evaluator`, not `to_operator`,
    so it must reject a non-boolean `FILTER` there too, or narrowing the
    predicate to a `BoolArray` aborts the process. The runtime lane's case
    is in `runtime/tests/test_aggregates.mojo`."""
    var b = record_batch([array([1, 2, 3], int64).copy()], names=["v"])
    var plan = table(b^).with_columns(
        ["s"],
        [
            col("v", int64)
            .sum()
            .filter(col("v", int64))
            .over(order_by=[col("v", int64)])
        ],
    )
    var raised = False
    try:
        _ = plan.execute()
    except:
        raised = True
    assert_true(raised)
