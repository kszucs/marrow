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

from std.os import remove
from std.python import Python
from std.testing import assert_true

from ...builders import array, nulls
from ...dtypes import int64, string
from ...tabular import record_batch
from ..builders import col, dense_rank, lit, rank, row_number, scan, table


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


def test_a_sum_over_an_all_null_frame_is_null() raises:
    """The window aggregate inherits the kernel's null rule rather than
    restating it.

    `SUM` skips nulls and answers null when it saw none, so the row whose frame
    is `{null}` is null while the row whose frame is `{null, 4}` is 4. Getting
    this from `SumFold` rather than from an accumulator written here is the
    reason the aggregate goes through its own operator.
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
def test_a_non_aggregate_cannot_take_a_frame() raises:
    """`col("v", int64).over(...)` is a mistake, and it gets a diagnostic.

    A per-row value has no frame, so evaluating one would either broadcast a
    copy or silently pick a row. Both are worse than raising.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["v"])
    var raised = False
    try:
        _ = table(b^).with_columns(
            ["x"], [col("v", int64).over(order_by=[col("v", int64)])]
        )
    except:
        raised = True
    assert_true(raised)


def test_a_window_column_cannot_shadow_an_existing_one() raises:
    """`Window` appends, so a repeated name would duplicate rather than replace.

    The `DynValue` overload of `with_columns` replaces in place because a
    `Project` names every output column anyway; this one raises instead, since
    a duplicated name makes every later read by name ambiguous.
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

    `Relation.to_operator` threads a `Pushdown` down the plan so a
    `ParquetScan` can skip row groups. Every node that decides *which rows
    exist* has to stop it -- `Aggregate` and `Limit` already do -- because a
    window function reads its whole partition: prune the scan and
    `row_number()` counts a smaller population, silently.

    `Sort` forwards it and is right to; reordering never removes a row. This
    node was written in that mould and belonged in the other one.

    The file holds `a` in `[0, 100)` across four disjoint row groups, so
    `a > 74` can prove three of the four away. With the pushdown stopped, the
    surviving rows keep the row numbers they had in the full ordering --
    76..100. If it descends, the window sees only the last group and numbers
    it 1..25 instead: a plausible answer, and the reason a wrong-population
    bug like this is invisible without an assertion on the *values*.
    """
    var path = String("/tmp/marrow_window_pushdown.parquet")
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
        .with_columns(["rn"], [row_number().over(order_by=[col("a", int64)])])
        .filter(col("a", int64) > lit(74, int64))
    )
    var out = plan.execute()

    assert_true(out.num_rows() == 25, "expected the 25 rows above 74")
    ref rn = out.column("rn").as_int64()
    assert_true(
        Int(rn[0].value()) == 76,
        "row_number restarted -- the pushdown reached the scan: got "
        + String(rn[0].value()),
    )
    assert_true(Int(rn[24].value()) == 100, String(rn[24].value()))
    remove(path)


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
