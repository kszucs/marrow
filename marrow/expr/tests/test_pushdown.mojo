"""Delivery: turning a pushed predicate into a read plan over a real Parquet
file.

Written against files produced by **pyarrow**, not by marrow's own writer.
Every statistic here is therefore decoded from a footer marrow did not write,
which is the case the six safety defects fixed in `9a491ce` all lived in — a
suite built only on marrow-written files never sees an absent `column_orders`,
a leaf under a struct, or a chunk with no statistics at all.

The load-bearing assertion in each case is the same one: **the rows returned
after pruning are exactly the rows returned without it.** A `groups_read`
comparison alone would pass trivially for a pruner that pruned nothing, and a
row-count comparison alone would pass for one that pruned everything; both are
asserted together.
"""

from std.os import remove
from std.python import Python
from std.testing import assert_equal, assert_false, assert_true

from ...dtypes import Int64Type, int64
from ...parquet.reader import ParquetFile
from ...scalars import Int64Scalar
from ...tabular import Table
from ..bindings import Bindings
from ..pruning import PrunePredicate, PruneStats, Truth
from ..pushdown import Pushdown, read_plan, row_group_stats
from .test_pruning import (
    _PAnd,
    _PColumn,
    _PComparison,
    _PLiteral,
    _gt_lit,
)
from ...kernels.numeric import LtKernel


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _write_banded(path: String, groups: Int, rows: Int) raises:
    """`groups` row groups of `rows` rows, where group `g` holds `a` values in
    `[g*rows, (g+1)*rows)` — so every row group's `[min, max]` is disjoint and
    a range predicate has something to prove."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    var b = Python.list()
    for g in range(groups):
        for i in range(rows):
            a.append(g * rows + i)
            b.append(i % 7)
    var tbl = pa.table(Python.dict(a=pa.array(a), b=pa.array(b)))
    pq.write_table(tbl, path, row_group_size=rows, compression="none")


def _count_above(t: Table, threshold: Int) raises -> Int:
    """Rows whose `a` is greater than `threshold` — the exact predicate,
    applied by hand so the comparison does not depend on the engine."""
    var n = 0
    for ref batch in t.to_batches():
        ref c = batch.columns[0].as_int64()
        for i in range(len(c)):
            if c.is_valid(i) and Int(c[i].value()) > threshold:
                n += 1
    return n


def _count_between(t: Table, lo: Int, hi: Int) raises -> Int:
    var n = 0
    for ref batch in t.to_batches():
        ref c = batch.columns[0].as_int64()
        for i in range(len(c)):
            if c.is_valid(i):
                var v = Int(c[i].value())
                if v > lo and v < hi:
                    n += 1
    return n


# ---------------------------------------------------------------------------
# row-group skipping
# ---------------------------------------------------------------------------
def test_pushdown_skips_row_groups_and_returns_the_same_rows() raises:
    """One predicate, two assertions that only pass together: strictly fewer
    row groups are read, **and** the rows that come back are the rows a full
    read would have produced."""
    var path = String("/tmp/marrow_pushdown_banded.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)

    var granules = row_group_stats(f)
    assert_equal(len(granules), 4)

    # `a > 150` — groups 0..2 hold [0,50) [50,100) [100,150); only group 3 can
    # match, and its bounds are [150, 200).
    var pushed = Pushdown().conjoined(PrunePredicate(_gt_lit("a", 150)))
    var keep = read_plan(granules, pushed, Bindings())
    assert_equal(len(keep), 1)
    assert_equal(keep[0], 3)

    var full = f.read()
    var partial = f.read(row_groups=Optional(keep.copy()))
    assert_equal(_count_above(full, 150), _count_above(partial, 150))
    assert_true(_count_above(full, 150) > 0)
    assert_true(partial.num_rows() < full.num_rows())

    remove(path)


def test_pushdown_an_empty_pushdown_reads_everything() raises:
    """The default, and what every plan does today: no predicate, no skipping.
    """
    var path = String("/tmp/marrow_pushdown_empty.parquet")
    _write_banded(path, groups=3, rows=20)
    var f = ParquetFile(path)

    var keep = read_plan(row_group_stats(f), Pushdown(), Bindings())
    assert_equal(len(keep), 3)
    remove(path)


def test_pushdown_conjoins_stacked_filters() raises:
    """`Filter(Filter(scan))` forwards two predicates, and a granule survives
    only if both say maybe — conjunction without a node that ANDs two erased
    boxes."""
    var path = String("/tmp/marrow_pushdown_stacked.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)
    var granules = row_group_stats(f)

    # a > 60  keeps groups 1,2,3 ; a < 140 keeps groups 0,1,2 -> intersection 1,2
    var below = _PComparison[
        LtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]
    ](
        _PColumn[Int64Type]("a"),
        _PLiteral[Int64Type](Scalar[int64.native](140)),
    )
    var pushed = (
        Pushdown()
        .conjoined(PrunePredicate(_gt_lit("a", 60)))
        .conjoined(PrunePredicate(below))
    )
    var keep = read_plan(granules, pushed, Bindings())
    assert_equal(len(keep), 2)
    assert_equal(keep[0], 1)
    assert_equal(keep[1], 2)

    var full = f.read()
    var partial = f.read(row_groups=Optional(keep.copy()))
    assert_equal(
        _count_between(full, 60, 140), _count_between(partial, 60, 140)
    )
    assert_true(_count_between(full, 60, 140) > 0)

    remove(path)


def test_pushdown_a_conjunction_inside_one_predicate_agrees() raises:
    """The same two conjuncts as one `_PAnd` rather than two pushed entries.
    Two spellings of a conjunction must give the same plan, or a rewrite that
    merges filters would change which groups are read."""
    var path = String("/tmp/marrow_pushdown_conj.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)
    var granules = row_group_stats(f)

    var below = _PComparison[
        LtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]
    ](
        _PColumn[Int64Type]("a"),
        _PLiteral[Int64Type](Scalar[int64.native](140)),
    )
    var one = Pushdown().conjoined(
        PrunePredicate(_PAnd(_gt_lit("a", 60), below.copy()))
    )
    var keep = read_plan(granules, one, Bindings())
    assert_equal(len(keep), 2)
    assert_equal(keep[0], 1)
    assert_equal(keep[1], 2)

    remove(path)


def test_pushdown_a_predicate_on_an_absent_column_prunes_nothing() raises:
    """A name the file does not carry is a missing statistic, not an error.
    The scan reads everything and the `Filter` above it raises — pruning
    degrades, resolution raises."""
    var path = String("/tmp/marrow_pushdown_absent.parquet")
    _write_banded(path, groups=3, rows=20)
    var f = ParquetFile(path)

    var pushed = Pushdown().conjoined(PrunePredicate(_gt_lit("zz", 10)))
    assert_equal(len(read_plan(row_group_stats(f), pushed, Bindings())), 3)
    remove(path)


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------
def test_pushdown_a_file_without_statistics_prunes_nothing() raises:
    """`write_statistics=False`. Every granule then has min/max absent and a
    null count of -1, so nothing is provable and the whole file is read."""
    var path = String("/tmp/marrow_pushdown_nostats.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for i in range(60):
        a.append(i)
    var tbl = pa.table(Python.dict(a=pa.array(a)))
    pq.write_table(
        tbl,
        path,
        row_group_size=20,
        compression="none",
        write_statistics=False,
    )

    var f = ParquetFile(path)
    var granules = row_group_stats(f)
    assert_equal(len(granules), 3)

    var pushed = Pushdown().conjoined(PrunePredicate(_gt_lit("a", 1000)))
    var keep = read_plan(granules, pushed, Bindings())
    assert_equal(len(keep), 3)
    remove(path)


def test_pushdown_a_nested_schema_prunes_nothing() raises:
    """The leaf-alignment guard.

    `ParquetFile.statistics()` is indexed by *leaf*, an expression names a
    *top-level column*, and the two agree only when there are as many leaves as
    fields. A struct with two members breaks that, and handing field `i` leaf
    `i`'s bounds is exactly defect D14 (`struct<x>`'s bounds handed to a
    top-level `x`). Here it must fall back to no statistics rather than to
    misattributed ones — including for the flat column sitting beside the
    struct, whose leaf index has shifted.
    """
    var path = String("/tmp/marrow_pushdown_nested.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var s = Python.list()
    var a = Python.list()
    for i in range(60):
        s.append(Python.dict(x=i, y=i * 2))
        a.append(i)
    var tbl = pa.table(
        Python.dict(
            s=pa.array(
                s,
                type=pa.struct(
                    [pa.field("x", pa.int64()), pa.field("y", pa.int64())]
                ),
            ),
            a=pa.array(a),
        )
    )
    pq.write_table(tbl, path, row_group_size=20, compression="none")

    var f = ParquetFile(path)
    var granules = row_group_stats(f)
    assert_equal(len(granules), 3)
    for ref g in granules:
        assert_equal(g.num_columns(), 0)

    var pushed = Pushdown().conjoined(PrunePredicate(_gt_lit("a", 1000)))
    assert_equal(len(read_plan(granules, pushed, Bindings())), 3)
    remove(path)


def test_pushdown_reads_null_counts_from_a_foreign_writer() raises:
    """An all-null row group written by pyarrow must prune exactly, and a
    partly-null one must not.

    This is the one prune driven by the null count rather than a bound, and it
    is also where `null_count`'s independence from `has_min_max` shows up: the
    all-null group carries a null count and no bounds at all.
    """
    var path = String("/tmp/marrow_pushdown_nulls.parquet")
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for _ in range(20):
        a.append(Python.none())
    for i in range(20):
        a.append(i)
    var tbl = pa.table(Python.dict(a=pa.array(a, type=pa.int64())))
    pq.write_table(tbl, path, row_group_size=20, compression="none")

    var f = ParquetFile(path)
    var granules = row_group_stats(f)
    assert_equal(len(granules), 2)
    assert_true(granules[0].all_null(0))
    assert_false(granules[1].all_null(0))

    # `a > -1` is true of every non-null row, so only the all-null group goes.
    var pushed = Pushdown().conjoined(PrunePredicate(_gt_lit("a", -1)))
    var keep = read_plan(granules, pushed, Bindings())
    assert_equal(len(keep), 1)
    assert_equal(keep[0], 1)

    remove(path)


# ---------------------------------------------------------------------------
# the Pushdown value type
# ---------------------------------------------------------------------------
def test_pushdown_conjoined_does_not_mutate_its_receiver() raises:
    """A `Pushdown` is as immutable as the plan it descends through: the
    `Limit` rule below depends on a node being able to hand its input a
    *cleared* pushdown without disturbing what its own parent holds."""
    var base = Pushdown()
    var one = base.conjoined(PrunePredicate(_gt_lit("a", 5)))
    var two = one.conjoined(PrunePredicate(_gt_lit("a", 9)))
    assert_equal(len(base), 0)
    assert_equal(len(one), 1)
    assert_equal(len(two), 2)
    assert_equal(len(Pushdown.cleared()), 0)


def test_pushdown_prunes_only_when_every_predicate_agrees() raises:
    var s = PruneStats(100, capacity=1)
    s.add("a", Optional(Int64Scalar(0).to_dyn()), Optional(Int64Scalar(3).to_dyn()), 0)

    var keeps = Pushdown().conjoined(PrunePredicate(_gt_lit("a", 1)))
    assert_true(keeps.prune(s, Bindings()) == Truth.maybe)

    var prunes = keeps.conjoined(PrunePredicate(_gt_lit("a", 5)))
    assert_true(prunes.prune(s, Bindings()) == Truth.never)
