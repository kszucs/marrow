"""Turning a scan's pruners into a read plan over a real Parquet file.

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

from ...dtypes import Int64Type, field, int64
from ...execution import ExecContext
from ...schema import schema
from ...parquet.reader import ParquetFile
from ...scalars import Int64Scalar
from ...tabular import Table
from ..bindings import Bindings
from ..builders import col, lit, scan
from ..logical import Filter, ParquetScan
from ..optimizer import AllRules, NoRules
from ..index import Index, page_selections
from ..logical import DynValue
from ..runtime.values import column, gt, literal
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


def _write_paged(path: String, rows: Int, page_rows: Int) raises:
    """One row group of `rows` ascending values, written in pages of
    `page_rows` — so the *page* index has something disjoint to prove where
    the row-group statistics cover the whole range and prove nothing.

    **`write_batch_size` is the load-bearing argument, not `data_page_size`.**
    parquet-cpp only checks whether the current page is full at the end of each
    write batch, and that batch defaults to 1024 values — so a 400-row group
    lands in a single page however small `data_page_size` is, and every case
    below then measures a page index with one entry in it. Setting both is what
    makes the fixture write the pages it claims to.
    """
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for i in range(rows):
        a.append(i)
    var tbl = pa.table(Python.dict(a=pa.array(a)))
    pq.write_table(
        tbl,
        path,
        row_group_size=rows,
        data_page_size=page_rows * 8,
        write_batch_size=page_rows,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )


# ---------------------------------------------------------------------------
# page skipping — the same decision, one granularity down
# ---------------------------------------------------------------------------
def test_scan_pruning_skips_pages_within_a_surviving_row_group() raises:
    """A single row group whose statistics prove nothing, and a page index that
    proves a great deal.

    The file is one group holding `[0, 400)`, so `a > 350` cannot skip the
    group -- `read_plan` keeps it, which is the precondition this case exists
    to test past. The page index then answers per page, and the selection
    drops every page that ends at or below 350.

    Both assertions have to hold together: strictly fewer rows decoded **and**
    the same rows a full read returns. A selection that kept everything would
    pass the second alone, and one that kept nothing would pass the first.
    """
    var path = String("/tmp/marrow_pages_banded.parquet")
    _write_paged(path, rows=400, page_rows=50)
    var f = ParquetFile(path)

    assert_true(
        len(f.page_bounds()[0][0]) > 1,
        "the fixture wrote one page, so this case would prove nothing",
    )

    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(350, int64))]
    var idx = Index.from_parquet(f)
    var keep = idx.read_plan(pushed)
    assert_equal(len(keep), 1, "the row group itself cannot be skipped")

    var sels = page_selections(f, idx, keep, pushed)
    assert_equal(len(sels), 1)
    assert_true(
        sels[0].num_selected() < sels[0].total_rows(),
        "the page index proved nothing",
    )

    var full = f.read()
    var partial = f.read(
        row_groups=Optional(keep.copy()), row_selections=Optional(sels.copy())
    )
    assert_equal(_count_above(full, 350), _count_above(partial, 350))
    assert_true(_count_above(full, 350) > 0)
    assert_true(partial.num_rows() < full.num_rows())
    remove(path)


def test_scan_pruning_page_selection_from_the_runtime_lane() raises:
    """Page selection is driven by `DynValue.mask`, so it does not care which
    lane built the predicate — and the two must agree.

    Worth pinning separately from the row-group case: the lanes reach the
    statistics differently, a fused node passing its own `T` where an
    interpreted one recovers the dtype from the index and dispatches. At page
    granularity that dispatch runs against a single-column `Index` built from
    `PageBounds`, which the row-group tests never exercise.
    """
    var path = String("/tmp/marrow_pages_runtime.parquet")
    _write_paged(path, rows=400, page_rows=50)
    var f = ParquetFile(path)
    var idx = Index.from_parquet(f)

    var fused: List[DynValue] = [DynValue(col("a", int64) > lit(350, int64))]
    var interpreted: List[DynValue] = [
        DynValue(gt(column("a"), literal(Int64Scalar(350).to_dyn())))
    ]
    var keep = idx.read_plan(fused)
    assert_equal(idx.read_plan(interpreted), keep, "same row groups")

    var a = page_selections(f, idx, keep, fused)
    var b = page_selections(f, idx, keep, interpreted)
    assert_equal(len(a), len(b))
    assert_true(a[0].num_selected() < a[0].total_rows(), "pages were skipped")
    assert_equal(a[0].num_selected(), b[0].num_selected(), "same rows survive")
    for row in range(a[0].total_rows()):
        assert_equal(
            a[0].selected(row), b[0].selected(row), "row " + String(row)
        )
    remove(path)


def test_scan_pruning_a_file_without_a_page_index_selects_everything() raises:
    """No page index, no selection — every row of every surviving group is
    read, which is what the engine did before this existed."""
    var path = String("/tmp/marrow_pages_none.parquet")
    _write_banded(path, groups=2, rows=40)
    var f = ParquetFile(path)

    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(-1, int64))]
    var sels = page_selections(f, Index.from_parquet(f), [0, 1], pushed)
    assert_equal(len(sels), 0, "no page index means no page pruning at all")
    remove(path)


def test_scan_pruning_a_plan_reads_fewer_rows_through_the_page_index() raises:
    """End to end, through `execute()`: the answer is the answer, and the
    operator asked the reader for a subset of the rows.

    Asserted on the *values* rather than on a count, because a selection that
    is misaligned by a page returns the right number of wrong rows -- the
    failure mode `RowSelection` exists to make impossible for flat files and
    which nested files are refused for.
    """
    var path = String("/tmp/marrow_pages_plan.parquet")
    _write_paged(path, rows=400, page_rows=50)

    var plan = scan(path, schema([field("a", int64)])).filter(
        col("a", int64) > lit(350, int64)
    )
    var out = plan.optimize[AllRules]().execute()
    assert_equal(out.num_rows(), 49)
    ref c = out.column("a").as_int64()
    assert_equal(Int(c[0].value()), 351)
    assert_equal(Int(c[48].value()), 399)
    remove(path)


# ---------------------------------------------------------------------------
# row-group skipping
# ---------------------------------------------------------------------------
def test_scan_pruning_skips_row_groups_and_returns_the_same_rows() raises:
    """One predicate, two assertions that only pass together: strictly fewer
    row groups are read, **and** the rows that come back are the rows a full
    read would have produced."""
    var path = String("/tmp/marrow_pushdown_banded.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)

    var idx = Index.from_parquet(f)
    assert_equal(idx.chunks, 4)

    # `a > 150` — groups 0..2 hold [0,50) [50,100) [100,150); only group 3 can
    # match, and its bounds are [150, 200).
    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(150, int64))]
    var keep = idx.read_plan(pushed)
    assert_equal(len(keep), 1)
    assert_equal(keep[0], 3)

    var full = f.read()
    var partial = f.read(row_groups=Optional(keep.copy()))
    assert_equal(_count_above(full, 150), _count_above(partial, 150))
    assert_true(_count_above(full, 150) > 0)
    assert_true(partial.num_rows() < full.num_rows())

    remove(path)


def test_scan_pruning_an_empty_pushdown_reads_everything() raises:
    """The default, and what every plan does today: no predicate, no skipping.
    """
    var path = String("/tmp/marrow_pushdown_empty.parquet")
    _write_banded(path, groups=3, rows=20)
    var f = ParquetFile(path)

    var keep = Index.from_parquet(f).read_plan([])
    assert_equal(len(keep), 3)
    remove(path)


def test_scan_pruning_conjoins_stacked_filters() raises:
    """`Filter(Filter(scan))` forwards two predicates, and a chunk survives
    only if both say maybe — conjunction without a node that ANDs two erased
    boxes."""
    var path = String("/tmp/marrow_pushdown_stacked.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)
    var idx = Index.from_parquet(f)

    # a > 60  keeps groups 1,2,3 ; a < 140 keeps groups 0,1,2 -> intersection 1,2
    var pushed: List[DynValue] = [
        DynValue(col("a", int64) > lit(60, int64)),
        DynValue(col("a", int64) < lit(140, int64)),
    ]
    var keep = idx.read_plan(pushed)
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


def test_scan_pruning_a_conjunction_inside_one_predicate_agrees() raises:
    """The same two conjuncts as one `AND` node rather than two entries.
    Two spellings of a conjunction must give the same plan, or a rewrite that
    merges filters would change which groups are read."""
    var path = String("/tmp/marrow_pushdown_conj.parquet")
    _write_banded(path, groups=4, rows=50)
    var f = ParquetFile(path)
    var idx = Index.from_parquet(f)

    var one: List[DynValue] = [
        DynValue(
            (col("a", int64) > lit(60, int64))
            & (col("a", int64) < lit(140, int64))
        )
    ]
    var keep = idx.read_plan(one)
    assert_equal(len(keep), 2)
    assert_equal(keep[0], 1)
    assert_equal(keep[1], 2)

    remove(path)


def test_scan_pruning_a_predicate_on_an_absent_column_prunes_nothing() raises:
    """A name the file does not carry is a missing statistic, not an error.
    The scan reads everything and the `Filter` above it raises — pruning
    degrades, resolution raises."""
    var path = String("/tmp/marrow_pushdown_absent.parquet")
    _write_banded(path, groups=3, rows=20)
    var f = ParquetFile(path)

    var pushed: List[DynValue] = [DynValue(col("zz", int64) > lit(10, int64))]
    assert_equal(len(Index.from_parquet(f).read_plan(pushed)), 3)
    remove(path)


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------
def test_scan_pruning_a_file_without_statistics_prunes_nothing() raises:
    """`write_statistics=False`. Every chunk then has min/max absent and a
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
    var idx = Index.from_parquet(f)
    assert_equal(idx.chunks, 3)

    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(1000, int64))]
    var keep = idx.read_plan(pushed)
    assert_equal(len(keep), 3)
    remove(path)


def test_scan_pruning_a_nested_schema_prunes_nothing() raises:
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
    var idx = Index.from_parquet(f)
    assert_equal(idx.chunks, 3)
    assert_equal(idx.zones.num_columns(), 0)

    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(1000, int64))]
    assert_equal(len(idx.read_plan(pushed)), 3)
    remove(path)


def test_scan_pruning_reads_null_counts_from_a_foreign_writer() raises:
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
    var idx = Index.from_parquet(f)
    assert_equal(idx.chunks, 2)
    var live = idx.defined(String("a"))
    assert_false(live[0].value(), "the all-null group has no live row")
    assert_true(live[1].value())

    # `a > -1` is true of every non-null row, so only the all-null group goes.
    var pushed: List[DynValue] = [DynValue(col("a", int64) > lit(-1, int64))]
    var keep = idx.read_plan(pushed)
    assert_equal(len(keep), 1)
    assert_equal(keep[0], 1)

    remove(path)


# ---------------------------------------------------------------------------
# the conjunction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# the optimizer as the delivery mechanism
# ---------------------------------------------------------------------------
def test_scan_pruning_the_optimizer_installs_a_pruner_that_skips_row_groups() raises:
    """End to end: `PushFilterIntoScan` puts a pruner on the scan node, that
    pruner skips row groups, and the plan still answers what it answered.

    All three together are the point. The rule firing proves nothing on its own
    — a pruner that prunes nothing installs just as cleanly — so the middle
    assertion runs the optimizer's own pruner through the same `read_plan` the
    scan operator calls, and the last one checks the rows against the
    unoptimized plan.
    """
    var path = String("/tmp/marrow_pushdown_optimized.parquet")
    _write_banded(path, groups=4, rows=50)

    var sch = schema([field("a", int64), field("b", int64)])
    var plan = scan(path.copy(), sch^).filter(col("a", int64) > lit(150, int64))

    var optimized = plan.optimize[AllRules]()
    assert_true(optimized.isa[Filter]())
    var below = optimized.get[Filter]().input[].copy()
    assert_true(below.isa[ParquetScan]())
    var pruners = below.get[ParquetScan]().pruners.copy()
    assert_equal(len(pruners), 1)

    # groups 0..2 hold [0,50) [50,100) [100,150); only group 3 can match.
    var f = ParquetFile(path)
    var keep = Index.from_parquet(f).read_plan(pruners)
    assert_equal(len(keep), 1)
    assert_equal(keep[0], 3)

    var ctx = ExecContext()
    var before = plan.optimize[NoRules]().execute(ctx)
    var after = optimized.execute(ctx)
    assert_equal(after.num_rows(), before.num_rows())
    assert_true(before.num_rows() > 0)

    remove(path)
