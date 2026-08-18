"""Pushdown into ParquetScan: predicate (row-group / page skipping) and
projection, plus the per-row-group streaming the scan is driven by.

`filter()` over a `parquet_scan` pushes the predicate into the scan (which prunes
row groups) while keeping the `Filter` for exact results. Correctness never
depends on pruning, so we check both the pruning decision and that the executed
result matches an exact filter.

The scan's schema is its projection, and it decodes one row group at a time —
so a morsel never straddles a row-group boundary, which is what the streaming
tests below pin down. Neither changes the rows produced."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from ...dtypes import Int64Type, int64, string, field
from ...schema import Schema, schema
from ...scalars import DynScalar
from ...parquet import read_table, ParquetFile, LeafSet
from ...expr.relations import (
    ParquetScan,
    parquet_scan,
    DynRelation,
    Filter,
    RELATION_PARQUET_SCAN,
)
from ...expr.builders import col, count_star, lit
from ...expr.values import BoxedValue
from ...expr.pruning import PruneStats


def _write_sorted(path: String, n: Int, rgsize: Int) raises:
    """A single int64 column `x` = 0..n-1, in row groups of `rgsize`."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(
                Python.evaluate("list(range(" + String(n) + "))"),
                type=pa.int64(),
            )
        )
    )
    pq.write_table(tbl, path, row_group_size=rgsize, compression="none")


def _write_wide(path: String, n: Int, rgsize: Int) raises:
    """Three columns — `x` = 0..n-1, `y` = 10*x, `s` = "r<x>" — in row groups of
    `rgsize`. `y` is what a projection keeps and `s` is what it must skip."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var rng = "list(range(" + String(n) + "))"
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate(rng), type=pa.int64()),
            y=pa.array(
                Python.evaluate("[10 * i for i in " + rng + "]"),
                type=pa.int64(),
            ),
            s=pa.array(
                Python.evaluate("['r%d' % i for i in " + rng + "]"),
                type=pa.string(),
            ),
        )
    )
    pq.write_table(tbl, path, row_group_size=rgsize, compression="none")


def test_read_table_row_groups() raises:
    var path = String("/tmp/marrow_pd_rg.parquet")
    _write_sorted(path, 3000, 1000)  # 3 row groups
    var rgs: List[Int] = [2]
    var t = read_table(path, row_groups=rgs^)  # last group only
    assert_equal(t.num_rows(), 1000)
    var b = t.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_int64()[0].value(), 2000)
    remove(path)


def test_row_group_prune_decision() raises:
    var path = String("/tmp/marrow_pd_decide.parquet")
    _write_sorted(path, 3000, 1000)
    var stats = ParquetFile(path).statistics()
    assert_equal(len(stats), 3)
    var pred = BoxedValue(col("x") > lit[Int64Type](Int64(1500)))
    var sch = schema([field("x", int64)])
    var keep = List[Bool]()
    for rg in range(len(stats)):
        var mins = List[Optional[DynScalar]]()
        var maxs = List[Optional[DynScalar]]()
        mins.append(stats[rg][0].min.copy())
        maxs.append(stats[rg][0].max.copy())
        var ps = PruneStats(Schema(copy=sch), mins^, maxs^)
        keep.append(pred.prune(ps).maybe_true)
    assert_false(keep[0])  # x in [0, 999] can never exceed 1500 -> prune
    assert_true(keep[1])  # [1000, 1999]
    assert_true(keep[2])  # [2000, 2999]
    remove(path)


def test_filter_pushes_predicate_into_scan() raises:
    # structural: filter() over a ParquetScan yields Filter(ParquetScan(pred=..))
    var sch = schema([field("x", int64)])
    var plan = parquet_scan("t", sch).filter(
        col("x") > lit[Int64Type](Int64(0))
    )
    var f = plan.downcast[Filter]()
    assert_equal(f[].input.kind(), RELATION_PARQUET_SCAN)
    var scan = f[].input.downcast[ParquetScan[LeafSet.all()]]()
    assert_true(Bool(scan[].predicate))  # predicate was pushed down


def test_pushdown_end_to_end() raises:
    var path = String("/tmp/marrow_pd_e2e.parquet")
    _write_sorted(path, 3000, 1000)
    var sch = schema([field("x", int64)])
    # RG0 is pruned; the Filter still yields exactly x > 1500
    var plan = parquet_scan(path, sch).filter(
        col("x") > lit[Int64Type](Int64(1500))
    )
    var result = plan.execute()
    assert_equal(result.num_rows(), 1499)  # 1501..2999
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 1501)
    remove(path)


def test_pushdown_page_skip() raises:
    # one row group, many small pages with a page index; a range predicate must
    # skip whole pages yet return exactly the matching rows (Filter re-applies)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate("list(range(10000))"), type=pa.int64())
        )
    )
    var path = String("/tmp/marrow_pd_pageskip.parquet")
    pq.write_table(
        tbl,
        path,
        data_page_size=256,
        row_group_size=1000000,
        write_page_index=True,
        compression="none",
    )
    var sch = schema([field("x", int64)])
    var plan = parquet_scan(path, sch).filter(
        col("x") > lit[Int64Type](Int64(7500))
    )
    var result = plan.execute()
    assert_equal(result.num_rows(), 2499)  # 7501..9999
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 7501)
    assert_equal(result.columns[0].copy().as_int64()[2498].value(), 9999)
    remove(path)


def test_scan_yields_one_row_group_per_morsel() raises:
    """With a morsel larger than a row group, each pull yields exactly one row
    group — the observable form of "only one row group is decoded at a time"."""
    var path = String("/tmp/marrow_pd_stream.parquet")
    _write_sorted(path, 3000, 1000)  # 3 row groups
    var sch = schema([field("x", int64)])
    var scan = parquet_scan(path, sch, morsel_size=1_000_000).to_processor()
    var sizes = List[Int]()
    var firsts = List[Int64]()
    while True:
        try:
            var m = scan.pull()
            sizes.append(m.num_rows())
            firsts.append(m.columns[0].copy().as_int64()[0].value())
        except:
            break
    var expected_sizes: List[Int] = [1000, 1000, 1000]
    var expected_firsts: List[Int64] = [0, 1000, 2000]
    assert_equal(String(sizes), String(expected_sizes))
    assert_equal(String(firsts), String(expected_firsts))
    remove(path)


def test_scan_morsels_do_not_span_row_groups() raises:
    """A morsel smaller than a row group splits it, and never carries rows of
    the next group: 1000-row groups at morsel 400 give 400/400/200 each."""
    var path = String("/tmp/marrow_pd_morsel.parquet")
    _write_sorted(path, 2000, 1000)
    var sch = schema([field("x", int64)])
    var scan = parquet_scan(path, sch, morsel_size=400).to_processor()
    var sizes = List[Int]()
    while True:
        try:
            sizes.append(scan.pull().num_rows())
        except:
            break
    var expected: List[Int] = [400, 400, 200, 400, 400, 200]
    assert_equal(String(sizes), String(expected))
    remove(path)


def test_scan_projects_to_its_own_schema() raises:
    """The scan reads only the columns its schema names, in that order — the
    other chunks are never decoded."""
    var path = String("/tmp/marrow_pd_project.parquet")
    _write_wide(path, 2000, 1000)
    var sch = schema([field("y", int64)])
    var result = parquet_scan(path, sch).execute()
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.num_rows(), 2000)
    assert_equal(result.columns[0].copy().as_int64()[7].value(), 70)
    # a projection may also reorder
    var reordered = parquet_scan(
        path, schema([field("y", int64), field("x", int64)])
    ).execute()
    assert_equal(reordered.schema.fields[0].name, "y")
    assert_equal(reordered.schema.fields[1].name, "x")
    assert_equal(reordered.columns[1].copy().as_int64()[7].value(), 7)
    remove(path)


def test_scan_projection_keeps_pruning_aligned() raises:
    """Statistics are indexed by *file leaf*, the predicate by the scan's own
    schema. A projection that drops the leading column must not shift one
    against the other: `y > 15000` still prunes group 0 and returns exactly the
    matching rows."""
    var path = String("/tmp/marrow_pd_proj_prune.parquet")
    _write_wide(path, 3000, 1000)  # y = 0, 10, ..., 29990
    var sch = schema([field("y", int64), field("s", string)])
    var plan = parquet_scan(path, sch).filter(
        col("y") > lit[Int64Type](Int64(15000))
    )
    var result = plan.execute()
    assert_equal(result.num_columns(), 2)
    assert_equal(result.num_rows(), 1499)  # y = 15010 .. 29990
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 15010)
    remove(path)


def test_pushdown_prunes_all_groups() raises:
    # a predicate no row can satisfy -> every group pruned -> empty result
    var path = String("/tmp/marrow_pd_none.parquet")
    _write_sorted(path, 3000, 1000)
    var sch = schema([field("x", int64)])
    var plan = parquet_scan(path, sch).filter(
        col("x") > lit[Int64Type](Int64(100000))
    )
    var result = plan.execute()
    assert_equal(result.num_rows(), 0)
    remove(path)


# ---------------------------------------------------------------------------
# Projection pushdown
#
# The assertions below are on the **rewritten scan's schema**, not on the rows:
# a test that only checks answers passes even when nothing was pushed, because
# reading a column nobody wants is slow rather than wrong. Each one therefore
# walks the optimized plan down to its `ParquetScan` and names the columns that
# survived — and a handful then also execute, so the rewrite is pinned to
# producing the same rows as the plan it replaced.
# ---------------------------------------------------------------------------


def _scan_schema_of(plan: DynRelation) raises -> Schema:
    """The schema of the first `ParquetScan` under `plan`, found by walking
    `children()` — the traversal primitive this work added, used here instead of
    a `downcast` chain that would have to name every intermediate node type."""
    if plan.kind() == RELATION_PARQUET_SCAN:
        return plan.schema()
    for ref c in plan.children():
        var found = _scan_schema_of(c)
        if len(found.fields) > 0:
            return found^
    return Schema()


def _scan_columns_of(plan: DynRelation) raises -> String:
    """The scan's surviving column names, comma-joined, for a one-line assert.
    """
    var out = String()
    for ref f in _scan_schema_of(plan).fields:
        if out.byte_length() > 0:
            out += ","
        out += f.name
    return out^


def _wide_scan(path: String) raises -> DynRelation:
    """A scan over the three-column file `_write_wide` builds."""
    return parquet_scan(
        path,
        schema([field("x", int64), field("y", int64), field("s", string)]),
    )


def test_relation_children_walks_the_plan() raises:
    """`children()` makes a plan walkable: root to leaf without a downcast."""
    var sch = schema([field("x", int64), field("y", int64)])
    var plan = (
        parquet_scan("t", sch)
        .filter(col("x") > lit[Int64Type](Int64(0)))
        .select("y")
    )
    assert_equal(len(plan.children()), 1)  # Project -> Filter
    var f = plan.children()[0].copy()
    assert_equal(len(f.children()), 1)  # Filter -> ParquetScan
    var scan = f.children()[0].copy()
    assert_equal(scan.kind(), RELATION_PARQUET_SCAN)
    assert_equal(len(scan.children()), 0)  # a leaf has none


def test_projection_pushdown_narrows_scan_to_selected_column() raises:
    """`select("y")` over a three-column scan reads only `y`."""
    var plan = _wide_scan("t").select("y").optimize()
    assert_equal(_scan_columns_of(plan), "y")


def test_projection_pushdown_keeps_predicate_columns() raises:
    """The predicate's columns are read even when the output drops them: the
    `Filter` still evaluates `x > 0`, so `x` must survive alongside `y`."""
    var plan = (
        _wide_scan("t")
        .filter(col("x") > lit[Int64Type](Int64(0)))
        .select("y")
        .optimize()
    )
    assert_equal(_scan_columns_of(plan), "x,y")


def test_projection_pushdown_keeps_sort_keys() raises:
    """Same rule for a `Sort`: `s` is ordered on but not emitted."""
    var asc: List[Bool] = [True]
    var plan = (
        _wide_scan("t").sort(keys=[col("s")], ascending=asc).select("y")
    ).optimize()
    assert_equal(_scan_columns_of(plan), "y,s")


def test_projection_pushdown_keeps_group_keys_and_agg_inputs() raises:
    """An aggregate needs both halves — the key `s` and the summed `y` — and
    nothing else, whatever its own output columns are called."""
    var plan = (
        _wide_scan("t")
        .aggregate(keys=[col("s")], aggs=[col("y").aggregate("sum")])
        .optimize()
    )
    assert_equal(_scan_columns_of(plan), "y,s")


def test_projection_pushdown_is_transitive_through_project() raises:
    """A `Project` drops the outputs nobody reads, so the scan below stops
    reading their inputs too — `select("x", "y")` under a `sum(y)` reads only
    `y`, not `x`."""
    var plan = (
        _wide_scan("t")
        .select("x", "y")
        .aggregate(keys=[], aggs=[col("y").aggregate("sum")])
        .optimize()
    )
    assert_equal(_scan_columns_of(plan), "y")


def test_projection_pushdown_leaves_full_scan_alone() raises:
    """A plan that emits every column must still read every column — the root
    seeds `needed` with its own schema, so nothing narrows."""
    var plan = _wide_scan("t").filter(col("x") > lit[Int64Type](Int64(0)))
    assert_equal(_scan_columns_of(plan.optimize()), "x,y,s")
    # ... and the root's own schema is untouched by the rewrite.
    assert_equal(String(plan.optimize().schema()), String(plan.schema()))


def test_projection_pushdown_keeps_one_column_for_count_star() raises:
    """`COUNT(*)` references no column at all, and a zero-column scan yields
    zero-row batches — which the streaming loop reads as end-of-file. So the
    empty case keeps exactly one column, the narrowest fixed-width one: `x` and
    `y` are both int64 and `s` is variable-width, so `x` wins on order."""
    var plan = _wide_scan("t").aggregate(keys=[], aggs=[count_star()])
    assert_equal(_scan_columns_of(plan.optimize()), "x")


def test_projection_pushdown_stops_at_a_join() raises:
    """A join's key *indices* are positions into its children's schemas, so
    narrowing a child would join on the wrong column. Both sides stay whole."""
    var left = _wide_scan("l")
    var right = parquet_scan("r", schema([field("x", int64)]))
    var plan = left.join(right, left_on=[col("x")], right_on=[col("x")])
    var optimized = plan.select("y").optimize()
    var join = optimized.children()[0].copy()
    assert_equal(len(join.children()), 2)
    assert_equal(_scan_columns_of(join.children()[0].copy()), "x,y,s")
    assert_equal(_scan_columns_of(join.children()[1].copy()), "x")


def test_projection_pushdown_preserves_results() raises:
    """End to end on a real file: the rewritten plan returns exactly the rows
    and columns the un-rewritten one does."""
    var path = String("/tmp/marrow_pd_projpush.parquet")
    _write_wide(path, 2000, 500)  # 4 row groups, y = 10 * x
    var plan = (
        _wide_scan(path)
        .filter(col("x") > lit[Int64Type](Int64(1996)))
        .select("y")
    )
    assert_equal(_scan_columns_of(plan.optimize()), "x,y")
    var result = plan.execute()
    assert_equal(result.num_columns(), 1)
    assert_equal(result.schema.fields[0].name, "y")
    assert_equal(result.num_rows(), 3)  # x = 1997, 1998, 1999
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 19970)
    remove(path)


def test_projection_pushdown_count_star_counts_every_row() raises:
    """The one-column fallback still sees every row: `COUNT(*)` over the whole
    file is the file's row count, not the count of one decoded chunk."""
    var path = String("/tmp/marrow_pd_countstar.parquet")
    _write_wide(path, 2000, 500)
    var plan = _wide_scan(path).aggregate(keys=[], aggs=[count_star()])
    assert_equal(_scan_columns_of(plan.optimize()), "x")
    var result = plan.execute()
    assert_equal(result.num_rows(), 1)
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 2000)
    remove(path)


def test_projection_pushdown_groups_on_the_narrowed_scan() raises:
    """A grouped aggregate over a wide file reads only its key and its input,
    and still computes what the wide plan computes."""
    var path = String("/tmp/marrow_pd_group.parquet")
    _write_wide(path, 100, 50)
    var plan = _wide_scan(path).aggregate(
        keys=[], aggs=[col("y").aggregate("sum").alias("total")]
    )
    assert_equal(_scan_columns_of(plan.optimize()), "y")
    var result = plan.execute()
    assert_equal(result.num_rows(), 1)
    # y = 10 * x for x in 0..99 -> 10 * (99 * 100 / 2)
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 49500)
    remove(path)
