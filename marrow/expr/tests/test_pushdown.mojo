"""Predicate pushdown into ParquetScan: row-group skipping via column stats.

`filter()` over a `parquet_scan` pushes the predicate into the scan (which prunes
row groups) while keeping the `Filter` for exact results. Correctness never
depends on pruning, so we check both the pruning decision and that the executed
result matches an exact filter."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.dtypes import Int64Type, int64, field
from marrow.schema import Schema, schema
from marrow.scalars import AnyScalar
from marrow.parquet import read_table, read_statistics
from marrow.expr.relations import (
    ParquetScan,
    parquet_scan,
    AnyRelation,
    Filter,
    execute,
    RELATION_PARQUET_SCAN,
)
from marrow.expr.dynamic import col, lit
from marrow.expr.values import AnyValue
from marrow.expr.pruning import PruneStats


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
    var stats = read_statistics(path)
    assert_equal(len(stats), 3)
    var pred = AnyValue(col("x") > lit[Int64Type](Int64(1500)))
    var sch = schema([field("x", int64)])
    var keep = List[Bool]()
    for rg in range(len(stats)):
        var mins = List[Optional[AnyScalar]]()
        var maxs = List[Optional[AnyScalar]]()
        mins.append(stats[rg][0].min.copy())
        maxs.append(stats[rg][0].max.copy())
        var ps = PruneStats(Schema(copy=sch), mins^, maxs^)
        keep.append(pred.prune_bound(ps).maybe_true)
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
    var scan = f[].input.downcast[ParquetScan]()
    assert_true(Bool(scan[].predicate))  # predicate was pushed down


def test_pushdown_end_to_end() raises:
    var path = String("/tmp/marrow_pd_e2e.parquet")
    _write_sorted(path, 3000, 1000)
    var sch = schema([field("x", int64)])
    # RG0 is pruned; the Filter still yields exactly x > 1500
    var plan = parquet_scan(path, sch).filter(
        col("x") > lit[Int64Type](Int64(1500))
    )
    var result = execute(plan)
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
    var result = execute(plan)
    assert_equal(result.num_rows(), 2499)  # 7501..9999
    assert_equal(result.columns[0].copy().as_int64()[0].value(), 7501)
    assert_equal(result.columns[0].copy().as_int64()[2498].value(), 9999)
    remove(path)


def test_pushdown_prunes_all_groups() raises:
    # a predicate no row can satisfy -> every group pruned -> empty result
    var path = String("/tmp/marrow_pd_none.parquet")
    _write_sorted(path, 3000, 1000)
    var sch = schema([field("x", int64)])
    var plan = parquet_scan(path, sch).filter(
        col("x") > lit[Int64Type](Int64(100000))
    )
    var result = execute(plan)
    assert_equal(result.num_rows(), 0)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
