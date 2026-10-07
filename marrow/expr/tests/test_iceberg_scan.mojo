# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`scan_iceberg` — an Iceberg table as a plan source, an `IcebergScan`.

Over the pyiceberg fixtures in `marrow/iceberg/tests/data`: a filter above
it, pushed into it as a pruner, `ColumnPruning` narrowing it, and time
travel to an older snapshot and schema.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import int64
from ...iceberg.catalog import IcebergTable
from ..builders import col, lit, scan_iceberg
from ..logical import IcebergScan
from ..optimizer import ColumnPruning, ScanPruning

comptime DATA = "marrow/iceberg/tests/data/"


def test_iceberg_scan_reads_the_current_snapshot() raises:
    var plan = scan_iceberg(DATA + "partitioned")
    assert_true(plan.isa[IcebergScan]())
    var out = plan.execute()
    assert_equal(out.num_rows(), 12)
    assert_equal(out.num_columns(), 4)


def test_iceberg_scan_filter_is_pushed_and_still_applied() raises:
    var plan = (
        scan_iceberg(DATA + "partitioned")
        .filter(col("id", int64) > lit(9, int64))
        .optimize[ScanPruning]()
    )
    assert_true(String(plan).find("pruned by 1") >= 0, String(plan))
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    var ids = out.column("id").copy()
    assert_true(
        ids == array([10, 11], int64).to_dyn()
        or ids == array([11, 10], int64).to_dyn()
    )


def test_iceberg_scan_pruning_skips_every_file() raises:
    """A predicate no file's bounds admit reads nothing, and still answers."""
    var plan = (
        scan_iceberg(DATA + "partitioned")
        .filter(col("id", int64) > lit(100, int64))
        .optimize[ScanPruning]()
    )
    assert_equal(plan.execute().num_rows(), 0)


def test_iceberg_scan_column_pruning_narrows_the_read() raises:
    var pruned = ColumnPruning.apply(scan_iceberg(DATA + "simple_v2"), ["id"])
    assert_true(pruned.isa[IcebergScan]())
    assert_equal(len(pruned.schema().fields), 1)
    var out = pruned.execute()
    assert_equal(out.num_columns(), 1)
    assert_equal(out.num_rows(), 20)


def test_iceberg_scan_time_travel_reads_the_old_schema() raises:
    var table = IcebergTable.open(DATA + "evolved")
    var first = table.metadata.snapshot_log[0].snapshot_id
    var old = scan_iceberg(table.copy(), first)
    assert_true(old.schema().names() == ["a", "b", "c", "gone", "s"])
    assert_equal(old.execute().num_rows(), 3)
    var now = scan_iceberg(table^)
    assert_true(now.schema().names() == ["a", "b", "c2", "s", "e"])
    assert_equal(now.execute().num_rows(), 4)
