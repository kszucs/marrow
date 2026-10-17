# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`scan_iceberg` — an Iceberg table as a plan source, an `IcebergScan`.

Over the pyiceberg fixtures in `marrow/iceberg/tests/data`: a filter above
it, pushed into it as a pruner, `ColumnPruning` narrowing it, and time
travel to an older snapshot and schema.
"""

from std.os.path import join
from std.python import Python
from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...dtypes import int64
from ...iceberg.catalog import IcebergTable
from ..builders import col, lit, scan_iceberg
from ..logical import IcebergScan
from ..optimizer import AllRules, ColumnPruning, ScanPruning
from ...utils.testing import ScratchDir

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


def test_iceberg_scan_reads_its_manifests_only_when_it_runs() raises:
    """Building, printing and optimizing a plan read nothing: a snapshot
    whose manifest list is gone fails only once the plan is explained or
    executed."""
    var table = IcebergTable.open(DATA + "partitioned")
    for i in range(len(table.metadata.snapshots)):
        table.metadata.snapshots[i].manifest_list = (
            DATA + "partitioned/metadata/no-such-manifest-list.avro"
        )
    var plan = scan_iceberg(table^).filter(col("id", int64) > lit(1, int64))
    var optimized = plan.optimize[AllRules]()
    _ = String(optimized)
    with assert_raises():
        _ = optimized.explain()
    with assert_raises():
        _ = optimized.execute()


def test_iceberg_scan_errors_name_the_data_file() raises:
    """A data file that cannot be read fails the scan with its path."""
    with ScratchDir() as dir:
        var root = join(dir, "table")
        var shutil = Python.import_module("shutil")
        var glob = Python.import_module("glob")
        _ = shutil.copytree(DATA + "simple_v2", root)
        for path in glob.glob(join(root, "data", "*.parquet")):
            var f = Python.import_module("builtins").open(path, "wb")
            _ = f.write(Python.import_module("builtins").bytes(16))
            _ = f.close()
        try:
            with assert_raises(contains=join(root, "data")):
                _ = scan_iceberg(root).execute()
        finally:
            # The scratch directory removes files, not a copied tree.
            _ = shutil.rmtree(root)
