# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""``marrow.read_iceberg`` over the pyiceberg fixture tables.

Every snapshot of every fixture reads back equal to pyiceberg's own scan,
stored beside it as ``expected/<snapshot-id>.parquet``. Rows are compared
sorted, since their order across data files is not defined.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow as ma
from marrow import col, lit

DATA = Path(__file__).parents[3] / "marrow" / "iceberg" / "tests" / "data"
TABLES = ["simple_v1", "simple_v2", "partitioned", "evolved", "no_field_ids"]


def _sorted_rows(table):
    rows = table.to_pylist()
    return sorted(rows, key=lambda r: json.dumps(r, default=str, sort_keys=True))


def _snapshots(name):
    return sorted(p.stem for p in (DATA / name / "expected").glob("*.parquet"))


@pytest.mark.parametrize(
    "name,snapshot",
    [(name, s) for name in TABLES for s in _snapshots(name)],
)
def test_read_iceberg_matches_pyiceberg(name, snapshot):
    expected = pq.read_table(DATA / name / "expected" / f"{snapshot}.parquet")
    got = ma.read_iceberg(DATA / name, snapshot_id=int(snapshot)).to_pyarrow()
    assert got.column_names == expected.column_names
    assert _sorted_rows(got) == _sorted_rows(expected)


def test_read_iceberg_filter_is_pushed_into_the_scan():
    plan = ma.read_iceberg(DATA / "partitioned").filter(col("id") > lit(9))
    optimized = plan.optimize()
    assert "IcebergScan" in str(optimized)
    assert "pruned by 1" in str(optimized)
    ids = sorted(optimized.to_pyarrow().column("id").to_pylist())
    assert ids == [10, 11]


def test_read_iceberg_in_sql():
    t = ma.read_iceberg(DATA / "partitioned")
    out = ma.sql(
        "SELECT category, COUNT(*) AS n FROM t GROUP BY category", t=t
    ).collect()
    counts = {r["category"]: r["n"] for r in out.to_pylist()}
    assert counts == {"a": 4, "b": 4, None: 4}


def test_read_iceberg_sql_still_takes_batches():
    batch = ma.record_batch({"x": ma.array([1, 2, 3], type=ma.int64())})
    out = ma.sql("SELECT x FROM t WHERE x > 1", t=batch).collect()
    assert [r["x"] for r in out.to_pylist()] == [2, 3]


def test_read_iceberg_unknown_table():
    with pytest.raises(Exception, match="neither a metadata file"):
        ma.read_iceberg(DATA / "nope")
