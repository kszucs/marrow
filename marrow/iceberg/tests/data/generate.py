# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Generate the Iceberg fixture tables under this directory.

pyiceberg is not a marrow dependency; run this in a throwaway environment:

    uvx --with 'pyiceberg[sql-sqlite,pyarrow]' python marrow/iceberg/tests/data/generate.py

Tables are written under WAREHOUSE and then copied here, so every path the
metadata records starts with WAREHOUSE. The reader resolves a moved table by
replacing the recorded `location` with the directory it was opened from.
"""

import shutil
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, DayTransform, IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
)

WAREHOUSE = Path("/tmp/marrow-iceberg-fixtures")
HERE = Path(__file__).parent

SIMPLE = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "name", StringType()),
    NestedField(3, "price", DoubleType()),
    NestedField(4, "ts", TimestampType()),
    NestedField(5, "tstz", TimestamptzType()),
    NestedField(6, "d", DateType()),
    NestedField(7, "amount", DecimalType(9, 2)),
    NestedField(8, "flag", BooleanType()),
)


def simple_batch(ids):
    n = len(ids)
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "name": pa.array([f"name-{i}" if i % 3 else None for i in ids]),
            "price": pa.array([i * 1.5 for i in ids], pa.float64()),
            "ts": pa.array(
                [datetime(2024, 1, 1 + i % 28, i % 24) for i in ids],
                pa.timestamp("us"),
            ),
            "tstz": pa.array(
                [datetime(2024, 1, 1 + i % 28, tzinfo=timezone.utc) for i in ids],
                pa.timestamp("us", "UTC"),
            ),
            "d": pa.array([date(2024, 1 + i % 12, 1) for i in ids], pa.date32()),
            "amount": pa.array(
                [Decimal(i) / 4 for i in ids], pa.decimal128(9, 2)
            ),
            "flag": pa.array([i % 2 == 0 for i in ids]),
        },
        schema=SIMPLE.as_arrow(),
    ).slice(0, n)


def simple(catalog, name, version):
    """Unpartitioned, two appends: two snapshots, two data files."""
    t = catalog.create_table(
        f"db.{name}", SIMPLE, properties={"format-version": str(version)}
    )
    t.append(simple_batch(list(range(0, 10))))
    t.append(simple_batch(list(range(10, 20))))


def partitioned(catalog):
    """identity(category), day(ts) and bucket[2](id)."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "category", StringType()),
        NestedField(3, "ts", TimestampType()),
        NestedField(4, "value", DoubleType()),
    )
    spec = PartitionSpec(
        PartitionField(2, 1000, IdentityTransform(), "category"),
        PartitionField(3, 1001, DayTransform(), "ts_day"),
        PartitionField(1, 1002, BucketTransform(2), "id_bucket"),
    )
    t = catalog.create_table("db.partitioned", schema, partition_spec=spec)
    ids = list(range(12))
    t.append(
        pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "category": pa.array(["a", "b", None][i % 3] for i in ids),
                "ts": pa.array(
                    [datetime(2024, 3, 1 + i % 2, i) for i in ids],
                    pa.timestamp("us"),
                ),
                "value": pa.array([float(i) for i in ids]),
            },
            schema=schema.as_arrow(),
        )
    )


def evolved(catalog):
    """Rename, add, drop, promote, and a struct gaining a child, across two
    appends, so the two data files disagree with the current schema."""
    schema = Schema(
        NestedField(1, "a", IntegerType()),
        NestedField(2, "b", FloatType()),
        NestedField(3, "c", StringType()),
        NestedField(4, "gone", StringType()),
        NestedField(
            5,
            "s",
            StructType(
                NestedField(6, "x", IntegerType()),
                NestedField(7, "y", StringType()),
            ),
        ),
    )
    t = catalog.create_table("db.evolved", schema)
    t.append(
        pa.table(
            {
                "a": pa.array([1, 2, None], pa.int32()),
                "b": pa.array([1.5, None, 3.5], pa.float32()),
                "c": ["one", "two", None],
                "gone": ["x", "y", "z"],
                "s": [{"x": 1, "y": "p"}, None, {"x": 3, "y": None}],
            },
            schema=schema.as_arrow(),
        )
    )
    with t.update_schema() as u:
        u.rename_column("c", "c2")
        u.update_column("a", LongType())
        u.update_column("b", DoubleType())
        u.delete_column("gone")
        u.add_column("e", LongType())
        u.add_column(("s", "z"), DoubleType())
    t.append(
        pa.table(
            {
                "a": pa.array([4], pa.int64()),
                "b": pa.array([4.5], pa.float64()),
                "c2": ["four"],
                "s": [{"x": 4, "y": "q", "z": 0.25}],
                "e": pa.array([40], pa.int64()),
            },
            schema=t.schema().as_arrow(),
        )
    )


def no_field_ids(catalog, scratch):
    """A Parquet file written without field ids, added as-is: readable only
    through the table's default name mapping."""
    path = scratch / "plain.parquet"
    pq.write_table(
        pa.table({"k": pa.array([1, 2, 3], pa.int64()), "v": ["a", "b", "c"]}),
        path,
    )
    schema = Schema(
        NestedField(1, "k", LongType()),
        NestedField(2, "v", StringType()),
    )
    t = catalog.create_table("db.no_field_ids", schema)
    t.add_files([str(path)])


def expected(table, root):
    """pyiceberg's answer for every snapshot, as plain Parquet:
    `expected/<snapshot-id>.parquet`. Row order across data files is not
    defined, so compare sorted."""
    out = root / "expected"
    out.mkdir()
    for snapshot in table.snapshots():
        result = table.scan(snapshot_id=snapshot.snapshot_id).to_arrow()
        pq.write_table(result, out / f"{snapshot.snapshot_id}.parquet")


def main():
    shutil.rmtree(WAREHOUSE, ignore_errors=True)
    WAREHOUSE.mkdir(parents=True)
    catalog = SqlCatalog(
        "fixtures",
        uri=f"sqlite:///{WAREHOUSE}/catalog.db",
        warehouse=f"file://{WAREHOUSE}",
    )
    catalog.create_namespace("db")
    simple(catalog, "simple_v1", 1)
    simple(catalog, "simple_v2", 2)
    partitioned(catalog)
    evolved(catalog)
    scratch = WAREHOUSE / "db" / "no_field_ids" / "data"
    scratch.mkdir(parents=True)
    no_field_ids(catalog, scratch)

    for ident in catalog.list_tables("db"):
        expected(catalog.load_table(ident), WAREHOUSE / "db" / ident[-1])

    for table in sorted((WAREHOUSE / "db").iterdir()):
        target = HERE / table.name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(table, target)
        print(table.name, sum(1 for _ in target.rglob("*") if _.is_file()), "files")


if __name__ == "__main__":
    main()
