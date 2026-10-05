# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro reader/writer bindings, verified against fastavro as the oracle.

PyArrow has no Avro support, so an independent implementation reads every
file marrow writes and writes files marrow reads: a mistake mirrored in
marrow's encoder and decoder would survive a marrow-only round trip, but not
this. Values are compared as fastavro represents them."""

import datetime
import decimal
import struct
import uuid
from pathlib import Path

import fastavro
import pyarrow as pa
import pytest

import marrow.avro as mavro

DATA = Path(__file__).parents[3] / "marrow" / "avro" / "tests" / "data"


CODECS = ["null", "deflate", "snappy", "zstandard"]


def _to_pa(marrow_table):
    return pa.RecordBatchReader.from_stream(marrow_table).read_all()


def _uuid_bytes(value):
    """`value` with each uuid as its 16 bytes. fastavro returns a uuid stored
    as text as a `UUID` and one stored as `fixed(16)` as bytes; marrow reads
    both as Arrow's `arrow.uuid`, which pyarrow returns as a `UUID`."""
    if isinstance(value, uuid.UUID):
        return value.bytes
    if isinstance(value, dict):
        return {k: _uuid_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_uuid_bytes(v) for v in value]
    return value


def _avro_value(value, t):
    """`value`, read from an Arrow column of type `t`, as fastavro represents
    the Avro value it was written as."""
    if value is None:
        return None
    if pa.types.is_map(t):
        items = [(k, _avro_value(v, t.item_type)) for k, v in value]
        if pa.types.is_string(t.key_type):
            return dict(items)
        # Iceberg's form for a map whose keys are not strings.
        return [{"key": k, "value": v} for k, v in items]
    if pa.types.is_list(t):
        return [_avro_value(v, t.value_type) for v in value]
    if pa.types.is_struct(t):
        return {f.name: _avro_value(value[f.name], f.type) for f in t}
    if pa.types.is_interval(t):
        # fastavro has no duration type: the 12 bytes as written.
        months, days, nanos = value
        return struct.pack("<III", months, days, nanos // 1_000_000)
    return value


def _avro_rows(table):
    """The rows of an Arrow table as fastavro reads them back. fastavro has no
    nanosecond timestamp, so those columns compare as their int64 values."""
    columns = [
        c.cast(pa.int64())
        if pa.types.is_timestamp(c.type) and c.type.unit == "ns"
        else c
        for c in table.columns
    ]
    rows = pa.table(columns, names=table.column_names).to_pylist()
    return _uuid_bytes(
        [
            {f.name: _avro_value(row[f.name], f.type) for f in table.schema}
            for row in rows
        ]
    )


def _fastavro_read(path):
    with open(path, "rb") as f:
        reader = fastavro.reader(f)
        return [_uuid_bytes(r) for r in reader], reader.metadata


def test_read_fixture():
    t = _to_pa(mavro.read_table(DATA / "alltypes_plain.avro"))
    assert t.num_rows == 8
    assert t.column("id").to_pylist() == [4, 5, 6, 7, 2, 3, 0, 1]
    assert t.schema.field("timestamp_col").type == pa.timestamp("us", tz="UTC")


def test_read_columns():
    t = _to_pa(
        mavro.read_table(DATA / "alltypes_plain.avro", columns=["bigint_col", "id"])
    )
    assert t.column_names == ["bigint_col", "id"]


def test_field_ids_and_header_metadata():
    t = _to_pa(mavro.read_table(DATA / "manifest-list-v2-1.avro"))
    assert t.schema.field("manifest_path").metadata == {b"field_id": b"500"}
    assert t.schema.metadata[b"format-version"] == b"2"


@pytest.mark.parametrize("codec", ["null", "deflate", "snappy", "zstandard"])
def test_roundtrip(tmp_path, codec):
    table = pa.table(
        {
            "i": pa.array([1, None, 3], pa.int64()),
            "s": pa.array(["a", None, "ccc"]),
            "l": pa.array([[1.5], None, []], pa.list_(pa.float64())),
            "m": pa.array([[("k", 1)], [], None], pa.map_(pa.string(), pa.int32())),
            "st": pa.array(
                [{"x": True}, None, {"x": None}], pa.struct([("x", pa.bool_())])
            ),
        }
    )
    path = tmp_path / "t.avro"
    mavro.write_table(table, path, codec=codec)
    assert _to_pa(mavro.read_table(path)).equals(table)


def test_invalid_codec(tmp_path):
    with pytest.raises(NotImplementedError, match="unknown codec 'lz4'"):
        mavro.write_table(pa.table({"a": [1]}), tmp_path / "t.avro", codec="lz4")


# ---------------------------------------------------------------------------
# Cross-checks against fastavro
# ---------------------------------------------------------------------------


def _wide_table():
    """Every type marrow writes, with nulls, empties and nesting."""
    utc = datetime.timezone.utc
    return pa.table(
        {
            "i": pa.array([1, None, -3, 2**31 - 1], pa.int32()),
            "l": pa.array([2**62, -1, None, 0], pa.int64()),
            "f": pa.array([1.5, None, float("inf"), -0.0], pa.float32()),
            "d": pa.array([0.1, 2.5, None, -1e300], pa.float64()),
            "b": pa.array([True, False, None, True]),
            "s": pa.array(["a", "héllo", None, ""]),
            "ls": pa.array(["x", None, "y", "z"], pa.large_string()),
            "dict": pa.array(["p", "q", None, "p"]).dictionary_encode(),
            "y": pa.array([b"\x00\x01", b"", None, b"z"], pa.binary()),
            "fx": pa.array([b"abcd", b"efgh", None, b"ijkl"], pa.binary(4)),
            "dt": pa.array([0, 19000, None, -1], pa.date32()),
            "tm": pa.array([0, 1000, None, 86399999], pa.time32("ms")),
            "tu": pa.array([0, 1, None, 86399999999], pa.time64("us")),
            "ts": pa.array(
                [
                    datetime.datetime(2020, 1, 2, 3, 4, 5, 6, tzinfo=utc),
                    None,
                    datetime.datetime(1960, 1, 1, tzinfo=utc),
                    datetime.datetime(2038, 1, 19, 3, 14, 8, tzinfo=utc),
                ],
                pa.timestamp("us", tz="UTC"),
            ),
            "tl": pa.array([0, 1, None, -1], pa.timestamp("ms")),
            "tn": pa.array([0, 1, None, -1], pa.timestamp("ns", tz="UTC")),
            "dec": pa.array(
                [
                    decimal.Decimal("1.23"),
                    decimal.Decimal("-99999.99"),
                    None,
                    decimal.Decimal("0.00"),
                ],
                pa.decimal128(7, 2),
            ),
            "big": pa.array(
                [
                    decimal.Decimal("1" * 50),
                    None,
                    decimal.Decimal("-1"),
                    decimal.Decimal("0"),
                ],
                pa.decimal256(50, 0),
            ),
            "iv": pa.array(
                [
                    pa.MonthDayNano([1, 2, 3_000_000]),
                    None,
                    pa.MonthDayNano([0, 0, 0]),
                    pa.MonthDayNano([12, 31, 999_000_000]),
                ],
                pa.month_day_nano_interval(),
            ),
            "li": pa.array([[1, None], [], None, [4]], pa.list_(pa.int64())),
            "m": pa.array(
                [[("k", 1)], [], None, [("a", None), ("b", 2)]],
                pa.map_(pa.string(), pa.int64()),
            ),
            "mi": pa.array(
                [[(1, "one")], None, [], [(2, None)]],
                pa.map_(pa.int32(), pa.string()),
            ),
            "st": pa.array(
                [{"x": 1, "y": "a"}, None, {"x": None, "y": None}, {"x": 3, "y": "c"}],
                pa.struct([("x", pa.int32()), ("y", pa.string())]),
            ),
            "nested": pa.array(
                [[{"p": [1, 2]}], None, [{"p": None}], []],
                pa.list_(pa.struct([("p", pa.list_(pa.int32()))])),
            ),
        }
    )


@pytest.mark.parametrize("codec", CODECS)
def test_fastavro_reads_what_marrow_writes(tmp_path, codec):
    table = _wide_table()
    path = tmp_path / "t.avro"
    mavro.write_table(table, path, codec=codec)
    records, metadata = _fastavro_read(path)
    assert metadata["avro.codec"] == codec
    assert records == _avro_rows(table)


# An Iceberg-shaped schema: field ids on every field, element/key/value ids,
# optional fields as ["null", T] with a null default, a map with int keys as
# an array of records, and the logical types Iceberg uses.
ICEBERG_SCHEMA = {
    "type": "record",
    "name": "manifest_entry",
    "fields": [
        {"name": "status", "type": "int", "field-id": 0},
        {
            "name": "snapshot_id",
            "type": ["null", "long"],
            "default": None,
            "field-id": 1,
        },
        {"name": "path", "type": "string", "field-id": 100},
        {
            "name": "format",
            "type": {
                "type": "enum",
                "name": "file_format",
                "symbols": ["AVRO", "ORC", "PARQUET"],
            },
            "field-id": 101,
        },
        {
            "name": "partition",
            "type": {
                "type": "record",
                "name": "r102",
                "fields": [
                    {
                        "name": "day",
                        "type": ["null", {"type": "int", "logicalType": "date"}],
                        "default": None,
                        "field-id": 1000,
                    },
                    {
                        "name": "bucket",
                        "type": ["null", "int"],
                        "default": None,
                        "field-id": 1001,
                    },
                ],
            },
            "field-id": 102,
        },
        {
            "name": "column_sizes",
            "type": [
                "null",
                {
                    "type": "array",
                    "logicalType": "map",
                    "items": {
                        "type": "record",
                        "name": "k117_v118",
                        "fields": [
                            {"name": "key", "type": "int", "field-id": 117},
                            {"name": "value", "type": "long", "field-id": 118},
                        ],
                    },
                },
            ],
            "default": None,
            "field-id": 108,
        },
        {
            "name": "lower_bounds",
            "type": [
                "null",
                {
                    "type": "array",
                    "logicalType": "map",
                    "items": {
                        "type": "record",
                        "name": "k126_v127",
                        "fields": [
                            {"name": "key", "type": "int", "field-id": 126},
                            {"name": "value", "type": "bytes", "field-id": 127},
                        ],
                    },
                },
            ],
            "default": None,
            "field-id": 125,
        },
        {
            "name": "properties",
            "type": [
                "null",
                {
                    "type": "map",
                    "values": ["null", "string"],
                    "key-id": 300,
                    "value-id": 301,
                },
            ],
            "default": None,
            "field-id": 299,
        },
        {
            "name": "split_offsets",
            "type": ["null", {"type": "array", "items": "long", "element-id": 133}],
            "default": None,
            "field-id": 132,
        },
        {
            "name": "price",
            "type": {
                "type": "fixed",
                "name": "fixed_9_2",
                "size": 4,
                "logicalType": "decimal",
                "precision": 9,
                "scale": 2,
            },
            "field-id": 200,
        },
        {
            "name": "ratio",
            "type": {
                "type": "bytes",
                "logicalType": "decimal",
                "precision": 20,
                "scale": 5,
            },
            "field-id": 201,
        },
        {
            "name": "ts",
            "type": {"type": "long", "logicalType": "timestamp-micros"},
            "field-id": 202,
        },
        {
            "name": "local",
            "type": ["null", {"type": "long", "logicalType": "local-timestamp-millis"}],
            "default": None,
            "field-id": 203,
        },
        {
            "name": "t",
            "type": {"type": "long", "logicalType": "time-micros"},
            "field-id": 204,
        },
        {
            "name": "id",
            "type": {
                "type": "fixed",
                "name": "uuid_fixed",
                "size": 16,
                "logicalType": "uuid",
            },
            "field-id": 205,
        },
        {
            "name": "sid",
            "type": {"type": "string", "logicalType": "uuid"},
            "field-id": 206,
        },
        {"name": "score", "type": ["float", "null"], "field-id": 207},
        {"name": "weight", "type": "double", "field-id": 208},
        {"name": "flag", "type": "boolean", "field-id": 209},
    ],
}


def _iceberg_records(n):
    utc = datetime.timezone.utc
    records = []
    for i in range(n):
        u = uuid.UUID(int=(i * 0x9E3779B97F4A7C15) % 2**128)
        records.append(
            {
                "status": i % 3,
                "snapshot_id": None if i % 5 == 0 else 1_000_000_000_000 + i,
                "path": f"s3://bucket/data/{i:08d}-é.parquet",
                "format": ["AVRO", "ORC", "PARQUET"][i % 3],
                "partition": {
                    "day": None
                    if i % 7 == 0
                    else datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 400),
                    "bucket": None if i % 4 == 0 else i % 16,
                },
                "column_sizes": None
                if i % 6 == 0
                else [{"key": k, "value": i * k} for k in range(i % 4)],
                "lower_bounds": None
                if i % 9 == 0
                else [{"key": 1, "value": i.to_bytes(8, "little")}],
                "properties": None
                if i % 8 == 0
                else {"k": None if i % 2 else f"v{i}", "z": "end"},
                "split_offsets": None if i % 3 == 0 else [4, 4 + i],
                "price": decimal.Decimal(i * 37 - 5000).scaleb(-2),
                "ratio": decimal.Decimal((-1) ** i * i * 12345).scaleb(-5),
                "ts": datetime.datetime(2024, 1, 1, tzinfo=utc)
                + datetime.timedelta(microseconds=i * 1_000_003),
                "local": None
                if i % 10 == 0
                else datetime.datetime(2000, 1, 1)
                + datetime.timedelta(milliseconds=i * 61),
                "t": datetime.time(i % 24, i % 60, i % 60, i % 1_000_000),
                "id": u.bytes,
                "sid": u,
                "score": None if i % 11 == 0 else float(i) / 4,
                "weight": i * 0.1,
                "flag": i % 2 == 0,
            }
        )
    return records


@pytest.mark.parametrize("codec", CODECS)
def test_marrow_reads_what_fastavro_writes(tmp_path, codec):
    path = tmp_path / "t.avro"
    records = _iceberg_records(500)
    with open(path, "wb") as f:
        # A small sync interval spreads the rows over many blocks.
        fastavro.writer(
            f,
            fastavro.parse_schema(ICEBERG_SCHEMA),
            records,
            codec=codec,
            sync_interval=2_000,
            metadata={"format-version": "2", "content": "data"},
        )
    expected, _ = _fastavro_read(path)
    table = _to_pa(mavro.read_table(path))
    assert _avro_rows(table) == expected
    assert table.schema.metadata[b"format-version"] == b"2"
    assert table.schema.field("path").metadata == {b"field_id": b"100"}
    sizes = table.schema.field("column_sizes").type
    assert sizes.key_field.metadata == {b"field_id": b"117"}
    props = table.schema.field("properties").type
    assert props.item_field.metadata == {b"field_id": b"301"}


@pytest.mark.parametrize("codec", CODECS)
def test_iceberg_shape_survives_marrow_rewrite(tmp_path, codec):
    """fastavro writes, marrow reads and writes again, fastavro reads: the
    records and every field id come out as they went in."""
    src, out = tmp_path / "src.avro", tmp_path / "out.avro"
    records = _iceberg_records(300)
    with open(src, "wb") as f:
        fastavro.writer(f, fastavro.parse_schema(ICEBERG_SCHEMA), records, codec=codec)
    expected, _ = _fastavro_read(src)
    mavro.write_table(mavro.read_table(src), out, codec=codec)
    again, _ = _fastavro_read(out)
    assert again == expected
    with open(out, "rb") as f:
        schema = fastavro.reader(f).writer_schema
    ids = {f["name"]: f.get("field-id") for f in schema["fields"]}
    assert ids == {f["name"]: f["field-id"] for f in ICEBERG_SCHEMA["fields"]}


@pytest.mark.parametrize("codec", CODECS)
def test_many_rows_both_ways(tmp_path, codec):
    """Enough rows for many blocks and for each codec's output to outgrow its
    first allocation, through both implementations."""
    n = 50_000
    table = pa.table(
        {
            "id": pa.array(range(n), pa.int64()),
            "name": pa.array(
                [None if i % 13 == 0 else "row-" * (i % 7) + str(i) for i in range(n)]
            ),
            "value": pa.array([float(i) * 0.5 for i in range(n)]),
            "tags": pa.array(
                [[str(i % 10)] * (i % 3) for i in range(n)], pa.list_(pa.string())
            ),
        }
    )
    ours = tmp_path / "ours.avro"
    mavro.write_table(table, ours, codec=codec)
    records, _ = _fastavro_read(ours)
    assert records == _avro_rows(table)

    theirs = tmp_path / "theirs.avro"
    with open(ours, "rb") as f:
        schema = fastavro.reader(f).writer_schema
    with open(theirs, "wb") as f:
        fastavro.writer(f, schema, records, codec=codec)
    assert _to_pa(mavro.read_table(theirs)).equals(table)
