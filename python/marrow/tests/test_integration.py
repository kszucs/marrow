# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`marrow.integration.read_json`, the binding the archery suite drives.

The reader is tested type by type in `marrow/tests/test_integration.mojo`; this checks
what the binding hands back and that pyarrow sees the same values."""

import json

import pyarrow as pa
import pytest

import marrow
import marrow.integration


def _int(name, width=32, nullable=True):
    return {
        "name": name,
        "nullable": nullable,
        "type": {"name": "int", "isSigned": True, "bitWidth": width},
        "children": [],
    }


def _write(tmp_path, document):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    return path


def test_read_json_returns_the_schema_and_the_batches(tmp_path):
    items = {
        "name": "l",
        "nullable": True,
        "type": {"name": "list"},
        "children": [_int("element", nullable=False)],
        "metadata": [{"key": "k", "value": "v"}],
    }
    document = {
        "schema": {"fields": [_int("a", 64), items]},
        "batches": [
            {
                "count": 2,
                "columns": [
                    {"name": "a", "count": 2, "VALIDITY": [1, 0], "DATA": ["7", "0"]},
                    {
                        "name": "l",
                        "count": 2,
                        "VALIDITY": [1, 1],
                        "OFFSET": [0, 2, 3],
                        "children": [
                            {
                                "name": "element",
                                "count": 3,
                                "VALIDITY": [1, 1, 1],
                                "DATA": [1, 2, 3],
                            }
                        ],
                    },
                ],
            }
        ],
    }
    schema, batches = marrow.integration.read_json(_write(tmp_path, document))

    assert isinstance(schema, marrow.Schema)
    element = pa.field("element", pa.int32(), nullable=False)
    expected_schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("l", pa.list_(element), metadata={"k": "v"}),
        ]
    )
    assert pa.schema(schema).equals(expected_schema, check_metadata=True)
    assert len(batches) == 1
    expected = pa.record_batch(
        [pa.array([7, None], pa.int64()), pa.array([[1, 2], [3]], pa.list_(element))],
        schema=expected_schema,
    )
    assert pa.record_batch(batches[0]).equals(expected, check_metadata=True)


def test_read_json_refuses_an_unsupported_type(tmp_path):
    union = {
        "name": "u",
        "nullable": True,
        "type": {"name": "union", "mode": "SPARSE", "typeIds": []},
        "children": [],
    }
    path = _write(tmp_path, {"schema": {"fields": [union]}, "batches": []})
    with pytest.raises(marrow.ArrowNotImplementedError, match="union"):
        marrow.integration.read_json(path)
