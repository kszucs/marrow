# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`scan_json` — NDJSON as a plan source, through `ExternalScan`.

The scan runs in a plan like any other: a filter above it, `ColumnPruning`
narrowing it, and a path bound per execution by a parameter.
"""

from std.os.path import join
from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import field, int64, string
from ...expr import ColumnPruning, ExternalScan, col, lit, param
from ...scalars import StringScalar
from ...schema import Schema
from ...utils.testing import ScratchDir

from ..scan import scan_json


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def _rows() -> String:
    return (
        '{"id": 1, "name": "a", "note": "x"}\n'
        + '{"id": 2, "name": "b", "note": "y"}\n'
        + '{"id": 3, "name": "c"}\n'
    )


def _schema() -> Schema:
    return Schema(
        fields=[
            field("id", int64),
            field("name", string),
            field("note", string),
        ]
    )


def test_scan_json_executes_through_a_filter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _rows())
        var out = (
            scan_json(path, _schema()).filter(col("id", int64) > lit(1, int64))
        ).execute()
        assert_equal(out.num_rows(), 2)
        assert_true(out.column("id") == array([2, 3], int64).to_dyn())


def test_scan_json_prints_as_an_external_scan() raises:
    var plan = scan_json("rows.jsonl", _schema())
    assert_true(plan.isa[ExternalScan]())
    assert_equal(String(plan), "ExternalScan[json](rows.jsonl)")


def test_scan_json_column_pruning_narrows_the_read() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _rows())
        var pruned = ColumnPruning.apply(scan_json(path, _schema()), ["name"])
        assert_true(pruned.isa[ExternalScan]())
        assert_equal(len(pruned.schema().fields), 1)
        # The narrowed scan skips `id` and `note` in every row carrying them.
        var out = pruned.execute()
        assert_equal(out.num_columns(), 1)
        assert_equal(out.num_rows(), 3)


def test_scan_json_path_is_a_parameter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _rows())
        var plan = scan_json(param("src", string), _schema())
        assert_equal(len(plan.params()), 1)
        var out = plan.execute(bindings={"src": StringScalar(path).to_dyn()})
        assert_equal(out.num_rows(), 3)
