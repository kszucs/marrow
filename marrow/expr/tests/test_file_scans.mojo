# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`scan_json` and `scan_ipc` — JSON and Arrow IPC files as plan sources,
a `JsonScan` and an `IpcScan`.

Each runs in a plan like any other: a filter above it, `ColumnPruning`
narrowing it, a path bound per execution by a parameter.
"""

from std.os.path import join
from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import field, int64, string
from ...ipc import write_ipc_file
from ...scalars import StringScalar
from ...schema import Schema
from ...tabular import RecordBatch, record_batch
from ...utils.testing import ScratchDir
from ..builders import col, lit, param, scan_ipc, scan_json
from ..logical import IpcScan, JsonScan
from ..optimizer import ColumnPruning


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def _json_rows() -> String:
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


def _ipc_batch() raises -> RecordBatch:
    var names: List[Optional[String]] = [String("a"), String("b"), String("c")]
    var notes: List[Optional[String]] = [String("x"), String("y"), None]
    return record_batch(
        [array([1, 2, 3], int64).to_dyn(), array(names^), array(notes^)],
        names=["id", "name", "note"],
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_scan_json_executes_through_a_filter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _json_rows())
        var out = (
            scan_json(path, _schema()).filter(col("id", int64) > lit(1, int64))
        ).execute()
        assert_equal(out.num_rows(), 2)
        assert_true(out.column("id") == array([2, 3], int64).to_dyn())


def test_scan_json_prints_as_a_json_scan() raises:
    var plan = scan_json("rows.jsonl", _schema())
    assert_true(plan.isa[JsonScan]())
    assert_equal(String(plan), "JsonScan(rows.jsonl)")


def test_scan_json_reads_several_files_in_order() raises:
    with ScratchDir() as dir:
        var first = join(dir, "a.jsonl")
        var second = join(dir, "b.jsonl")
        _write(first, _json_rows())
        _write(second, '{"id": 4, "name": "d"}\n')
        var plan = scan_json([first, second], _schema())
        assert_equal(
            String(plan), String("JsonScan(", first, ", ", second, ")")
        )
        var out = plan.execute()
        assert_true(out.column("id") == array([1, 2, 3, 4], int64).to_dyn())


def test_scan_json_column_pruning_narrows_the_read() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _json_rows())
        var pruned = ColumnPruning.apply(scan_json(path, _schema()), ["name"])
        assert_true(pruned.isa[JsonScan]())
        assert_equal(len(pruned.schema().fields), 1)
        # The narrowed scan skips `id` and `note` in every row carrying them.
        var out = pruned.execute()
        assert_equal(out.num_columns(), 1)
        assert_equal(out.num_rows(), 3)


def test_scan_json_path_is_a_parameter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.jsonl")
        _write(path, _json_rows())
        var plan = scan_json(param("src", string), _schema())
        assert_equal(len(plan.params()), 1)
        var out = plan.execute(bindings={"src": StringScalar(path).to_dyn()})
        assert_equal(out.num_rows(), 3)


# ---------------------------------------------------------------------------
# Arrow IPC
# ---------------------------------------------------------------------------


def test_scan_ipc_executes_through_a_filter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.arrow")
        write_ipc_file(path, [_ipc_batch(), _ipc_batch()])
        var out = (
            scan_ipc(path, _schema()).filter(col("id", int64) > lit(1, int64))
        ).execute()
        assert_equal(out.num_rows(), 4)
        assert_true(out.column("id") == array([2, 3, 2, 3], int64).to_dyn())


def test_scan_ipc_prints_as_an_ipc_scan() raises:
    var plan = scan_ipc("rows.arrow", _schema())
    assert_true(plan.isa[IpcScan]())
    assert_equal(String(plan), "IpcScan(rows.arrow)")


def test_scan_ipc_column_pruning_selects_by_name() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.arrow")
        write_ipc_file(path, [_ipc_batch()])
        var pruned = ColumnPruning.apply(scan_ipc(path, _schema()), ["note"])
        assert_true(pruned.isa[IpcScan]())
        var out = pruned.execute()
        assert_equal(out.num_columns(), 1)
        assert_equal(out.schema.fields[0].name, "note")
        assert_equal(out.num_rows(), 3)


def test_scan_ipc_path_is_a_parameter() raises:
    with ScratchDir() as dir:
        var path = join(dir, "rows.arrow")
        write_ipc_file(path, [_ipc_batch()])
        var plan = scan_ipc(param("src", string), _schema())
        assert_equal(len(plan.params()), 1)
        var out = plan.execute(bindings={"src": StringScalar(path).to_dyn()})
        assert_equal(out.num_rows(), 3)
