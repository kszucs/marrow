# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Typed extension types over `Field` metadata: the canonical types, their
storage validation, and a round trip through an IPC file read by PyArrow."""

from std.os.path import join
from std.python import Python
from std.testing import assert_equal, assert_false, assert_true

from ..arrays import DynArray
from ..builders import FixedSizeBinaryBuilder, array
from ..dtypes import (
    EXTENSION_METADATA,
    EXTENSION_NAME,
    Field,
    field,
    fixed_size_binary_,
    int8,
    int64,
    large_string,
    string,
    string_view,
)
from ..errors import InvalidError
from ..extensions import Bool8Type, JsonType, UuidType
from ..formats.ipc import RecordBatchFileWriter, read_ipc_file
from ..schema import Schema
from ..tabular import RecordBatch
from ..utils.testing import ScratchDir


def test_extension_uuid_roundtrip() raises:
    var f = field("id", fixed_size_binary_(16)).with_extension(UuidType())
    assert_equal(f.metadata[EXTENSION_NAME], "arrow.uuid")
    assert_equal(f.metadata[EXTENSION_METADATA], "")
    assert_equal(f.extension_name().value(), "arrow.uuid")
    assert_true(Bool(f.extension[UuidType]()))
    assert_true(f.dtype == fixed_size_binary_(16))


def test_extension_json_roundtrip() raises:
    var f = field("doc", string).with_extension(JsonType())
    assert_equal(f.extension_name().value(), "arrow.json")
    assert_true(Bool(f.extension[JsonType]()))


def test_extension_bool8_roundtrip() raises:
    var f = field("flag", int8).with_extension(Bool8Type())
    assert_equal(f.extension_name().value(), "arrow.bool8")
    assert_true(Bool(f.extension[Bool8Type]()))


def test_extension_json_storages() raises:
    var storages = [
        string.to_dyn(),
        large_string.to_dyn(),
        string_view.to_dyn(),
    ]
    for ref storage in storages:
        var f = field("doc", storage.copy()).with_extension(
            JsonType(storage.copy())
        )
        var json = f.extension[JsonType]()
        assert_true(json.value().storage == storage)
        assert_true(f.dtype == storage)


def test_extension_json_accepts_empty_object_metadata() raises:
    var f = field("doc", string)
    f.metadata[EXTENSION_NAME] = "arrow.json"
    f.metadata[EXTENSION_METADATA] = "{}"
    assert_true(Bool(f.extension[JsonType]()))


def test_extension_without_extension_is_none() raises:
    var f = field("n", int64)
    assert_false(Bool(f.extension_name()))
    assert_false(Bool(f.extension[UuidType]()))


def test_extension_other_name_is_none() raises:
    var f = field("id", fixed_size_binary_(16)).with_extension(UuidType())
    assert_false(Bool(f.extension[JsonType]()))
    assert_false(Bool(f.extension[Bool8Type]()))


def test_extension_wrong_storage_raises() raises:
    var raised = 0
    try:
        _ = field("id", fixed_size_binary_(8)).with_extension(UuidType())
    except e:
        raised += 1
    try:
        _ = field("doc", int64).with_extension(Bool8Type())
    except e:
        raised += 1
    assert_equal(raised, 2)

    # A field carrying the name over the wrong storage fails on read.
    var f = field("id", fixed_size_binary_(8))
    f.metadata[EXTENSION_NAME] = "arrow.uuid"
    var read_raised = False
    try:
        _ = f.extension[UuidType]()
    except e:
        read_raised = True
    assert_true(read_raised)

    var bad_json = False
    try:
        _ = JsonType(int64)
    except e:
        bad_json = True
    assert_true(bad_json)


def _extension_batch() raises -> RecordBatch:
    var ids = FixedSizeBinaryBuilder(16)
    ids.append("0123456789abcdef".as_bytes())
    ids.append_null()
    var fields: List[Field] = [
        field("u", fixed_size_binary_(16)).with_extension(UuidType()),
        field("j", string).with_extension(JsonType()),
        field("b", int8).with_extension(Bool8Type()),
    ]
    var columns = List[DynArray]()
    columns.append(ids.finish())
    columns.append(array(["{}", "[1]"]))
    columns.append(array([1, 0], int8))
    return RecordBatch(Schema(fields=fields^), columns^)


def _write_extension_file(path: String) raises:
    var batch = _extension_batch()
    var writer = RecordBatchFileWriter(path, batch.schema)
    writer.write(batch)
    writer.close()


def test_extension_ipc_file_roundtrip() raises:
    with ScratchDir() as dir:
        var path = join(dir, "ext.arrow")
        _write_extension_file(path)
        var got = read_ipc_file(path)
        assert_equal(len(got), 1)
        ref schema = got[0].schema
        assert_true(Bool(schema.fields[0].extension[UuidType]()))
        assert_true(Bool(schema.fields[1].extension[JsonType]()))
        assert_true(Bool(schema.fields[2].extension[Bool8Type]()))
        assert_true(got[0] == _extension_batch())


def test_extension_pyarrow_reads_ipc_file() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var path = join(dir, "ext.arrow")
        _write_extension_file(path)
        var schema = pa.ipc.open_file(path).schema
        assert_true(Bool(py=schema.field("u").type.equals(pa.uuid())))
        assert_true(Bool(py=schema.field("j").type.equals(pa.json_())))
        assert_true(Bool(py=schema.field("b").type.equals(pa.bool8())))
