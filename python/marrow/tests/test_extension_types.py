# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Extension types survive marrow's interop paths, checked against pyarrow.

An extension type is a field annotation over a storage-typed array
(`ARROW:extension:name` / `ARROW:extension:metadata` field metadata), so every
path that moves a field must keep that metadata.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow as ma
import marrow.parquet as mpq


class Tagged(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.int64(), "marrow.test.tagged")

    def __arrow_ext_serialize__(self):
        return b"v1"

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


@pytest.fixture(scope="module", autouse=True)
def _registered():
    pa.register_extension_type(Tagged())
    yield
    pa.unregister_extension_type("marrow.test.tagged")


def _uuid_array():
    storage = pa.array([b"\x01" * 16, None, b"\x02" * 16], pa.binary(16))
    return pa.ExtensionArray.from_storage(pa.uuid(), storage)


def _tagged_array():
    return pa.ExtensionArray.from_storage(Tagged(), pa.array([1, None, 3]))


def _batch():
    uuid, tagged = _uuid_array(), _tagged_array()
    return pa.record_batch(
        {
            "uuid": uuid,
            "tagged": tagged,
            "struct": pa.StructArray.from_arrays(
                [tagged], fields=[pa.field("t", tagged.type)]
            ),
            "list": pa.ListArray.from_arrays(
                pa.array([0, 1, 2, 3], pa.int32()), tagged
            ),
            "map": pa.MapArray.from_arrays(
                pa.array([0, 1, 2, 3], pa.int32()),
                pa.array(["a", "b", "c"]),
                tagged,
            ),
        }
    )


def _through_marrow(batch):
    return pa.record_batch(ma.record_batch(batch))


def test_cdata_batch_roundtrip():
    batch = _batch()
    assert _through_marrow(batch).equals(batch)
    assert _through_marrow(batch).schema == batch.schema


@pytest.mark.xfail(
    reason="a bare array has no Field to carry the extension metadata",
    strict=True,
)
def test_cdata_bare_array_roundtrip():
    arr = _uuid_array()
    assert pa.array(ma.array(arr)).type == arr.type


@pytest.mark.parametrize("kind", ["file", "stream"])
def test_ipc_roundtrip(tmp_path, kind):
    batch = _batch()
    path = str(tmp_path / f"ext.{kind}")
    if kind == "file":
        ma.write_ipc_file(path, batches=[ma.record_batch(batch)])
        got = ma.read_ipc_file(path)
    else:
        ma.write_ipc_stream(path, batches=[ma.record_batch(batch)])
        got = ma.read_ipc_stream(path)
    assert pa.record_batch(got[0]).schema == batch.schema


def test_ipc_pyarrow_written_marrow_read(tmp_path):
    batch = _batch()
    path = str(tmp_path / "ext.arrow")
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, batch.schema) as w:
        w.write_batch(batch)
    got = ma.read_ipc_file(path)
    assert pa.record_batch(got[0]).schema == batch.schema


def test_parquet_pyarrow_written_marrow_read(tmp_path):
    table = pa.Table.from_batches([_batch()])
    path = str(tmp_path / "ext.parquet")
    pq.write_table(table, path)
    got = pa.RecordBatchReader.from_stream(mpq.read_table(path)).read_all()
    assert got.schema == table.schema


def test_parquet_marrow_roundtrip(tmp_path):
    table = pa.Table.from_batches([_batch()])
    path = str(tmp_path / "ext.parquet")
    mpq.write_table(table, path)
    got = pa.RecordBatchReader.from_stream(mpq.read_table(path)).read_all()
    assert got.schema == table.schema
    # PyArrow itself does not restore an extension type under a map on read.
    assert (
        pq.read_table(path).drop_columns(["map"]).schema
        == table.drop_columns(["map"]).schema
    )
