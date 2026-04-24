"""MarrowTester — registers Marrow as a participant in archery integration tests.

All IPC reading and writing goes through Marrow (never pa.ipc.*).
PyArrow is used only as the comparison oracle in validate().
"""

import json
import os

import marrow as ma
import pyarrow as pa

from archery.integration.tester import Tester, CDataExporter, CDataImporter


# ---------------------------------------------------------------------------
# Arrow JSON integration format → Marrow
# ---------------------------------------------------------------------------


def _json_type_to_ma(type_obj: dict, children_fields: list):
    """Convert Arrow JSON type descriptor to a marrow DataType. Returns None for unsupported types."""
    name = type_obj["name"]
    if name == "bool":
        return ma.bool_()
    if name == "int":
        bw = type_obj["bitWidth"]
        prefix = "int" if type_obj["isSigned"] else "uint"
        return getattr(ma, prefix + str(bw))()
    if name == "floatingpoint":
        return {"HALF": ma.float16(), "SINGLE": ma.float32(), "DOUBLE": ma.float64()}[
            type_obj["precision"]
        ]
    if name == "binary":
        return ma.binary()
    if name == "utf8":
        return ma.string()
    if name == "list":
        child_type = _json_type_to_ma(
            children_fields[0]["type"], children_fields[0].get("children") or []
        )
        return None if child_type is None else ma.list_(child_type)
    if name == "fixedsizelist":
        child_type = _json_type_to_ma(
            children_fields[0]["type"], children_fields[0].get("children") or []
        )
        return None if child_type is None else ma.fixed_size_list_(child_type, type_obj["listSize"])
    if name == "struct":
        ma_fields = [_json_field_to_ma(f) for f in children_fields]
        return None if any(f is None for f in ma_fields) else ma.struct_(ma_fields)
    return None


def _json_field_to_ma(field_obj: dict):
    """Convert Arrow JSON field to a marrow Field. Returns None for unsupported types."""
    if "dictionary" in field_obj:
        return None
    ma_type = _json_type_to_ma(
        field_obj["type"], field_obj.get("children") or []
    )
    if ma_type is None:
        return None
    return ma.field(field_obj["name"], type=ma_type, nullable=field_obj.get("nullable", True))


def _json_col_to_ma(col_obj: dict, field_obj: dict, ma_type) -> object:
    """Convert an Arrow JSON column to a marrow array.

    Dispatches on field_obj["type"]["name"] (the JSON type descriptor) to avoid
    relying on Python class names for marrow DataType objects.
    """
    type_obj = field_obj["type"]
    type_name = type_obj["name"]
    n = col_obj["count"]
    validity = col_obj.get("VALIDITY")
    mask_list = None if validity is None else [v == 0 for v in validity]
    mask = [1 - v for v in validity] if validity else None

    if type_name in ("bool", "int", "floatingpoint"):
        data = col_obj.get("DATA", [])
        data = [int(v) if isinstance(v, str) else v for v in data]
        values = [None if (mask_list and mask_list[i]) else data[i] for i in range(n)]
        return ma.array(values, type=ma_type)

    if type_name == "utf8":
        data = col_obj.get("DATA", [])
        values = [None if (mask_list and mask_list[i]) else data[i] for i in range(n)]
        return ma.array(values, type=ma_type)

    if type_name == "binary":
        data = col_obj.get("DATA", [])
        values = [
            None if (mask_list and mask_list[i]) else bytes.fromhex(v) if v else b""
            for i, v in enumerate(data)
        ]
        return ma.array(values, type=ma_type)

    if type_name == "list":
        offsets = col_obj.get("OFFSET", [0])
        child_json_field = (field_obj.get("children") or [{}])[0]
        child_type = _json_type_to_ma(
            child_json_field["type"], child_json_field.get("children") or []
        )
        child_arr = _json_col_to_ma(col_obj["children"][0], child_json_field, child_type)
        return ma.list_array_from_arrays(offsets, values=child_arr, type=ma_type, mask=mask)

    if type_name == "fixedsizelist":
        child_json_field = (field_obj.get("children") or [{}])[0]
        child_type = _json_type_to_ma(
            child_json_field["type"], child_json_field.get("children") or []
        )
        child_arr = _json_col_to_ma(col_obj["children"][0], child_json_field, child_type)
        return ma.fixed_size_list_array_from_arrays(child_arr, type=ma_type, mask=mask)

    if type_name == "struct":
        child_json_fields = field_obj.get("children") or []
        child_ma_fields = [_json_field_to_ma(f) for f in child_json_fields]
        child_arrs = [
            _json_col_to_ma(
                col_obj["children"][i],
                child_json_fields[i],
                _json_type_to_ma(child_json_fields[i]["type"], child_json_fields[i].get("children") or []),
            )
            for i in range(len(child_json_fields))
        ]
        return ma.struct_array_from_arrays(child_arrs, fields=child_ma_fields, mask=mask)

    raise ValueError(f"Unsupported type in marrow JSON converter: {type_name}")


def _read_json(json_path: os.PathLike) -> dict:
    with open(json_path, "rb") as f:
        return json.loads(f.read())


def _json_to_ma_schema(json_dict: dict) -> list:
    """Return list of supported marrow Fields from JSON schema."""
    return [f for f in (_json_field_to_ma(f) for f in json_dict["schema"]["fields"]) if f is not None]


def _json_to_ma_batch(json_dict: dict, num_batch: int, ma_fields: list):
    """Build a marrow RecordBatch from a JSON batch."""
    batch_obj = json_dict["batches"][num_batch]
    supported_names = {f.name() for f in ma_fields}
    arrays = []
    for col_obj, json_field in zip(batch_obj["columns"], json_dict["schema"]["fields"]):
        if col_obj["name"] not in supported_names:
            continue
        mf = _json_field_to_ma(json_field)
        if mf is None:
            continue
        ma_type = _json_type_to_ma(json_field["type"], json_field.get("children") or [])
        arrays.append(_json_col_to_ma(col_obj, json_field, ma_type))
    return ma.record_batch(arrays, schema=ma.schema(ma_fields))


def _cffi_ptr_to_int(cffi_ptr) -> int:
    import cffi
    return int(cffi.FFI().cast("uintptr_t", cffi_ptr))


# ---------------------------------------------------------------------------
# MarrowTester
# ---------------------------------------------------------------------------


class MarrowTester(Tester):
    """Marrow implementation for archery integration tests."""

    PRODUCER = True
    CONSUMER = True
    C_DATA_SCHEMA_EXPORTER = True
    C_DATA_ARRAY_EXPORTER = True
    C_DATA_SCHEMA_IMPORTER = True
    C_DATA_ARRAY_IMPORTER = True
    name = "Marrow"

    def make_c_data_exporter(self):
        return MarrowCDataExporter()

    def make_c_data_importer(self):
        return MarrowCDataImporter()

    def json_to_file(self, json_path, arrow_path):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        if not ma_fields:
            raise NotImplementedError("no supported columns in this test case")
        n_batches = len(json_dict.get("batches", []))
        empty_rb = ma.record_batch(
            [ma.array([], type=f.type()) for f in ma_fields],
            schema=ma.schema(ma_fields),
        )
        batches = [_json_to_ma_batch(json_dict, i, ma_fields) for i in range(n_batches)]
        ma.write_ipc_file(str(arrow_path), schema=empty_rb, batches=batches)

    def validate(self, json_path, arrow_path, quirks=None):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        if not ma_fields:
            raise NotImplementedError("no supported columns in this test case")
        ma_batches = list(ma.read_ipc_file(str(arrow_path)))
        n_expected = len(json_dict.get("batches", []))
        assert len(ma_batches) == n_expected, (
            f"Expected {n_expected} batches, got {len(ma_batches)}"
        )
        for i, ma_batch in enumerate(ma_batches):
            expected = pa.record_batch(_json_to_ma_batch(json_dict, i, ma_fields))
            result = pa.record_batch(ma_batch)
            assert expected.equals(result), (
                f"Batch {i} mismatch:\n"
                f"  expected schema: {expected.schema}\n"
                f"  got schema: {result.schema}\n"
                f"  rows: {expected.num_rows}"
            )

    def stream_to_file(self, stream_path, file_path):
        ma_batches = list(ma.read_ipc_stream(str(stream_path)))
        if ma_batches:
            ma.write_ipc_file(str(file_path), batches=ma_batches)
        else:
            schema_rb = ma.read_ipc_stream_schema(str(stream_path))
            ma.write_ipc_file(str(file_path), schema=schema_rb, batches=[])

    def file_to_stream(self, file_path, stream_path):
        ma_batches = list(ma.read_ipc_file(str(file_path)))
        if ma_batches:
            ma.write_ipc_stream(str(stream_path), batches=ma_batches)
        else:
            schema_rb = ma.read_ipc_file_schema(str(file_path))
            ma.write_ipc_stream(str(stream_path), schema=schema_rb, batches=[])


# ---------------------------------------------------------------------------
# C Data exporter
# ---------------------------------------------------------------------------


class MarrowCDataExporter(CDataExporter):
    """Export test data from Arrow JSON format through Marrow's C Data Interface."""

    @property
    def supports_releasing_memory(self) -> bool:
        return False

    def export_schema_from_json(self, json_path, c_schema_ptr):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        empty_rb = ma.record_batch(
            [ma.array([], type=f.type()) for f in ma_fields],
            schema=ma.schema(ma_fields),
        )
        pa.record_batch(empty_rb).schema._export_to_c(_cffi_ptr_to_int(c_schema_ptr))

    def export_batch_from_json(self, json_path, num_batch: int, c_array_ptr):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        pa.record_batch(_json_to_ma_batch(json_dict, num_batch, ma_fields))._export_to_c(
            _cffi_ptr_to_int(c_array_ptr)
        )


# ---------------------------------------------------------------------------
# C Data importer
# ---------------------------------------------------------------------------


class MarrowCDataImporter(CDataImporter):
    """Import test data from CFFI struct through Marrow's C Data Interface."""

    @property
    def supports_releasing_memory(self) -> bool:
        return False

    def import_schema_and_compare_to_json(self, json_path, c_schema_ptr):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        expected_schema = pa.record_batch(
            ma.record_batch(
                [ma.array([], type=f.type()) for f in ma_fields],
                schema=ma.schema(ma_fields),
            )
        ).schema

        imported_schema = pa.Schema._import_from_c(_cffi_ptr_to_int(c_schema_ptr))
        empty_batch = pa.record_batch(
            [pa.array([], type=f.type) for f in imported_schema], schema=imported_schema
        )
        result_schema = pa.record_batch(ma.record_batch(empty_batch)).schema

        assert expected_schema.equals(result_schema), (
            f"Schema mismatch:\n  expected: {expected_schema}\n  got: {result_schema}"
        )

    def import_batch_and_compare_to_json(self, json_path, num_batch: int, c_array_ptr):
        json_dict = _read_json(json_path)
        ma_fields = _json_to_ma_schema(json_dict)
        expected = pa.record_batch(_json_to_ma_batch(json_dict, num_batch, ma_fields))

        pa_batch = pa.RecordBatch._import_from_c(
            _cffi_ptr_to_int(c_array_ptr), expected.schema
        )
        result = pa.record_batch(ma.record_batch(pa_batch))

        assert expected.equals(result), (
            f"Batch {num_batch} mismatch:\n"
            f"  schema: {expected.schema}\n"
            f"  rows: {expected.num_rows}"
        )
