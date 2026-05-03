"""MarrowTester — registers Marrow as a participant in archery integration tests.

All IPC reading and writing goes through Marrow (never pa.ipc.*).
PyArrow is used only as the comparison oracle in validate(), and as the
JSON-integration-format intermediate (since pyarrow's `pa.list_(field)`
preserves nested-field names that marrow's `ma.list_(value_type)` cannot).

The bridge from pyarrow to marrow is the C Data Interface: any pa.RecordBatch
can be wrapped as a marrow.RecordBatch via `ma.record_batch(pa_batch)`.
"""

import json
import os

import numpy as np
import marrow as ma
import pyarrow as pa

from archery.integration.tester import Tester, CDataExporter, CDataImporter


# ---------------------------------------------------------------------------
# Arrow JSON integration format → PyArrow (preserves nested-field names)
# ---------------------------------------------------------------------------


def _json_type_to_pa(type_obj: dict, children_fields: list) -> pa.DataType | None:
    """Convert an Arrow JSON type descriptor to a pyarrow DataType.

    Returns None when any nested type is unsupported (we filter the outer
    field out instead of materialising an unsupported pa type).
    """
    name = type_obj["name"]
    if name == "null":
        return pa.null()
    if name == "bool":
        return pa.bool_()
    if name == "int":
        bw = type_obj["bitWidth"]
        prefix = "int" if type_obj["isSigned"] else "uint"
        return getattr(pa, f"{prefix}{bw}")()
    if name == "floatingpoint":
        return {"HALF": pa.float16(), "SINGLE": pa.float32(), "DOUBLE": pa.float64()}[
            type_obj["precision"]
        ]
    if name == "binary":
        return pa.binary()
    if name == "fixedsizebinary":
        return pa.binary(type_obj["byteWidth"])
    if name == "utf8":
        return pa.utf8()
    if name == "date":
        return pa.date32() if type_obj.get("unit") == "DAY" else pa.date64()
    if name == "time":
        _UNIT_MAP = {"SECOND": "s", "MILLISECOND": "ms", "MICROSECOND": "us", "NANOSECOND": "ns"}
        unit = _UNIT_MAP[type_obj["unit"]]
        if type_obj.get("bitWidth", 32) == 32:
            return pa.time32(unit)
        else:
            return pa.time64(unit)
    if name == "timestamp":
        _UNIT_MAP = {"SECOND": "s", "MILLISECOND": "ms", "MICROSECOND": "us", "NANOSECOND": "ns"}
        unit = _UNIT_MAP[type_obj["unit"]]
        tz = type_obj.get("timezone")
        return pa.timestamp(unit, tz=tz)
    if name == "duration":
        _UNIT_MAP = {"SECOND": "s", "MILLISECOND": "ms", "MICROSECOND": "us", "NANOSECOND": "ns"}
        return pa.duration(_UNIT_MAP[type_obj["unit"]])
    if name == "list":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.list_(child)
    if name == "fixedsizelist":
        child = _json_field_to_pa(children_fields[0])
        return (
            None
            if child is None
            else pa.list_(child, type_obj["listSize"])
        )
    if name == "struct":
        pa_fields = [_json_field_to_pa(f) for f in children_fields]
        return (
            None
            if any(f is None for f in pa_fields)
            else pa.struct(pa_fields)
        )
    if name == "decimal":
        precision = type_obj["precision"]
        scale = type_obj["scale"]
        bit_width = type_obj.get("bitWidth", 128)
        if bit_width == 32:
            return pa.decimal32(precision, scale)
        elif bit_width == 64:
            return pa.decimal64(precision, scale)
        elif bit_width == 256:
            return pa.decimal256(precision, scale)
        else:
            return pa.decimal128(precision, scale)
    return None


def _json_field_to_pa(field_obj: dict) -> pa.Field | None:
    """Convert an Arrow JSON field to a pyarrow Field (None if unsupported)."""
    if "dictionary" in field_obj:
        return None
    pa_type = _json_type_to_pa(
        field_obj["type"], field_obj.get("children") or []
    )
    if pa_type is None:
        return None
    metadata = {
        kv["key"]: kv["value"] for kv in field_obj.get("metadata") or []
    }
    return pa.field(
        field_obj["name"],
        pa_type,
        nullable=field_obj.get("nullable", True),
        metadata=metadata or None,
    )


def _json_col_to_pa(col_obj: dict, pa_type: pa.DataType) -> pa.Array:
    """Convert an Arrow JSON column to a pyarrow Array.

    Encoding quirks:
      - int64 / uint64 DATA are JSON strings (JSON can't represent 64-bit ints)
      - binary DATA are hex strings
      - all other primitives are native Python types
    """
    n = col_obj["count"]
    validity = col_obj.get("VALIDITY")
    mask_pa = (
        None
        if validity is None
        else pa.array(~np.array(validity, dtype=bool), type=pa.bool_())
    )
    mask_np = None if validity is None else ~np.array(validity, dtype=bool)

    if pa.types.is_null(pa_type):
        return pa.array([None] * n, type=pa_type)

    if pa.types.is_boolean(pa_type) or pa.types.is_integer(pa_type):
        data = [int(v) if isinstance(v, str) else v for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_floating(pa_type):
        return pa.array(col_obj.get("DATA", []), type=pa_type, mask=mask_np)

    if pa.types.is_binary(pa_type):
        data = [bytes.fromhex(v) if v else b"" for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_fixed_size_binary(pa_type):
        data = [bytes.fromhex(v) if v else b"" for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if (
        pa.types.is_date(pa_type)
        or pa.types.is_time(pa_type)
        or pa.types.is_timestamp(pa_type)
        or pa.types.is_duration(pa_type)
    ):
        data = [int(v) if isinstance(v, str) else v for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_string(pa_type):
        return pa.array(col_obj.get("DATA", []), type=pa_type, mask=mask_np)

    if pa.types.is_decimal(pa_type):
        from decimal import Decimal as _Decimal
        scale = pa_type.scale
        # Use exponential notation to avoid Python's default 28-digit Decimal
        # precision limit — large decimal256 values need up to 76 digits.
        data = [
            _Decimal(f"{v}E-{scale}") if v is not None else None
            for v in col_obj.get("DATA", [])
        ]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_list(pa_type) and not pa.types.is_fixed_size_list(pa_type):
        offsets = col_obj.get("OFFSET", [])
        child_arr = _json_col_to_pa(col_obj["children"][0], pa_type.value_type)
        return pa.ListArray.from_arrays(offsets, child_arr, mask=mask_pa)

    if pa.types.is_fixed_size_list(pa_type):
        child_arr = _json_col_to_pa(col_obj["children"][0], pa_type.value_type)
        return pa.FixedSizeListArray.from_arrays(
            child_arr, pa_type.list_size, mask=mask_pa
        )

    if pa.types.is_struct(pa_type):
        field_arrs = [
            _json_col_to_pa(col_obj["children"][i], pa_type.field(i).type)
            for i in range(pa_type.num_fields)
        ]
        return pa.StructArray.from_arrays(
            field_arrs, fields=list(pa_type), mask=mask_pa
        )

    raise ValueError(f"Unsupported pa type in JSON converter: {pa_type}")


# ---------------------------------------------------------------------------
# JSON → Marrow (via PyArrow + C Data Interface)
# ---------------------------------------------------------------------------


def _read_json(json_path: os.PathLike) -> dict:
    with open(json_path, "rb") as f:
        return json.loads(f.read())


def _json_to_pa_schema(json_dict: dict) -> pa.Schema | None:
    """Build a pa.Schema from the supported subset of JSON fields, or None."""
    pa_fields = [
        _json_field_to_pa(f) for f in json_dict["schema"]["fields"]
    ]
    pa_fields = [f for f in pa_fields if f is not None]
    if not pa_fields:
        return None
    schema_meta = {
        kv["key"]: kv["value"]
        for kv in json_dict["schema"].get("metadata") or []
    }
    return pa.schema(pa_fields, metadata=schema_meta or None)


def _empty_ma_batch(pa_schema: pa.Schema):
    """Build an empty marrow RecordBatch matching the given pa.Schema."""
    pa_batch = pa.record_batch(
        [pa.array([], type=f.type) for f in pa_schema], schema=pa_schema
    )
    return ma.record_batch(pa_batch)


def _json_to_ma_batch(json_dict: dict, num_batch: int):
    """Build a marrow RecordBatch from a JSON batch.

    Build a pa.RecordBatch (which preserves nested-field names) and bridge to
    marrow via the C Data Interface.  Columns are matched positionally — the
    JSON schema may contain duplicate field names (e.g. the duplicate_fieldnames
    test case has two "ints" columns of different widths).
    """
    pa_schema = _json_to_pa_schema(json_dict)
    assert pa_schema is not None, "no supported fields"
    batch_obj = json_dict["batches"][num_batch]
    # Pair each surviving (supported) JSON field with its column at the same
    # index in the original schema, preserving order across drops.
    json_fields = json_dict["schema"]["fields"]
    json_cols = batch_obj["columns"]
    arrays: list[pa.Array] = []
    pa_field_iter = iter(pa_schema)
    for jf, jc in zip(json_fields, json_cols):
        if _json_field_to_pa(jf) is None:
            continue
        pa_field = next(pa_field_iter)
        arrays.append(_json_col_to_pa(jc, pa_field.type))
    pa_batch = pa.record_batch(arrays, schema=pa_schema)
    return ma.record_batch(pa_batch)


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
    name = "Mojo"

    def make_c_data_exporter(self):
        return MarrowCDataExporter()

    def make_c_data_importer(self):
        return MarrowCDataImporter()

    def json_to_file(self, json_path, arrow_path):
        json_dict = _read_json(json_path)
        # Strict: refuse if any field is unsupported (a partial-coverage write
        # would round-trip differently than the JSON expects).
        if any(
            _json_field_to_pa(f) is None for f in json_dict["schema"]["fields"]
        ):
            raise NotImplementedError("test case has unsupported column types")
        empty_rb = _empty_ma_batch(_json_to_pa_schema(json_dict))
        n_batches = len(json_dict.get("batches", []))
        batches = [_json_to_ma_batch(json_dict, i) for i in range(n_batches)]
        ma.write_ipc_file(str(arrow_path), schema=empty_rb, batches=batches)

    def validate(self, json_path, arrow_path, quirks=None):
        json_dict = _read_json(json_path)
        if _json_to_pa_schema(json_dict) is None:
            raise NotImplementedError("no supported columns in this test case")
        ma_batches = list(ma.read_ipc_file(str(arrow_path)))
        n_expected = len(json_dict.get("batches", []))
        assert len(ma_batches) == n_expected, (
            f"Expected {n_expected} batches, got {len(ma_batches)}"
        )
        for i, ma_batch in enumerate(ma_batches):
            expected = pa.record_batch(_json_to_ma_batch(json_dict, i))
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
        empty_rb = _empty_ma_batch(_json_to_pa_schema(json_dict))
        pa.record_batch(empty_rb).schema._export_to_c(_cffi_ptr_to_int(c_schema_ptr))

    def export_batch_from_json(self, json_path, num_batch: int, c_array_ptr):
        json_dict = _read_json(json_path)
        pa.record_batch(_json_to_ma_batch(json_dict, num_batch))._export_to_c(
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
        expected_schema = pa.record_batch(
            _empty_ma_batch(_json_to_pa_schema(json_dict))
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
        expected = pa.record_batch(_json_to_ma_batch(json_dict, num_batch))

        pa_batch = pa.RecordBatch._import_from_c(
            _cffi_ptr_to_int(c_array_ptr), expected.schema
        )
        result = pa.record_batch(ma.record_batch(pa_batch))

        assert expected.equals(result), (
            f"Batch {num_batch} mismatch:\n"
            f"  schema: {expected.schema}\n"
            f"  rows: {expected.num_rows}"
        )
