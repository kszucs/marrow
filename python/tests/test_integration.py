"""Arrow integration tests for Marrow.

Validates Marrow's C Data Interface implementation against Arrow's official
integration test suite — archery's generated test cases and golden .arrow_file
files from apache/arrow.

Track 1: self-contained; no IPC reader/writer needed in Mojo — PyArrow handles
the Arrow IPC binary format, and Marrow participates via the C Data Interface.

Roundtrip pattern:
  PyArrow RecordBatch
    → marrow.record_batch()        (uses __arrow_c_record_batch__ from PyArrow)
    → pa.record_batch()            (uses __arrow_c_array__ from Marrow)
    → assert original.equals(result)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

try:
    from archery.integration.datagen import get_generated_json_files

    HAS_ARCHERY = True
except ImportError:
    HAS_ARCHERY = False

import marrow

ARROW_TESTING_DIR = os.environ.get("ARROW_TESTING_DIR", "")

# Test-case names (datagen file names) to skip entirely — no supported columns
UNSUPPORTED_CASE_PATTERNS = [
    "dictionary",
    "union",
    "map",
    "decimal",
    "interval",
    "duration",
    "timestamp",
    "extension",
    "large",
    "view",
    "run_end",
    "null",
]


def _case_unsupported(name: str) -> bool:
    return any(p in name for p in UNSUPPORTED_CASE_PATTERNS)


# ---------------------------------------------------------------------------
# Type-level support check
# ---------------------------------------------------------------------------


def _type_supported(t: pa.DataType) -> bool:
    """Return True if Marrow's C Data Interface implementation supports this type."""
    if pa.types.is_boolean(t):
        return True
    if pa.types.is_integer(t):
        return True
    if pa.types.is_floating(t):
        return True
    if (
        pa.types.is_binary(t)
        and not pa.types.is_large_binary(t)
        and not pa.types.is_fixed_size_binary(t)
    ):
        return True
    if pa.types.is_string(t) and not pa.types.is_large_string(t):
        return True
    if (
        pa.types.is_list(t)
        and not pa.types.is_large_list(t)
        and not pa.types.is_fixed_size_list(t)
    ):
        return _type_supported(t.value_type)
    if pa.types.is_fixed_size_list(t):
        return _type_supported(t.value_type)
    if pa.types.is_struct(t):
        return all(_type_supported(t.field(i).type) for i in range(t.num_fields))
    return False


def _project_supported(batch: pa.RecordBatch) -> pa.RecordBatch | None:
    """Return a new RecordBatch with only the columns whose types Marrow supports.

    Returns None if no columns are supported (caller should skip the batch).
    """
    indices = [i for i, f in enumerate(batch.schema) if _type_supported(f.type)]
    if not indices:
        return None
    return batch.select(indices)


# ---------------------------------------------------------------------------
# Arrow JSON integration format → PyArrow converter
# ---------------------------------------------------------------------------
# Archery's datagen produces a JSON dict in Arrow's integration format.
# PyArrow cannot read this format directly so we convert it here.
#
# Encoding quirks:
#   • int64 / uint64 DATA are JSON strings (JSON can't represent 64-bit ints)
#   • binary DATA are hex strings (e.g. "27DD17")
#   • all other primitives are native Python types
#   • ListArray.from_arrays() requires mask as pa.Array[bool], not numpy


def _json_type_to_pa(type_obj: dict, children_fields: list) -> pa.DataType | None:
    name = type_obj["name"]
    if name == "bool":
        return pa.bool_()
    if name == "int":
        bw = type_obj["bitWidth"]
        prefix = "int" if type_obj["isSigned"] else "uint"
        return getattr(pa, prefix + str(bw))()
    if name == "floatingpoint":
        return {"HALF": pa.float16(), "SINGLE": pa.float32(), "DOUBLE": pa.float64()}[
            type_obj["precision"]
        ]
    if name == "binary":
        return pa.binary()
    if name == "utf8":
        return pa.utf8()
    if name == "list":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.list_(child)
    if name == "fixedsizelist":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.list_(child, type_obj["listSize"])
    if name == "struct":
        pa_fields = [_json_field_to_pa(f) for f in children_fields]
        return None if any(f is None for f in pa_fields) else pa.struct(pa_fields)
    return None


def _json_field_to_pa(field_obj: dict) -> pa.Field | None:
    pa_type = _json_type_to_pa(field_obj["type"], field_obj.get("children") or [])
    if pa_type is None:
        return None
    return pa.field(
        field_obj["name"], pa_type, nullable=field_obj.get("nullable", True)
    )


def _json_col_to_pa(col_obj: dict, pa_type: pa.DataType) -> pa.Array:
    n = col_obj["count"]
    validity = col_obj.get("VALIDITY")
    # pa.ListArray.from_arrays requires pa.Array[bool], not numpy
    mask_pa = (
        None
        if validity is None
        else pa.array(~np.array(validity, dtype=bool), type=pa.bool_())
    )
    mask_np = None if validity is None else ~np.array(validity, dtype=bool)

    if pa.types.is_boolean(pa_type) or pa.types.is_integer(pa_type):
        data = [int(v) if isinstance(v, str) else v for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_floating(pa_type):
        return pa.array(col_obj.get("DATA", []), type=pa_type, mask=mask_np)

    if pa.types.is_binary(pa_type):
        data = [bytes.fromhex(v) if v else b"" for v in col_obj.get("DATA", [])]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_string(pa_type):
        return pa.array(col_obj.get("DATA", []), type=pa_type, mask=mask_np)

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

    raise ValueError(f"Unsupported type in converter: {pa_type}")


def _json_dict_to_pa_batches(json_dict: dict) -> list[pa.RecordBatch]:
    """Convert an Arrow JSON integration dict to a list of PyArrow RecordBatches.

    Silently drops fields whose types are unsupported (converter returns None).
    Returns [] if no supported fields remain.
    """
    schema_fields = json_dict["schema"]["fields"]
    supported = [
        (i, _json_field_to_pa(f))
        for i, f in enumerate(schema_fields)
        if _json_field_to_pa(f) is not None
    ]
    if not supported:
        return []
    pa_schema = pa.schema([pf for _, pf in supported])
    batches = []
    for batch_obj in json_dict.get("batches", []):
        arrays = [
            _json_col_to_pa(batch_obj["columns"][i], pf.type) for i, pf in supported
        ]
        batches.append(pa.record_batch(arrays, schema=pa_schema))
    return batches


# ---------------------------------------------------------------------------
# Roundtrip helper
# ---------------------------------------------------------------------------


def _roundtrip(batch: pa.RecordBatch) -> pa.RecordBatch:
    """PyArrow → Marrow → PyArrow via C Data Interface."""
    return pa.record_batch(marrow.record_batch(batch))


# ---------------------------------------------------------------------------
# Track 1a: Generated test cases via archery datagen
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ARCHERY, reason="archery not installed")
class TestGeneratedCases:
    """Validate all archery-generated test cases that Marrow supports."""

    def test_all_generated(self):
        files = get_generated_json_files()
        failures = []
        skipped = []
        tested = []
        for f in files:
            if _case_unsupported(f.name):
                skipped.append(f.name)
                continue
            json_dict = f.get_json()
            batches = _json_dict_to_pa_batches(json_dict)
            if not batches:
                skipped.append(f.name)
                continue
            for i, batch in enumerate(batches):
                try:
                    result = _roundtrip(batch)
                    assert batch.equals(result), (
                        f"Batch mismatch in {f.name}[{i}]:\n"
                        f"  schema: {batch.schema}\n"
                        f"  rows: {batch.num_rows}"
                    )
                    tested.append(f.name)
                except Exception as e:
                    failures.append(f"{f.name}[{i}]: {e}")

        assert not failures, "Failures:\n" + "\n".join(failures)
        assert tested, "No generated cases were tested"


# ---------------------------------------------------------------------------
# Track 1b: Golden .arrow_file files from apache/arrow/testing/
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARROW_TESTING_DIR, reason="ARROW_TESTING_DIR not set")
class TestGoldenFiles:
    """Validate Marrow against pre-generated golden .arrow_file files.

    These files are generated by the C++ Arrow reference implementation and
    represent the authoritative test data for backward compatibility.

    Set ARROW_TESTING_DIR to the path of apache/arrow/testing/data before running.
    """

    def _golden_dir(self) -> Path:
        return (
            Path(ARROW_TESTING_DIR) / "arrow-ipc-stream" / "integration" / "cpp-21.0.0"
        )

    def test_all_golden(self):
        golden_dir = self._golden_dir()
        if not golden_dir.exists():
            pytest.skip(f"Golden dir not found: {golden_dir}")

        failures = []
        skipped = []
        tested = []
        for fpath in sorted(golden_dir.iterdir()):
            if fpath.suffix != ".arrow_file":
                continue
            name = fpath.stem.removeprefix("generated_")
            if _case_unsupported(name):
                skipped.append(name)
                continue
            reader = pa.ipc.open_file(fpath)
            for i in range(reader.num_record_batches):
                original = reader.get_batch(i)
                projected = _project_supported(original)
                if projected is None:
                    skipped.append(name)
                    continue
                try:
                    result = _roundtrip(projected)
                    assert projected.equals(result), (
                        f"Batch mismatch in {name}[{i}]:\n"
                        f"  schema: {projected.schema}\n"
                        f"  rows: {projected.num_rows}"
                    )
                    tested.append(name)
                except Exception as e:
                    failures.append(f"{name}[{i}]: {e}")

        assert not failures, f"Failures ({len(failures)}):\n" + "\n".join(failures)
        assert tested, "No golden files were tested — check the golden directory"

    @pytest.mark.parametrize(
        "name",
        [
            "primitive",
            "primitive_zerolength",
            "primitive_no_batches",
            "binary",
            "binary_zerolength",
            "binary_no_batches",
            "nested",
            "recursive_nested",
            "custom_metadata",
            "duplicate_fieldnames",
        ],
    )
    def test_golden_case(self, name: str):
        """Individual parametrized test per supported golden file for clearer CI output."""
        golden_dir = self._golden_dir()
        path = golden_dir / f"generated_{name}.arrow_file"
        if not path.exists():
            pytest.skip(f"Golden file not found: {path}")
        reader = pa.ipc.open_file(path)
        for i in range(reader.num_record_batches):
            original = reader.get_batch(i)
            projected = _project_supported(original)
            if projected is None:
                pytest.skip(f"No supported columns in {name}[{i}]")
            result = _roundtrip(projected)
            assert projected.equals(result), (
                f"Batch {i} mismatch in {name}:\n"
                f"  schema: {projected.schema}\n"
                f"  rows: {projected.num_rows}"
            )


# ---------------------------------------------------------------------------
# Track 2: IPC file and stream round-trips via Marrow Python bindings
# ---------------------------------------------------------------------------


def _make_pa_batch() -> pa.RecordBatch:
    return pa.record_batch(
        {
            "a": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "b": pa.array([1.1, 2.2, 3.3, 4.4, 5.5], type=pa.float64()),
        }
    )


class TestIPCRoundtrip:
    """IPC file and stream round-trips using marrow.read/write_ipc_*."""

    def _roundtrip_file(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            marrow.write_ipc_file(f.name, batches=[marrow.record_batch(batch)])
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 1
        return pa.record_batch(result_batches[0])

    def _roundtrip_stream(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        with tempfile.NamedTemporaryFile(suffix=".arrows") as f:
            marrow.write_ipc_stream(f.name, batches=[marrow.record_batch(batch)])
            result_batches = marrow.read_ipc_stream(f.name)
        assert len(result_batches) == 1
        return pa.record_batch(result_batches[0])

    def test_file_primitives(self):
        batch = _make_pa_batch()
        assert batch.equals(self._roundtrip_file(batch))

    def test_stream_primitives(self):
        batch = _make_pa_batch()
        assert batch.equals(self._roundtrip_stream(batch))

    def test_file_multi_batch(self):
        b1 = _make_pa_batch()
        b2 = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            marrow.write_ipc_file(
                f.name,
                batches=[marrow.record_batch(b1), marrow.record_batch(b2)],
            )
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 2
        assert b1.equals(pa.record_batch(result_batches[0]))
        assert b2.equals(pa.record_batch(result_batches[1]))

    def test_file_bool_and_string(self):
        batch = pa.record_batch(
            {
                "flags": pa.array([True, False, True], type=pa.bool_()),
                "name": pa.array(["x", "y", "z"], type=pa.utf8()),
            }
        )
        assert batch.equals(self._roundtrip_file(batch))

    def test_file_nullable(self):
        batch = pa.record_batch({"x": pa.array([10, None, 30, None], type=pa.int32())})
        result = self._roundtrip_file(batch)
        assert batch.equals(result)
        assert result.column("x").null_count == 2

    def test_stream_to_file(self):
        """write_ipc_stream + read_ipc_stream + write_ipc_file + read_ipc_file round-trip."""
        batch = _make_pa_batch()
        with (
            tempfile.NamedTemporaryFile(suffix=".arrows") as sf,
            tempfile.NamedTemporaryFile(suffix=".arrow") as ff,
        ):
            marrow.write_ipc_stream(sf.name, batches=[marrow.record_batch(batch)])
            ma_batches = marrow.read_ipc_stream(sf.name)
            marrow.write_ipc_file(ff.name, batches=list(ma_batches))
            result_batches = marrow.read_ipc_file(ff.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))

    def test_pyarrow_writes_marrow_reads_file(self):
        """PyArrow writes IPC file, Marrow reads it."""
        batch = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            with pa.ipc.new_file(f.name, batch.schema) as writer:
                writer.write_batch(batch)
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))

    def test_pyarrow_writes_marrow_reads_stream(self):
        """PyArrow writes IPC stream, Marrow reads it."""
        batch = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrows") as f:
            with pa.ipc.new_stream(f.name, batch.schema) as writer:
                writer.write_batch(batch)
            result_batches = marrow.read_ipc_stream(f.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))
