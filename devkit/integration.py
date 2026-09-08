"""Arrow protocol conformance, via apache/arrow's archery.

Registers marrow as a participant in the integration suite and runs it against
the C++, Rust and Go implementations.  All IPC reading and writing goes through
marrow (never `pa.ipc.*`); pyarrow is the comparison oracle in `validate()` and
the JSON-integration-format intermediate, since `pa.list_(field)` preserves
nested-field names that marrow's `ma.list_(value_type)` cannot.  The bridge from
pyarrow to marrow is the C Data Interface: any `pa.RecordBatch` can be wrapped as
a `marrow.RecordBatch` via `ma.record_batch(pa_batch)`.

**This module needs `archery`, which only the `integration` environment has**,
and archery in turn needs the apache/arrow clone that `pixi run -e integration
clone_arrow` fetches.  `devkit.cli` therefore imports it inside the command,
never at module scope.

The `_json_*` conversion functions below are the Arrow JSON integration format
read into pyarrow.  Neither archery nor pyarrow exposes that conversion, so it
is written out here; it is the one part of the suite that is pure pyarrow and
has its own unit tests in `devkit/tests/test_integration.py`.
"""

import contextlib
import functools
import io
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
from rich import box
from rich.console import Console
from rich.table import Table
from archery.integration.tester import CDataExporter, CDataImporter, Tester


class _LazyMarrow:
    """`marrow`, imported on first attribute access rather than at module scope.

    Importing it here would load `libmarrow.so`, and the two halves of this
    module have very different needs: the Arrow-JSON bridge and the report are
    pure pyarrow and are unit-tested, while only the tester classes actually
    drive marrow.  A module-scope import puts a shared-library build in front of
    `pixi run selftest`, which is meant to be a one-second command -- and makes
    those unit tests pass or fail on whether a stale `.so` happens to be on disk.
    """

    def __getattr__(self, name):
        import marrow

        return getattr(marrow, name)


ma = _LazyMarrow()


# ---------------------------------------------------------------------------
# Arrow JSON integration format → PyArrow (preserves nested-field names)
# ---------------------------------------------------------------------------

#: The JSON format spells a temporal unit out; pyarrow abbreviates it.
_UNITS = {
    "SECOND": "s",
    "MILLISECOND": "ms",
    "MICROSECOND": "us",
    "NANOSECOND": "ns",
}


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
    if name == "largebinary":
        return pa.large_binary()
    if name == "largeutf8":
        return pa.large_utf8()
    if name == "largelist":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.large_list(child)
    if name == "date":
        return pa.date32() if type_obj.get("unit") == "DAY" else pa.date64()
    if name == "time":
        unit = _UNITS[type_obj["unit"]]
        if type_obj.get("bitWidth", 32) == 32:
            return pa.time32(unit)
        else:
            return pa.time64(unit)
    if name == "timestamp":
        unit = _UNITS[type_obj["unit"]]
        tz = type_obj.get("timezone")
        return pa.timestamp(unit, tz=tz)
    if name == "duration":
        return pa.duration(_UNITS[type_obj["unit"]])
    if name == "list":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.list_(child)
    if name == "fixedsizelist":
        child = _json_field_to_pa(children_fields[0])
        return None if child is None else pa.list_(child, type_obj["listSize"])
    if name == "map":
        # The single child is the entries struct field; its two fields carry the
        # key/value names, which `map_non_canonical` deliberately varies.
        entries = _json_field_to_pa(children_fields[0])
        if entries is None or not pa.types.is_struct(entries.type):
            return None
        return pa.map_(
            entries.type.field(0),
            entries.type.field(1),
            keys_sorted=type_obj.get("keysSorted", False),
        )
    if name == "interval":
        # Only MONTH_DAY_NANO is reachable: pyarrow 23 exposes
        # `month_day_nano_interval` and has no type at all for YEAR_MONTH or
        # DAY_TIME, and this converter builds every column through pyarrow
        # before bridging to marrow over the C Data Interface.  Marrow itself
        # handles all three -- it reads them back from C++/Rust/Go -- so the
        # limit here is the bridge, not the library.
        return (
            pa.month_day_nano_interval()
            if type_obj.get("unit") == "MONTH_DAY_NANO"
            else None
        )
    if name == "struct":
        pa_fields = [_json_field_to_pa(f) for f in children_fields]
        return None if any(f is None for f in pa_fields) else pa.struct(pa_fields)
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
    metadata = {kv["key"]: kv["value"] for kv in field_obj.get("metadata") or []}
    dict_info = field_obj.get("dictionary")
    if dict_info is not None:
        value_type = _json_type_to_pa(
            field_obj["type"], field_obj.get("children") or []
        )
        if value_type is None:
            return None
        idx = dict_info["indexType"]
        bw = idx["bitWidth"]
        index_type = getattr(pa, ("int" if idx["isSigned"] else "uint") + str(bw))()
        ordered = dict_info.get("isOrdered", False)
        pa_type = pa.dictionary(index_type, value_type, ordered=ordered)
    else:
        pa_type = _json_type_to_pa(field_obj["type"], field_obj.get("children") or [])
        if pa_type is None:
            return None
    return pa.field(
        field_obj["name"],
        pa_type,
        nullable=field_obj.get("nullable", True),
        metadata=metadata or None,
    )


def _find_field_for_dict_id(json_fields: list, dict_id: int) -> dict | None:
    """Recursively search schema fields for the one referencing the given dict_id."""
    for f in json_fields:
        if (di := f.get("dictionary")) is not None and di["id"] == dict_id:
            return f
        found = _find_field_for_dict_id(f.get("children") or [], dict_id)
        if found is not None:
            return found
    return None


def _json_col_to_pa(
    col_obj: dict,
    pa_type: pa.DataType,
    dict_cache: dict | None = None,
    json_field: dict | None = None,
) -> pa.Array:
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

    if (
        pa.types.is_binary(pa_type)
        or pa.types.is_large_binary(pa_type)
        or pa.types.is_fixed_size_binary(pa_type)
    ):
        # Binary DATA arrives hex-encoded, whatever the width.
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

    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
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

    if pa.types.is_large_list(pa_type):
        offsets = [int(v) for v in col_obj.get("OFFSET", [])]
        child_jf = (
            json_field["children"][0]
            if json_field and json_field.get("children")
            else None
        )
        child_arr = _json_col_to_pa(
            col_obj["children"][0], pa_type.value_type, dict_cache, child_jf
        )
        return pa.LargeListArray.from_arrays(offsets, child_arr, mask=mask_pa)

    if pa.types.is_list(pa_type) and not pa.types.is_fixed_size_list(pa_type):
        offsets = col_obj.get("OFFSET", [])
        child_jf = (
            json_field["children"][0]
            if json_field and json_field.get("children")
            else None
        )
        child_arr = _json_col_to_pa(
            col_obj["children"][0], pa_type.value_type, dict_cache, child_jf
        )
        return pa.ListArray.from_arrays(offsets, child_arr, mask=mask_pa)

    if pa.types.is_fixed_size_list(pa_type):
        child_jf = (
            json_field["children"][0]
            if json_field and json_field.get("children")
            else None
        )
        child_arr = _json_col_to_pa(
            col_obj["children"][0], pa_type.value_type, dict_cache, child_jf
        )
        return pa.FixedSizeListArray.from_arrays(
            child_arr, pa_type.list_size, mask=mask_pa
        )

    if pa.types.is_map(pa_type):
        offsets = [int(v) for v in col_obj.get("OFFSET", [])]
        child_jf = (
            json_field["children"][0]
            if json_field and json_field.get("children")
            else None
        )
        entries = _json_col_to_pa(
            col_obj["children"][0],
            pa.struct([pa_type.key_field, pa_type.item_field]),
            dict_cache,
            child_jf,
        )
        return pa.MapArray.from_arrays(
            offsets,
            entries.field(0),
            entries.field(1),
            type=pa_type,
            mask=mask_pa,
        )

    if pa.types.is_interval(pa_type):
        # DATA is a list of {'months', 'days', 'nanoseconds'} objects.
        data = [
            None
            if v is None
            else pa.MonthDayNano([v["months"], v["days"], int(v["nanoseconds"])])
            for v in col_obj.get("DATA", [])
        ]
        return pa.array(data, type=pa_type, mask=mask_np)

    if pa.types.is_struct(pa_type):
        field_arrs = []
        for i in range(pa_type.num_fields):
            child_jf = (
                json_field["children"][i]
                if json_field
                and json_field.get("children")
                and i < len(json_field["children"])
                else None
            )
            field_arrs.append(
                _json_col_to_pa(
                    col_obj["children"][i], pa_type.field(i).type, dict_cache, child_jf
                )
            )
        return pa.StructArray.from_arrays(
            field_arrs, fields=list(pa_type), mask=mask_pa
        )

    if pa.types.is_dictionary(pa_type):
        if dict_cache is None or json_field is None:
            raise ValueError(
                "dict_cache and json_field required for dictionary columns"
            )
        dict_id = json_field["dictionary"]["id"]
        indices = pa.array(
            [int(v) for v in col_obj.get("DATA", [])],
            type=pa_type.index_type,
            mask=mask_np,
        )
        return pa.DictionaryArray.from_arrays(indices, dict_cache[dict_id])

    raise ValueError(f"Unsupported pa type in JSON converter: {pa_type}")


# ---------------------------------------------------------------------------
# JSON → Marrow (via PyArrow + C Data Interface)
# ---------------------------------------------------------------------------


def _read_json(json_path: os.PathLike) -> dict:
    with open(json_path, "rb") as f:
        return json.loads(f.read())


def _json_to_pa_schema(json_dict: dict) -> pa.Schema | None:
    """Build a pa.Schema from the supported subset of JSON fields, or None."""
    pa_fields = [_json_field_to_pa(f) for f in json_dict["schema"]["fields"]]
    pa_fields = [f for f in pa_fields if f is not None]
    if not pa_fields:
        return None
    schema_meta = {
        kv["key"]: kv["value"] for kv in json_dict["schema"].get("metadata") or []
    }
    return pa.schema(pa_fields, metadata=schema_meta or None)


def _empty_ma_batch(pa_schema: pa.Schema):
    """Build an empty marrow RecordBatch matching the given pa.Schema."""
    arrays = []
    for f in pa_schema:
        try:
            arrays.append(pa.array([], type=f.type))
        except Exception:
            arrays.append(pa.nulls(0, type=f.type))
    pa_batch = pa.record_batch(arrays, schema=pa_schema)
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
    # Build dict_cache sorted by id so inner (lower-id) dicts are ready when
    # outer dict values reference them (nested dictionary case).
    dict_cache: dict[int, pa.Array] = {}
    for d in sorted(json_dict.get("dictionaries", []), key=lambda x: x["id"]):
        d_id = d["id"]
        dict_field = _find_field_for_dict_id(json_dict["schema"]["fields"], d_id)
        if dict_field is not None:
            value_type = _json_type_to_pa(
                dict_field["type"], dict_field.get("children") or []
            )
            if value_type is not None:
                dict_cache[d_id] = _json_col_to_pa(
                    d["data"]["columns"][0], value_type, dict_cache, dict_field
                )
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
        arrays.append(_json_col_to_pa(jc, pa_field.type, dict_cache, jf))
    pa_batch = pa.record_batch(arrays, schema=pa_schema)
    return ma.record_batch(pa_batch)


@functools.cache
def _ffi():
    """cffi's FFI is stateless here and costs 106 us to build; build it once."""
    import cffi

    return cffi.FFI()


def _cffi_ptr_to_int(cffi_ptr) -> int:
    return int(_ffi().cast("uintptr_t", cffi_ptr))


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
        if any(_json_field_to_pa(f) is None for f in json_dict["schema"]["fields"]):
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Tee:
    """Write to several streams at once -- the terminal and a capture buffer."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


class ArcheryReport:
    """A summary table out of archery's stdout.

    `run_all_tests` prints its results and returns nothing structured, so the
    only way to summarise a run is to scrape the stream it printed.  That is
    why `ArcherySuite.run` tees stdout rather than just letting it through.
    """

    PHASE = re.compile(
        r"##########################################################\n"
        r"([^\n]+)\n"
        r"##########################################################"
    )
    FILE_CASE = re.compile(r"Testing file .*?(?:generated_)?([^/]+?)\.json\s*$")
    C_DATA_CASE = re.compile(r"Testing C Arrow(?:Schema|Array) from file '([^']+)'")

    def __init__(self, log):
        self.phases = self._parse(log)

    @classmethod
    def _parse(cls, log):
        """Group archery's stdout by phase -> {pass: {cases}, skip: {cases}}.

        The banner spans three lines, so phases are located over the whole log
        and used to slice it; a per-line scan cannot match them.  Inside a
        phase, a skip consumes the case and a pass does not -- one file yields
        several `... with record batch` lines, and they are all the same case.
        """
        banners = [
            (match.start(), match.group(1).strip()) for match in cls.PHASE.finditer(log)
        ]
        banners.append((len(log), None))

        phases = defaultdict(lambda: {"pass": set(), "skip": set()})
        for index in range(len(banners) - 1):
            start, phase = banners[index]
            body = log[start : banners[index + 1][0]]
            case = None
            for line in body.splitlines():
                match = cls.FILE_CASE.match(line) or cls.C_DATA_CASE.match(line)
                if match is not None:
                    case = match.group(1)
                    continue
                if case is None:
                    continue
                if line.startswith("-- Skipping"):
                    phases[phase]["skip"].add(case)
                    case = None
                elif line.startswith("-- Validating") or line.startswith(
                    "... with record batch"
                ):
                    phases[phase]["pass"].add(case)
        return dict(phases)

    def coverage(self):
        """`{case: phases it passed}`, over every case the run mentioned.

        Seeded from pass *and* skip, so a case that passed nothing still gets a
        row reading 0.  Dropping it would quietly hide the cases most worth
        looking at -- an implementation that skips a layout everywhere is
        exactly what this table exists to surface.
        """
        cases = set()
        for counts in self.phases.values():
            cases |= counts["pass"] | counts["skip"]
        return {
            case: sum(1 for counts in self.phases.values() if case in counts["pass"])
            for case in cases
        }

    def render(self):
        if not self.phases:
            return
        console = Console(highlight=False)

        phases = Table(title="Per-phase summary", box=box.SIMPLE_HEAD)
        phases.add_column("Phase", no_wrap=True)
        for column in ("Pass", "Skip"):
            phases.add_column(column, justify="right")
        total_pass = total_skip = 0
        for phase, counts in sorted(self.phases.items()):
            phases.add_row(phase, str(len(counts["pass"])), str(len(counts["skip"])))
            total_pass += len(counts["pass"])
            total_skip += len(counts["skip"])
        phases.add_section()
        phases.add_row("TOTAL", str(total_pass), str(total_skip))
        console.print(phases)

        # Out of how many phases, because that denominator is the vocabulary
        # everything about this suite is written in -- "14/14", "10/14", "7/14".
        total = len(self.phases)
        cases = Table(title="Per-case coverage", box=box.SIMPLE_HEAD)
        cases.add_column("Case", no_wrap=True)
        cases.add_column("Phases passing", justify="right")
        coverage = self.coverage()
        for case, count in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0])):
            cases.add_row(case, f"{count} / {total}")
        console.print(cases)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


class ArcherySuite:
    """Runs archery's integration tests with marrow as a participant.

    The skip sets below are patched into `archery.integration.datagen` rather
    than carried upstream, which is what keeps the apache/arrow clone pristine.
    Patching is an explicit call rather than an import side effect, so importing
    this module cannot silently change archery's behaviour for something else.
    """

    #: Layouts marrow genuinely does not implement.
    #:
    #: `interval` is different: it is a *harness* limit, not a library one.
    #: pyarrow has no type for YEAR_MONTH or DAY_TIME, so `_json_field_to_pa`
    #: returns None, `json_to_file` refuses the case, and marrow's IPC writer is
    #: never reached.  Marrow reads all three back from C++, Rust and Go
    #: correctly.  Measured 2026-08-14: un-skipping `interval` scores 10/14 with
    #: every failure on the `Mojo producing` side, and 10/14 fails the job.
    UNSUPPORTED = frozenset(
        {
            "interval",
            "union",
            "binary_view",
            "list_view",
            "extension",
            "run_end_encoded",
        }
    )

    #: PyArrow cannot construct empty arrays for nested-dictionary types
    #: (ArrowNotImplementedError), so the C Data phases are skipped for Mojo.
    #: IPC is unaffected: the other implementations validate directly without
    #: going through pyarrow.  The case shows 7/14.
    SKIP_C_DATA = frozenset({"nested_dictionary"})

    # Expected partial coverage, and not marrow bugs:
    #
    # decimal32 / decimal64 (6/14): Rust and Go implement neither type in IPC
    # or C Data, so four IPC phases and four C Data phases are skipped by them.
    #
    # binary_no_batches / primitive_no_batches (7/14): these files contain zero
    # record batches.  The IPC phases pass because the schema is still
    # exchanged; the C Data array phases iterate over batches, and zero batches
    # produce zero results, which archery counts as zero passes rather than one.

    def __init__(self):
        self._patched = False

    def patch_datagen(self):
        """Mark the cases marrow does not implement as skipped, for Mojo only."""
        if self._patched:
            return
        from archery.integration import datagen
        from archery.integration.util import SKIP_C_ARRAY, SKIP_C_SCHEMA

        original = datagen.get_generated_json_files

        def patched(tempdir=None):
            files = original(tempdir)
            for entry in files:
                if entry.name in self.UNSUPPORTED:
                    entry.skip_tester("Mojo")
                if entry.name in self.SKIP_C_DATA:
                    entry.skip_format(SKIP_C_SCHEMA, "Mojo")
                    entry.skip_format(SKIP_C_ARRAY, "Mojo")
            return files

        datagen.get_generated_json_files = patched
        self._patched = True

    @staticmethod
    def _other_testers(with_cpp, with_rust, with_go):
        wanted = [
            (with_cpp, "archery.integration.tester_cpp", "CppTester"),
            (with_rust, "archery.integration.tester_rust", "RustTester"),
            (with_go, "archery.integration.tester_go", "GoTester"),
        ]
        testers = []
        for enabled, module, attribute in wanted:
            if not enabled:
                continue
            try:
                imported = __import__(module, fromlist=[attribute])
                testers.append(getattr(imported, attribute)())
            except Exception as error:
                print(f"Warning: could not load {attribute}: {error}", file=sys.stderr)
        return testers

    def run(
        self,
        *,
        run_ipc,
        run_c_data,
        with_cpp=False,
        with_rust=False,
        with_go=False,
        stop_on_error=True,
        match=None,
        gold_dirs=None,
    ):
        from archery.integration.runner import run_all_tests

        self.patch_datagen()

        # Tee stdout to a buffer so the report can summarise a stream the user
        # is still watching live.
        buffer = io.StringIO()
        passed = True
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
                run_all_tests(
                    testers=[MarrowTester()],
                    other_testers=self._other_testers(with_cpp, with_rust, with_go),
                    run_ipc=run_ipc,
                    run_c_data=run_c_data,
                    stop_on_error=stop_on_error,
                    match=match,
                    gold_dirs=gold_dirs,
                )
        except SystemExit as exit_request:
            # archery signals failure by exiting; the report is still wanted.
            passed = not exit_request.code
        ArcheryReport(buffer.getvalue()).render()
        return passed
