# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow protocol conformance, via apache/arrow's archery.

Registers marrow as a participant in the integration suite and runs it against
the C++, Rust and Go implementations.  Marrow does every step itself: it reads
archery's JSON (`marrow.integration`), reads and writes IPC, and exports and
imports the C Data structs at the addresses archery hands over.  pyarrow is
only the equality oracle -- `RecordBatch.equals` on both sides of a comparison,
reached over the PyCapsule interface -- because marrow's own `equals` compares
layouts, not values.

**This module needs `archery`, which only the `integration` environment has**,
and archery in turn needs the apache/arrow clone that `pixi run -e integration
clone_arrow` fetches.  `devkit.cli` therefore imports it inside the command,
never at module scope.
"""

import contextlib
import gc
import io
import os
from pathlib import Path
import re
import sys
from collections import defaultdict

import pyarrow as pa
from rich import box
from rich.console import Console
from rich.table import Table
from archery.integration.tester import CDataExporter, CDataImporter, Tester


class _LazyMarrow:
    """`marrow`, imported on first attribute access rather than at module scope.

    Importing it here would load `libmarrow.so`, while only the tester classes
    drive marrow; the report and the skip sets are unit-tested without it.  A
    module-scope import puts a shared-library build in front of `pixi run
    selftest`, which is meant to be a one-second command -- and makes those
    unit tests pass or fail on whether a stale `.so` happens to be on disk.
    """

    def __getattr__(self, name):
        import marrow.integration  # noqa: F401 - binds `marrow.integration`
        import marrow

        return getattr(marrow, name)


ma = _LazyMarrow()


def _address(ptr):
    """Archery's cffi struct pointer, as the integer address marrow takes."""
    from archery.integration import cdata

    return int(cdata.ffi().cast("uintptr_t", ptr))


def _assert_equal(expected, result, what):
    """Compare two pyarrow schemas or batches, metadata included."""
    assert expected.equals(result, check_metadata=True), (
        f"{what} mismatch:\n  expected: {expected}\n  got: {result}"
    )


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
        schema, batches = ma.integration.read_json(json_path)
        ma.write_ipc_file(str(arrow_path), schema=schema, batches=batches)

    def validate(self, json_path, arrow_path, quirks=None):
        schema, expected = ma.integration.read_json(json_path)
        result = list(ma.read_ipc_file(str(arrow_path)))
        assert len(result) == len(expected), (
            f"Expected {len(expected)} batches, got {len(result)}"
        )
        _assert_equal(
            pa.schema(schema),
            pa.schema(ma.read_ipc_file_schema(str(arrow_path)).schema),
            "Schema",
        )
        for i, (e, r) in enumerate(zip(expected, result)):
            _assert_equal(pa.record_batch(e), pa.record_batch(r), f"Batch {i}")

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
# C Data exporter and importer
# ---------------------------------------------------------------------------


class MarrowCDataExporter(CDataExporter):
    """Export the JSON's schema and batches through marrow's C Data export.

    archery samples `record_allocation_state` before an export and after the
    importer is done: the count of heap blocks marrow's C Data structs hold must
    return to where it was, or the consumer leaked what marrow exported.
    """

    @property
    def supports_releasing_memory(self) -> bool:
        return True

    def record_allocation_state(self):
        return ma.integration.c_data_allocations()

    def run_gc(self):
        gc.collect()

    def export_schema_from_json(self, json_path, c_schema_ptr):
        schema, _ = ma.integration.read_json(json_path)
        schema._export_to_c(_address(c_schema_ptr))

    def export_batch_from_json(self, json_path, num_batch: int, c_array_ptr):
        _, batches = ma.integration.read_json(json_path)
        batches[num_batch]._export_to_c(_address(c_array_ptr))


class MarrowCDataImporter(CDataImporter):
    """Import archery's structs through marrow's C Data import, then validate
    every value before comparing, as Arrow C++ and arrow-rs do."""

    @property
    def supports_releasing_memory(self) -> bool:
        return True

    def run_gc(self):
        gc.collect()

    def import_schema_and_compare_to_json(self, json_path, c_schema_ptr):
        schema, _ = ma.integration.read_json(json_path)
        result = ma.Schema._import_from_c(_address(c_schema_ptr))
        _assert_equal(pa.schema(schema), pa.schema(result), "Schema")

    def import_batch_and_compare_to_json(self, json_path, num_batch: int, c_array_ptr):
        schema, batches = ma.integration.read_json(json_path)
        result = ma.RecordBatch._import_from_c(_address(c_array_ptr), schema)
        result.validate(full=True)
        _assert_equal(
            pa.record_batch(batches[num_batch]),
            pa.record_batch(result),
            f"Batch {num_batch}",
        )


# ---------------------------------------------------------------------------
# C Stream and C Device, against Arrow C++
# ---------------------------------------------------------------------------


class InterfacePhases:
    """The C Stream and C Device interfaces, which archery does not exercise,
    against Arrow C++ through pyarrow -- the way arrow-rs tests its C Stream.

    Every case of the corpus goes both ways through each interface. The log
    follows archery's, so `ArcheryReport` counts these phases with the rest.
    """

    BANNER = "#" * 58

    def __init__(self, files, match=None):
        from archery.integration.util import SKIP_C_ARRAY, SKIP_C_SCHEMA

        def usable(case):
            return (match is None or match in case.name) and not any(
                case.should_skip(tester, fmt)
                for tester in ("Mojo", "C++")
                for fmt in (SKIP_C_SCHEMA, SKIP_C_ARRAY)
            )

        self.files = [case for case in files if usable(case)]

    def run(self):
        """Every phase; whether all of them passed."""
        phases = [
            ("C Stream: Mojo exporting, C++ importing", self._stream_to_cpp),
            ("C Stream: C++ exporting, Mojo importing", self._stream_from_cpp),
            ("C Device: Mojo exporting, C++ importing", self._device_to_cpp),
            ("C Device: C++ exporting, Mojo importing", self._device_from_cpp),
        ]
        passed = True
        for title, check in phases:
            print(f"{self.BANNER}\n{title}\n{self.BANNER}")
            for case in self.files:
                print("=" * 70)
                print(f"Testing file {case.path}")
                schema, batches = ma.integration.read_json(case.path)
                try:
                    check(schema, batches)
                    print("-- Validating")
                except Exception as error:
                    passed = False
                    print(f"-- FAILED: {type(error).__name__}: {error}")
        return passed

    @staticmethod
    def _expected(schema, batches):
        return pa.Table.from_batches(
            [pa.record_batch(b) for b in batches], schema=pa.schema(schema)
        )

    def _stream_to_cpp(self, schema, batches):
        table = ma.Table.from_batches(batches, schema=schema)
        result = pa.RecordBatchReader.from_stream(table).read_all()
        _assert_equal(self._expected(schema, batches), result, "Stream")

    def _stream_from_cpp(self, schema, batches):
        expected = self._expected(schema, batches)
        _assert_equal(expected, pa.table(ma.table(expected)), "Stream")

    @staticmethod
    def _structs(*kinds):
        from pyarrow.cffi import ffi

        return [ffi.new(f"struct {kind} *") for kind in kinds]

    def _device_to_cpp(self, schema, batches):
        for i, batch in enumerate(batches):
            c_schema, c_array = self._structs("ArrowSchema", "ArrowDeviceArray")
            batch._export_to_c_device(_address(c_array), _address(c_schema))
            imported = pa.Schema._import_from_c(_address(c_schema))
            result = pa.RecordBatch._import_from_c_device(_address(c_array), imported)
            _assert_equal(pa.record_batch(batch), result, f"Batch {i}")

    def _device_from_cpp(self, schema, batches):
        for i, batch in enumerate(batches):
            expected = pa.record_batch(batch)
            (c_array,) = self._structs("ArrowDeviceArray")
            expected._export_to_c_device(_address(c_array))
            result = ma.RecordBatch._import_from_c_device(_address(c_array), schema)
            _assert_equal(expected, pa.record_batch(result), f"Batch {i}")


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

        phases = defaultdict(lambda: {"pass": set(), "skip": set(), "seen": set()})
        for index in range(len(banners) - 1):
            start, phase = banners[index]
            body = log[start : banners[index + 1][0]]
            case = None
            for line in body.splitlines():
                match = cls.FILE_CASE.match(line) or cls.C_DATA_CASE.match(line)
                if match is not None:
                    case = match.group(1)
                    phases[phase]["seen"].add(case)
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
            cases |= counts["pass"] | counts["skip"] | counts["seen"]
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

        # Out of the phases that ran the case: a gold file runs in one phase,
        # and a case that failed somewhere must still show the phase it lost.
        cases = Table(title="Per-case coverage", box=box.SIMPLE_HEAD)
        cases.add_column("Case", no_wrap=True)
        cases.add_column("Phases passing", justify="right")
        coverage = self.coverage()
        for case, count in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0])):
            ran = sum(1 for counts in self.phases.values() if case in counts["seen"])
            cases.add_row(case, f"{count} / {ran}")
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

    #: archery's gold files: IPC written by earlier Arrow C++ releases, in the
    #: `testing` submodule (apache/arrow-testing, pinned to Arrow 24's commit).
    GOLD_ROOT = (
        Path(__file__).resolve().parents[1]
        / "testing/data/arrow-ipc-stream/integration"
    )

    #: Gold directories marrow does not read: it refuses big-endian IPC.
    UNSUPPORTED_GOLD = frozenset({"1.0.0-bigendian"})

    #: Layouts marrow does not implement.
    UNSUPPORTED = frozenset(
        {
            "union",
            "list_view",
            "extension",
            "run_end_encoded",
        }
    )

    # Expected partial coverage, and not marrow bugs:
    #
    # decimal32 / decimal64: Rust and Go implement neither type in IPC or C
    # Data, so their phases are skipped.
    #
    # binary_no_batches / primitive_no_batches: these files contain zero
    # record batches.  The IPC phases pass because the schema is still
    # exchanged; the C Data array phases iterate over batches, and zero batches
    # produce zero results, which archery counts as zero passes rather than one.

    def __init__(self):
        self._patched = False

    @staticmethod
    def marrow_cases(dg):
        """The cases archery's corpus leaves out, built from its own field
        classes so every implementation reads them as it reads the corpus.

        The corpus never generates `float16`, and nests little beyond `int32`
        and `utf8`: these put every layout marrow implements inside a struct,
        and structs inside every container.
        """
        import numpy as np

        class HalfFloatField(dg.FloatingPointField):
            """`float16`, spelled as Arrow C++ reads it: the uint16 bits."""

            def __init__(self, name, **kwargs):
                super().__init__(name, 16, **kwargs)

            def generate_column(self, size, name=None):
                halves = (np.random.randn(size) * 100).astype(np.float16)
                bits = [int(b) for b in halves.view(np.uint16)]
                return dg.PrimitiveColumn(
                    name or self.name, size, self._make_is_valid(size), bits
                )

        def flat():
            """One field of every non-nested type Rust and Go implement too."""
            names = [
                "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                "uint32", "uint64", "float32", "float64", "binary", "utf8",
                "largebinary", "largeutf8", "fixedsizebinary_7",
            ]  # fmt: skip
            return [
                dg.NullField("null"),
                *(dg.get_field(name, name) for name in names),
                dg.DecimalField("decimal128", 20, 3, 128),
                dg.DecimalField("decimal256", 60, 5, 256),
                dg.DateField("date32", dg.DateField.DAY),
                dg.DateField("date64", dg.DateField.MILLISECOND),
                dg.TimeField("time32", "ms"),
                dg.TimeField("time64", "ns"),
                dg.TimestampField("timestamp", "us", tz="UTC"),
                dg.DurationIntervalField("duration", "ns"),
                dg.YearMonthIntervalField("year_month"),
                dg.DayTimeIntervalField("day_time"),
                dg.MonthDayNanoIntervalField("month_day_nano"),
            ]

        def nested(dictionary):
            """One field of every nested type, and a dictionary."""
            return [
                dg.ListField("list", dg.get_field("item", "int32")),
                dg.LargeListField("large_list", dg.get_field("item", "utf8")),
                dg.FixedSizeListField(
                    "fixed_size_list", dg.get_field("item", "int16"), 3
                ),
                dg.MapField(
                    "map",
                    dg.get_field("key", "utf8", nullable=False),
                    dg.get_field("value", "float64"),
                ),
                dg.StructField(
                    "struct", [dg.get_field("a", "int32"), dg.get_field("b", "utf8")]
                ),
                dg.DictionaryField("dictionary", dg.get_field("", "int16"), dictionary),
            ]

        def case(name, fields, dictionaries=()):
            batches = [
                dg.RecordBatch(size, [f.generate_column(size) for f in fields])
                for size in (7, 10)
            ]
            return dg.File(name, dg.Schema(fields), batches, list(dictionaries))

        every = dg.Dictionary(0, dg.StringField("dictionary0"), size=5)
        inner = dg.Dictionary(0, dg.StringField("dictionary0"), size=5)
        row = [
            dg.get_field("int64", "int64"),
            dg.DecimalField("decimal128", 20, 3, 128),
            dg.TimestampField("timestamp", "ns", tz="Europe/Paris"),
            *nested(inner),
        ]
        return [
            # Only C++ reads `HALF` as bits; Go reads the same numbers as
            # values and Rust not at all, which is why archery leaves it out.
            case(
                "marrow_half_float",
                [HalfFloatField("f16"), HalfFloatField("f16_nn", nullable=False)],
            )
            .skip_tester("Rust")
            .skip_tester("Go"),
            case(
                "marrow_struct_of_every_type",
                [dg.StructField("every_type", flat() + nested(every))],
                [every],
            ),
            case(
                "marrow_structs_in_containers",
                [
                    dg.ListField("list", dg.StructField("item", row)),
                    dg.LargeListField("large_list", dg.StructField("item", row)),
                    dg.FixedSizeListField(
                        "fixed_size_list", dg.StructField("item", row), 2
                    ),
                    dg.MapField(
                        "map",
                        dg.get_field("key", "utf8", nullable=False),
                        dg.StructField("value", row),
                    ),
                    dg.StructField("struct", [dg.StructField("inner", row)]),
                ],
                [inner],
            ),
            # Rust implements neither view type; see archery's `binary_view`.
            case(
                "marrow_nested_views",
                [
                    dg.StructField(
                        "views",
                        [dg.StringViewField("utf8"), dg.BinaryViewField("binary")],
                    ),
                    dg.ListField("list", dg.StringViewField("item")),
                ],
            ).skip_tester("Rust"),
            # Rust and Go implement neither width; see archery's `decimal32`.
            case(
                "marrow_nested_small_decimals",
                [
                    dg.StructField(
                        "decimals",
                        [
                            dg.DecimalField("decimal32", 7, 2, 32),
                            dg.DecimalField("decimal64", 15, 3, 64),
                        ],
                    )
                ],
            )
            .skip_tester("Rust")
            .skip_tester("Go"),
        ]

    @classmethod
    def gold_dirs(cls):
        """Every gold directory in the `testing` submodule, or none if it is
        not checked out."""
        if not cls.GOLD_ROOT.is_dir():
            print(
                "arrow-testing is not checked out; gold files skipped "
                "(git submodule update --init testing)"
            )
            return []
        return sorted(str(d) for d in cls.GOLD_ROOT.iterdir() if d.is_dir())

    def patch_archery(self):
        """Add marrow's cases, and skip what marrow does not implement, in
        both the generated corpus and the gold files."""
        if self._patched:
            return
        from archery.integration import datagen, runner

        original_gold = runner.IntegrationRunner._gold_tests

        def gold_tests(runner_self, gold_dir):
            prefix = os.path.basename(os.path.normpath(gold_dir))
            for case in original_gold(runner_self, gold_dir):
                if prefix in self.UNSUPPORTED_GOLD or case.name in self.UNSUPPORTED:
                    case.skip_tester("Mojo")
                yield case

        runner.IntegrationRunner._gold_tests = gold_tests

        original = datagen.get_generated_json_files

        def patched(tempdir=None):
            files = original(tempdir)
            tempdir = os.path.dirname(files[0].path)
            for case in self.marrow_cases(datagen):
                case.write(os.path.join(tempdir, f"generated_{case.name}.json"))
                files.append(case)
            for entry in files:
                if entry.name in self.UNSUPPORTED:
                    entry.skip_tester("Mojo")
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

        self.patch_archery()
        if gold_dirs is None:
            gold_dirs = self.gold_dirs()

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
        if run_c_data and with_cpp:
            from archery.integration import datagen

            files = datagen.get_generated_json_files()
            with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
                passed &= InterfacePhases(files, match=match).run()
        ArcheryReport(buffer.getvalue()).render()
        return passed
