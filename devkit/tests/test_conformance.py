"""The Arrow-JSON bridge, and that the archery module is importable at all.

`archery` is only installed in the `integration` environment, and it in turn
needs the apache/arrow clone.  These tests stub the three base classes marrow
subclasses so the module can be imported anywhere -- which is what catches a
name error or a bad signature without a ninety-minute Arrow build.

The JSON conversion itself is pure pyarrow and is tested for real: neither
archery nor pyarrow exposes that conversion, so it is marrow's own code and the
only part of the suite that can be checked here.
"""

import sys
import types

import pyarrow as pa
import pytest


@pytest.fixture(scope="module")
def integration():
    """Import `devkit.conformance` against stubbed archery base classes.

    The stubs come back out afterwards.  Left in `sys.modules` they outlive this
    file and tell every later test in the session that archery is installed,
    when what is installed is four empty modules -- so a test that means to
    assert "this environment has no archery" quietly asserts nothing, and which
    way it goes depends on collection order.
    """
    installed = {}
    if "archery.integration.tester" not in sys.modules:
        tester = types.ModuleType("archery.integration.tester")
        for name in ("Tester", "CDataExporter", "CDataImporter"):
            setattr(tester, name, type(name, (), {}))
        util = types.ModuleType("archery.integration.util")
        util.SKIP_C_ARRAY = "c_array"
        util.SKIP_C_SCHEMA = "c_schema"
        installed = {
            "archery": types.ModuleType("archery"),
            "archery.integration": types.ModuleType("archery.integration"),
            "archery.integration.tester": tester,
            "archery.integration.util": util,
        }
        sys.modules.update(installed)
    import devkit
    import devkit.conformance as module

    yield module

    if installed:
        for name in installed:
            del sys.modules[name]
        # Both halves: `devkit` keeps its own attribute for the submodule, so
        # dropping only the `sys.modules` entry still hands a later
        # `devkit.conformance` the module that was built on the stubs.
        del sys.modules["devkit.conformance"]
        del devkit.conformance


def test_the_module_imports_and_exposes_the_participants(integration):
    """A name error here is otherwise only visible to a 90-minute CI job."""
    assert integration.MarrowTester.name == "Mojo"
    # Marrow produces and consumes both wire formats; Flight is out of scope.
    for flag in ("PRODUCER", "CONSUMER", "C_DATA_SCHEMA_EXPORTER"):
        assert getattr(integration.MarrowTester, flag) is True
    assert integration.MarrowCDataExporter and integration.MarrowCDataImporter
    assert integration.ArcherySuite and integration.ArcheryReport


def test_the_skip_sets_say_why(integration):
    suite = integration.ArcherySuite
    # `interval` is a harness limit, not a library one -- pyarrow has no type
    # for YEAR_MONTH or DAY_TIME, so the bridge cannot build the column.
    assert "interval" in suite.UNSUPPORTED
    assert "union" in suite.UNSUPPORTED
    assert suite.SKIP_C_DATA == frozenset({"nested_dictionary"})


# ---------------------------------------------------------------------------
# The JSON bridge
# ---------------------------------------------------------------------------


def field(name, type_obj, **extra):
    return {"name": name, "type": type_obj, "nullable": True, "children": [], **extra}


def test_primitive_types_convert(integration):
    convert = integration._json_field_to_pa
    cases = {
        "int64": ({"name": "int", "bitWidth": 64, "isSigned": True}, pa.int64()),
        "uint32": ({"name": "int", "bitWidth": 32, "isSigned": False}, pa.uint32()),
        "double": ({"name": "floatingpoint", "precision": "DOUBLE"}, pa.float64()),
        "single": ({"name": "floatingpoint", "precision": "SINGLE"}, pa.float32()),
        "utf8": ({"name": "utf8"}, pa.string()),
        "binary": ({"name": "binary"}, pa.binary()),
        "bool": ({"name": "bool"}, pa.bool_()),
        "null": ({"name": "null"}, pa.null()),
    }
    for label, (type_obj, expected) in cases.items():
        assert convert(field(label, type_obj)).type == expected, label


def test_unsupported_types_answer_none_rather_than_raising(integration):
    """The outer schema filters them out; raising would fail the whole case."""
    assert integration._json_field_to_pa(field("u", {"name": "union"})) is None


def test_nested_types_keep_their_child_field_names(integration):
    """The whole reason pyarrow is the intermediate: marrow's `list_` cannot."""
    child = field("item", {"name": "int", "bitWidth": 32, "isSigned": True})
    listed = integration._json_field_to_pa(
        {
            "name": "l",
            "type": {"name": "list"},
            "nullable": True,
            "children": [child],
        }
    )
    assert pa.types.is_list(listed.type)
    assert listed.type.value_field.name == "item"


def test_struct_children_are_preserved(integration):
    struct = integration._json_field_to_pa(
        {
            "name": "s",
            "type": {"name": "struct"},
            "nullable": True,
            "children": [
                field("a", {"name": "int", "bitWidth": 64, "isSigned": True}),
                field("b", {"name": "utf8"}),
            ],
        }
    )
    assert struct.type == pa.struct(
        [pa.field("a", pa.int64()), pa.field("b", pa.string())]
    )


def test_a_64_bit_column_arrives_as_strings(integration):
    """JSON cannot represent a 64-bit integer, so the format sends them quoted."""
    array = integration._json_col_to_pa(
        {
            "name": "v",
            "count": 3,
            "VALIDITY": [1, 0, 1],
            "DATA": ["1", "2", "9223372036854775807"],
        },
        pa.int64(),
    )
    assert array.to_pylist() == [1, None, 9223372036854775807]


def test_a_binary_column_arrives_as_hex(integration):
    array = integration._json_col_to_pa(
        {"name": "v", "count": 2, "VALIDITY": [1, 1], "DATA": ["00FF", ""]},
        pa.binary(),
    )
    assert array.to_pylist() == [b"\x00\xff", b""]


def test_validity_becomes_nulls(integration):
    array = integration._json_col_to_pa(
        {"name": "v", "count": 3, "VALIDITY": [1, 0, 1], "DATA": [True, False, True]},
        pa.bool_(),
    )
    assert array.to_pylist() == [True, None, True]


def test_a_schema_drops_only_the_unsupported_fields(integration):
    schema = integration._json_to_pa_schema(
        {
            "schema": {
                "fields": [
                    field("keep", {"name": "int", "bitWidth": 64, "isSigned": True}),
                    field("drop", {"name": "union"}),
                    field("also_keep", {"name": "utf8"}),
                ]
            }
        }
    )
    assert schema.names == ["keep", "also_keep"]


def test_the_dictionary_id_search_recurses(integration):
    fields = [
        {
            "name": "outer",
            "children": [
                {"name": "inner", "dictionary": {"id": 7}, "children": []},
            ],
        }
    ]
    found = integration._find_field_for_dict_id(fields, 7)
    assert found is not None and found["name"] == "inner"
    assert integration._find_field_for_dict_id(fields, 99) is None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_scrapes_phases_and_cases(integration):
    """`run_all_tests` prints its results and returns nothing structured."""
    banner = "#" * 58
    log = "\n".join(
        [
            banner,
            "Mojo producing, C++ consuming",
            banner,
            "Testing file /tmp/generated_primitive.json",
            "-- Validating file",
            "Testing file /tmp/generated_union.json",
            "-- Skipping test because producer Mojo does not support union",
        ]
    )
    report = integration.ArcheryReport(log)
    [(phase, counts)] = report.phases.items()
    assert phase == "Mojo producing, C++ consuming"
    assert counts["pass"] == {"primitive"}
    assert counts["skip"] == {"union"}


def test_a_case_that_passed_nothing_still_gets_a_row(integration):
    """An implementation that skips a layout everywhere is the interesting case.

    Counting only passes would drop it from the coverage table entirely.
    """
    banner = "#" * 58
    log = "\n".join(
        [
            banner,
            "Mojo producing, C++ consuming",
            banner,
            "Testing file /tmp/generated_union.json",
            "-- Skipping test because producer Mojo does not support union",
            banner,
            "C++ producing, Mojo consuming",
            banner,
            "Testing file /tmp/generated_union.json",
            "-- Skipping test because consumer Mojo does not support union",
        ]
    )
    assert integration.ArcheryReport(log).coverage() == {"union": 0}


def test_repeated_batches_count_as_one_passing_case(integration):
    """One file yields several `... with record batch` lines."""
    banner = "#" * 58
    log = "\n".join(
        [
            banner,
            "Mojo producing, C++ consuming",
            banner,
            "Testing file /tmp/generated_primitive.json",
            "-- Validating file",
            "... with record batch 0",
            "... with record batch 1",
        ]
    )
    assert integration.ArcheryReport(log).coverage() == {"primitive": 1}


def test_an_unparseable_log_renders_nothing(integration):
    integration.ArcheryReport("").render()  # must not raise


# ---------------------------------------------------------------------------
# The suite's verdict -- this is the CI gate
# ---------------------------------------------------------------------------


def run_suite(integration, monkeypatch, outcome):
    """Drive `ArcherySuite.run` with a stubbed `run_all_tests`."""
    import sys
    import types

    runner = types.ModuleType("archery.integration.runner")

    def run_all_tests(**kwargs):
        runner.seen = kwargs
        if outcome is not None:
            raise outcome

    runner.run_all_tests = run_all_tests
    monkeypatch.setitem(sys.modules, "archery.integration.runner", runner)

    suite = integration.ArcherySuite()
    monkeypatch.setattr(suite, "patch_datagen", lambda: None)
    monkeypatch.setattr(integration, "MarrowTester", lambda: object())
    return suite.run(run_ipc=True, run_c_data=False), runner


def test_a_clean_run_passes(integration, monkeypatch):
    passed, _ = run_suite(integration, monkeypatch, None)
    assert passed is True


def test_archery_signalling_failure_is_not_a_pass(integration, monkeypatch):
    """archery reports failure by exiting.

    Returning True regardless makes `pixi run integration` green forever, and
    nothing else in the suite would notice.
    """
    passed, _ = run_suite(integration, monkeypatch, SystemExit(1))
    assert not passed


def test_a_zero_exit_is_still_a_pass(integration, monkeypatch):
    passed, _ = run_suite(integration, monkeypatch, SystemExit(0))
    assert passed


def test_the_skip_sets_are_applied_to_the_generated_files(integration):
    """Asserting the sets' *contents* does not prove anything consults them."""
    import sys
    import types

    class Entry:
        def __init__(self, name):
            self.name = name
            self.skipped_testers = []
            self.skipped_formats = []

        def skip_tester(self, tester):
            self.skipped_testers.append(tester)

        def skip_format(self, fmt, tester):
            self.skipped_formats.append((fmt, tester))

    entries = [Entry("union"), Entry("nested_dictionary"), Entry("primitive")]
    datagen = types.ModuleType("archery.integration.datagen")
    datagen.get_generated_json_files = lambda tempdir=None: entries
    saved = sys.modules.get("archery.integration.datagen")
    sys.modules["archery.integration.datagen"] = datagen
    try:
        suite = integration.ArcherySuite()
        suite.patch_datagen()
        datagen.get_generated_json_files()
    finally:
        if saved is None:
            del sys.modules["archery.integration.datagen"]
        else:
            sys.modules["archery.integration.datagen"] = saved

    union, nested, primitive = entries
    assert union.skipped_testers == ["Mojo"]
    assert nested.skipped_formats and all(
        t == "Mojo" for _, t in nested.skipped_formats
    )
    assert not primitive.skipped_testers and not primitive.skipped_formats
