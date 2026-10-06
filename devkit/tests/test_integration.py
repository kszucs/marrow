# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""That the archery module is importable at all, its skip sets, and its report.

`archery` is only installed in the `integration` environment, and it in turn
needs the apache/arrow clone.  These tests stub the three base classes marrow
subclasses so the module can be imported anywhere -- which is what catches a
name error or a bad signature without a ninety-minute Arrow build.

The testers themselves drive `libmarrow.so` and run only in the suite; the
integration JSON reader they rely on is tested in `marrow/tests/test_integration.mojo`.
"""

import sys
import types

import pytest


@pytest.fixture(scope="module")
def integration():
    """Import `devkit.integration` against stubbed archery base classes.

    The stubs come back out afterwards.  Left in `sys.modules` they outlive this
    file and tell every later test in the session that archery is installed,
    when what is installed is three empty modules -- so a test that means to
    assert "this environment has no archery" quietly asserts nothing, and which
    way it goes depends on collection order.
    """
    installed = {}
    if "archery.integration.tester" not in sys.modules:
        tester = types.ModuleType("archery.integration.tester")
        for name in ("Tester", "CDataExporter", "CDataImporter"):
            setattr(tester, name, type(name, (), {}))
        installed = {
            "archery": types.ModuleType("archery"),
            "archery.integration": types.ModuleType("archery.integration"),
            "archery.integration.tester": tester,
        }
        sys.modules.update(installed)
    import devkit
    import devkit.integration as module

    yield module

    if installed:
        for name in installed:
            del sys.modules[name]
        # Both halves: `devkit` keeps its own attribute for the submodule, so
        # dropping only the `sys.modules` entry still hands a later
        # `devkit.integration` the module that was built on the stubs.
        del sys.modules["devkit.integration"]
        del devkit.integration


def test_the_module_imports_and_exposes_the_participants(integration):
    """A name error here is otherwise only visible to a 90-minute CI job."""
    assert integration.MarrowTester.name == "Mojo"
    # Marrow produces and consumes both wire formats; Flight is out of scope.
    for flag in ("PRODUCER", "CONSUMER", "C_DATA_SCHEMA_EXPORTER"):
        assert getattr(integration.MarrowTester, flag) is True
    assert integration.MarrowCDataExporter and integration.MarrowCDataImporter
    assert integration.ArcherySuite and integration.ArcheryReport


def test_the_skip_sets_name_only_unimplemented_layouts(integration):
    unsupported = integration.ArcherySuite.UNSUPPORTED
    assert unsupported == {"union", "list_view", "extension", "run_end_encoded"}


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
    monkeypatch.setattr(suite, "patch_archery", lambda: None)
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


def test_the_skip_sets_are_applied_to_the_generated_files(integration, monkeypatch):
    """Asserting the sets' *contents* does not prove anything consults them:
    marrow's own cases land beside the generated ones, and the skips reach
    both those and the gold files."""

    class Entry:
        def __init__(self, name):
            self.name = name
            self.path = f"/tmp/generated/generated_{name}.json"
            self.skipped_testers = []

        def skip_tester(self, tester):
            self.skipped_testers.append(tester)

        def write(self, path):
            self.path = path

    entries = [Entry("union"), Entry("nested_dictionary"), Entry("primitive")]
    extra = Entry("marrow_struct_of_every_type")
    datagen = types.ModuleType("archery.integration.datagen")
    datagen.get_generated_json_files = lambda tempdir=None: list(entries)
    monkeypatch.setitem(sys.modules, "archery.integration.datagen", datagen)

    class IntegrationRunner:
        def _gold_tests(self, gold_dir):
            yield from (Entry("primitive"), Entry("union"))

    runner = types.ModuleType("archery.integration.runner")
    runner.IntegrationRunner = IntegrationRunner
    monkeypatch.setitem(sys.modules, "archery.integration.runner", runner)
    monkeypatch.setattr(
        integration.ArcherySuite, "marrow_cases", staticmethod(lambda dg: [extra])
    )
    integration.ArcherySuite().patch_archery()
    files = datagen.get_generated_json_files()

    assert [f.name for f in files][-1] == "marrow_struct_of_every_type"
    assert extra.path == "/tmp/generated/generated_marrow_struct_of_every_type.json"
    assert [f.skipped_testers for f in files] == [["Mojo"], [], [], []]

    def gold(prefix):
        cases = IntegrationRunner()._gold_tests(f"/gold/{prefix}")
        return [case.skipped_testers for case in cases]

    assert gold("1.0.0-bigendian") == [["Mojo"], ["Mojo"]]
    assert gold("1.0.0-littleendian") == [[], ["Mojo"]]
