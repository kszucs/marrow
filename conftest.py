"""The pytest boundary.

Every decision this file makes lives in `devkit/`; what stays here is the part
that is pytest API by definition -- the hooks, the collector classes for
`.mojo` files, and one `Harness` that holds the session's state.

Two modules under `devkit/` reach into pytest, both from inside a function
rather than at import: `benches` needs `pytest_benchmark` to inject Mojo
timings, and `golden` needs `pytest.mark.skip` to carry a case's `-- skip
python` marker into the runtime lane.  Nothing else does, which is what lets
the rest be tested with plain objects instead of a fake config.
"""

import contextlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from devkit.benches import (  # noqa: E402 - must follow the path insertion
    BenchmarkEnvelope,
    BenchmarkGrouping,
    BenchmarkHistory,
    BenchmarkInjector,
    CompetitionReport,
)
from devkit.mojo import (  # noqa: E402
    AsanRuntime,
    BuildOptions,
    MojoToolchain,
    ProcessRunner,
    Repo,
)
from devkit.progress import ConsoleProgress  # noqa: E402
from devkit.runner import (  # noqa: E402
    CaseRef,
    CaseScanner,
    DriverGenerator,
    LaneSelector,
    RunnerOptions,
    Selection,
    SuiteRunner,
)

# This file used to append "throughput" to `pytest_benchmark.utils.ALLOWED_COLUMNS`
# so that `--benchmark-columns=throughput` would be accepted.  It never worked and
# has been removed: `--benchmark-columns` carries `type=parse_columns`, and argparse
# runs a type converter during `Config._preparse`, which is *before* the rootdir
# conftest is imported.  A conftest cannot patch it in time; only a plugin
# registered by entry point or `-p` could.  Verified against the pre-refactor file
# -- both fail with "Invalid column name(s): throughput".
#
# The throughput figure reaches the report the other way, and always did: as an
# `extra_info` key, written by `BenchmarkGrouping.annotate_throughput` and read by
# `CompetitionReport`.


class MojoTestFailure(Exception):
    pass


class Harness:
    """Everything one pytest session needs from devkit, built once.

    Both suites are lazy and memoised: the first item to run compiles the whole
    selection as a single unit, and every later item just reads its own entry.
    """

    def __init__(self, config):
        self.config = config
        self.repo = Repo(config.rootpath)
        self.options = RunnerOptions.from_config(config)
        self.lanes = LaneSelector(
            self.options,
            config.args,
            self.repo,
            # Where pytest was launched, which is what its relative path
            # arguments are resolved against -- not the rootdir.
            invocation_dir=config.invocation_params.dir,
        )
        self.toolchain = MojoToolchain(
            ProcessRunner(
                self.repo.root,
                ConsoleProgress(),
                timeout=self.options.mojo_timeout,
                suspend=self._lift_capture,
            ),
            self._asan_runtime(),
        )
        self._selections = {}
        self._results = {}

    def _asan_runtime(self):
        if not self.options.asan:
            return None
        runtime = AsanRuntime.locate()
        if runtime is None:
            pytest.exit(
                "ASAN requested but no compatible ASAN runtime found. "
                "Install libcompiler-rt via conda-forge.",
                returncode=1,
            )
        return runtime

    def _lift_capture(self):
        """Suspend pytest's output capture so the progress display is visible."""
        capman = self.config.pluginmanager.getplugin("capturemanager")
        return (
            capman.global_and_fixture_disabled() if capman else contextlib.nullcontext()
        )

    # -- the Mojo suites ----------------------------------------------------

    def select(self, kind, items):
        """Record which cases the *kind* runner has to cover."""
        self._selections[kind] = Selection(
            CaseRef(Path(item.fspath), item.name) for item in items
        )

    def results(self, kind):
        if kind not in self._results:
            selection = self._selections.get(kind) or Selection()
            self._results[kind] = self._suite(kind).run(selection)
        return self._results[kind]

    def _suite(self, kind):
        build = BuildOptions.for_benches if kind == "bench" else BuildOptions.for_tests
        return SuiteRunner(
            self.toolchain,
            DriverGenerator(self.repo, kind),
            build(gpu=self.options.gpu, asan=self.options.asan),
            notify=lambda message: print(message, flush=True),
        )

    # -- the Python lane ----------------------------------------------------

    def build_libmarrow(self):
        result = self.toolchain.build_shared_lib(
            self.repo.bindings_entry,
            self.repo.libmarrow,
            BuildOptions.for_shared_lib(
                bench=self.options.benchmark, asan=self.options.asan
            ),
            f"compiling {self.repo.libmarrow.relative_to(self.repo.root)}",
        )
        if not result.ok:
            pytest.exit(result.failure(f"Failed to build {self.repo.libmarrow}"), 1)

    # -- benchmarks ---------------------------------------------------------

    @property
    def benchmark_session(self):
        return getattr(self.config, "_benchmarksession", None)

    @property
    def injector(self):
        session = self.benchmark_session
        return BenchmarkInjector(session) if session is not None else None

    def history(self):
        return BenchmarkHistory(
            self.repo.root / self.options.save_benchmarks,
            self.options.benchmark_history,
            root=self.repo.root,
        )


def harness(config):
    return config._devkit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    RunnerOptions.declare(parser)


def pytest_configure(config):
    config.addinivalue_line("markers", "mojo: Mojo language tests")
    config.addinivalue_line("markers", "python: Python tests")
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmarks (skipped by default, run with --benchmark)",
    )
    config._devkit = Harness(config)


def pytest_sessionstart(session):
    """Rebuild `libmarrow.so` when this session will run something that imports it."""
    config = session.config
    if hasattr(config, "workerinput"):
        return  # an xdist worker; the controller already built it
    if harness(config).lanes.needs_libmarrow():
        harness(config).build_libmarrow()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def pytest_ignore_collect(collection_path, config):
    if not (
        collection_path.suffix == ".py"
        and collection_path.name.startswith(("test_", "bench_"))
    ):
        return None
    # Asked here as well as during gating, because this hook runs before items
    # exist and so is the only place that can answer it.  Without it
    # `pytest --mojo` collects none of the harness's own tests, which is the
    # same hole `owns` exists to close, one hook earlier.
    if harness(config).lanes.owns(collection_path):
        return None
    if harness(config).lanes.python_excluded():
        return True
    # Python bench files are collected only under --benchmark, mirroring the
    # treatment of Mojo bench_*.mojo files.
    if (
        collection_path.name.startswith("bench_")
        and not harness(config).options.benchmark
    ):
        return True
    # None, never False: the hook is firstresult, so False would *force*
    # collection and stop every other plugin's opinion rather than withhold
    # one of our own.
    return None


def pytest_collect_file(parent, file_path):
    if file_path.suffix != ".mojo":
        return None
    if file_path.name.startswith("test_"):
        return MojoTestFile.from_parent(parent, path=file_path)
    if file_path.name.startswith("bench_"):
        return MojoBenchFile.from_parent(parent, path=file_path)
    return None


def pytest_itemcollected(item):
    """Mark a Python item with its lane -- unless it is one of devkit's own.

    The devkit suite tests the harness; it is not marrow's Python lane, and
    gating it on `--mojo`/`--python` would make the harness's own tests subject
    to the code they exist to check.  That is not theoretical: a one-character
    mutation in `skip_reason` that skips *every* item survived the suite,
    because it skipped the tests that would have caught it, and pytest exits 0
    on an all-skipped run.
    """
    if item.fspath.ext != ".py":
        return
    if harness(item.config).lanes.owns(Path(item.fspath)):
        return
    item.add_marker(pytest.mark.python)
    if item.fspath.basename.startswith("bench_"):
        item.add_marker(pytest.mark.benchmark)


def pytest_collection_modifyitems(config, items):
    lanes = harness(config).lanes
    for item in items:
        # The harness's own tests are never gated: they check this very
        # function, so letting it reach them lets a bug here skip the tests
        # that would have caught it -- and an all-skipped run exits 0.
        #
        # This reads as redundant with `pytest_itemcollected` declining to mark
        # them, and is not: that only makes `skip_reason` answer None *while it
        # is correct*.  Removing this line and mutating `skip_reason` to skip
        # everything gives "158 skipped" and exit 0 -- which is the failure
        # this guards against, so it has to be asked here too.
        if lanes.owns(Path(item.fspath)):
            continue
        is_gpu = "gpu" in item.keywords
        reason = lanes.skip_reason(
            is_gpu=is_gpu,
            is_mojo="mojo" in item.keywords and not is_gpu,
            is_python="python" in item.keywords,
            is_benchmark="benchmark" in item.keywords,
        )
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_collection_finish(session):
    """Pre-compute the selection each generated runner has to cover.

    Tests and benchmarks are recorded separately: they build at different
    optimization levels, so they cannot share a compilation unit.

    Split on the item's own `KIND`.  The two item classes are siblings under
    `MojoItem` precisely so this can be a plain attribute read rather than an
    isinstance test that a shared base would make ambiguous.
    """
    for kind in ("test", "bench"):
        harness(session.config).select(
            kind,
            [
                item
                for item in session.items
                if isinstance(item, MojoItem)
                and item.KIND == kind
                and not any(mark.name == "skip" for mark in item.iter_markers())
            ],
        )


# ---------------------------------------------------------------------------
# Mojo collectors
# ---------------------------------------------------------------------------


class MojoItem(pytest.Item):
    """One case in a generated Mojo runner.

    `MojoTestItem` and `MojoBenchItem` are *siblings*, not parent and child:
    they share how a result is looked up, but a benchmark is not a kind of
    test -- it compiles at a different optimization level, into a different
    unit.  Making one inherit the other put every benchmark in the test
    selection under `isinstance`, which is why the split reads `KIND`.
    """

    #: Which runner this case belongs to; also the key `Harness.results` uses.
    KIND = ""

    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_marker(pytest.mark.mojo)

    def result(self):
        # The first item to run compiles and executes the runner for the whole
        # selection; every later item just reads its own entry.  Case names are
        # unique across the suite, so the name alone identifies the result.
        found = harness(self.config).results(self.KIND).get(self.name)
        if found is None:
            raise MojoTestFailure(f"{self.name} did not appear in the runner output")
        if found.failed:
            raise MojoTestFailure(found.error)
        return found

    def runtest(self):
        self.result()

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::{self.KIND}::{self.name}"


class MojoTestFile(pytest.File):
    def collect(self):
        is_gpu = CaseScanner.is_gpu(self.path)
        for name in CaseScanner.cases(self.path, "test"):
            yield MojoTestItem.from_parent(self, name=name, is_gpu=is_gpu)


class MojoTestItem(MojoItem):
    KIND = "test"

    def __init__(self, name, parent, is_gpu=False):
        super().__init__(name, parent)
        if is_gpu:
            self.add_marker(pytest.mark.gpu)


class MojoBenchFile(pytest.File):
    def collect(self):
        is_gpu = CaseScanner.is_gpu(self.path)
        for name in CaseScanner.cases(self.path, "bench"):
            yield MojoBenchItem.from_parent(self, name=name, is_gpu=is_gpu)


class MojoBenchItem(MojoItem):
    """One `def bench_*(mut b: Benchmark)`, timed by the Mojo runner.

    The selection compiles and runs as a single -O3 unit; the timings it reports
    are injected into pytest-benchmark afterwards, so a Mojo benchmark lands in
    the same table as a Python one.
    """

    KIND = "bench"

    def __init__(self, name, parent, is_gpu=False):
        super().__init__(name, parent)
        self.add_marker(pytest.mark.benchmark)
        if is_gpu:
            self.add_marker(pytest.mark.gpu)

    def runtest(self):
        result = self.result()
        injector = harness(self.config).injector
        if injector is not None:
            injector.inject(self.name, self._nodeid, result.entry)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def pytest_benchmark_group_stats(config, benchmarks, group_by):
    return BenchmarkGrouping.group(benchmarks, group_by)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not harness(config).options.competition:
        return
    session = harness(config).benchmark_session
    if session is None or not session.benchmarks:
        return
    terminalreporter.ensure_newline()
    terminalreporter.write_line("")
    for line in CompetitionReport(session.benchmarks).render():
        terminalreporter.write_line(line)


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not harness(config).options.save_benchmarks:
        return
    if hasattr(config, "workerinput"):
        return  # only the controller writes results
    benchmark_session = harness(config).benchmark_session
    if benchmark_session is None or not benchmark_session.benchmarks:
        return

    history = harness(config).history()
    envelope = BenchmarkEnvelope.from_benchmarks(
        harness(config).repo.vcs, benchmark_session.benchmarks
    )
    written, count, runs = history.save(envelope)
    print(f"\n--save-benchmarks: {count} entries written to {written}")
    print(f"--save-benchmarks: {runs} run(s) in {history.history_file}")
