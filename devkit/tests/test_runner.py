# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Options, lanes, selections, drivers and the suite runner."""

import argparse
from pathlib import Path

import pytest

from devkit.mojo import BuildOptions, CommandResult, Repo
from devkit.runner import (
    CaseScanner,
    DriverGenerator,
    LaneSelector,
    RunnerOptions,
    Selection,
    SuiteRunner,
    define_arg,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / Repo.MARKER).touch()
    return Repo(tmp_path)


def selection(repo, **files):
    """`selection(repo, marrow__tests__test_arrays=["a", "b"])` -> a Selection."""
    return Selection.of(
        {repo.root / path.replace("__", "/"): names for path, names in files.items()}
    )


def two_file_selection(repo):
    return Selection.of(
        {
            repo.root / "marrow/kernels/tests/test_sort.mojo": ["test_sort_one"],
            repo.root / "marrow/tests/test_arrays.mojo": [
                "test_arrays_one",
                "test_arrays_two",
            ],
        }
    )


# ---------------------------------------------------------------------------
# RunnerOptions
# ---------------------------------------------------------------------------


class FakeConfig:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name, default=False):
        return self._options.get(name.lstrip("-").replace("-", "_"), default)


def test_options_read_the_config():
    options = RunnerOptions.from_config(FakeConfig(gpu=True, mojo_timeout=60))
    assert options.gpu and options.mojo_timeout == 60
    assert not options.asan and not options.benchmark


def test_saving_benchmarks_implies_running_them():
    """Saving results is pointless without producing them."""
    options = RunnerOptions.from_config(FakeConfig(save_benchmarks="results"))
    assert options.benchmark
    assert options.save_benchmarks == "results"


@pytest.mark.parametrize(
    "option", [{"competition_winner": "comptime"}, {"competition_json": "out.json"}]
)
def test_a_competition_verdict_or_record_implies_the_table_and_the_run(option):
    options = RunnerOptions.from_config(FakeConfig(**option))
    assert options.competition and options.benchmark


def test_the_competition_table_alone_runs_nothing_extra():
    options = RunnerOptions.from_config(FakeConfig(competition=True))
    assert options.competition and not options.benchmark


def test_defines_are_read_as_a_tuple():
    options = RunnerOptions.from_config(
        FakeConfig(define=["MARROW_GROUPBY_RADIX_MIN_ROWS=0", "X=1"])
    )
    assert options.define == ("MARROW_GROUPBY_RADIX_MIN_ROWS=0", "X=1")
    assert RunnerOptions.from_config(FakeConfig()).define == ()


def test_defines_are_refused_with_saved_benchmarks():
    """A define builds something other than the tree; its numbers must not land
    in the history under the tree's names."""
    with pytest.raises(ValueError, match="--save-benchmarks"):
        RunnerOptions.from_config(FakeConfig(define=["X=1"], save_benchmarks="out"))
    with pytest.raises(ValueError, match="--competition-json"):
        RunnerOptions.from_config(FakeConfig(define=["X=1"], competition_json="o"))


def test_leaks_are_refused_without_asan():
    """LeakSanitizer runs inside ASAN: alone, the flag would check nothing."""
    with pytest.raises(ValueError, match="--asan"):
        RunnerOptions.from_config(FakeConfig(leaks=True))
    assert RunnerOptions.from_config(FakeConfig(leaks=True, asan=True)).leaks


@pytest.mark.parametrize("text", ["X", "=1", "1X=2", "A-B=1"])
def test_define_arg_rejects_malformed_input(text):
    with pytest.raises(argparse.ArgumentTypeError):
        define_arg(text)


def test_define_arg_accepts_name_value():
    assert define_arg("MARROW_GPU=true") == "MARROW_GPU=true"
    assert define_arg("EMPTY=") == "EMPTY="


def test_options_declare_every_flag_it_reads():
    """A declared-but-unread or read-but-undeclared option is a silent no-op."""

    class RecordingParser:
        def __init__(self):
            self.declared = []

        def addoption(self, name, **kwargs):
            self.declared.append(name.lstrip("-").replace("-", "_"))

    parser = RecordingParser()
    RunnerOptions.declare(parser)
    fields = set(RunnerOptions().__dataclass_fields__)
    assert set(parser.declared) == fields


# ---------------------------------------------------------------------------
# LaneSelector
# ---------------------------------------------------------------------------


def lanes(rootpath, args=(), **options):
    # Relative arguments resolve against the *invocation* directory, which for
    # these is the scratch repository rather than the real cwd.
    return LaneSelector(
        RunnerOptions(**options), args, Repo(rootpath), invocation_dir=rootpath
    )


def test_cpu_selects_both_non_gpu_lanes(tmp_path):
    selector = lanes(tmp_path, cpu=True)
    assert selector.select_mojo and selector.select_python
    assert (
        selector.skip_reason(
            is_gpu=False, is_mojo=True, is_python=False, is_benchmark=False
        )
        is None
    )
    assert (
        selector.skip_reason(
            is_gpu=False, is_mojo=False, is_python=True, is_benchmark=False
        )
        is None
    )


def test_naming_one_lane_skips_the_other(tmp_path):
    selector = lanes(tmp_path, mojo=True)
    assert (
        selector.skip_reason(
            is_gpu=False, is_mojo=True, is_python=False, is_benchmark=False
        )
        is None
    )
    assert "Python tests excluded" in selector.skip_reason(
        is_gpu=False, is_mojo=False, is_python=True, is_benchmark=False
    )


def test_naming_no_lane_runs_everything_but_gpu_and_benchmarks(tmp_path):
    selector = lanes(tmp_path)
    assert not selector.selective
    base = dict(is_gpu=False, is_mojo=False, is_python=False, is_benchmark=False)
    for kind in ("is_mojo", "is_python"):
        assert selector.skip_reason(**{**base, kind: True}) is None
    assert "GPU tests excluded" in selector.skip_reason(
        is_gpu=True, is_mojo=False, is_python=False, is_benchmark=False
    )
    assert "benchmarks excluded" in selector.skip_reason(
        is_gpu=False, is_mojo=True, is_python=False, is_benchmark=True
    )


def test_the_negative_selectors_exclude_their_lane(tmp_path):
    """`--no-mojo` and `--no-python` are what CI uses to split a job."""
    assert "Mojo tests excluded" in lanes(tmp_path, no_mojo=True).skip_reason(
        is_gpu=False, is_mojo=True, is_python=False, is_benchmark=False
    )
    assert "Python tests excluded" in lanes(tmp_path, no_python=True).skip_reason(
        is_gpu=False, is_mojo=False, is_python=True, is_benchmark=False
    )


def test_no_gpu_beats_gpu(tmp_path):
    """CI passes both; the exclusion has to win or the job cannot pass."""
    selector = lanes(tmp_path, gpu=True, no_gpu=True)
    assert "GPU tests excluded" in selector.skip_reason(
        is_gpu=True, is_mojo=False, is_python=False, is_benchmark=False
    )


def test_benchmarks_are_skipped_before_any_lane_check(tmp_path):
    """--mojo --benchmark must still run Mojo benchmarks."""
    selector = lanes(tmp_path, mojo=True, benchmark=True)
    assert (
        selector.skip_reason(
            is_gpu=False, is_mojo=True, is_python=False, is_benchmark=True
        )
        is None
    )


def test_python_excluded_by_the_option_matrix(tmp_path):
    assert lanes(tmp_path, no_python=True).python_excluded()
    assert lanes(tmp_path, mojo=True).python_excluded()
    assert lanes(tmp_path, gpu=True).python_excluded()
    # ...unless the Python lane is named as well.
    assert not lanes(tmp_path, mojo=True, python=True).python_excluded()
    assert not lanes(tmp_path, gpu=True, cpu=True).python_excluded()
    assert not lanes(tmp_path).python_excluded()


def test_python_excluded_by_the_selected_paths(tmp_path):
    (tmp_path / "marrow" / "tests").mkdir(parents=True)
    (tmp_path / "marrow" / "tests" / "test_arrays.mojo").touch()
    (tmp_path / "python" / "marrow" / "tests").mkdir(parents=True)
    (tmp_path / "python" / "marrow" / "tests" / "test_array.py").touch()

    assert lanes(tmp_path, ["marrow/tests/test_arrays.mojo"]).python_excluded()
    assert lanes(tmp_path, ["marrow/tests"]).python_excluded()
    assert not lanes(tmp_path, ["python/marrow/tests/test_array.py"]).python_excluded()
    assert not lanes(tmp_path, ["python"]).python_excluded()
    # A node id still names its file.
    assert not lanes(
        tmp_path, ["python/marrow/tests/test_array.py::test_one"]
    ).python_excluded()


def test_devkit_tests_do_not_need_libmarrow(tmp_path):
    """They are Python tests that import nothing from marrow.

    Building the shared library for them would put a multi-minute Mojo compile
    in front of a suite that runs in seconds.
    """
    (tmp_path / "devkit" / "tests").mkdir(parents=True)
    (tmp_path / "devkit" / "tests" / "test_runner.py").touch()
    assert not lanes(tmp_path, ["devkit/tests"]).needs_libmarrow()
    assert not lanes(tmp_path, ["devkit"]).needs_libmarrow()


def test_the_repository_root_needs_libmarrow(tmp_path):
    """`pytest .` collects the Python suite, so it must rebuild the library.

    `relative_to` answers `Path(".")` for the root and its `parts` is empty;
    reading that as "names no tree" ran the whole Python suite against whatever
    `.so` happened to be on disk -- collected, never rebuilt, silently
    measuring the previous build.
    """
    for tree in ("python", "golden"):
        (tmp_path / tree).mkdir(parents=True, exist_ok=True)
        (tmp_path / tree / "test_x.py").touch()
    for argument in (".", "./", str(tmp_path)):
        assert lanes(tmp_path, [argument]).needs_libmarrow(), argument


def test_a_python_suite_outside_the_tree_needs_libmarrow(tmp_path):
    """Better a redundant build than a stale one.

    A path with no Python tests at all is excluded before this is asked, so the
    directory has to hold one for the question to arise.
    """
    outside = tmp_path.parent / f"elsewhere_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    (outside / "test_x.py").touch()
    selector = lanes(tmp_path, [str(outside)])
    assert not selector.python_excluded()
    assert selector.needs_libmarrow()


def test_relative_arguments_resolve_against_the_invocation_directory(tmp_path):
    """pytest run from a subdirectory names its files relative to *there*.

    Resolving against the rootdir instead turns every such argument into a path
    that does not exist, and the whole selection reads as "no Python tests".
    """
    tests = tmp_path / "python" / "marrow" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_arrays.py").touch()

    selector = LaneSelector(
        RunnerOptions(), ["test_arrays.py"], Repo(tmp_path), invocation_dir=tests
    )
    assert not selector.python_excluded()
    assert selector.needs_libmarrow()


def test_the_harness_owns_its_own_tests(tmp_path):
    """They test the lane logic, so the lane logic must not gate them."""
    selector = lanes(tmp_path)
    assert selector.owns(tmp_path / "devkit" / "tests" / "test_runner.py")
    assert not selector.owns(tmp_path / "python" / "marrow" / "tests" / "test_x.py")
    assert not selector.owns(tmp_path / "golden" / "test_cases.py")


def test_the_trees_that_do_need_libmarrow(tmp_path):
    for tree in ("python", "golden"):
        (tmp_path / tree).mkdir(parents=True, exist_ok=True)
        (tmp_path / tree / "test_x.py").touch()
        assert lanes(tmp_path, [tree]).needs_libmarrow()
    # A whole-repository run needs it too.
    assert lanes(tmp_path).needs_libmarrow()
    # A Mojo-only selection does not.
    assert not lanes(tmp_path, mojo=True).needs_libmarrow()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_groups_by_file_in_a_fixed_order(repo):
    grouped = two_file_selection(repo).by_file()
    assert [path.name for path in grouped] == ["test_sort.mojo", "test_arrays.mojo"]
    assert grouped[repo.root / "marrow/tests/test_arrays.mojo"] == [
        "test_arrays_one",
        "test_arrays_two",
    ]


def test_selection_halves_keeping_each_file_together(repo):
    left, right = two_file_selection(repo).halve()
    assert len(left) + len(right) == 3
    assert set(left.names()) | set(right.names()) == {
        "test_sort_one",
        "test_arrays_one",
        "test_arrays_two",
    }


def test_halving_a_single_case_terminates(repo):
    one = Selection.of({repo.root / "a.mojo": ["only"]})
    left, right = one.halve()
    assert (len(left), len(right)) == (0, 1)


# ---------------------------------------------------------------------------
# DriverGenerator
# ---------------------------------------------------------------------------


def test_driver_imports_every_selected_case(repo):
    driver = DriverGenerator(repo, "test").write(two_file_selection(repo))
    source = driver.read_text()

    assert driver.name.startswith("_test_driver_") and driver.suffix == ".mojo"
    assert "from marrow.utils.testing import TestSuite" in source
    assert (
        "from marrow.tests.test_arrays import test_arrays_one, test_arrays_two"
        in source
    )
    assert "from marrow.kernels.tests.test_sort import test_sort_one" in source
    assert "TestSuite.run[" in source
    for case in ("test_sort_one", "test_arrays_one", "test_arrays_two"):
        assert f"            {case},\n" in source


def test_driver_module_path_backticks_reserved_words(repo):
    generator = DriverGenerator(repo, "test")
    plain = repo.root / "marrow/expr/tests/test_relations.mojo"
    assert generator.module_path(plain) == "marrow.expr.tests.test_relations"

    # `marrow/expr/comptime/` is a deliberate package name (the AOT lane), and
    # `comptime` is a keyword -- unbackticked the import fails to parse.
    reserved = repo.root / "marrow/expr/comptime/tests/test_x.mojo"
    assert generator.module_path(reserved) == "marrow.expr.`comptime`.tests.test_x"


def test_defines_give_a_selection_its_own_driver(repo):
    """Two builds of one selection under different `-D`s must not share a
    content-addressed path -- the ASAN binary is named after it."""
    plain = DriverGenerator(repo, "bench").write(two_file_selection(repo))
    forced = DriverGenerator(repo, "bench", ("X=1",)).write(two_file_selection(repo))
    assert plain != forced
    assert forced.read_text().startswith("# built with -D X=1\n")
    assert "# built with" not in plain.read_text()


def test_driver_is_deterministic(repo):
    """Same selection => same path and byte-identical source, so the cache hits.

    File order is not part of the selection; case order within a file is, since
    it is source order and the report reads back in it.
    """
    generator = DriverGenerator(repo, "test")
    first = generator.write(two_file_selection(repo))
    by_file = two_file_selection(repo).by_file()
    again = generator.write(Selection.of(dict(reversed(list(by_file.items())))))
    assert again == first
    assert again.read_text() == first.read_text()


def test_driver_name_is_unique_per_selection(repo):
    """Concurrent sessions must not overwrite each other's driver mid-compile."""
    generator = DriverGenerator(repo, "test")
    base = two_file_selection(repo)
    extra = Selection(
        list(base) + list(selection(repo, marrow__tests__test_extra=["test_extra_one"]))
    )
    assert generator.write(base) != generator.write(extra)
    # A bench selection never collides with a test selection either.
    assert DriverGenerator(repo, "bench").write(base).name.startswith("_bench_driver_")


def test_driver_leaves_an_unchanged_file_alone(repo):
    """Rewriting identical bytes would bump mtime and defeat the cache."""
    generator = DriverGenerator(repo, "test")
    driver = generator.write(two_file_selection(repo))
    before = driver.stat().st_mtime_ns
    assert generator.write(two_file_selection(repo)).stat().st_mtime_ns == before


def test_bench_driver_uses_the_bench_suite(repo):
    source = (
        DriverGenerator(repo, "bench")
        .write(selection(repo, marrow__tests__bench_bitmap=["bench_and"]))
        .read_text()
    )
    # Benchmarks get their own driver: they cannot share -O3 with -O1 tests.
    assert "from marrow.utils.testing import BenchSuite" in source
    assert "BenchSuite.run[" in source


def test_driver_skips_files_without_cases(repo):
    cases = Selection(
        list(two_file_selection(repo))
        + list(Selection.of({repo.root / "marrow/tests/test_all_skipped.mojo": []}))
    )
    assert "test_all_skipped" not in DriverGenerator(repo, "test").render(cases)


# ---------------------------------------------------------------------------
# SuiteRunner
# ---------------------------------------------------------------------------


def result(stdout="", stderr="", returncode=0):
    return CommandResult(
        argv=("mojo",),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=0.0,
    )


class FakeToolchain:
    """Answers with canned command results instead of invoking a compiler."""

    def __init__(self, respond):
        self._respond = respond
        self.calls = []
        self.steps = []
        self.envs = []
        self._built = ""

    def run(self, source, options, args=(), label=""):
        self.calls.append(Path(source).name)
        self.steps.append("run")
        return self._respond(Path(source).read_text())

    def build(self, source, out, options, label):
        self.calls.append(Path(source).name)
        self.steps.append("build")
        self._built = Path(source).read_text()
        return self._respond(self._built)

    def execute(self, program, args=(), label="", env=None):
        self.steps.append("execute")
        self.envs.append(env)
        return self._respond(self._built)


def run_suite(repo, cases, respond, options=None, repeat=1, leaks=False):
    toolchain = FakeToolchain(respond)
    runner = SuiteRunner(
        toolchain,
        DriverGenerator(repo, "test"),
        options or BuildOptions.for_tests(),
        repeat=repeat,
        leaks=leaks,
    )
    return runner.run(cases), toolchain


def test_a_build_failure_fails_every_case(repo):
    """A runner that fails to build must fail its cases, not vanish silently."""
    results, _ = run_suite(
        repo, two_file_selection(repo), lambda _: result("", "boom", returncode=1)
    )
    assert set(results) == {"test_sort_one", "test_arrays_one", "test_arrays_two"}
    assert all(r.failed and "boom" in r.error for r in results.values())


def test_a_case_the_runner_never_reported_is_a_failure(repo):
    """Not a silent pass."""
    cases = selection(repo, marrow__tests__test_arrays=["test_ran", "test_vanished"])
    results, _ = run_suite(
        repo, cases, lambda _: result('[{"name": "test_ran", "status": "PASS"}]')
    )
    assert results["test_ran"].status == "PASS"
    assert results["test_vanished"].failed
    assert "no result reported" in results["test_vanished"].error


def test_a_compiler_crash_splits_the_unit(repo):
    """A crash is a size problem -- the halves must still be run."""

    def respond(source):
        names = [
            line for line in source.splitlines() if line.startswith("from marrow.")
        ]
        cases = source.count(",\n")
        if cases > 1:
            return result("", "Please submit a bug report to https://...", returncode=1)
        name = source.split("import ")[-1].split("\n")[0].strip()
        return (
            result(f'[{{"name": "{name}", "status": "PASS"}}]')
            if names
            else result("[]")
        )

    results, toolchain = run_suite(repo, two_file_selection(repo), respond)
    assert set(results) == {"test_sort_one", "test_arrays_one", "test_arrays_two"}
    assert all(r.status == "PASS" for r in results.values())
    assert len(toolchain.calls) > 1  # the whole selection, then its halves


def test_a_crash_that_is_not_size_dependent_does_not_split(repo):
    """A missing Metal toolchain reports "Please submit a bug report" for *any*
    input, so bisecting it costs 2N-1 full elaborations of marrow to reach the
    same answer.  One probe on the smallest possible unit settles it.
    """
    metal = "Metal Compiler failed to compile metallib. Please submit a bug report."
    cases = Selection.of(
        {repo.root / "marrow/tests/test_x.mojo": [f"test_{i}" for i in range(8)]}
    )
    results, toolchain = run_suite(repo, cases, lambda _: result("", metal, 1))
    # The whole unit, then a single-case probe.  Nothing more.
    assert len(toolchain.calls) == 2
    assert len(results) == 8
    assert all(r.failed and metal in r.error for r in results.values())


def test_a_size_dependent_crash_still_splits(repo):
    """The probe must not disarm the bisection it guards."""

    def respond(source):
        if source.count(",\n") > 1:
            return result("", "Please submit a bug report to https://...", 1)
        name = source.split("import ")[-1].split("\n")[0].strip()
        return result(f'[{{"name": "{name}", "status": "PASS"}}]')

    cases = Selection.of(
        {repo.root / "marrow/tests/test_x.mojo": [f"test_{i}" for i in range(4)]}
    )
    results, toolchain = run_suite(repo, cases, respond)
    assert len(toolchain.calls) > 2
    assert set(results) == {f"test_{i}" for i in range(4)}
    assert all(r.status == "PASS" for r in results.values())


def test_a_compile_error_does_not_split(repo):
    """A diagnostic fails identically in every subset -- splitting is waste."""
    detail = "marrow/kernels/filter.mojo:12:5: error: use of unknown 'x'"
    results, toolchain = run_suite(
        repo, two_file_selection(repo), lambda _: result("", detail, returncode=1)
    )
    assert len(toolchain.calls) == 1
    assert all(r.failed for r in results.values())


def test_a_failing_case_carries_its_error(repo):
    entry = '[{"name": "test_one", "status": "FAIL", "error": "assert failed"}]'
    results, _ = run_suite(
        repo,
        selection(repo, marrow__tests__test_x=["test_one"]),
        lambda _: result(entry),
    )
    assert results["test_one"].failed
    assert results["test_one"].error == "assert failed"


def test_the_raw_entry_survives_for_benchmarks(repo):
    """Bench timings ride in the same JSON entry the status came from."""
    entry = '[{"name": "bench_one", "status": "PASS", "runs": [1, 2], "unit": "ns"}]'
    results, _ = run_suite(
        repo, selection(repo, marrow__tests__b=["bench_one"]), lambda _: result(entry)
    )
    assert results["bench_one"].entry["runs"] == [1, 2]


def test_unparseable_output_reports_the_exit_code(repo):
    results, _ = run_suite(
        repo,
        selection(repo, marrow__tests__t=["test_one"]),
        lambda _: result("", "", 7),
    )
    assert "exit code 7" in results["test_one"].error


def test_crash_detection_needs_a_crash_marker():
    assert SuiteRunner.compiler_crashed("... Please submit a bug report ...")
    assert SuiteRunner.compiler_crashed("Stack dump:\n0.\t...")
    assert not SuiteRunner.compiler_crashed("x.mojo:1:1: error: nope")
    assert not SuiteRunner.compiler_crashed("")
    assert not SuiteRunner.compiler_crashed(None)


def test_an_empty_selection_compiles_nothing(repo):
    """A driver with no imports and an empty case tuple is not valid Mojo."""
    results, toolchain = run_suite(repo, Selection(), lambda _: result("[]"))
    assert results == {}
    assert toolchain.steps == []


def test_asan_builds_a_binary_and_then_runs_it(repo):
    """The sanitizer runtime has to be linked into a real binary.

    `mojo run` cannot do that, and a binary is also what gives symbolicated
    crash traces -- so the ASAN path is build-then-execute, not run.
    """
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    results, toolchain = run_suite(
        repo,
        cases,
        lambda _: result('[{"name": "test_one", "status": "PASS"}]'),
        options=BuildOptions.for_tests(asan=True),
    )
    assert toolchain.steps == ["build", "execute"]
    assert results["test_one"].status == "PASS"


def test_tsan_builds_a_binary_and_then_runs_it(repo):
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    results, toolchain = run_suite(
        repo,
        cases,
        lambda _: result('[{"name": "test_one", "status": "PASS"}]'),
        options=BuildOptions.for_tests(tsan=True),
    )
    assert toolchain.steps == ["build", "execute"]
    assert results["test_one"].status == "PASS"


def test_a_failed_asan_build_never_reaches_the_binary(repo):
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    results, toolchain = run_suite(
        repo,
        cases,
        lambda _: result("", "link failed", returncode=1),
        options=BuildOptions.for_tests(asan=True),
    )
    assert toolchain.steps == ["build"]
    assert results["test_one"].failed
    assert "link failed" in results["test_one"].error


def outputs(*runs):
    """Answer the build, then each run in turn with the next of *runs*."""
    answers = iter((result(), *runs))
    return lambda _: next(answers)


def test_repeat_builds_once_and_runs_each_time(repo):
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    passed = result('[{"name": "test_one", "status": "PASS"}]')
    results, toolchain = run_suite(
        repo, cases, outputs(passed, passed, passed), repeat=3
    )
    assert toolchain.steps == ["build", "execute", "execute", "execute"]
    assert results["test_one"].status == "PASS"


def test_repeat_fails_a_case_that_failed_in_any_run(repo):
    """A race that shows once in N runs is a failure, named by its run."""
    cases = selection(repo, marrow__tests__test_x=["test_one", "test_two"])
    both = '[{"name": "test_one", "status": "PASS"}, {"name": "test_two", "status": "PASS"}]'
    racy = (
        '[{"name": "test_one", "status": "FAIL", "error": "lost a wake-up"},'
        ' {"name": "test_two", "status": "PASS"}]'
    )
    results, _ = run_suite(
        repo, cases, outputs(result(both), result(racy), result(both)), repeat=3
    )
    assert results["test_one"].failed
    assert results["test_one"].error == "run 2 of 3: lost a wake-up"
    assert results["test_two"].status == "PASS"


def test_repeat_stops_at_a_run_that_crashed(repo):
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    passed = result('[{"name": "test_one", "status": "PASS"}]')
    crashed = result("", "Segmentation fault", returncode=-11)
    results, toolchain = run_suite(
        repo, cases, outputs(passed, crashed, passed), repeat=3
    )
    assert toolchain.steps == ["build", "execute", "execute"]
    assert results["test_one"].failed
    assert results["test_one"].error.startswith("run 2 of 3: ")
    assert "Segmentation fault" in results["test_one"].error


def tsan_report(*frames):
    """A ThreadSanitizer report as the runtime prints it, one Mojo frame per
    `(function, location)`, each symbol as long as a real one."""
    lines = [
        "==================",
        "WARNING: ThreadSanitizer: data race (pid=7)",
        "  Read of size 8 at 0x000108f04040 by main thread:",
    ]
    for depth, (function, location) in enumerate(frames):
        lines.append(
            f"    #{depth} {function}(::Arc[$0]$),T=[typevalue<{'x' * 300}>] "
            f"{location} (driver.tsan:arm64+0x{depth})"
        )
    lines += ["SUMMARY: ThreadSanitizer: data race", "=================="]
    return "\n".join(lines) + "\n"


def test_a_tsan_report_fails_the_case_it_names(repo):
    """TSAN reports and carries on, so the JSON says every case passed."""
    cases = selection(repo, marrow__tests__test_x=["test_one", "test_two"])
    both = '[{"name": "test_one", "status": "PASS"}, {"name": "test_two", "status": "PASS"}]'
    stderr = (
        tsan_report(
            (
                "std::memory::arc_pointer::ArcPointer::__deinit__",
                "arc_pointer.mojo:280",
            ),
            ("marrow::tests::test_x::test_one", "test_x.mojo:12"),
        )
        + "ThreadSanitizer: reported 1 warnings\n"
    )
    results, _ = run_suite(
        repo,
        cases,
        outputs(result(both, stderr, returncode=66)),
        options=BuildOptions.for_tests(tsan=True),
    )
    assert results["test_one"].failed
    error = results["test_one"].error
    assert (
        "std::memory::arc_pointer::ArcPointer::__deinit__ arc_pointer.mojo:280" in error
    )
    assert "marrow::tests::test_x::test_one test_x.mojo:12" in error
    assert "x" * 300 not in error
    assert results["test_two"].status == "PASS"


def test_a_tsan_report_naming_no_case_fails_them_all(repo):
    cases = selection(repo, marrow__tests__test_x=["test_one", "test_two"])
    both = '[{"name": "test_one", "status": "PASS"}, {"name": "test_two", "status": "PASS"}]'
    stderr = tsan_report(("marrow::utils::threads::_Shared::work", "threads.mojo:9"))
    results, _ = run_suite(
        repo,
        cases,
        outputs(result(both, stderr, returncode=66)),
        options=BuildOptions.for_tests(tsan=True),
    )
    assert results["test_one"].failed and results["test_two"].failed
    assert "_Shared::work threads.mojo:9" in results["test_two"].error


LEAKED = (
    "=================================================================\n"
    "==1==ERROR: LeakSanitizer: detected memory leaks\n\n"
    "Direct leak of 48 byte(s) in 1 object(s) allocated from:\n"
    "    #0 0x0001037e2f28 in malloc+0x70\n"
    "    #5 0x000102f55e20 in _create_pool threads.mojo:1038 (driver.asan:arm64+0x1)\n"
    "\nSUMMARY: AddressSanitizer: 48 byte(s) leaked in 1 allocation(s).\n"
)


def test_leaks_fail_every_case_on_a_leak_report(repo):
    """A leak belongs to the process, not to the case that allocated it."""
    cases = selection(repo, marrow__tests__test_x=["test_one", "test_two"])
    both = '[{"name": "test_one", "status": "PASS"}, {"name": "test_two", "status": "PASS"}]'
    results, toolchain = run_suite(
        repo,
        cases,
        outputs(result(both, LEAKED, returncode=23)),
        options=BuildOptions.for_tests(asan=True),
        leaks=True,
    )
    assert results["test_one"].failed and results["test_two"].failed
    assert "_create_pool threads.mojo:1038" in results["test_one"].error
    assert "48 byte(s) leaked" in results["test_one"].error
    assert "detect_leaks=1" in toolchain.envs[0]["ASAN_OPTIONS"]


def test_leaks_fail_every_case_when_leak_sanitizer_dies(repo):
    """A LeakSanitizer that could not check is no evidence of no leak."""
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    passed = '[{"name": "test_one", "status": "PASS"}]'
    died = (
        "==1==LeakSanitizer has encountered a fatal error.\n"
        "==1==HINT: LeakSanitizer does not work under ptrace (strace, gdb, etc)\n"
    )
    results, _ = run_suite(
        repo,
        cases,
        outputs(result(passed, died, returncode=1)),
        options=BuildOptions.for_tests(asan=True),
        leaks=True,
    )
    assert results["test_one"].failed
    assert results["test_one"].error.startswith("==1==LeakSanitizer has encountered")


def test_a_leak_report_passes_without_leaks(repo):
    """Linux runs LeakSanitizer under ASAN by default; only --leaks fails on it."""
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    passed = '[{"name": "test_one", "status": "PASS"}]'
    results, toolchain = run_suite(
        repo,
        cases,
        outputs(result(passed, LEAKED, returncode=23)),
        options=BuildOptions.for_tests(asan=True),
    )
    assert results["test_one"].status == "PASS"
    assert toolchain.envs == [None]


def test_the_non_asan_path_compiles_and_runs_in_one_step(repo):
    """`mojo run` leaves no artifact behind; compilation is what takes the minutes."""
    cases = selection(repo, marrow__tests__test_x=["test_one"])
    _, toolchain = run_suite(
        repo, cases, lambda _: result('[{"name": "test_one", "status": "PASS"}]')
    )
    assert toolchain.steps == ["run"]


def test_a_gpu_file_is_recognised_by_its_name():
    """`*_gpu.mojo` needs a device.

    Without the marker it runs by default and fails on every machine that has
    none -- and `_gpu` anywhere but the end of the stem is not that.
    """
    assert CaseScanner.is_gpu("marrow/kernels/tests/test_filter_gpu.mojo")
    assert not CaseScanner.is_gpu("marrow/kernels/tests/test_filter.mojo")
    assert not CaseScanner.is_gpu("marrow/tests/test_gpu_helpers.mojo")


def test_a_benchmark_entry_carries_no_status(repo):
    """`marrow/utils/testing.mojo` emits no `status` field for a benchmark.

    Defaulting it to anything but PASS fails every Mojo benchmark in the suite.
    """
    entry = '[{"name": "bench_one", "runs": [1, 2], "unit": "ns"}]'
    results, _ = run_suite(
        repo, selection(repo, marrow__tests__b=["bench_one"]), lambda _: result(entry)
    )
    assert results["bench_one"].status == "PASS"
    assert not results["bench_one"].failed
