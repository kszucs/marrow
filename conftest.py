import os
import re
import subprocess
from pathlib import Path

import pytest


collect_ignore = ["marrow/tests/bench_popcount_py.py"]

_TEST_FN_RE = re.compile(r"^def\s+(test_\w+)\s*\(", re.MULTILINE)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RESULT_RE = re.compile(r"^\s+(PASS|FAIL|SKIP)\s+\[\s+[\d.]+\s+\]\s+(\S+)")


def _find_asan_lib():
    """Locate the upstream LLVM ASAN runtime (libclang_rt.asan_osx_dynamic.dylib).

    Searches in order:
    1. $CONDA_PREFIX/lib  (pixi/conda environment)
    2. clang resource dirs reported by any clang on PATH
    3. Known Xcode/CommandLineTools paths (Apple's clang — only if version matches)
    """
    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libclang_rt.asan_osx_dynamic.dylib")

    for clang in ["clang", "clang-18", "clang-17", "clang-16"]:
        try:
            result = subprocess.run(
                [clang, "--print-runtime-dir"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                candidates.append(
                    Path(result.stdout.strip()) / "libclang_rt.asan_osx_dynamic.dylib"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def _mojo_run_cmd(config, fspath, test_names=None):
    """Return the command to run a Mojo source file with optional test filtering.

    Uses `mojo run` directly so the Mojo compiler handles build caching.
    When *test_names* is provided, appends `--only name1 name2 ...` so that
    TestSuite skips unselected tests.
    """
    opt = "-O3" if config.getoption("--benchmark") else "-O1"
    cmd = ["mojo", "run", opt, "-I", ".", str(fspath)]

    if config.getoption("--asan"):
        asan_lib = _find_asan_lib()
        if asan_lib is None:
            pytest.exit(
                "ASAN requested but no compatible libclang_rt.asan_osx_dynamic.dylib found. "
                "Install libcompiler-rt via conda-forge.",
                returncode=1,
            )
        cmd += ["--sanitize", "address", "--shared-libasan"]
        if asan_lib.endswith(".dylib"):
            cmd += ["-Xlinker", asan_lib]

    if test_names:
        cmd += ["--only"] + list(test_names)

    return cmd


def _run_mojo_file(config, fspath, test_names):
    """Run a Mojo test file and return {test_name: (status, error_text)}."""
    cmd = _mojo_run_cmd(config, fspath, test_names)
    result = subprocess.run(
        cmd, cwd=config.rootpath,
        capture_output=True, text=True,
    )
    parsed = {}
    current_name = None
    current_status = None
    error_lines = []
    for line in _ANSI_RE.sub("", result.stdout).splitlines():
        m = _RESULT_RE.match(line)
        if m:
            if current_name is not None:
                parsed[current_name] = (current_status, "\n".join(error_lines))
            current_status, current_name = m.group(1), m.group(2)
            error_lines = []
        elif current_status == "FAIL" and current_name is not None:
            error_lines.append(line)
    if current_name is not None:
        parsed[current_name] = (current_status, "\n".join(error_lines))
    if result.returncode != 0:
        for name in test_names:
            if name not in parsed:
                parsed[name] = ("FAIL", result.stderr)
    return parsed


def pytest_addoption(parser):
    parser.addoption("--mojo", action="store_true", default=False, help="Select Mojo tests")
    parser.addoption("--no-mojo", action="store_true", default=False, help="Exclude Mojo tests")
    parser.addoption("--python", action="store_true", default=False, help="Select Python tests")
    parser.addoption("--no-python", action="store_true", default=False, help="Exclude Python tests")
    parser.addoption("--cpu", action="store_true", default=False, help="Select CPU tests (non-GPU Mojo + Python)")
    parser.addoption("--gpu", action="store_true", default=False, help="Select GPU tests")
    parser.addoption("--no-gpu", action="store_true", default=False, help="Exclude GPU tests")
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Include benchmarks (Python pytest-benchmark and Mojo bench_*.mojo); skipped by default",
    )
    parser.addoption(
        "--asan",
        action="store_true",
        default=False,
        help="Run Mojo tests under AddressSanitizer (ASAN)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mojo: Mojo language tests")
    config.addinivalue_line("markers", "python: Python tests")
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmarks (skipped by default, run with --benchmark)",
    )


def pytest_collection_modifyitems(config, items):
    sel_cpu = config.getoption("--cpu")
    sel_mojo = config.getoption("--mojo")
    sel_python = config.getoption("--python")
    sel_gpu = config.getoption("--gpu")
    no_mojo = config.getoption("--no-mojo")
    no_python = config.getoption("--no-python")
    no_gpu = config.getoption("--no-gpu")
    run_benchmark = config.getoption("--benchmark")

    # --cpu implies both --mojo and --python (all non-GPU tests)
    if sel_cpu:
        sel_mojo = True
        sel_python = True

    selective = sel_mojo or sel_python or sel_gpu

    for item in items:
        is_gpu = "gpu" in item.keywords
        is_mojo = "mojo" in item.keywords and not is_gpu
        is_python = "python" in item.keywords
        is_benchmark = "benchmark" in item.keywords

        if is_benchmark and not run_benchmark:
            item.add_marker(pytest.mark.skip(reason="benchmarks excluded; pass --benchmark to include"))
        elif is_gpu and (no_gpu or not sel_gpu):
            item.add_marker(pytest.mark.skip(reason="GPU tests excluded; pass --gpu to include"))
        elif (no_mojo and is_mojo) or (selective and is_mojo and not sel_mojo):
            item.add_marker(pytest.mark.skip(reason="Mojo tests excluded; pass --mojo to include"))
        elif (no_python and is_python) or (selective and is_python and not sel_python):
            item.add_marker(pytest.mark.skip(reason="Python tests excluded; pass --python to include"))


def pytest_collection_finish(session):
    """Pre-compute per-file test groups for use in runtest()."""
    file_groups = {}
    for item in session.items:
        if isinstance(item, MojoTestItem) and not any(m.name == "skip" for m in item.iter_markers()):
            key = str(item.fspath)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(item.name)
    session.config._mojo_file_groups = file_groups
    session.config._mojo_results = {}


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".mojo" and file_path.name.startswith("test_"):
        return MojoTestFile.from_parent(parent, path=file_path)
    if file_path.suffix == ".mojo" and file_path.name.startswith("bench_"):
        return MojoBenchFile.from_parent(parent, path=file_path)


def pytest_itemcollected(item):
    if item.fspath.ext == ".py":
        item.add_marker(pytest.mark.python)
        if item.fspath.basename.startswith("bench_"):
            item.add_marker(pytest.mark.benchmark)


class MojoTestFile(pytest.File):
    def collect(self):
        is_gpu = self.path.stem.endswith("_gpu")
        source = self.path.read_text()
        test_names = _TEST_FN_RE.findall(source)
        for name in test_names:
            yield MojoTestItem.from_parent(self, name=name, is_gpu=is_gpu)


class MojoTestItem(pytest.Item):
    def __init__(self, name, parent, is_gpu=False):
        super().__init__(name, parent)
        self.is_gpu = is_gpu
        self.add_marker(pytest.mark.mojo)
        if is_gpu:
            self.add_marker(pytest.mark.gpu)

    def runtest(self):
        results = self.config._mojo_results
        fspath = str(self.fspath)
        if fspath not in results:
            names = self.config._mojo_file_groups.get(fspath, [self.name])
            results[fspath] = _run_mojo_file(self.config, fspath, names)
        file_results = results[fspath]
        if self.name not in file_results:
            raise MojoTestFailure(f"{self.name} did not appear in test runner output")
        status, error = file_results[self.name]
        if status == "FAIL":
            raise MojoTestFailure(error)

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::{self.name}"


class MojoBenchFile(pytest.File):
    def collect(self):
        yield MojoBenchItem.from_parent(self, name=self.path.stem)


class MojoBenchItem(pytest.Item):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_marker(pytest.mark.mojo)
        self.add_marker(pytest.mark.benchmark)

    def runtest(self):
        cmd = _mojo_run_cmd(self.config, self.fspath)
        capman = self.config.pluginmanager.getplugin("capturemanager")
        with capman.global_and_fixture_disabled():
            result = subprocess.run(cmd, cwd=self.config.rootpath)
        if result.returncode != 0:
            raise MojoTestFailure(f"exit code {result.returncode}")

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::bench::{self.name}"


class MojoTestFailure(Exception):
    pass
