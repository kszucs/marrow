import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _spinning(message, done=None):
    """Show an ASCII spinner with *message* until the block exits.

    *done*: optional callable(elapsed_seconds) -> str, or plain str.
    If provided, its result is printed after the spinner stops (line kept).
    If None, the spinner line is erased.
    """
    frames = r"|/-\\"
    stop = threading.Event()
    width = len(message) + 5  # "  X " prefix + message
    start = time.monotonic()

    def _spin():
        i = 0
        print()  # newline so the spinner gets its own line
        while not stop.is_set():
            print(f"\r  {frames[i % 4]} {message}", end="", flush=True)
            i += 1
            stop.wait(0.08)
        if done is not None:
            elapsed = time.monotonic() - start
            msg = done(elapsed) if callable(done) else done
            print(f"\r  {msg}" + " " * max(0, width - len(msg) - 2), flush=True)
        else:
            print("\r" + " " * width + "\r", end="", flush=True)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


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


def _parse_mojo_output(stdout):
    """Parse TestSuite output into {test_name: (status, error_text)}."""
    lines = _ANSI_RE.sub("", stdout).splitlines()
    results = {}
    current_name = None
    current_status = None
    error_lines = []

    for line in lines:
        m = _RESULT_RE.match(line)
        if m:
            if current_name is not None:
                results[current_name] = (current_status, "\n".join(error_lines))
            current_status, current_name = m.group(1), m.group(2)
            error_lines = []
        elif current_status == "FAIL" and current_name is not None:
            error_lines.append(line)

    if current_name is not None:
        results[current_name] = (current_status, "\n".join(error_lines))
    return results


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


def pytest_addoption(parser):
    parser.addoption("--mojo", action="store_true", default=False, help="Select Mojo tests")
    parser.addoption("--no-mojo", action="store_true", default=False, help="Exclude Mojo tests")
    parser.addoption("--python", action="store_true", default=False, help="Select Python tests")
    parser.addoption("--no-python", action="store_true", default=False, help="Exclude Python tests")
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
    sel_mojo = config.getoption("--mojo")
    sel_python = config.getoption("--python")
    sel_gpu = config.getoption("--gpu")
    no_mojo = config.getoption("--no-mojo")
    no_python = config.getoption("--no-python")
    no_gpu = config.getoption("--no-gpu")
    run_benchmark = config.getoption("--benchmark")
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
    """Compile and run Mojo test files after collection, before any test output."""
    mojo_items = [
        i for i in session.items
        if isinstance(i, MojoTestItem)
        and not any(m.name == "skip" for m in i.iter_markers())
    ]
    if not mojo_items:
        return

    results = {}
    session.config._mojo_results = results

    # Group items by source file.
    file_groups = {}
    for item in mojo_items:
        key = str(item.fspath)
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(item)

    files = list(file_groups.keys())
    label = Path(files[0]).name if len(files) == 1 else f"{len(files)} files"

    try:
        with _spinning(
            f"mojo: running {label}",
            done=lambda s: f"mojo: ran {label} in {s:.1f}s",
        ):
            for fspath, items in file_groups.items():
                cmd = _mojo_run_cmd(
                    session.config, fspath, [i.name for i in items]
                )
                result = subprocess.run(
                    cmd, cwd=session.config.rootpath,
                    capture_output=True, text=True,
                )
                parsed = _parse_mojo_output(result.stdout) if result.stdout else {}
                if result.returncode != 0:
                    for item in items:
                        if item.name not in parsed:
                            parsed[item.name] = ("FAIL", result.stderr)
                results.update(parsed)
    except MojoTestFailure as e:
        session.config._mojo_results = {item.name: ("FAIL", str(e)) for item in mojo_items}


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
        results = getattr(self.config, "_mojo_results", {})
        if self.name not in results:
            raise MojoTestFailure(f"{self.name} did not appear in test runner output")
        status, error = results[self.name]
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
