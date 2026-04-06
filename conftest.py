import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

collect_ignore = ["marrow/tests/bench_popcount_py.py"]

_TEST_FN_RE = re.compile(r"^def\s+(test_\w+)\s*\(", re.MULTILINE)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RESULT_RE = re.compile(r"^\s+(PASS|FAIL|SKIP)\s+\[\s+[\d.]+\s+\]\s+(\S+)")

# Directories containing test modules, mapped to their Mojo import prefix.
_TEST_DIRS = {
    "marrow/tests": "marrow.tests",
    "marrow/kernels/tests": "marrow.kernels.tests",
    "marrow/expr/tests": "marrow.expr.tests",
}



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


def _module_for_path(file_path, rootpath):
    """Map a test file path to its Mojo import module name, or None."""
    rel = file_path.relative_to(rootpath)
    for dir_rel, prefix in _TEST_DIRS.items():
        if str(rel).startswith(dir_rel + "/"):
            return f"{prefix}.{file_path.stem}"
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


_RUNNER_DIR = Path(".test_runners")
_MAX_RUNNERS = 10


def _generate_test_runner(items, rootpath):
    """Generate a Mojo test runner source file for the given test items.

    The runner imports each selected test module and registers exactly the
    requested test functions with TestSuite.  It is written to
    .test_runners/test_runner_<hash>.mojo where <hash> is an MD5 digest of
    the file content — so the same test selection always produces the same
    file, and a different selection produces a different file.

    This stable naming is load-bearing for build caching: _build_mojo_cmd
    compiles each runner to a binary at .test_runners/test_runner_<hash>
    (no extension).  On subsequent runs with the same selection, mojo build
    detects the existing binary and skips recompilation, cutting cold-start
    time from ~5 s to ~1 s.

    Up to _MAX_RUNNERS runner files (and their compiled binaries) are kept
    on disk for inspection; older ones are pruned automatically.
    """
    # Group items by module.
    modules = {}  # {module_name: (alias, [fn_name, ...])}
    for item in items:
        module = item._mojo_module
        if module not in modules:
            alias = "_" + module.replace(".", "_")
            modules[module] = (alias, [])
        modules[module][1].append(item.name)

    lines = [
        "# AUTO-GENERATED — do not edit.",
        "from std.testing import TestSuite",
        "",
    ]
    for module, (alias, _) in modules.items():
        lines.append(f"import {module} as {alias}")

    lines.append("")
    lines.append("")
    lines.append("def main() raises:")
    lines.append("    var suite = TestSuite()")
    for module, (alias, fns) in modules.items():
        for fn in fns:
            lines.append(f"    suite.test[{alias}.{fn}]()")
    lines.append("    suite^.run()")
    lines.append("")

    runner_dir = rootpath / _RUNNER_DIR
    runner_dir.mkdir(parents=True, exist_ok=True)

    # Prune old runners (and their compiled binaries), keeping only the last (_MAX_RUNNERS - 1).
    existing = sorted(runner_dir.glob("test_runner_*.mojo"), key=lambda p: p.stat().st_mtime)
    for old in existing[:max(0, len(existing) - _MAX_RUNNERS + 1)]:
        old.unlink()
        old.with_suffix("").unlink(missing_ok=True)  # remove compiled binary if present

    content = "\n".join(lines)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    path = runner_dir / f"test_runner_{content_hash}.mojo"
    path.write_text(content)
    return str(path)


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


def _build_mojo_cmd(config, fspath):
    """Compile a Mojo source file and return the command to run the resulting binary.

    Always uses `mojo build` (never `mojo run`) so that the compiled binary is
    written to .test_runners/<stem> alongside its source.  Because
    _generate_test_runner names runners by content hash, the binary path is
    stable across runs with the same test selection: mojo build detects the
    existing binary and skips recompilation, cutting ~5 s → ~1 s.

    Optimization level:
      -O1  normal test runs  (fast compile, catches correctness issues)
      -O3  --benchmark runs  (full optimization, representative perf numbers)

    ASAN (--asan):
      Appends --sanitize address --shared-libasan to the build command.
      On macOS the ASAN dylib must also be linked explicitly via -Xlinker
      because mojo build's JIT ignores -Xlinker flags otherwise.

    Returns (cmd, None).  None signals that the binary should NOT be deleted
    after the run — it is kept for the caching benefit described above.
    Cleanup is handled by _generate_test_runner's pruning logic.
    """
    opt = "-O3" if config.getoption("--benchmark") else "-O1"
    runner_dir = config.rootpath / _RUNNER_DIR
    runner_dir.mkdir(parents=True, exist_ok=True)
    binary = str(runner_dir / Path(fspath).stem)  # always in .test_runners/<stem>
    build_cmd = ["mojo", "build", opt, "-I", ".", str(fspath), "-o", binary]

    if config.getoption("--asan"):
        asan_lib = _find_asan_lib()
        if asan_lib is None:
            pytest.exit(
                "ASAN requested but no compatible libclang_rt.asan_osx_dynamic.dylib found. "
                "Install libcompiler-rt via conda-forge.",
                returncode=1,
            )
        build_cmd += ["--sanitize", "address", "--shared-libasan"]
        if asan_lib.endswith(".dylib"):
            build_cmd += ["-Xlinker", asan_lib]

    result = subprocess.run(build_cmd, cwd=config.rootpath)
    if result.returncode != 0:
        raise MojoTestFailure(f"mojo build failed with exit code {result.returncode}")
    return [binary], None


class MojoTestFile(pytest.File):
    def collect(self):
        is_gpu = self.path.stem.endswith("_gpu")
        source = self.path.read_text()
        test_names = _TEST_FN_RE.findall(source)
        module = _module_for_path(self.path, self.config.rootpath)
        for name in test_names:
            yield MojoTestItem.from_parent(
                self, name=name, is_gpu=is_gpu, mojo_module=module,
            )


class MojoTestItem(pytest.Item):
    def __init__(self, name, parent, is_gpu=False, mojo_module=None):
        super().__init__(name, parent)
        self.is_gpu = is_gpu
        self._mojo_module = mojo_module
        self.add_marker(pytest.mark.mojo)
        if is_gpu:
            self.add_marker(pytest.mark.gpu)

    def _get_session_cache(self):
        if not hasattr(self.config, "_mojo_results"):
            self.config._mojo_results = {}
        return self.config._mojo_results

    def runtest(self):
        self._run_via_centralized()

    def _run_via_centralized(self):
        cache = self._get_session_cache()

        if "centralized" not in cache:
            selected = [
                item for item in self.session.items
                if isinstance(item, MojoTestItem)
                and item._mojo_module is not None
                and not any(m.name == "skip" for m in item.iter_markers())
            ]
            if not selected:
                cache["centralized"] = {}
            else:
                runner_path = _generate_test_runner(selected, self.config.rootpath)
                cmd, tmp = _build_mojo_cmd(self.config, runner_path)
                try:
                    capman = self.config.pluginmanager.getplugin("capturemanager")
                    with capman.global_and_fixture_disabled():
                        result = subprocess.run(
                            cmd, cwd=self.config.rootpath,
                            capture_output=True, text=True,
                        )
                    parsed = _parse_mojo_output(result.stdout) if result.stdout else {}
                    if result.returncode != 0:
                        for item in selected:
                            if item.name not in parsed:
                                parsed[item.name] = ("FAIL", result.stderr)
                    cache["centralized"] = parsed
                finally:
                    if tmp:
                        Path(tmp).unlink(missing_ok=True)

        results = cache["centralized"]
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
        cmd, tmp = _build_mojo_cmd(self.config, self.fspath)
        capman = self.config.pluginmanager.getplugin("capturemanager")
        try:
            with capman.global_and_fixture_disabled():
                result = subprocess.run(cmd, cwd=self.config.rootpath)
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        if result.returncode != 0:
            raise MojoTestFailure(f"exit code {result.returncode}")

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::bench::{self.name}"


def pytest_terminal_summary(terminalreporter):
    cache = getattr(terminalreporter.config, "_mojo_results", {})
    runner_path = cache.get("runner_path")
    runner_cmd = cache.get("runner_cmd")
    if runner_path:
        terminalreporter.section("mojo test runner")
        terminalreporter.write_line(f"generated: {runner_path}")
        if runner_cmd:
            terminalreporter.write_line(f"command:   {' '.join(str(c) for c in runner_cmd)}")


class MojoTestFailure(Exception):
    pass
