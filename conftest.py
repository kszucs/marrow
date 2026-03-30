import os
import subprocess
from pathlib import Path

import pytest

collect_ignore = ["marrow/tests/bench_popcount_py.py"]


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
        elif (no_gpu and is_gpu) or (selective and is_gpu and not sel_gpu):
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
    """Return (cmd, tmp_binary_path).

    With --asan on macOS, mojo run's JIT ignores -Xlinker flags, so we
    compile to a temp binary with the ASAN lib linked in, then run that.
    On other platforms mojo run with --sanitize address works directly.
    """
    if not config.getoption("--asan"):
        return ["mojo", "run", "-I", ".", str(fspath)], None

    asan_lib = _find_asan_lib()
    if asan_lib is None:
        pytest.exit(
            "ASAN requested but no compatible libclang_rt.asan_osx_dynamic.dylib found. "
            "Install libcompiler-rt via conda-forge.",
            returncode=1,
        )

    # On macOS (.dylib), mojo run's JIT ignores -Xlinker, so compile to a temp
    # binary first. On Linux (.so), mojo run --sanitize address works directly.
    if not asan_lib.endswith(".dylib"):
        return ["mojo", "run", "--sanitize", "address", "--shared-libasan", "-I", ".", str(fspath)], None

    import tempfile
    fd, tmp = tempfile.mkstemp(prefix="mojo_asan_")
    os.close(fd)
    build_cmd = [
        "mojo", "build",
        "--sanitize", "address", "--shared-libasan",
        "-Xlinker", asan_lib,
        "-I", ".", str(fspath), "-o", tmp,
    ]
    result = subprocess.run(build_cmd, cwd=config.rootpath)
    if result.returncode != 0:
        raise MojoTestFailure(f"mojo build failed with exit code {result.returncode}")
    return [tmp], tmp


class MojoTestFile(pytest.File):
    def collect(self):
        is_gpu = self.path.stem.endswith("_gpu")
        yield MojoTestItem.from_parent(self, name=self.path.stem, is_gpu=is_gpu)


class MojoTestItem(pytest.Item):
    def __init__(self, name, parent, is_gpu=False):
        super().__init__(name, parent)
        self.is_gpu = is_gpu
        self.add_marker(pytest.mark.mojo)
        if is_gpu:
            self.add_marker(pytest.mark.gpu)

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


class MojoTestFailure(Exception):
    pass
