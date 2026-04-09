import json
import operator
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import pytest_benchmark.session as _bm_session
import pytest_benchmark.utils as _bm_utils
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_benchmark.utils import NameWrapper

# Patch pytest-benchmark to support a "throughput" column before argparse runs.
if "throughput" not in _bm_utils.ALLOWED_COLUMNS:
    _bm_utils.ALLOWED_COLUMNS.append("throughput")

_TEST_FN_RE = re.compile(r"^def\s+(test_\w+)\s*\(", re.MULTILINE)
_BENCH_FN_RE = re.compile(r"^def\s+(bench_\w+)\s*\(", re.MULTILINE)


class MojoRunner:
    """Builds and executes Mojo test/benchmark commands."""

    @staticmethod
    def find_asan_lib():
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

    @staticmethod
    def build_cmd(config, fspath, test_names=None):
        """Return the command to run a Mojo source file with optional test filtering.

        Uses `mojo run` directly so the Mojo compiler handles build caching.
        When *test_names* is provided, appends `--only name1 name2 ...` so that
        TestSuite skips unselected tests.
        """
        opt = "-O3" if config.getoption("--benchmark") else "-O1"
        cmd = ["mojo", "run", opt, "-I", ".", str(fspath)]

        if config.getoption("--asan"):
            asan_lib = MojoRunner.find_asan_lib()
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

    @staticmethod
    def run_tests(config, fspath, test_names):
        """Run a Mojo test file with ``--json`` and return {name: (status, error)}."""
        cmd = MojoRunner.build_cmd(config, fspath, test_names)
        cmd.append("--json")
        result = subprocess.run(
            cmd, cwd=config.rootpath,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Try to parse JSON even on failure (tests may have run partially).
            try:
                entries = json.loads(result.stdout)
            except (json.JSONDecodeError, ValueError):
                # Couldn't parse — mark all requested tests as failed.
                return {name: ("FAIL", result.stderr) for name in test_names}
            parsed = {}
            for entry in entries:
                parsed[entry["name"]] = (entry["status"], entry.get("error", ""))
            # Mark any missing test names as failed.
            for name in test_names:
                if name not in parsed:
                    parsed[name] = ("FAIL", result.stderr)
            return parsed

        try:
            entries = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return {name: ("FAIL", f"failed to parse JSON:\n{result.stdout}") for name in test_names}

        return {entry["name"]: (entry["status"], entry.get("error", "")) for entry in entries}

    @staticmethod
    def run_benches(config, fspath, bench_names=None):
        """Run a Mojo benchmark file with ``--json`` and return parsed entries.

        When *bench_names* is provided, passes ``--only name1 name2 ...`` so that
        BenchSuite skips unselected benchmarks (same pattern as tests).
        """
        cmd = MojoRunner.build_cmd(config, fspath, test_names=bench_names)
        cmd.append("--json")
        result = subprocess.run(
            cmd, cwd=config.rootpath,
            capture_output=True, text=True,
        )
        if result.stderr:
            sys.stderr.write(result.stderr)

        if result.returncode != 0:
            return {"_error": result.stderr or f"exit code {result.returncode}"}

        try:
            entries = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return {"_error": f"failed to parse JSON output:\n{result.stdout}"}

        return {e["name"]: e for e in entries}


class BenchmarkTable:
    """Formats and renders a benchmark results table."""

    @staticmethod
    def to_seconds(value, unit):
        """Convert a benchmark timing value to seconds."""
        if unit == "ns":
            return value / 1e9
        if unit == "us":
            return value / 1e6
        if unit == "ms":
            return value / 1e3
        return value

    @staticmethod
    def format_time(seconds):
        """Format seconds as a compact human-readable string with unit."""
        ns = seconds * 1e9
        if ns < 1_000:
            return f"{ns:.1f}", "ns"
        if ns < 1_000_000:
            return f"{ns / 1_000:.2f}", "us"
        if ns < 1_000_000_000:
            return f"{ns / 1_000_000:.2f}", "ms"
        return f"{ns / 1_000_000_000:.2f}", "s"

    @staticmethod
    def throughput_str(bench):
        """Extract throughput string from benchmark extra_info."""
        ei = bench.get("extra_info", {})
        for key, value in ei.items():
            if "(" in key:
                unit = key.split("(")[-1].rstrip(")")
                return f"{value:.2f} {unit}"
            return f"{value:.2f}"
        return ""

    @staticmethod
    def display(tr, benchmarks):
        """Render a benchmark results table with optional throughput column."""
        benchmarks = sorted(benchmarks, key=operator.itemgetter("mean"))

        # Determine the time unit from the fastest benchmark.
        _, time_unit = BenchmarkTable.format_time(benchmarks[0]["mean"])

        # Pre-format all values.
        rows = []
        has_throughput = False
        for bench in benchmarks:
            name = bench["name"]
            # Use consistent unit for all rows.
            if time_unit == "ns":
                mean_v = bench["mean"] * 1e9
                std_v = bench["stddev"] * 1e9
            elif time_unit == "us":
                mean_v = bench["mean"] * 1e6
                std_v = bench["stddev"] * 1e6
            elif time_unit == "ms":
                mean_v = bench["mean"] * 1e3
                std_v = bench["stddev"] * 1e3
            else:
                mean_v = bench["mean"]
                std_v = bench["stddev"]

            mean_s = f"{mean_v:,.4f}"
            std_s = f"{std_v:,.4f}"
            rounds_s = str(bench["rounds"])
            tp_s = BenchmarkTable.throughput_str(bench)
            if tp_s:
                has_throughput = True
            rows.append((name, mean_s, std_s, rounds_s, tp_s))

        # Column widths.
        hdr_name = f"Name (time in {time_unit})"
        w_name = max(len(hdr_name), max(len(r[0]) for r in rows)) + 3
        w_mean = max(4, max(len(r[1]) for r in rows)) + 2
        w_std = max(6, max(len(r[2]) for r in rows)) + 2
        w_rounds = max(6, max(len(r[3]) for r in rows)) + 2
        w_tp = 0
        if has_throughput:
            w_tp = max(10, max(len(r[4]) for r in rows)) + 2

        # Build header.
        hdr = (
            f"{hdr_name:<{w_name}}"
            f"{'Mean':>{w_mean}}"
            f"{'StdDev':>{w_std}}"
            f"{'Rounds':>{w_rounds}}"
        )
        if has_throughput:
            hdr += f"{'Throughput':>{w_tp}}"

        total_w = len(hdr)
        sep = "-" * total_w

        tr.ensure_newline()
        tr.write_line("")
        tr.write_line(
            f" benchmark: {len(benchmarks)} tests ".center(total_w, "-"),
            yellow=True,
        )
        tr.write_line(hdr)
        tr.write_line(sep, yellow=True)

        for name, mean_s, std_s, rounds_s, tp_s in rows:
            line = (
                f"{name:<{w_name}}"
                f"{mean_s:>{w_mean}}"
                f"{std_s:>{w_std}}"
                f"{rounds_s:>{w_rounds}}"
            )
            if has_throughput:
                line += f"{tp_s:>{w_tp}}"
            tr.write_line(line)

        tr.write_line(sep, yellow=True)
        tr.write_line("")


class MojoTestFailure(Exception):
    pass


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


def pytest_collection_finish(session):
    """Pre-compute per-file groups for tests and benchmarks."""
    # Test groups (existing).
    file_groups = {}
    for item in session.items:
        if isinstance(item, MojoTestItem) and not any(
            m.name == "skip" for m in item.iter_markers()
        ):
            key = str(item.fspath)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(item.name)
    session.config._mojo_file_groups = file_groups
    session.config._mojo_results = {}

    # Benchmark groups — collect non-skipped bench names per file.
    bench_groups = {}
    for item in session.items:
        if isinstance(item, MojoBenchItem) and not any(
            m.name == "skip" for m in item.iter_markers()
        ):
            key = str(item.fspath)
            if key not in bench_groups:
                bench_groups[key] = []
            bench_groups[key].append(item.name)
    session.config._mojo_bench_groups = bench_groups
    session.config._mojo_bench_results = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "mojo: Mojo language tests")
    config.addinivalue_line("markers", "python: Python tests")
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmarks (skipped by default, run with --benchmark)",
    )

    # Replace the benchmark table display with our own that supports throughput.
    bs = getattr(config, "_benchmarksession", None)
    if bs is not None:
        bs.columns = ["mean", "stddev", "rounds", "throughput"]

    def _display_benchmarks(self, tr):
        if not self.groups:
            return
        for _, benchmarks in self.groups:
            BenchmarkTable.display(tr, benchmarks)

    _bm_session.BenchmarkSession.display = _display_benchmarks


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
            results[fspath] = MojoRunner.run_tests(self.config, fspath, names)
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
    """Collect individual benchmark items from a bench_*.mojo file.

    Files using BenchSuite yield one item per ``def bench_*(mut b: Bencher)``
    function discovered in the source (mirroring MojoTestFile).  Files without
    discoverable bench functions fall back to a single item per file.

    The Mojo file is compiled and executed once per file; results are cached
    and individual timings injected into pytest-benchmark.
    """

    def collect(self):
        source = self.path.read_text()
        bench_names = _BENCH_FN_RE.findall(source)
        if bench_names:
            for name in bench_names:
                yield MojoBenchItem.from_parent(self, name=name)
        else:
            # Fallback for old-style files without discoverable bench_* fns.
            yield MojoBenchItem.from_parent(self, name=self.path.stem)


class MojoBenchItem(pytest.Item):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_marker(pytest.mark.mojo)
        self.add_marker(pytest.mark.benchmark)

    def runtest(self):
        # Run the bench file once per file, cache results (same as MojoTestItem).
        results = self.config._mojo_bench_results
        fspath = str(self.fspath)
        if fspath not in results:
            bench_names = self.config._mojo_bench_groups.get(fspath, None)
            results[fspath] = MojoRunner.run_benches(self.config, fspath, bench_names)
        entries = results[fspath]

        if "_error" in entries:
            raise MojoTestFailure(entries["_error"])

        # Look up this benchmark's entry.
        entry = entries.get(self.name)
        if entry is not None:
            self._inject_one(self.name, entry)
            return

        # Not found by exact name.  For old-style files (dump_report output),
        # the first item for the file injects all entries; subsequent items
        # from the same file become no-ops.
        injected_key = fspath + ":_injected"
        if injected_key not in results and entries:
            results[injected_key] = True
            self._inject_all(entries)

    def _inject_one(self, bench_name, entry):
        """Inject pre-measured timings into pytest-benchmark.

        When the entry contains a ``runs`` list (from BenchSuite), each run
        is injected as a separate round so pytest-benchmark computes proper
        min/max/stddev statistics.  Otherwise falls back to a single mean.
        """
        bs = self.config._benchmarksession
        if bs.disabled:
            return

        runs = entry.get("runs")
        unit = entry.get("unit", "ns")

        if runs:
            # Multiple per-iteration measurements — inject each as a round.
            durations_s = [BenchmarkTable.to_seconds(v, unit) for v in runs]
        else:
            # Legacy single-value format.
            durations_s = [BenchmarkTable.to_seconds(entry["value"], unit)]

        # Build a fake timer that yields (0, d1, 0, d2, ...) for each round.
        # pedantic() calls timer() twice per round: start then end.
        timer_seq = []
        for d in durations_s:
            timer_seq.append(0.0)
            timer_seq.append(d)
        timer_it = iter(timer_seq)
        fake_timer = NameWrapper(lambda: next(timer_it))

        node = types.SimpleNamespace(name=bench_name, _nodeid=self._nodeid)
        noop = lambda *_: None
        fixture = BenchmarkFixture(
            node=node,
            add_stats=bs.benchmarks.append,
            logger=noop,
            warner=noop,
            disabled=bs.disabled,
            timer=fake_timer,
            disable_gc=False,
            min_rounds=1,
            min_time=0,
            max_time=0,
            calibration_precision=10,
            warmup=False,
            warmup_iterations=0,
            cprofile=False,
            cprofile_loops=None,
            cprofile_dump=None,
        )
        fixture.pedantic(
            lambda: None,
            rounds=len(durations_s),
            iterations=1,
            warmup_rounds=0,
        )

        # Compute and attach throughput if the entry has metric data.
        tp_count = entry.get("throughput_count")
        if tp_count and fixture.stats:
            mean_s = fixture.stats.stats.mean
            if mean_s > 0:
                metric_name = entry.get("throughput_metric", "throughput")
                metric_unit = entry.get("throughput_unit", "GElems/s")
                rate = tp_count * 1e-9 / mean_s
                fixture.extra_info[f"{metric_name} ({metric_unit})"] = round(rate, 4)

    def _inject_all(self, entries):
        """Inject all entries (fallback for old-style files)."""
        for bench_name, entry in entries.items():
            self._inject_one(bench_name, entry)

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::bench::{self.name}"
