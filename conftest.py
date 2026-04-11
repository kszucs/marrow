import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
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
    def asan_flags(config):
        """Return ASAN-related compiler flags, or an empty list if not requested."""
        if not config.getoption("--asan"):
            return []
        asan_lib = MojoRunner.find_asan_lib()
        if asan_lib is None:
            pytest.exit(
                "ASAN requested but no compatible libclang_rt.asan_osx_dynamic.dylib found. "
                "Install libcompiler-rt via conda-forge.",
                returncode=1,
            )
        flags = ["--sanitize", "address", "--shared-libasan"]
        if asan_lib.endswith(".dylib"):
            flags += ["-Xlinker", asan_lib]
        return flags

    @staticmethod
    def build_cmd(config, fspath, test_names=None):
        """Return the command to run a Mojo source file with optional test filtering.

        Uses `mojo run` directly so the Mojo compiler handles build caching.
        When *test_names* is provided, appends `--only name1 name2 ...` so that
        TestSuite skips unselected tests.
        """
        opt = "-O3" if config.getoption("--benchmark") else "-O1"
        cmd = ["mojo", "run", opt, "-I", "."] + MojoRunner.asan_flags(config) + [str(fspath)]

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


def _to_seconds(value, unit):
    """Convert a benchmark timing value to seconds."""
    if unit == "ns":
        return value / 1e9
    if unit == "us":
        return value / 1e6
    if unit == "ms":
        return value / 1e3
    return value


# Strips the leading "[n=NNN]" or "[n=NNN-" prefix from a parametrized pytest
# test ID.  The n fixture always comes first: "[n=10000]" or "[n=10000-case]".
_N_PREFIX_RE = re.compile(r'\[n=\d+(-|\])')


class CompetitionReport:
    """Side-by-side benchmark comparison table for all measured libraries."""

    @staticmethod
    def _parse(bench):
        """Return ``(lib, operation, n)`` using ``extra_info``."""
        ei = bench.get("extra_info", {})
        lib = ei.get("lib")
        n_val = ei.get("n")
        if not (lib and n_val is not None):
            return None, None, None
        name = bench["name"]
        prefix = f"test_{lib}_"
        op = name[len(prefix):] if name.startswith(prefix) else name
        # "[n=10000]"       → ""            (fixture-only, no mark suffix)
        # "[n=10000-inner]" → "[inner]"     (fixture + mark suffix)
        op = _N_PREFIX_RE.sub(lambda m: "[" if m.group(1) == "-" else "", op)
        return lib, op, n_val

    @staticmethod
    def _fmt(seconds):
        ns = seconds * 1e9
        if ns < 1_000:
            return f"{ns:.1f} ns"
        if ns < 1_000_000:
            return f"{ns / 1_000:.2f} µs"
        if ns < 1_000_000_000:
            return f"{ns / 1_000_000:.2f} ms"
        return f"{ns / 1_000_000_000:.2f} s"

    @classmethod
    def display(cls, tr, benchmarks):
        from rich.console import Console
        from rich.table import Table
        from rich import box

        # Keys from extra_info that are not shown as columns (internal bookkeeping).
        _hidden = frozenset({"lib", "n", _THROUGHPUT_KEY})

        # Collect (op, n) → {lib: mean_seconds} and metadata per (op, n).
        data: dict[tuple, dict[str, float]] = {}
        meta: dict[tuple, dict] = {}
        for b in benchmarks:
            lib, op, n = cls._parse(b)
            if lib is None:
                continue
            data.setdefault((op, n), {})[lib] = b["mean"]
            ei = b.get("extra_info", {})
            row_meta = {k: v for k, v in ei.items() if k not in _hidden}
            if row_meta:
                meta.setdefault((op, n), {}).update(row_meta)

        # Discover all libs and metadata keys in stable insertion order.
        libs: list[str] = []
        meta_keys: list[str] = []
        for lib_data in data.values():
            for lib in lib_data:
                if lib not in libs:
                    libs.append(lib)
        for row_meta in meta.values():
            for k in row_meta:
                if k not in meta_keys:
                    meta_keys.append(k)

        if not libs:
            tr.write_line("No benchmarks with lib metadata found.")
            return

        # Build rows: only include (op, n) pairs that have at least two libs.
        rows = []
        for (op, n), lib_data in sorted(data.items()):
            if len(lib_data) < 2:
                continue
            best_t = min(lib_data.values())
            best_lib = min(lib_data, key=lib_data.get)
            rows.append((op, n, lib_data, best_lib, best_t))

        if not rows:
            tr.write_line("No operations with multiple libs measured.")
            return

        # Win counters per lib.
        wins = {lib: 0 for lib in libs}
        ties = 0
        for op, n, lib_data, best_lib, best_t in rows:
            present = [l for l in libs if l in lib_data]
            if len(present) < 2:
                continue
            times = [lib_data[l] for l in present]
            fastest_t = min(times)
            ratio = max(times) / fastest_t if fastest_t > 0 else 1.0
            if ratio < 1.05:
                ties += 1
            else:
                wins[best_lib] += 1

        table = Table(title="Competition", box=box.SIMPLE_HEAD, show_footer=True)
        table.add_column("Operation", no_wrap=True)
        table.add_column("n", justify="right")
        table.add_column("", no_wrap=True)  # before first lib
        for i, lib in enumerate(libs):
            if i > 0:
                table.add_column("", no_wrap=True)
            footer = f"[bold green]{wins[lib]} wins[/]" if wins[lib] else ""
            table.add_column(lib.capitalize(), justify="right", footer=footer, no_wrap=True)
        table.add_column("", no_wrap=True)  # after last lib
        table.add_column("Fastest", justify="right", footer=f"[dim]{ties} ties[/]")
        for k in meta_keys:
            table.add_column(k.capitalize(), justify="right", no_wrap=True)

        current_group = None
        for op, n, lib_data, best_lib, best_t in rows:
            grp = op.split("[")[0]
            if grp != current_group:
                if current_group is not None:
                    table.add_section()
                current_group = grp

            present = {lib: t for lib in libs if (t := lib_data.get(lib)) is not None}
            fastest_t = min(present.values()) if present else None
            if len(present) >= 2 and fastest_t and fastest_t > 0:
                worst_t = max(present.values())
                ratio = worst_t / fastest_t
                is_tie = ratio < 1.05
            else:
                ratio = 1.0
                is_tie = True

            if is_tie:
                fastest_markup = "[dim]~tie[/dim]"
            else:
                fastest_markup = f"[bold green]{best_lib} {ratio:.1f}x[/bold green]"

            sep = "[dim]│[/]"
            row = [op.replace("[", "\\["), f"{n:,}", sep]
            for i, lib in enumerate(libs):
                if i > 0:
                    row.append(sep)
                t = present.get(lib)
                if t is None:
                    row.append("—")
                elif is_tie:
                    row.append(cls._fmt(t))
                elif t == fastest_t:
                    row.append(f"[bold green]{cls._fmt(t)} 1.0x[/bold green]")
                else:
                    rel = t / fastest_t
                    row.append(f"{cls._fmt(t)} {rel:.1f}x")
            row.append(sep)
            row.append(fastest_markup)
            row_meta = meta.get((op, n), {})
            for k in meta_keys:
                v = row_meta.get(k)
                row.append(str(v) if v is not None else "")
            table.add_row(*row)

        console = Console(highlight=False, width=220)
        tr.ensure_newline()
        tr.write_line("")
        with console.capture() as cap:
            console.print(table)
        for line in cap.get().splitlines():
            tr.write_line(line)


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
    parser.addoption(
        "--competition",
        action="store_true",
        default=False,
        help="After benchmarks, print a side-by-side comparison table for all measured libs.",
    )


def _python_excluded(config) -> bool:
    """Return True if Python tests/files should be excluded from this session."""
    if config.getoption("--no-python"):
        return True
    sel_mojo = config.getoption("--mojo")
    sel_gpu = config.getoption("--gpu")
    sel_python = config.getoption("--python")
    sel_cpu = config.getoption("--cpu")
    if (sel_mojo or sel_gpu) and not (sel_python or sel_cpu):
        return True
    # Specific paths given → check whether any lead to Python test files.
    if config.args:
        for arg in config.args:
            path_str = str(arg).split("::")[0]
            p = Path(path_str)
            if not p.is_absolute():
                p = config.rootpath / p
            if p.is_file():
                if p.suffix == ".py":
                    return False
            elif p.is_dir():
                if any(p.rglob("test_*.py")) or any(p.rglob("bench_*.py")):
                    return False
        return True
    return False


def pytest_ignore_collect(collection_path, config):
    """Skip collecting Python test/bench files when Python tests are not needed."""
    if collection_path.suffix == ".py" and collection_path.name.startswith(
        ("test_", "bench_")
    ):
        if _python_excluded(config):
            return True


def pytest_sessionstart(session):
    """Rebuild python/marrow.so before the session when Python tests will run."""
    config = session.config

    # xdist workers inherit the already-built library from the controller.
    if hasattr(config, "workerinput"):
        return

    # Skip build when Python tests are excluded.
    if _python_excluded(config):
        return

    print("building python/marrow.so ...", flush=True)
    opt = "-O3" if config.getoption("--benchmark") else "-O1"
    cmd = (
        ["mojo", "build", opt, "-I", "."]
        + MojoRunner.asan_flags(config)
        + ["python/lib.mojo", "--emit", "shared-lib", "-o", "python/marrow.so"]
    )
    result = subprocess.run(cmd, cwd=config.rootpath, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.exit(
            f"Failed to build python/marrow.so:\n{result.stderr}",
            returncode=1,
        )
    print("python/marrow.so built successfully", flush=True)


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
            durations_s = [_to_seconds(v, unit) for v in runs]
        else:
            # Legacy single-value format.
            durations_s = [_to_seconds(entry["value"], unit)]

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


_THROUGHPUT_KEY = "throughput (GElems/s)"


def pytest_benchmark_group_stats(config, benchmarks, group_by):  # config: required by pytest hook signature
    """Group benchmarks by the native benchmark group marker for display.
    Within each group, benchmarks are sorted by ``(n, name, mean)`` so rows
    are ordered by size then by operation name.  Throughput is computed and
    injected into ``extra_info`` for each benchmark that has an ``n`` value.

    Only activates for the default ``group_by="group"``; custom
    ``--benchmark-group-by`` values are passed through unchanged.
    """
    if group_by != "group":
        return None  # honour explicit --benchmark-group-by choices

    groups: dict[str, list] = {}
    for bench in benchmarks:
        key = bench.get("group") or bench["name"].split("[")[0]
        groups.setdefault(key, []).append(bench)

    for group_benchmarks in groups.values():
        group_benchmarks.sort(key=lambda b: (
            b.get("extra_info", {}).get("n", 0),
            b["name"],
            b["mean"],
        ))
        for bench in group_benchmarks:
            ei = bench.get("extra_info", {})
            n_val = ei.get("n")
            mean_s = bench.get("mean", 0)
            if n_val and mean_s > 0 and _THROUGHPUT_KEY not in ei:
                ei[_THROUGHPUT_KEY] = round(n_val / mean_s / 1e9, 4)

    return sorted(groups.items(), key=lambda pair: pair[0] or "")


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):  # exitstatus: required by pytest hook signature
    if not config.getoption("--competition", default=False):
        return
    bs = getattr(config, "_benchmarksession", None)
    if bs is None or not bs.benchmarks:
        return
    CompetitionReport.display(terminalreporter, bs.benchmarks)


