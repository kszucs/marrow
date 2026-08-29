import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from datetime import datetime, timezone
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

# Holds the generated drivers (and, under --asan, the binaries built from them).
RUNNER_DIR = ".test_runners"

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _rss_bytes(pid):
    """Resident size of *pid* in bytes, or 0 if it can no longer be read."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return int(out) * 1024 if out else 0
    except (ValueError, OSError):
        return 0


def _progress(pid, label, stream, stop, peak):
    """Tick a spinner until *stop* is set, tracking the compiler's memory.

    Elaboration is the slow part and its cost shows up as resident memory, so
    the live RSS is the most informative thing to show while waiting minutes for
    a single compile.  Sampled once a second — `ps` per frame would be silly.
    """
    start = time.monotonic()
    rss = 0
    for tick in range(sys.maxsize):
        if tick % 10 == 0:
            rss = _rss_bytes(pid) or rss
            peak[0] = max(peak[0], rss)
        elapsed = time.monotonic() - start
        mins, secs = divmod(int(elapsed), 60)
        clock = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        mem = f", {rss / 1e9:.1f} GB" if rss else ""
        frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
        stream.write(f"\r\033[K{frame} {label} — {clock}{mem}")
        stream.flush()
        if stop.wait(0.1):
            return


def run_with_progress(config, cmd, cwd, label):
    """Run *cmd* to completion, showing progress while it works.

    A single Mojo compile here runs for minutes with no output of its own, which
    is indistinguishable from a hang.  On a terminal this shows a live spinner
    (pytest's capture has to be lifted for that); otherwise it prints a plain
    start/finish pair so CI logs still show what happened.
    """
    capman = config.pluginmanager.getplugin("capturemanager") if config else None
    lifted = (
        capman.global_and_fixture_disabled() if capman else contextlib.nullcontext()
    )

    started = time.monotonic()
    peak = [0]
    with lifted:
        # isatty() has to be asked *inside* the lifted block: pytest captures at
        # the file-descriptor level, so until capture is suspended fd 2 points at
        # a temp file and every terminal looks non-interactive.
        stream = sys.__stderr__
        interactive = stream is not None and stream.isatty()
        # Start on a line of our own — pytest is mid-way through writing its
        # per-file progress line, and appending to it reads as garbage.
        if interactive:
            stream.write("\n")
            stream.flush()
        else:
            print(f"\n{label} ...", flush=True)
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stop = threading.Event()
        ticker = None
        if interactive:
            ticker = threading.Thread(
                target=_progress,
                args=(proc.pid, label, stream, stop, peak),
                daemon=True,
            )
            ticker.start()
        timeout = config.getoption("--mojo-timeout") if config else 0
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            # A hung Mojo process emits nothing and never exits, so it is
            # indistinguishable from a slow compile until the deadline passes.
            # Kill it and turn the hang into an ordinary failure -- otherwise
            # this call blocks forever and takes CI with it (backlog B23).
            timed_out = True
            proc.kill()
            out, err = proc.communicate()
            err = (
                (err or "") + f"\n\nTIMEOUT: killed after {timeout}s with no exit.\n"
                "A Mojo compile or run that produces no output for this long is "
                "hung, not slow -- compare elapsed time against CPU time with "
                "`ps -o etime,time <pid>` to confirm. Raise --mojo-timeout if "
                "this selection is legitimately slower than the deadline.\n"
            )
        stop.set()
        if ticker is not None:
            ticker.join()

        elapsed = time.monotonic() - started
        mark = "✓" if proc.returncode == 0 and not timed_out else "✗"
        mem = f", peak {peak[0] / 1e9:.1f} GB" if peak[0] else ""
        suffix = " (TIMED OUT)" if timed_out else ""
        line = f"{mark} {label} — {elapsed:.0f}s{mem}{suffix}"
        if interactive:
            # Overwrite the spinner, then leave the cursor at column 0 so pytest
            # resumes its progress line cleanly underneath.
            stream.write(f"\r\033[K{line}\n")
            stream.flush()
        else:
            print(line, flush=True)

    # A killed process reports a signal returncode; force a plain non-zero so
    # callers that only check `!= 0` treat a hang like any other failure.
    code = 124 if timed_out else proc.returncode
    return subprocess.CompletedProcess(cmd, code, out, err)


class MojoRunner:
    """Generates, builds and executes the Mojo runner for a selection."""

    @staticmethod
    def find_asan_lib():
        """Locate the upstream LLVM ASAN runtime shared library.

        Searches in order:
        1. $CONDA_PREFIX/lib  (pixi/conda environment)
        2. clang resource dirs reported by any clang on PATH

        Returns the path as a string, or None if not found.
        """
        is_macos = sys.platform == "darwin"
        lib_names = (
            ["libclang_rt.asan_osx_dynamic.dylib"]
            if is_macos
            else ["libclang_rt.asan-x86_64.so", "libclang_rt.asan.so"]
        )

        candidates = []

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            for name in lib_names:
                candidates.append(Path(conda_prefix) / "lib" / name)

        for clang in ["clang", "clang-18", "clang-17", "clang-16"]:
            try:
                result = subprocess.run(
                    [clang, "--print-runtime-dir"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for name in lib_names:
                        candidates.append(Path(result.stdout.strip()) / name)
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
                "ASAN requested but no compatible ASAN runtime found. "
                "Install libcompiler-rt via conda-forge.",
                returncode=1,
            )
        # --shared-libasan is a Clang-only flag; on Linux the system `cc` is
        # GCC and rejects it.  The explicit -Xlinker path is sufficient there.
        flags = ["--sanitize", "address"]
        if sys.platform == "darwin":
            flags += ["--shared-libasan"]
            # Ensure the conda env's lib dir appears first in the binary's
            # rpath so dyld resolves libclang_rt.asan_osx_dynamic.dylib from
            # the pixi env rather than the incompatible Xcode toolchain copy.
            lib_dir = str(Path(asan_lib).parent)
            flags += ["-Xlinker", "-rpath", "-Xlinker", lib_dir]
        flags += ["-Xlinker", asan_lib]
        return flags

    @staticmethod
    def flags(config, bench):
        """Compiler flags for a runner.

        Optimization level follows the *kind*, not the session: a benchmark must
        measure optimized code, a test only has to be correct.  -O1 is the floor
        — at -O0 the masked-gather intrinsic in `filter`/`take` fails to lower
        ("failed to produce an archive for the module").

        GPU codegen is opt-in (`marrow.utils.GPU_ENABLED` defaults to False), so
        `--gpu` runs have to ask for it explicitly — without this the device
        paths are elaborated away and every GPU test would exercise the CPU
        fallback, or raise "no GPU accelerator available", instead of failing
        honestly.
        """
        opt = "-O3" if bench else "-O1"
        assert_flag = [] if bench else ["-D", "ASSERT=all"]
        gpu = ["-D", "MARROW_GPU=true"] if config.getoption("--gpu") else []
        # -lm: mojo doesn't auto-link libm on Linux (needed for log10f etc.);
        # harmless on macOS where libm is part of libSystem.
        lm = [] if sys.platform == "darwin" else ["-Xlinker", "-lm"]
        return (
            [opt, "-g1", "-I", "."]
            + assert_flag
            + gpu
            + MojoRunner.asan_flags(config)
            + lm
        )

    #: Mojo keywords that are legal directory names but illegal in an import
    #: path unless backticked.  ``marrow/expr/comptime/`` is a deliberate
    #: package name (the AOT lane), so the driver has to spell it ``\`comptime\```
    #: or the import fails with ``expected module name``.
    RESERVED = {"comptime", "fn", "def", "var", "trait", "struct", "alias"}

    @staticmethod
    def module_path(rootpath, fspath):
        """``marrow/expr/tests/test_relations.mojo`` -> ``marrow.expr.tests.test_relations``.

        A path component that is a Mojo keyword is backticked, so
        ``marrow/expr/comptime/tests/x.mojo`` becomes
        ``marrow.expr.`comptime`.tests.x``.
        """
        rel = Path(fspath).resolve().relative_to(Path(rootpath).resolve())
        return ".".join(
            f"`{part}`" if part in MojoRunner.RESERVED else part
            for part in rel.with_suffix("").parts
        )

    @staticmethod
    def write_driver(rootpath, groups, kind):
        """Generate one Mojo module that imports and runs the whole selection.

        The compiler accepts a single input file per invocation and
        re-elaborates all of marrow for each one, so compiling per test file
        costs that elaboration N times over.  Collapsing the selection into one
        unit pays it once: measured on the former expression-layer suite, its
        nine files (280 cases) took 4m43s together, against ~200s *each* when
        built separately.

        Output is deterministic — modules sorted, cases in source order — so an
        unchanged selection regenerates byte-identical source and hits the Mojo
        compiler's own artifact cache instead of recompiling from scratch.

        The file is *named* after that content, so concurrent sessions with
        different selections cannot overwrite each other's driver mid-compile,
        while the same selection keeps resolving to the same path (and the same
        cached artifact).
        """
        suite = "TestSuite" if kind == "test" else "BenchSuite"
        lines = [f"from marrow.utils.testing import {suite}"]
        names = []
        for fspath in sorted(groups):
            cases = groups[fspath]
            if not cases:
                continue
            lines.append(
                f"from {MojoRunner.module_path(rootpath, fspath)} import "
                + ", ".join(cases)
            )
            names.extend(cases)

        cases = ",\n            ".join(names)
        source = "\n".join(lines) + (
            f"\n\n\ndef main() raises:\n"
            f"    {suite}.run[\n        (\n            {cases},\n        )\n    ]()\n"
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:12]
        driver = Path(rootpath) / RUNNER_DIR / f"_{kind}_driver_{digest}.mojo"
        driver.parent.mkdir(exist_ok=True)
        if not driver.exists() or driver.read_text() != source:
            driver.write_text(source)
        return driver

    @staticmethod
    def run_suite(config, groups, kind):
        """Build the selection as one unit, run it, and return its JSON entries.

        Returns ``(entries, detail)``; *entries* is None when the runner could
        not be built or its output could not be parsed, and *detail* then holds
        the compiler/runtime output to report against every selected case.

        ``--asan`` goes through ``mojo build`` because the sanitizer runtime has
        to be linked into a real binary (and a binary is what gives symbolicated
        crash traces).  Everything else uses ``mojo run``, which compiles and
        executes in one step without leaving an artifact behind.
        """
        rootpath = Path(config.rootpath)
        driver = MojoRunner.write_driver(rootpath, groups, kind)
        flags = MojoRunner.flags(config, bench=kind == "bench")

        cases = sum(len(c) for c in groups.values())
        files = sum(1 for c in groups.values() if c)
        noun = "benchmarks" if kind == "bench" else "tests"
        label = f"compiling {cases} {noun} from {files} files"

        if config.getoption("--asan"):
            # Same content-addressed stem as the driver, so parallel sessions
            # never link over each other's binary either.
            binary = driver.with_suffix("")
            built = run_with_progress(
                config,
                ["mojo", "build"] + flags + [str(driver), "-o", str(binary)],
                rootpath,
                f"{label} (asan)",
            )
            if built.returncode != 0:
                return None, f"mojo build failed for {driver}:\n{built.stderr}"
            result = subprocess.run(
                [str(binary), "--json"], cwd=rootpath, capture_output=True, text=True
            )
        else:
            # `mojo run` compiles and executes in one step, so the spinner covers
            # the run too — but compilation is what takes the minutes.
            result = run_with_progress(
                config,
                ["mojo", "run"] + flags + [str(driver), "--json"],
                rootpath,
                label,
            )

        if result.stderr and result.returncode == 0:
            sys.stderr.write(result.stderr)
        try:
            return json.loads(result.stdout), result.stderr
        except (json.JSONDecodeError, ValueError):
            detail = "\n".join(
                part for part in (result.stderr, result.stdout) if part.strip()
            )
            return None, detail or f"exit code {result.returncode}"

    @staticmethod
    def compiler_crashed(detail):
        """True when the compiler *died* rather than rejecting the source.

        A crash prints a bug-report dump and no diagnostic, and it depends on
        how much the unit elaborates — the same cases build in smaller units.
        A real `error:` does not: it would be reported identically in every
        subset, so splitting on it would only multiply the compile time.
        """
        return any(
            marker in (detail or "")
            for marker in ("Please submit a bug report", "Stack dump:")
        )

    @staticmethod
    def halve(groups):
        """Split a selection into two, keeping each file's cases in order."""
        flat = [(f, case) for f in sorted(groups) for case in groups[f]]
        mid = len(flat) // 2
        halves = []
        for part in (flat[:mid], flat[mid:]):
            group = {}
            for fspath, case in part:
                group.setdefault(fspath, []).append(case)
            halves.append(group)
        return halves

    @staticmethod
    def collect(config, groups, kind):
        """Run the selection and return ``{case name: JSON entry}``.

        One unit is the fast path — marrow's elaboration dominates, so it is
        paid once for the whole selection.  When the *compiler crashes* on that
        unit the selection is halved and each half compiled on its own, down to
        a single case; a case that still cannot be built reports the crash as
        its own failure instead of failing every case in the selection with it.
        """
        entries, detail = MojoRunner.run_suite(config, groups, kind)
        if entries is not None:
            return {e["name"]: e for e in entries}
        names = [name for cases in groups.values() for name in cases]
        if len(names) > 1 and MojoRunner.compiler_crashed(detail):
            print(
                f"compiler crashed on {len(names)} cases — splitting the unit",
                flush=True,
            )
            collected = {}
            for half in MojoRunner.halve(groups):
                collected.update(MojoRunner.collect(config, half, kind))
            return collected
        return {
            name: {"name": name, "status": "FAIL", "error": detail} for name in names
        }

    @staticmethod
    def run_tests(config, groups):
        """Run every selected test case and return ``{name: (status, error)}``."""
        entries = MojoRunner.collect(config, groups, "test")
        parsed = {
            name: (e["status"], e.get("error", "")) for name, e in entries.items()
        }
        for cases in groups.values():
            for name in cases:
                parsed.setdefault(name, ("FAIL", "no result reported"))
        return parsed

    @staticmethod
    def run_benches(config, groups):
        """Run every selected benchmark and return ``{name: entry}``."""
        return MojoRunner.collect(config, groups, "bench")


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
_N_PREFIX_RE = re.compile(r"\[n=\d+(-|\])")


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
        op = name[len(prefix) :] if name.startswith(prefix) else name
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
            table.add_column(
                lib.capitalize(), justify="right", footer=footer, no_wrap=True
            )
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
    parser.addoption(
        "--mojo", action="store_true", default=False, help="Select Mojo tests"
    )
    parser.addoption(
        "--no-mojo", action="store_true", default=False, help="Exclude Mojo tests"
    )
    parser.addoption(
        "--python", action="store_true", default=False, help="Select Python tests"
    )
    parser.addoption(
        "--no-python", action="store_true", default=False, help="Exclude Python tests"
    )
    parser.addoption(
        "--cpu",
        action="store_true",
        default=False,
        help="Select CPU tests (non-GPU Mojo + Python)",
    )
    parser.addoption(
        "--gpu", action="store_true", default=False, help="Select GPU tests"
    )
    parser.addoption(
        "--no-gpu", action="store_true", default=False, help="Exclude GPU tests"
    )
    parser.addoption(
        "--morsel-size",
        type=int,
        default=8192,
        metavar="N",
        help="Rows per morsel for golden-corpus sources (default 8192). A small "
        "value drives every operator across morsel boundaries.",
    )
    parser.addoption(
        "--num-threads",
        type=int,
        default=0,
        metavar="N",
        help="CPU worker budget for golden-corpus execution (0 = auto).",
    )
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
        "--mojo-timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help=(
            "Kill a Mojo compile/run that exceeds SECONDS and report it as a "
            "failure (default: 1800). 0 disables the timeout. The harness "
            "already recovers from a compiler *crash* by splitting the "
            "selection, because a crash produces a signal; a hang produces "
            "none, so without this the run blocks forever -- see backlog B23."
        ),
    )
    parser.addoption(
        "--competition",
        action="store_true",
        default=False,
        help="After benchmarks, print a side-by-side comparison table for all measured libs.",
    )
    parser.addoption(
        "--save-benchmarks",
        metavar="DIR",
        default=None,
        help="Save benchmark results as a JSON envelope to DIR/<commit>.json (implies --benchmark).",
    )
    parser.addoption(
        "--benchmark-history",
        metavar="FILE",
        default=None,
        help="Path to the rolling benchmark history JSON file (default: benchmarks/data.json).",
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
        # Python bench files are only collected when --benchmark is active,
        # mirroring the behaviour of Mojo bench_*.mojo files.
        if collection_path.name.startswith("bench_") and not config.getoption(
            "--benchmark"
        ):
            return True


def pytest_sessionstart(session):
    """Rebuild python/marrow/libmarrow.so before the session when Python tests will run."""
    config = session.config

    if _python_excluded(config):
        return

    if not hasattr(config, "workerinput"):
        opt = "-O3" if config.getoption("--benchmark") else "-O1"
        cmd = (
            ["mojo", "build", opt, "-g0", "-I", "."]
            + MojoRunner.asan_flags(config)
            + [
                "python/bindings/lib.mojo",
                "--emit",
                "shared-lib",
                "-o",
                "python/marrow/libmarrow.so",
            ]
        )
        result = run_with_progress(
            config, cmd, config.rootpath, "compiling python/marrow/libmarrow.so"
        )
        if result.returncode != 0:
            pytest.exit(
                f"Failed to build python/marrow/libmarrow.so:\n{result.stderr}",
                returncode=1,
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
            item.add_marker(
                pytest.mark.skip(
                    reason="benchmarks excluded; pass --benchmark to include"
                )
            )
        elif is_gpu and (no_gpu or not sel_gpu):
            item.add_marker(
                pytest.mark.skip(reason="GPU tests excluded; pass --gpu to include")
            )
        elif (no_mojo and is_mojo) or (selective and is_mojo and not sel_mojo):
            item.add_marker(
                pytest.mark.skip(reason="Mojo tests excluded; pass --mojo to include")
            )
        elif (no_python and is_python) or (selective and is_python and not sel_python):
            item.add_marker(
                pytest.mark.skip(
                    reason="Python tests excluded; pass --python to include"
                )
            )


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
    """Pre-compute the selection each generated runner has to cover.

    ``{file: [case, ...]}`` for tests and benchmarks separately — the two build
    at different optimization levels, so they cannot share a runner.  Results
    start as None (not {}) because an empty dict is also a legitimate result and
    must not re-trigger the run.
    """
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
    session.config._mojo_results = None

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
    session.config._mojo_bench_results = None


def pytest_configure(config):
    config.addinivalue_line("markers", "mojo: Mojo language tests")
    config.addinivalue_line("markers", "python: Python tests")
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmarks (skipped by default, run with --benchmark)",
    )
    # --save-benchmarks implies --benchmark.
    if config.getoption("--save-benchmarks", default=None):
        config.option.benchmark = True


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
        # The first item to run builds and executes the runner for the whole
        # selection; every later item just reads its own entry.  Case names are
        # unique across the suite, so the name alone identifies the result.
        config = self.config
        if config._mojo_results is None:
            config._mojo_results = MojoRunner.run_tests(
                config, config._mojo_file_groups
            )
        if self.name not in config._mojo_results:
            raise MojoTestFailure(f"{self.name} did not appear in test runner output")
        status, error = config._mojo_results[self.name]
        if status == "FAIL":
            raise MojoTestFailure(error)

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::{self.name}"


class MojoBenchFile(pytest.File):
    """Collect individual benchmark items from a bench_*.mojo file.

    One item per ``def bench_*(mut b: Benchmark)`` in the source, mirroring
    MojoTestFile.  The selection is compiled and executed as a single runner at
    -O3; individual timings are injected into pytest-benchmark afterwards.
    """

    def collect(self):
        for name in _BENCH_FN_RE.findall(self.path.read_text()):
            yield MojoBenchItem.from_parent(self, name=name)


class MojoBenchItem(pytest.Item):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_marker(pytest.mark.mojo)
        self.add_marker(pytest.mark.benchmark)

    def runtest(self):
        # As in MojoTestItem: the first item runs the whole selection at once.
        config = self.config
        if config._mojo_bench_results is None:
            config._mojo_bench_results = MojoRunner.run_benches(
                config, config._mojo_bench_groups
            )
        entry = config._mojo_bench_results.get(self.name)

        if entry is None:
            raise MojoTestFailure(f"{self.name} did not appear in bench runner output")
        if entry.get("status") == "FAIL":
            # The unit this benchmark was in could not be built or run.
            raise MojoTestFailure(entry.get("error", ""))
        self._inject_one(self.name, entry)

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

    def repr_failure(self, excinfo):
        return str(excinfo.value)

    def reportinfo(self):
        return self.fspath, 0, f"mojo::bench::{self.name}"


_THROUGHPUT_KEY = "throughput (GElems/s)"


def pytest_benchmark_group_stats(
    config, benchmarks, group_by
):  # config: required by pytest hook signature
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
        group_benchmarks.sort(
            key=lambda b: (
                b.get("extra_info", {}).get("n", 0),
                b["name"],
                b["mean"],
            )
        )
        for bench in group_benchmarks:
            ei = bench.get("extra_info", {})
            n_val = ei.get("n")
            mean_s = bench.get("mean", 0)
            if n_val and mean_s > 0 and _THROUGHPUT_KEY not in ei:
                ei[_THROUGHPUT_KEY] = round(n_val / mean_s / 1e9, 4)

    return sorted(groups.items(), key=lambda pair: pair[0] or "")


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(
    terminalreporter, exitstatus, config
):  # exitstatus: required by pytest hook signature
    if not config.getoption("--competition", default=False):
        return
    bs = getattr(config, "_benchmarksession", None)
    if bs is None or not bs.benchmarks:
        return
    CompetitionReport.display(terminalreporter, bs.benchmarks)


# ---------------------------------------------------------------------------
# --save-benchmarks: write result envelope + update rolling history
# ---------------------------------------------------------------------------


class BenchmarkHistory:
    """Rolling history of benchmark runs, persisted as JSON.

    Each run is an envelope: ``{commit, timestamp, ref, results: [...]}``.
    The history merges envelopes into ``benchmarks/data.json`` and keeps
    individual per-commit snapshots under a results directory.
    """

    MAX_RUNS = 200

    def __init__(self, root, results_dir, history_file=None):
        self._root = Path(root)
        self._results_dir = Path(results_dir)
        self._history_file = (
            Path(history_file)
            if history_file
            else self._root / "benchmarks" / "data.json"
        )

    # -- git metadata -------------------------------------------------------

    @staticmethod
    def _git(root, *args):
        try:
            return (
                subprocess.run(
                    ["git", "-C", str(root)] + list(args),
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or "unknown"
            )
        except Exception:
            return "unknown"

    # -- envelope -----------------------------------------------------------

    def _make_envelope(self, benchmarks):
        """Convert pytest-benchmark Metadata objects into a result envelope."""
        commit = self._git(self._root, "rev-parse", "HEAD")
        ref = self._git(self._root, "rev-parse", "--abbrev-ref", "HEAD")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        results = []
        for b in benchmarks:
            # bs.benchmarks contains Metadata objects (not flat dicts).
            stats = b.stats
            mean_s = stats.mean if stats else 0.0
            throughput = b.extra_info.get(_THROUGHPUT_KEY)

            # Extract source file from fullname (pytest node ID)
            # e.g. "python/tests/bench_compute.py::test_marrow_add[...]" → "bench_compute.py"
            fullname = getattr(b, "fullname", "") or ""
            file = (
                fullname.split("::")[0].rsplit("/", 1)[-1] if "::" in fullname else None
            )

            # Collect extra_info fields (excluding internal keys)
            extra = {}
            for k, v in b.extra_info.items():
                if k != _THROUGHPUT_KEY:
                    extra[k] = v

            result = {
                "name": b.name,
                "file": file,
                "mean_ns": mean_s * 1e9,
                "throughput_gelems_s": throughput,
            }

            if stats:
                result["min_ns"] = stats.min * 1e9
                result["max_ns"] = stats.max * 1e9
                result["median_ns"] = stats.median * 1e9
                result["stddev_ns"] = stats.stddev * 1e9
                result["rounds"] = stats.rounds

            if extra:
                result["extra_info"] = extra

            results.append(result)
        return {
            "commit": commit,
            "timestamp": timestamp,
            "ref": ref,
            "results": results,
        }

    # -- persistence --------------------------------------------------------

    def _write_envelope(self, envelope):
        """Write per-commit result file and latest.json."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        out_file = self._results_dir / f"{envelope['commit']}.json"
        for path in [out_file, self._results_dir / "latest.json"]:
            with path.open("w") as f:
                json.dump(envelope, f, indent=2)
                f.write("\n")
        return out_file

    def _update_history(self, envelope):
        """Merge envelope into the rolling history JSON file."""
        data_file = self._history_file
        if data_file.exists():
            with data_file.open() as f:
                history = json.load(f)
        else:
            history = {"runs": [], "operations": []}

        existing_commits = {r["commit"] for r in history["runs"]}
        if envelope["commit"] not in existing_commits:
            run_results = {}
            for r in envelope["results"]:
                entry = {
                    "mean_ns": r["mean_ns"],
                    "throughput_gelems_s": r["throughput_gelems_s"],
                }
                for key in (
                    "file",
                    "min_ns",
                    "max_ns",
                    "median_ns",
                    "stddev_ns",
                    "rounds",
                    "extra_info",
                ):
                    if key in r:
                        entry[key] = r[key]
                run_results[r["name"]] = entry
            history["runs"].append(
                {
                    "commit": envelope["commit"],
                    "short_commit": envelope["commit"][:7],
                    "timestamp": envelope["timestamp"],
                    "ref": envelope["ref"],
                    "results": run_results,
                }
            )

        history["runs"].sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        history["runs"] = history["runs"][: self.MAX_RUNS]

        seen = {}
        for run in history["runs"]:
            for name in run.get("results", {}):
                seen[name] = None
        history["operations"] = list(seen)

        data_file.parent.mkdir(parents=True, exist_ok=True)
        with data_file.open("w") as f:
            json.dump(history, f, indent=2)
            f.write("\n")

        return len(history["runs"])

    # -- public API ---------------------------------------------------------

    def save(self, benchmarks):
        """Build envelope from pytest-benchmark results, write files, update history."""
        envelope = self._make_envelope(benchmarks)
        out_file = self._write_envelope(envelope)
        total = self._update_history(envelope)
        count = len(envelope["results"])
        print(f"\n--save-benchmarks: {count} entries written to {out_file}")
        print(f"--save-benchmarks: {total} run(s) in {self._history_file}")


def pytest_sessionfinish(session, exitstatus):
    save_dir = session.config.getoption("--save-benchmarks", default=None)
    if not save_dir:
        return
    # Only run on the controller (not xdist workers).
    if hasattr(session.config, "workerinput"):
        return

    bs = getattr(session.config, "_benchmarksession", None)
    if bs is None or not bs.benchmarks:
        return

    history_file = session.config.getoption("--benchmark-history", default=None)
    history = BenchmarkHistory(session.config.rootpath, save_dir, history_file)
    history.save(bs.benchmarks)


# ---------------------------------------------------------------------------
# Self-tests for BenchmarkHistory (run with: python conftest.py)
# ---------------------------------------------------------------------------


class _FakeStats:
    def __init__(self, mean):
        self.mean = mean
        self.min = mean * 0.9
        self.max = mean * 1.1
        self.median = mean
        self.stddev = mean * 0.02
        self.rounds = 10


class _FakeBenchmark:
    def __init__(self, name, mean, throughput=None, fullname=None, extra_info=None):
        self.name = name
        self.fullname = fullname or f"tests/bench_test.py::{name}"
        self.stats = _FakeStats(mean)
        self.extra_info = dict(extra_info or {})
        if throughput is not None:
            self.extra_info[_THROUGHPUT_KEY] = throughput


def _make_history(tmp):
    root = Path(tmp)
    return BenchmarkHistory(root, root / "results")


def _make_envelope(h, commit="abc123def456", timestamp="2026-04-16T00:00:00Z"):
    benchmarks = [
        _FakeBenchmark("bench_add_10k", 0.001, throughput=1.234),
        _FakeBenchmark("bench_add_100k", 0.01),
    ]
    envelope = h._make_envelope(benchmarks)
    envelope["commit"] = commit
    envelope["timestamp"] = timestamp
    return envelope


def test_make_envelope(tmp):
    h = _make_history(tmp)
    benchmarks = [
        _FakeBenchmark(
            "bench_add_10k",
            0.001,
            throughput=1.234,
            fullname="python/tests/bench_compute.py::bench_add_10k",
            extra_info={"lib": "marrow", "n": 10000},
        ),
        _FakeBenchmark("bench_add_100k", 0.01),
    ]
    envelope = h._make_envelope(benchmarks)

    r0 = envelope["results"][0]
    assert r0["name"] == "bench_add_10k"
    assert r0["mean_ns"] == 0.001 * 1e9
    assert r0["throughput_gelems_s"] == 1.234
    assert r0["file"] == "bench_compute.py"
    assert r0["min_ns"] == 0.001 * 0.9 * 1e9
    assert r0["max_ns"] == 0.001 * 1.1 * 1e9
    assert r0["median_ns"] == 0.001 * 1e9
    assert r0["stddev_ns"] == 0.001 * 0.02 * 1e9
    assert r0["rounds"] == 10
    assert r0["extra_info"] == {"lib": "marrow", "n": 10000}

    r1 = envelope["results"][1]
    assert r1["name"] == "bench_add_100k"
    assert r1["mean_ns"] == 0.01 * 1e9
    assert r1["throughput_gelems_s"] is None
    assert r1["file"] == "bench_test.py"
    assert "extra_info" not in r1  # no extra_info when empty

    assert "commit" in envelope
    assert "timestamp" in envelope
    assert "ref" in envelope


def test_write_envelope(tmp):
    h = _make_history(tmp)
    envelope = _make_envelope(h)
    out_file = h._write_envelope(envelope)
    assert out_file.exists()
    assert (h._results_dir / "latest.json").exists()
    with out_file.open() as f:
        assert json.load(f) == envelope
    with (h._results_dir / "latest.json").open() as f:
        assert json.load(f) == envelope


def test_update_history_first_run(tmp):
    h = _make_history(tmp)
    envelope = _make_envelope(h)
    total = h._update_history(envelope)
    assert total == 1
    with (h._history_file).open() as f:
        history = json.load(f)
    run = history["runs"][0]
    assert run["commit"] == "abc123def456"
    assert run["short_commit"] == "abc123d"
    r = run["results"]["bench_add_10k"]
    assert r["mean_ns"] == 0.001 * 1e9
    assert r["throughput_gelems_s"] == 1.234
    assert r["file"] == "bench_test.py"
    assert r["stddev_ns"] == 0.001 * 0.02 * 1e9
    assert r["min_ns"] == 0.001 * 0.9 * 1e9
    assert r["max_ns"] == 0.001 * 1.1 * 1e9
    assert r["median_ns"] == 0.001 * 1e9
    assert r["rounds"] == 10
    assert set(history["operations"]) == {"bench_add_10k", "bench_add_100k"}


def test_update_history_idempotent(tmp):
    h = _make_history(tmp)
    envelope = _make_envelope(h)
    h._update_history(envelope)
    total = h._update_history(envelope)
    assert total == 1


def test_update_history_appends(tmp):
    h = _make_history(tmp)
    h._update_history(_make_envelope(h, commit="aaa"))
    h._update_history(_make_envelope(h, commit="bbb", timestamp="2026-04-16T00:00:01Z"))
    with (h._history_file).open() as f:
        history = json.load(f)
    commits = [r["commit"] for r in history["runs"]]
    assert "aaa" in commits
    assert "bbb" in commits
    assert len(commits) == 2


def test_update_history_max_runs(tmp):
    h = _make_history(tmp)
    h.MAX_RUNS = 2
    h._update_history(_make_envelope(h, commit="a", timestamp="2026-04-16T00:00:00Z"))
    h._update_history(_make_envelope(h, commit="b", timestamp="2026-04-16T00:00:01Z"))
    total = h._update_history(
        _make_envelope(h, commit="c", timestamp="2026-04-16T00:00:02Z")
    )
    assert total == 2


class _FakeConfig:
    """Minimal stand-in for a pytest config: options plus a rootpath."""

    def __init__(self, rootpath, **options):
        self.rootpath = Path(rootpath)
        self._options = options

    def getoption(self, name, default=False):
        return self._options.get(name.lstrip("-").replace("-", "_"), default)


def test_flags_test_vs_bench(tmp):
    config = _FakeConfig(tmp)
    test_flags = MojoRunner.flags(config, bench=False)
    bench_flags = MojoRunner.flags(config, bench=True)

    # A test must keep its assertions; a benchmark must measure optimized code.
    assert "-O1" in test_flags and "-O3" not in test_flags
    assert "ASSERT=all" in test_flags
    assert "-O3" in bench_flags and "-O1" not in bench_flags
    assert "ASSERT=all" not in bench_flags
    # Both compile against the source tree, never a precompiled package.
    assert test_flags[test_flags.index("-I") + 1] == "."


def test_flags_gpu_is_opt_in(tmp):
    """GPU codegen is off by default, so --gpu must ask for it explicitly."""
    assert "MARROW_GPU=true" not in MojoRunner.flags(_FakeConfig(tmp), bench=False)
    gpu_flags = MojoRunner.flags(_FakeConfig(tmp, gpu=True), bench=False)
    assert gpu_flags[gpu_flags.index("MARROW_GPU=true") - 1] == "-D"
    # Benchmarks honour it too — a GPU benchmark must measure the device path.
    assert "MARROW_GPU=true" in MojoRunner.flags(_FakeConfig(tmp, gpu=True), bench=True)


def test_flags_asan_only_when_requested(tmp):
    assert "--sanitize" not in MojoRunner.flags(_FakeConfig(tmp), bench=False)
    if MojoRunner.find_asan_lib() is not None:
        flags = MojoRunner.flags(_FakeConfig(tmp, asan=True), bench=False)
        assert flags[flags.index("--sanitize") + 1] == "address"


def test_module_path(tmp):
    path = Path(tmp) / "marrow" / "expr" / "tests" / "test_relations.mojo"
    path.parent.mkdir(parents=True)
    path.touch()
    assert MojoRunner.module_path(tmp, path) == "marrow.expr.tests.test_relations"

    reserved = Path(tmp) / "marrow" / "expr" / "comptime" / "tests" / "test_x.mojo"
    reserved.parent.mkdir(parents=True, exist_ok=True)
    reserved.touch()
    assert (
        MojoRunner.module_path(tmp, reserved) == "marrow.expr.`comptime`.tests.test_x"
    )


def _driver_groups(tmp):
    return {
        f"{tmp}/marrow/kernels/tests/test_sort.mojo": ["test_sort_one"],
        f"{tmp}/marrow/tests/test_arrays.mojo": ["test_arrays_one", "test_arrays_two"],
    }


def test_write_driver_imports_every_selected_case(tmp):
    driver = MojoRunner.write_driver(tmp, _driver_groups(tmp), "test")
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


def test_write_driver_is_deterministic(tmp):
    """Same selection => same path and byte-identical source, so the cache hits."""
    groups = _driver_groups(tmp)
    first = MojoRunner.write_driver(tmp, groups, "test")
    again = MojoRunner.write_driver(tmp, dict(reversed(list(groups.items()))), "test")
    assert again == first
    assert again.read_text() == first.read_text()


def test_write_driver_name_is_unique_per_selection(tmp):
    """Concurrent sessions must not overwrite each other's driver mid-compile."""
    groups = _driver_groups(tmp)
    other = dict(groups)
    other[f"{tmp}/marrow/tests/test_extra.mojo"] = ["test_extra_one"]
    assert MojoRunner.write_driver(tmp, groups, "test") != MojoRunner.write_driver(
        tmp, other, "test"
    )
    # A bench selection never collides with a test selection either.
    assert MojoRunner.write_driver(tmp, groups, "bench").name.startswith(
        "_bench_driver_"
    )


def test_write_driver_leaves_unchanged_file_alone(tmp):
    """Rewriting identical bytes would bump mtime and defeat the cache."""
    groups = _driver_groups(tmp)
    driver = MojoRunner.write_driver(tmp, groups, "test")
    before = driver.stat().st_mtime_ns
    assert MojoRunner.write_driver(tmp, groups, "test").stat().st_mtime_ns == before


def test_write_driver_bench_kind_uses_bench_suite(tmp):
    groups = {f"{tmp}/marrow/tests/bench_bitmap.mojo": ["bench_and"]}
    driver = MojoRunner.write_driver(tmp, groups, "bench")
    source = driver.read_text()

    # Benchmarks get their own driver: they cannot share -O3 with -O1 tests.
    assert driver.name.startswith("_bench_driver_")
    assert "from marrow.utils.testing import BenchSuite" in source
    assert "BenchSuite.run[" in source


def test_write_driver_skips_files_without_cases(tmp):
    groups = dict(_driver_groups(tmp))
    groups[f"{tmp}/marrow/tests/test_all_skipped.mojo"] = []
    assert (
        "test_all_skipped"
        not in MojoRunner.write_driver(tmp, groups, "test").read_text()
    )


def test_run_tests_reports_build_failure_against_every_case(tmp):
    """A runner that fails to build must fail its cases, not vanish silently."""
    groups = _driver_groups(tmp)
    original = MojoRunner.run_suite
    MojoRunner.run_suite = staticmethod(lambda *a, **k: (None, "boom"))
    try:
        results = MojoRunner.run_tests(_FakeConfig(tmp), groups)
    finally:
        MojoRunner.run_suite = original

    assert set(results) == {"test_sort_one", "test_arrays_one", "test_arrays_two"}
    assert all(
        status == "FAIL" and error == "boom" for status, error in results.values()
    )


def test_run_tests_marks_missing_cases_failed(tmp):
    """A case the runner never reported is a failure, not a silent pass."""
    groups = {f"{tmp}/marrow/tests/test_arrays.mojo": ["test_ran", "test_vanished"]}
    original = MojoRunner.run_suite
    MojoRunner.run_suite = staticmethod(
        lambda *a, **k: ([{"name": "test_ran", "status": "PASS"}], "")
    )
    try:
        results = MojoRunner.run_tests(_FakeConfig(tmp), groups)
    finally:
        MojoRunner.run_suite = original

    assert results["test_ran"][0] == "PASS"
    assert results["test_vanished"][0] == "FAIL"


def test_run_tests_splits_the_unit_when_the_compiler_crashes(tmp):
    """A compiler crash is a size problem — the halves must still be run."""
    groups = _driver_groups(tmp)
    original = MojoRunner.run_suite
    calls = []

    def fake(config, groups, kind):
        names = [name for cases in groups.values() for name in cases]
        calls.append(tuple(names))
        if len(names) > 1:
            return None, "Please submit a bug report to https://..."
        return [{"name": names[0], "status": "PASS"}], ""

    MojoRunner.run_suite = staticmethod(fake)
    try:
        results = MojoRunner.run_tests(_FakeConfig(tmp), groups)
    finally:
        MojoRunner.run_suite = original

    assert set(results) == {"test_sort_one", "test_arrays_one", "test_arrays_two"}
    assert all(status == "PASS" for status, _ in results.values())
    assert len(calls) > 1  # the whole selection, then its halves


def test_run_tests_does_not_split_on_a_compile_error(tmp):
    """A diagnostic fails identically in every subset — splitting is waste."""
    groups = _driver_groups(tmp)
    original = MojoRunner.run_suite
    calls = []

    def fake(config, groups, kind):
        calls.append(1)
        return None, "marrow/kernels/filter.mojo:12:5: error: use of unknown 'x'"

    MojoRunner.run_suite = staticmethod(fake)
    try:
        results = MojoRunner.run_tests(_FakeConfig(tmp), groups)
    finally:
        MojoRunner.run_suite = original

    assert len(calls) == 1
    assert all(status == "FAIL" for status, _ in results.values())


def _run_history_selftests():
    tests = [
        test_make_envelope,
        test_write_envelope,
        test_update_history_first_run,
        test_update_history_idempotent,
        test_update_history_appends,
        test_update_history_max_runs,
        test_flags_test_vs_bench,
        test_flags_gpu_is_opt_in,
        test_flags_asan_only_when_requested,
        test_module_path,
        test_write_driver_imports_every_selected_case,
        test_write_driver_is_deterministic,
        test_write_driver_name_is_unique_per_selection,
        test_write_driver_leaves_unchanged_file_alone,
        test_write_driver_bench_kind_uses_bench_suite,
        test_write_driver_skips_files_without_cases,
        test_run_tests_reports_build_failure_against_every_case,
        test_run_tests_marks_missing_cases_failed,
        test_run_tests_splits_the_unit_when_the_compiler_crashes,
        test_run_tests_does_not_split_on_a_compile_error,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp)
    print(f"All {len(tests)} conftest selftests passed.")


if __name__ == "__main__":
    # `selftest` runs this file's own BenchmarkHistory tests.  Compiling the
    # library is `pixi run precompile`, a plain `mojo precompile marrow`.
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        _run_history_selftests()
    else:
        sys.exit(f"conftest.py: unknown command {command!r}")
