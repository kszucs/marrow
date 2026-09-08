"""Turning a selection of Mojo cases into a compiled driver and a set of results.

This is the harness proper: which lanes a session runs, which cases a driver has
to cover, how that driver is generated, and how one compilation unit's JSON maps
back onto individual cases.

Nothing here imports pytest.  `RunnerOptions` declares and reads the command
line through duck-typed `parser`/`config` objects, and `conftest.py` builds a
`Selection` out of collected items -- so every decision this module makes is
reachable from a plain unit test.
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .mojo import write_if_changed


# ---------------------------------------------------------------------------
# Options and lanes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerOptions:
    """The harness's command line, resolved once.

    Declared and read here rather than in `conftest.py` so the option surface
    sits next to the code that consumes it, and so a test can construct the
    combination it cares about instead of faking a pytest config.
    """

    mojo: bool = False
    no_mojo: bool = False
    python: bool = False
    no_python: bool = False
    cpu: bool = False
    gpu: bool = False
    no_gpu: bool = False
    benchmark: bool = False
    asan: bool = False
    mojo_timeout: int = 1800
    competition: bool = False
    save_benchmarks: str = ""
    benchmark_history: str = ""
    num_threads: int = 0

    @staticmethod
    def flag(name):
        """`mojo_timeout` -> `--mojo-timeout`."""
        return "--" + name.replace("_", "-")

    @staticmethod
    def declare(parser):
        """Add every harness option to *parser* (anything with `.addoption`).

        Written out flat rather than driven from a table: the payload of these
        lines is the help text, and a table would only add a
        `type -> argparse kwargs` translation nobody reads.  It also stays
        greppable by flag name, which is how anyone actually finds one.
        """
        add = parser.addoption
        add("--mojo", action="store_true", default=False, help="Select Mojo tests")
        add("--no-mojo", action="store_true", default=False, help="Exclude Mojo tests")
        add("--python", action="store_true", default=False, help="Select Python tests")
        add(
            "--no-python",
            action="store_true",
            default=False,
            help="Exclude Python tests",
        )
        add(
            "--cpu",
            action="store_true",
            default=False,
            help="Select CPU tests (non-GPU Mojo + Python)",
        )
        add("--gpu", action="store_true", default=False, help="Select GPU tests")
        add("--no-gpu", action="store_true", default=False, help="Exclude GPU tests")
        add(
            "--num-threads",
            type=int,
            default=0,
            metavar="N",
            help="CPU worker budget for golden-corpus execution (0 = auto).",
        )
        add(
            "--benchmark",
            action="store_true",
            default=False,
            help="Include benchmarks (Python pytest-benchmark and Mojo "
            "bench_*.mojo); skipped by default",
        )
        add(
            "--asan",
            action="store_true",
            default=False,
            help="Run Mojo tests under AddressSanitizer (ASAN)",
        )
        add(
            "--mojo-timeout",
            type=int,
            default=1800,
            metavar="SECONDS",
            help=(
                "Kill a Mojo compile/run that exceeds SECONDS and report it as "
                "a failure (default: 1800). 0 disables the timeout. The harness "
                "already recovers from a compiler *crash* by splitting the "
                "selection, because a crash produces a signal; a hang produces "
                "none, so without this the run blocks forever."
            ),
        )
        add(
            "--competition",
            action="store_true",
            default=False,
            help="After benchmarks, print a side-by-side comparison table for "
            "all measured libs.",
        )
        add(
            "--save-benchmarks",
            metavar="DIR",
            default="",
            help="Save benchmark results as a JSON envelope to DIR/<commit>.json "
            "(implies --benchmark).",
        )
        add(
            "--benchmark-history",
            metavar="FILE",
            default="",
            help="Path to the rolling benchmark history JSON file "
            "(default: benchmarks/data.json).",
        )

    @classmethod
    def from_config(cls, config):
        """Read every option off *config* (anything with `.getoption`).

        Derived from the fields rather than spelled out a third time: the flag
        for a field is mechanically its name with dashes, and the default is
        already on the field.  Writing them out again is a drift surface, which
        is what `test_options_declare_every_flag_it_reads` exists to police.
        """
        values = {
            name: config.getoption(cls.flag(name), default=spec.default)
            for name, spec in cls.__dataclass_fields__.items()
        }
        # argparse hands back None for an unset string option.
        for name in ("save_benchmarks", "benchmark_history"):
            values[name] = values[name] or ""
        # Saving results is pointless without producing them, so the one
        # implies the other rather than failing on the combination.
        values["benchmark"] = bool(values["benchmark"] or values["save_benchmarks"])
        return cls(**values)


class LaneSelector:
    """Which lanes a session runs, and why anything else is skipped.

    Four positive selectors (`--mojo`, `--python`, `--cpu`, `--gpu`) and three
    negative ones (`--no-mojo`, `--no-python`, `--no-gpu`).  Naming any positive
    selector makes the session *selective*: a lane nobody named is skipped.
    `--cpu` is shorthand for every non-GPU lane, so it implies both `--mojo` and
    `--python`.
    """

    #: Trees whose tests import the package's *Python* bindings, and so need
    #: `libmarrow.so` built before the session runs.  Policy rather than
    #: layout, which is why it lives here and not on `Repo`: this is where the
    #: decision is made and where a test looks for it.
    BINDING_TREES = ("python", "golden", "examples", "docs")

    def __init__(self, options, args, repo, invocation_dir=None):
        self._options = options
        self._args = [str(arg) for arg in args]
        self._repo = repo
        # pytest resolves its path arguments against the directory it was
        # launched from, which is not the rootdir when it is run from a
        # subdirectory.  Resolving against the rootdir instead silently turns
        # every relative argument into a path that does not exist.
        self._invocation_dir = Path(invocation_dir or Path.cwd())

    @property
    def select_mojo(self):
        return self._options.mojo or self._options.cpu

    @property
    def select_python(self):
        return self._options.python or self._options.cpu

    @property
    def selective(self):
        options = self._options
        return options.mojo or options.python or options.gpu or options.cpu

    def skip_reason(self, *, is_gpu, is_mojo, is_python, is_benchmark):
        """Why this item should be skipped, or None to run it."""
        options = self._options
        if is_benchmark and not options.benchmark:
            return "benchmarks excluded; pass --benchmark to include"
        if is_gpu and (options.no_gpu or not options.gpu):
            return "GPU tests excluded; pass --gpu to include"
        if is_mojo and (options.no_mojo or (self.selective and not self.select_mojo)):
            return "Mojo tests excluded; pass --mojo to include"
        if is_python and (
            options.no_python or (self.selective and not self.select_python)
        ):
            return "Python tests excluded; pass --python to include"
        return None

    def owns(self, path):
        """True when *path* is one of the harness's own tests.

        Those are never gated on a lane: they test this code, so subjecting
        them to it lets a bug in `skip_reason` skip the tests that would have
        caught it -- and an all-skipped pytest run exits 0.
        """
        try:
            Path(path).resolve().relative_to(self._repo.devkit_dir.resolve())
        except ValueError:
            return False
        return True

    def python_excluded(self):
        """True when no Python test file should even be collected."""
        options = self._options
        if options.no_python:
            return True
        if (options.mojo or options.gpu) and not (options.python or options.cpu):
            return True
        if self._args:
            # Specific paths given -- check whether any lead to Python tests.
            for path in self._selected_paths():
                if path.is_file() and path.suffix == ".py":
                    return False
                if path.is_dir() and (
                    any(path.rglob("test_*.py")) or any(path.rglob("bench_*.py"))
                ):
                    return False
            return True
        return False

    def needs_libmarrow(self):
        """True when the session will run something that imports `marrow`.

        `python_excluded` alone does not answer this: `devkit/tests` are Python
        tests that import nothing from marrow, and building the shared library
        for them would put a multi-minute Mojo compile in front of a suite that
        runs in seconds.
        """
        if self.python_excluded():
            return False
        if not self._args:
            return True
        return any(self._in_binding_tree(path) for path in self._selected_paths())

    def _selected_paths(self):
        for arg in self._args:
            path = Path(arg.split("::")[0])
            yield path if path.is_absolute() else self._invocation_dir / path

    def _in_binding_tree(self, path):
        """Whether *path* can reach the package's Python bindings.

        The repository root itself reaches everything: `relative_to` answers
        `Path(".")` for it, whose `parts` is empty, and reading that as "names
        no tree" is how `pytest .` came to run the Python suite against a stale
        `libmarrow.so` -- collected, never rebuilt, and silently measuring the
        previous build.  An unrelated path outside the tree answers True for
        the same reason: better a redundant build than a stale one.
        """
        try:
            relative = path.resolve().relative_to(self._repo.root.resolve())
        except ValueError:
            return True
        if not relative.parts:
            return True
        return relative.parts[0] in self.BINDING_TREES


# ---------------------------------------------------------------------------
# The selection and its driver
# ---------------------------------------------------------------------------


class CaseScanner:
    """Finds the case names a `.mojo` file defines.

    A regex over the source text rather than a parse, and it has to be: a test
    file carries no `main()`, so it cannot be compiled on its own, and there is
    nothing to import a name list out of.  Collection therefore reads the same
    `def test_*(` line a human would.
    """

    @staticmethod
    def cases(path, kind):
        """Every `def {kind}_*(` the file defines, in source order."""
        pattern = re.compile(rf"^def\s+({kind}_\w+)\s*\(", re.MULTILINE)
        return pattern.findall(Path(path).read_text())

    @staticmethod
    def is_gpu(path):
        """A `*_gpu.mojo` file needs a device, and is skipped without `--gpu`."""
        return Path(path).stem.endswith("_gpu")


@dataclass(frozen=True)
class CaseRef:
    """One case: the file it is defined in, and its name."""

    path: Path
    name: str


class Selection:
    """The cases one generated driver has to cover.

    Ordering is fixed -- files by path, cases in source order -- so an unchanged
    selection renders byte-identical source and hits the Mojo compiler's own
    artifact cache instead of recompiling from scratch.
    """

    def __init__(self, cases=()):
        self._cases = tuple(cases)

    @classmethod
    def of(cls, mapping):
        """Build from a `{path: [name, ...]}` mapping."""
        return cls(
            CaseRef(Path(path), name)
            for path, names in mapping.items()
            for name in names
        )

    def __len__(self):
        return len(self._cases)

    def __bool__(self):
        return bool(self._cases)

    def __iter__(self):
        return iter(self._cases)

    def __repr__(self):
        return f"Selection({len(self)} cases in {self.file_count} files)"

    def _ordered(self):
        """Files by path, cases in source order.  `sorted` is stable, so the
        within-file order the caller supplied survives."""
        return sorted(self._cases, key=lambda case: str(case.path))

    def by_file(self):
        grouped = {}
        for case in self._ordered():
            grouped.setdefault(case.path, []).append(case.name)
        return grouped

    def names(self):
        return [case.name for case in self._ordered()]

    @property
    def file_count(self):
        return len(self.by_file())

    def halve(self):
        """Split in two, keeping each file's cases in order."""
        flat = self._ordered()
        middle = len(flat) // 2
        return Selection(flat[:middle]), Selection(flat[middle:])


class DriverGenerator:
    """Writes the single Mojo module that imports and runs a whole selection.

    The compiler accepts one input file per invocation and re-elaborates all of
    marrow for each, so compiling per test file pays that elaboration N times
    over.  Collapsing the selection into one unit pays it once.

    The file is *named* after its own content, so concurrent sessions with
    different selections cannot overwrite each other's driver mid-compile, while
    the same selection keeps resolving to the same path and the same cached
    artifact.
    """

    #: Mojo keywords that are legal directory names but illegal in an import
    #: path unless backticked.  `marrow/expr/comptime/` is a deliberate package
    #: name (the AOT lane), so the driver has to spell it `` `comptime` `` or
    #: the import fails with "expected module name".
    RESERVED = frozenset({"comptime", "fn", "def", "var", "trait", "struct", "alias"})

    SUITES = {"test": "TestSuite", "bench": "BenchSuite"}

    def __init__(self, repo, kind):
        self.repo = repo
        self.kind = kind
        self.suite = self.SUITES[kind]

    def module_path(self, path):
        """`marrow/expr/tests/test_relations.mojo` -> `marrow.expr.tests.test_relations`."""
        relative = Path(path).resolve().relative_to(self.repo.root)
        return ".".join(
            f"`{part}`" if part in self.RESERVED else part
            for part in relative.with_suffix("").parts
        )

    def render(self, selection):
        lines = [f"from {self.repo.testing_module} import {self.suite}"]
        names = []
        for path, cases in selection.by_file().items():
            if not cases:
                continue
            lines.append(f"from {self.module_path(path)} import " + ", ".join(cases))
            names.extend(cases)
        cases = ",\n            ".join(names)
        return "\n".join(lines) + (
            f"\n\n\ndef main() raises:\n"
            f"    {self.suite}.run[\n        (\n            {cases},\n        )\n    ]()\n"
        )

    def write(self, selection):
        source = self.render(selection)
        digest = hashlib.sha256(source.encode()).hexdigest()[:12]
        return write_if_changed(
            self.repo.runner_dir / f"_{self.kind}_driver_{digest}.mojo", source
        )


# ---------------------------------------------------------------------------
# Running the selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """What the runner reported for one case."""

    name: str
    status: str
    error: str = ""
    entry: dict = None

    @property
    def failed(self):
        return self.status == "FAIL"


class SuiteRunner:
    """Compiles one selection as a single unit and maps its JSON back to cases.

    Compilation dominates, and almost all of it is elaborating marrow rather
    than the case bodies, so N files in one unit cost about what one file costs.
    That is why the whole selection becomes one driver -- and why a compile
    error fails every case in the run.

    A compiler *crash* is different: it depends on how much the unit elaborates,
    so the same cases build in smaller units.  The selection is halved and each
    half compiled on its own, down to a single case, and a case that still
    cannot be built reports the crash as its own failure.  A real `error:` never
    splits -- it would be reported identically in every subset, so splitting
    would only multiply the compile time.
    """

    CRASH_MARKERS = ("Please submit a bug report", "Stack dump:")

    def __init__(self, toolchain, driver, options, notify=None):
        self._toolchain = toolchain
        self._driver = driver
        self._options = options
        self._notify = notify if notify is not None else (lambda message: None)

    @classmethod
    def compiler_crashed(cls, detail):
        """True when the compiler *died* rather than rejecting the source."""
        return any(marker in (detail or "") for marker in cls.CRASH_MARKERS)

    def run(self, selection):
        """Run every case in *selection* and return `{name: CaseResult}`."""
        if not selection:
            # An empty selection would render a driver with no imports and an
            # empty case tuple, which is not valid Mojo.  Nothing to run is not
            # a failure.
            return {}
        results = self._run_recursive(selection)
        for name in selection.names():
            results.setdefault(
                name, CaseResult(name, "FAIL", "no result reported by the runner")
            )
        return results

    def _run_recursive(self, selection, probed=False):
        """Compile *selection* as one unit, splitting only if that would help.

        *probed* records that an ancestor already established the crash is
        size-dependent, so the probe is paid once for the whole descent rather
        than at every node.
        """
        entries, detail = self._run_once(selection)
        if entries is not None:
            return self._results(entries)
        if len(selection) > 1 and self.compiler_crashed(detail):
            if probed or self._size_dependent(selection):
                self._notify(
                    f"compiler crashed on {len(selection)} cases — splitting the unit"
                )
                results = {}
                for half in selection.halve():
                    results.update(self._run_recursive(half, probed=True))
                return results
            self._notify(
                "the compiler crashes on a single case too — not a size "
                "problem, so the unit is not split"
            )
        return {name: CaseResult(name, "FAIL", detail) for name in selection.names()}

    @staticmethod
    def _results(entries):
        return {
            entry["name"]: CaseResult(
                entry["name"],
                # A benchmark entry carries no `status`: the Mojo runner only
                # emits one for a test.
                entry.get("status", "PASS"),
                entry.get("error", ""),
                entry,
            )
            for entry in entries
        }

    def _size_dependent(self, selection):
        """Whether the crash depends on how much the unit elaborates.

        Bisection assumes it does: halve until the offending case is alone.
        Some failures carry a crash marker and are *not* size-dependent -- a
        missing Metal toolchain reports "Metal Compiler failed to compile
        metallib. Please submit a bug report." for any input at all -- and
        bisecting those costs 2N-1 full elaborations of marrow to reach the
        same answer.

        One probe settles it, and only one: the answer holds for the whole
        descent, so `_run_recursive` carries it down rather than re-asking.
        """
        _, detail = self._run_once(Selection(list(selection)[:1]))
        return not self.compiler_crashed(detail)

    def _run_once(self, selection):
        """Build and run the selection; return `(entries, detail)`.

        *entries* is None when the runner could not be built or its output could
        not be parsed, and *detail* then holds the compiler or runtime output to
        report against every selected case.
        """
        driver = self._driver.write(selection)
        label = self._label(selection)

        if self._options.asan:
            # ASAN goes through `mojo build` because the sanitizer runtime has
            # to be linked into a real binary -- and a binary is what gives
            # symbolicated crash traces.  The content-addressed stem is shared
            # with the driver, so parallel sessions never link over each other.
            binary = driver.with_suffix("")
            built = self._toolchain.build(
                driver, binary, self._options, f"{label} (asan)"
            )
            if not built.ok:
                return None, f"mojo build failed for {driver}:\n{built.stderr}"
            result = self._toolchain.execute(binary, ("--json",), f"running {label}")
        else:
            # `mojo run` compiles and executes in one step without leaving an
            # artifact behind; compilation is what takes the minutes.
            result = self._toolchain.run(driver, self._options, ("--json",), label)

        # Warnings on a successful build would otherwise be swallowed.
        if result.stderr and result.ok:
            sys.stderr.write(result.stderr)
        try:
            return json.loads(result.stdout), result.stderr
        except ValueError:
            return None, result.output or f"exit code {result.returncode}"

    def _label(self, selection):
        noun = "benchmarks" if self._driver.kind == "bench" else "tests"
        return f"compiling {len(selection)} {noun} from {selection.file_count} files"
