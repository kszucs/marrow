"""The Mojo compiler and the subprocess layer underneath it.

Everything that spawns a process lives here.  `MojoToolchain` is the only place
in the repository that invokes `mojo`, so the opt-level policy, the include path
and the sanitizer wiring each have exactly one definition to read and one to
change.  `ProcessRunner` is how anything long-running is run, because it owns
the progress display and the timeout; `Vcs` and `AsanRuntime.locate` call
`subprocess` directly instead, since both are sub-second probes that want no
display and no deadline.

Nothing in this module imports pytest, and nothing in it imports a third-party
package either.  The harness lifts pytest's output capture around a compile, and
that arrives as an injected context-manager factory (`ProcessRunner(suspend=...)`)
rather than as a dependency; the live display arrives the same way, from
`devkit.progress`.  The stdlib-only rule is load-bearing rather than tidy --
`python/build.py` is a hatchling build hook, and cibuildwheel gives it an
environment holding `hatchling` and the Mojo compiler and nothing else, so this
module staying importable there is what lets the wheel share one definition of
the shared-library recipe instead of keeping a second copy.
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# The checkout
# ---------------------------------------------------------------------------


def write_if_changed(path, text):
    """Write *text* to *path*, leaving the file alone if it already says that.

    Rewriting identical bytes bumps the mtime, which invalidates both the Mojo
    compiler's artifact cache and the harness's own content-addressed driver
    cache -- so an unchanged selection would recompile from scratch every run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


class Vcs:
    """Read-only version-control metadata for one working tree.

    Every reader is best-effort: a benchmark run must not fail because it was
    launched from a tarball, so an unavailable binary answers "unknown" rather
    than raising.
    """

    def __init__(self, root):
        self._root = Path(root)

    def read(self, *args):
        try:
            completed = subprocess.run(
                ["git", "-C", str(self._root), *args],
                capture_output=True,
                text=True,
            )
        except OSError:
            return "unknown"
        return completed.stdout.strip() or "unknown"

    @property
    def commit(self):
        return self.read("rev-parse", "HEAD")

    @property
    def ref(self):
        return self.read("rev-parse", "--abbrev-ref", "HEAD")


class Repo:
    """A marrow checkout: every path and name the tooling needs.

    The single source of layout truth.  Nothing else in `devkit` spells a
    directory or a Mojo namespace -- they ask here -- which is what keeps a
    rename from leaving stale strings scattered across the tooling.
    """

    #: What identifies the repository root.  `pytest.ini` fixes pytest's own
    #: rootdir, so anchoring to it means the tooling and pytest can never
    #: disagree about where the tree starts.
    MARKER = "pytest.ini"

    #: The Mojo package under test.  Everything else about it is derived.
    PACKAGE = "marrow"

    #: The corpus and the conformance suite are directories *and* Python import
    #: roots, so their names appear in generated source; keep them together.
    GOLDEN = "golden"

    #: This tooling's own package.
    DEVKIT = "devkit"

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.vcs = Vcs(self.root)

    def __repr__(self):
        return f"Repo({str(self.root)!r})"

    @classmethod
    def locate(cls, start=None):
        """Walk up from *start* (this file by default) to the marker."""
        origin = Path(start or __file__).resolve()
        for candidate in (origin, *origin.parents):
            if (candidate / cls.MARKER).exists():
                return cls(candidate)
        raise RuntimeError(f"no {cls.MARKER} at or above {origin}")

    # -- generated and build artefacts --------------------------------------

    @property
    def runner_dir(self):
        """Where generated test and bench drivers go.

        Deliberately *not* where `precompile` writes: Mojo adds a source file's
        own directory to the import search path, so an artifact left beside a
        driver shadows the whole package source tree for every later run.
        """
        return self.root / ".test_runners"

    @property
    def precompile_dir(self):
        return self.root / ".precompile"

    @property
    def artifact(self):
        """The distributable package, as `package/<name>.mojoc`."""
        return self.root / "package" / f"{self.PACKAGE}.mojoc"

    @property
    def precompiled(self):
        """The same artifact from a build-only check, kept out of the way."""
        return self.precompile_dir / f"{self.PACKAGE}.mojoc"

    # -- sources ------------------------------------------------------------

    @property
    def package_dir(self):
        """The Mojo package's own source tree."""
        return self.root / self.PACKAGE

    @property
    def python_dir(self):
        return self.root / "python"

    @property
    def bindings_entry(self):
        return self.python_dir / "bindings" / "lib.mojo"

    @property
    def libmarrow(self):
        return self.python_dir / self.PACKAGE / "libmarrow.so"

    @property
    def golden_dir(self):
        return self.root / self.GOLDEN

    @property
    def devkit_dir(self):
        return self.root / self.DEVKIT

    @property
    def benchmarks_dir(self):
        return self.root / "benchmarks"

    @property
    def footprint_dir(self):
        """Where the binary-size gate programs live."""
        return self.benchmarks_dir / "binary_size"

    # -- Mojo namespaces ----------------------------------------------------

    @property
    def testing_module(self):
        """Where the generated driver imports `TestSuite` / `BenchSuite` from."""
        return f"{self.PACKAGE}.utils.testing"


# ---------------------------------------------------------------------------
# Running a command
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """What one subprocess did."""

    argv: tuple
    returncode: int
    stdout: str
    stderr: str
    elapsed: float
    peak_rss: int = 0
    timed_out: bool = False

    @property
    def ok(self):
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self):
        """stderr and stdout joined, for reporting a failure against a case."""
        return "\n".join(part for part in (self.stderr, self.stdout) if part.strip())

    def failure(self, what):
        """A message naming *what* failed and everything the command said.

        Both streams, because Mojo splits its diagnostics across them: reading
        `stderr` alone gives an empty report for a build that failed on stdout.
        """
        return f"{what}:\n{self.output}"


class SilentProgress:
    """Reports nothing.  What tests want, and the default.

    Also the whole of the progress protocol `ProcessRunner` speaks -- `start`,
    `attach`, `finish` and `peak_rss`.  `devkit.progress.ConsoleProgress` is the
    implementation that actually shows something, and lives apart because it
    needs `rich` and `psutil`.
    """

    peak_rss = 0

    def start(self, label):
        pass

    def attach(self, pid):
        pass

    def finish(self, label, result):
        pass


class ProcessRunner:
    """Runs one command to completion, showing progress while it works.

    The timeout is what makes a *hang* reportable.  The harness already recovers
    from a compiler crash by splitting the selection, because a crash produces a
    signal; a hang produces none, so without a deadline the run blocks forever
    and takes CI with it.  A killed process reports a signal returncode, so the
    result carries a plain 124 instead -- callers that only check `!= 0` then
    treat a hang like any other failure.
    """

    TIMEOUT_NOTE = (
        "\n\nTIMEOUT: killed after {timeout}s with no exit.\n"
        "A Mojo compile or run that produces no output for this long is hung, "
        "not slow -- compare elapsed time against CPU time with "
        "`ps -o etime,time <pid>` to confirm. Raise the timeout if this "
        "selection is legitimately slower than the deadline.\n"
    )

    def __init__(self, cwd, progress=None, timeout=0, suspend=None):
        self._cwd = Path(cwd)
        self._progress = progress if progress is not None else SilentProgress()
        self._timeout = timeout
        self._suspend = suspend

    def run(self, argv, label):
        argv = [str(part) for part in argv]
        started = time.monotonic()
        # None until `_communicate` returns one.  `finish` accepts that: a
        # command that never ran still has to tear the display down, and a
        # *previous* run's result must not be reported in its place.
        result = None
        with self._suspended():
            self._progress.start(label)
            # Everything from here is guarded: an exception between `start` and
            # `finish` -- `mojo` missing from PATH is enough -- would otherwise
            # leave rich's live region running, and a live region that is never
            # torn down swallows the rest of the session's stdout.
            try:
                result = self._communicate(argv, started)
            finally:
                self._progress.finish(label, result)
        return result

    def _communicate(self, argv, started):
        process = subprocess.Popen(
            argv,
            cwd=self._cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Its own process group, so a timeout can take the whole tree down.
            # `mojo` spawns children; killing only the parent leaves them
            # holding the inherited pipes, and the second `communicate` then
            # blocks until *they* exit -- which is precisely the hang the
            # deadline exists to bound.
            start_new_session=True,
        )
        self._progress.attach(process.pid)
        timed_out = False
        try:
            out, err = process.communicate(timeout=self._timeout or None)
        except subprocess.TimeoutExpired:
            # A hung Mojo process emits nothing and never exits, so it is
            # indistinguishable from a slow compile until the deadline passes.
            # Kill it and turn the hang into an ordinary failure.
            timed_out = True
            self._terminate(process)
            out, err = process.communicate()
            err = (err or "") + self.TIMEOUT_NOTE.format(timeout=self._timeout)
        return CommandResult(
            argv=tuple(argv),
            returncode=124 if timed_out else process.returncode,
            stdout=out or "",
            stderr=err or "",
            elapsed=time.monotonic() - started,
            peak_rss=self._progress.peak_rss,
            timed_out=timed_out,
        )

    @staticmethod
    def _terminate(process):
        """Kill the process and everything it started."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()

    def _suspended(self):
        if self._suspend is None:
            return contextlib.nullcontext()
        return self._suspend()


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


class AsanRuntime:
    """The upstream LLVM AddressSanitizer runtime, as found on this machine."""

    MACOS_LIBS = ("libclang_rt.asan_osx_dynamic.dylib",)
    LINUX_LIBS = ("libclang_rt.asan-x86_64.so", "libclang_rt.asan.so")
    CLANGS = ("clang", "clang-18", "clang-17", "clang-16")

    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self):
        return f"AsanRuntime({str(self.path)!r})"

    @classmethod
    def locate(cls, env=None):
        """Search `$CONDA_PREFIX/lib`, then every clang resource dir on PATH."""
        env = os.environ if env is None else env
        names = cls.MACOS_LIBS if sys.platform == "darwin" else cls.LINUX_LIBS

        candidates = []
        prefix = env.get("CONDA_PREFIX")
        if prefix:
            candidates += [Path(prefix) / "lib" / name for name in names]
        for clang in cls.CLANGS:
            try:
                completed = subprocess.run(
                    [clang, "--print-runtime-dir"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if completed.returncode == 0:
                directory = Path(completed.stdout.strip())
                candidates += [directory / name for name in names]

        for candidate in candidates:
            if candidate.exists():
                return cls(candidate)
        return None

    def flags(self):
        """Compiler and linker flags that link this runtime in.

        `--shared-libasan` is Clang-only; on Linux the system `cc` is GCC and
        rejects it, where the explicit `-Xlinker` path is sufficient on its own.
        The rpath entry puts the conda env's lib dir first so dyld resolves the
        runtime from the pixi environment rather than the incompatible Xcode
        toolchain copy.
        """
        flags = ["--sanitize", "address"]
        if sys.platform == "darwin":
            flags += ["--shared-libasan"]
            flags += ["-Xlinker", "-rpath", "-Xlinker", str(self.path.parent)]
        return flags + ["-Xlinker", str(self.path)]


@dataclass(frozen=True)
class BuildOptions:
    """One compilation's flags, named by what the compilation is for.

    Optimization level follows the *kind* of program, not the session: a
    benchmark must measure optimized code, a test only has to be correct.  -O1
    is the floor -- at -O0 the masked-gather intrinsic in `filter`/`take` fails
    to lower ("failed to produce an archive for the module").
    """

    opt: str = "-O1"
    debug: str = "-g1"
    asserts: bool = False
    gpu: bool = False
    asan: bool = False
    include: tuple = (".",)
    debug_info_language: str = ""
    link_libm: bool = False

    @classmethod
    def for_tests(cls, *, gpu=False, asan=False):
        return cls(
            opt="-O1", debug="-g1", asserts=True, gpu=gpu, asan=asan, link_libm=True
        )

    @classmethod
    def for_benches(cls, *, gpu=False, asan=False):
        return cls(
            opt="-O3", debug="-g1", asserts=False, gpu=gpu, asan=asan, link_libm=True
        )

    @classmethod
    def for_shared_lib(cls, *, bench=False, asan=False):
        return cls(opt="-O3" if bench else "-O1", debug="-g0", asan=asan)

    @classmethod
    def for_size_gate(cls):
        return cls(opt="-O3", debug="-g0")

    @classmethod
    def for_docs(cls):
        """Compile a documentation listing to prove it still builds.

        Nothing runs the result, so there is no debug info to carry and no
        reason to optimize past the -O1 floor.
        """
        return cls(opt="-O1", debug="-g0")

    @classmethod
    def for_profiling(cls):
        """-O1 keeps frame pointers; -O2 omits them and the profile loses its
        call tree.  `--debug-info-language C` is what makes Instruments and
        `sample` resolve Mojo frames at all.
        """
        return cls(opt="-O1", debug="-g", debug_info_language="C")

    def flags(self, asan_runtime=None):
        flags = [self.opt, self.debug]
        for path in self.include:
            flags += ["-I", path]
        if self.debug_info_language:
            flags += ["--debug-info-language", self.debug_info_language]
        if self.asserts:
            flags += ["-D", "ASSERT=all"]
        if self.gpu:
            # GPU codegen is opt-in (`marrow.execution.GPU_ENABLED` defaults to
            # False), so a --gpu run has to ask for it explicitly -- without
            # this the device paths are elaborated away and every GPU test
            # would exercise the CPU fallback instead of failing honestly.
            flags += ["-D", "MARROW_GPU=true"]
        if self.asan:
            if asan_runtime is None:
                raise RuntimeError(
                    "ASAN requested but no compatible ASAN runtime found. "
                    "Install libcompiler-rt via conda-forge."
                )
            flags += asan_runtime.flags()
        if self.link_libm and sys.platform != "darwin":
            # mojo does not auto-link libm on Linux (log10f and friends);
            # harmless on macOS where libm is part of libSystem.
            flags += ["-Xlinker", "-lm"]
        return flags


class MojoToolchain:
    """Every `mojo` invocation in the repository.

    The compiler is resolved from PATH, so the toolchain follows whichever pixi
    environment the caller was launched in.  Naming an absolute path under
    `.pixi/envs/default/bin` -- which the profiler used to do -- silently
    compiles with the wrong environment's compiler under `-e bench` or
    `-e asan`.
    """

    def __init__(self, runner, asan_runtime=None, executable="mojo"):
        self._runner = runner
        self._asan = asan_runtime
        self._exe = executable

    def build(self, source, out, options, label):
        return self._runner.run(
            [self._exe, "build", *options.flags(self._asan), source, "-o", out],
            label,
        )

    def run(self, source, options, args=(), label=""):
        return self._runner.run(
            [self._exe, "run", *options.flags(self._asan), source, *args],
            label,
        )

    def build_shared_lib(self, source, out, options, label):
        return self._runner.run(
            [
                self._exe,
                "build",
                *options.flags(self._asan),
                source,
                "--emit",
                "shared-lib",
                "-o",
                out,
            ],
            label,
        )

    def execute(self, program, args=(), label=""):
        """Run an artifact this toolchain already built.

        Not a `mojo` invocation, but it belongs to the same collaborator: an
        ASAN suite is built to a binary and then run, and both halves want the
        same progress display and the same timeout.
        """
        return self._runner.run([program, *args], label)

    def precompile(self, package, out, label="precompiling marrow"):
        """Build-only check over a whole package.

        `mojo precompile` rejects `-D`, so this is always the CPU-only
        configuration and takes no `BuildOptions`.
        """
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return self._runner.run([self._exe, "precompile", package, "-o", out], label)

    @staticmethod
    def reports_errors(result):
        """Whether a precompile actually failed.

        **Judge it by its output, never by its exit status.**  A `use of unknown
        declaration` failure ends with `mojo: error: failed to parse the
        provided Mojo source module` and still exits 0, so a caller checking
        only the status reads a broken tree as a clean one.
        """
        return not result.ok or "error:" in result.output
