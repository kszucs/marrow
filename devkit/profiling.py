"""Profiling a Mojo or Python workload on macOS.

Two recorders, because they answer different questions.  `xctrace` gives an
Instruments trace with a full timeline; macOS `sample` gives a plain-text call
tree that resolves Mojo frames more reliably and can be read in any editor or
turned into a flamegraph.

Both need the target built with `BuildOptions.for_profiling()`: -O1 keeps frame
pointers, and `--debug-info-language C` is what makes either tool resolve a Mojo
frame at all.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .mojo import BuildOptions


class ProfileTarget:
    """What is being profiled, and how to launch it.

    A `.mojo` file is compiled to its own binary.  A `.py` file runs against a
    freshly built `libmarrow.so` -- the shared library is the thing under test
    there, so it has to carry the same debug info as a Mojo target would.
    """

    def __init__(self, repo, path):
        self.repo = repo
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"script not found: {self.path}")
        if self.path.suffix not in (".mojo", ".py"):
            raise ValueError(f"unsupported file type: {self.path.suffix}")

    @property
    def stem(self):
        return self.path.stem

    @property
    def is_mojo(self):
        return self.path.suffix == ".mojo"

    def prepare(self, toolchain, workdir):
        """Build what the target needs and return `(argv, env, artifact)`.

        *artifact* is the binary to keep beside a trace for symbolication, or
        None when there is nothing separate to keep.
        """
        options = BuildOptions.for_profiling()
        if self.is_mojo:
            binary = Path(workdir) / self.stem
            self._build(
                toolchain.build(
                    self.path, binary, options, f"building {self.path.name}"
                )
            )
            return [str(binary)], {}, binary

        self._build(
            toolchain.build_shared_lib(
                self.repo.bindings_entry,
                self.repo.libmarrow,
                options,
                "building libmarrow.so",
            )
        )
        return (
            [sys.executable, str(self.path)],
            {"PYTHONPATH": str(self.repo.python_dir)},
            None,
        )

    @staticmethod
    def _build(result):
        if not result.ok:
            raise RuntimeError(f"mojo build failed:\n{result.output}")


class TraceRecorder:
    """Instruments' CPU Profiler, via `xctrace`."""

    SUFFIX = ".trace"
    #: Instruments resolves symbols when the trace is *opened*, so the binary
    #: has to outlive the run that produced it.
    KEEPS_BINARY = True

    def record(self, argv, out, env):
        with tempfile.TemporaryDirectory(prefix="devkit_profile_") as workdir:
            trace = Path(workdir) / "profile.trace"
            completed = subprocess.run(
                [
                    "xcrun",
                    "xctrace",
                    "record",
                    "--template",
                    "CPU Profiler",
                    "--output",
                    str(trace),
                    "--launch",
                    "--",
                    *argv,
                ],
                env={**os.environ, **env},
                capture_output=True,
                text=True,
            )
            if not trace.exists():
                raise RuntimeError(
                    f"xctrace failed (exit {completed.returncode})\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            if out.exists():
                shutil.rmtree(out)
            shutil.move(str(trace), str(out))
        return out


class SampleRecorder:
    """macOS `sample`: a call tree with self and inclusive counts.

    Attaches to the target for the duration of the run rather than launching it,
    which is why the target is started first and given a moment to load its
    dylibs before `sample` sees it.
    """

    SUFFIX = ".sample.txt"
    #: `sample` writes resolved symbol names into its own output, so nothing
    #: has to be kept beside it.
    KEEPS_BINARY = False

    #: Generous upper bound; `sample` exits when the target does.
    MAX_SECONDS = "600"
    INTERVAL_MS = "1"

    def record(self, argv, out, env):
        target = subprocess.Popen(argv, env={**os.environ, **env})
        time.sleep(0.3)
        sampler = subprocess.Popen(
            [
                "sample",
                str(target.pid),
                self.MAX_SECONDS,
                self.INTERVAL_MS,
                "-f",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        target.wait()
        sampler.wait()
        if not out.exists():
            raise RuntimeError("sample failed to write output")
        return out


class Profiler:
    """Builds a target, records it, and leaves the result in the repository root."""

    def __init__(self, repo, toolchain, recorder):
        self.repo = repo
        self._toolchain = toolchain
        self._recorder = recorder

    def run(self, target):
        destination = self.repo.root / f"{target.stem}{self._recorder.SUFFIX}"
        with tempfile.TemporaryDirectory(prefix="devkit_profile_") as workdir:
            argv, env, artifact = target.prepare(self._toolchain, workdir)
            self._recorder.record(argv, destination, env)
            kept = self._keep_artifact(artifact)
        return destination, kept

    def _keep_artifact(self, artifact):
        """Copy the binary out of the tempdir when the recorder still needs it."""
        # Read the flag rather than defaulting it: a recorder that forgot to
        # declare one would otherwise silently lose its symbolication.
        if artifact is None or not self._recorder.KEEPS_BINARY:
            return None
        kept = self.repo.root / artifact.name
        shutil.copy2(artifact, kept)
        return kept
