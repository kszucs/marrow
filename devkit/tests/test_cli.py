"""The click CLI, and the profiler's command construction.

`--help` on every command is not a formality: each command imports its heavy
dependency inside its own body, so a broken import shows up nowhere else until
somebody runs that command for real.
"""

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from devkit.cli import Context, cli
from devkit.mojo import BuildOptions, Repo
from devkit.profiling import ProfileTarget, Profiler, SampleRecorder, TraceRecorder


def walk(command, path=()):
    """Every (path, command) pair in the group, leaves included."""
    yield path, command
    for name, child in getattr(command, "commands", {}).items():
        yield from walk(child, (*path, name))


@pytest.mark.parametrize(
    "path",
    [path for path, _ in walk(cli) if path],
    ids=lambda path: " ".join(path),
)
def test_every_command_has_help(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_the_group_lists_every_area():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for area in ("build", "size", "profile", "golden", "integration"):
        assert area in result.output


def test_the_integration_task_flags_reach_the_suite(monkeypatch):
    """The exact command line `pixi run integration` uses, against a stub.

    The suite itself needs archery, which only the `integration` environment
    has, so `devkit.conformance` is replaced by a module that records what it
    was asked to run.  That is what makes the assertion worth making: parsing
    alone would still pass if a flag reached `ArcherySuite.run` under the wrong
    keyword, or reached it inverted -- `--no-stop-on-error` is passed as
    `stop_on_error=not no_stop_on_error`, and nothing about the command line
    would look different if that negation were dropped.

    Stubbing is also what keeps this safe in the `integration` environment,
    where the real class is importable and would start a ninety-minute suite.
    """
    asked = {}

    class RecordingSuite:
        def run(self, **kwargs):
            asked.update(kwargs)
            return True

    stub = types.ModuleType("devkit.conformance")
    stub.ArcherySuite = RecordingSuite
    monkeypatch.setitem(sys.modules, "devkit.conformance", stub)

    result = CliRunner().invoke(
        cli,
        [
            "integration",
            "run",
            "--run-ipc",
            "--run-c-data",
            "--with-rust",
            "--with-go",
            "--with-cpp",
        ],
    )
    assert result.exit_code == 0, result.output
    assert asked == {
        "run_ipc": True,
        "run_c_data": True,
        "with_cpp": True,
        "with_rust": True,
        "with_go": True,
        "stop_on_error": True,
        "match": None,
        "gold_dirs": None,
    }


def test_integration_needs_a_protocol_to_test():
    result = CliRunner().invoke(cli, ["integration", "run"])
    assert result.exit_code != 0
    assert "--run-ipc" in result.output


class StubContext(Context):
    """A context over a scratch repository whose builds always fail.

    Lets a command body run end to end without a compiler, so the wiring
    between the CLI and `devkit` is exercised rather than just the signature.
    """

    def __init__(self, root):
        super().__init__(repo=Repo(root))

    @property
    def toolchain(self):
        from devkit.mojo import CommandResult

        class FailingToolchain:
            def build(self, source, out, options, label):
                return CommandResult(
                    argv=(), returncode=1, stdout="", stderr="no compiler", elapsed=0.0
                )

        return FailingToolchain()


def size_repo(tmp_path, gates=("query_streaming",)):
    """A scratch repository holding real gate programs and a recorded floor.

    The programs have to exist on disk: the gate list is discovered, not
    declared, so an empty directory means there is nothing to measure.
    """
    directory = tmp_path / "benchmarks" / "binary_size"
    directory.mkdir(parents=True)
    for gate in gates:
        (directory / f"{gate}.mojo").touch()
    (directory / "baseline.json").write_text(
        json.dumps({"threshold_pct": 0.5, "gates": dict.fromkeys(gates, 1_000_000)})
    )
    return tmp_path


def test_size_check_runs_its_body(tmp_path):
    """Reaches the build and reports it, rather than failing on CLI wiring.

    `--help` never enters a command body, so a broken call into `devkit` -- an
    unpacked return value, a renamed attribute -- is invisible to every other
    test here and shows up only when someone runs `pixi run binary_size_check`.
    """
    result = CliRunner().invoke(
        cli, ["size", "check"], obj=StubContext(size_repo(tmp_path))
    )
    assert not isinstance(result.exception, TypeError), result.exception
    assert "did not build" in result.output


def test_size_compare_runs_its_body(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["size", "compare", "query_streaming"],
        obj=StubContext(size_repo(tmp_path)),
    )
    assert not isinstance(result.exception, TypeError), result.exception
    assert "did not build" in result.output


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def test_profile_target_rejects_what_it_cannot_build(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProfileTarget(Repo(tmp_path), tmp_path / "missing.mojo")
    other = tmp_path / "notes.txt"
    other.touch()
    with pytest.raises(ValueError, match="unsupported file type"):
        ProfileTarget(Repo(tmp_path), other)


class RecordingToolchain:
    def __init__(self):
        self.calls = []

    def _ok(self):
        from devkit.mojo import CommandResult

        return CommandResult(argv=(), returncode=0, stdout="", stderr="", elapsed=0.0)

    def build(self, source, out, options, label):
        self.calls.append(("build", str(source), str(out), options))
        Path(out).write_bytes(b"")  # a build leaves a binary behind
        return self._ok()

    def build_shared_lib(self, source, out, options, label):
        self.calls.append(("shared_lib", str(source), str(out), options))
        return self._ok()


def test_a_mojo_target_is_built_to_its_own_binary(tmp_path):
    script = tmp_path / "profile_sort.mojo"
    script.touch()

    toolchain = RecordingToolchain()
    argv, env, artifact = ProfileTarget(Repo(tmp_path), script).prepare(
        toolchain, tmp_path
    )

    kind, source, out, options = toolchain.calls[0]
    assert kind == "build" and source == str(script)
    # -O1 keeps frame pointers and C debug info is what resolves Mojo frames.
    assert options == BuildOptions.for_profiling()
    assert argv == [str(tmp_path / "profile_sort")] and env == {}
    assert artifact is not None  # Instruments symbolicates against it


def test_a_python_target_rebuilds_the_shared_library(tmp_path):
    script = tmp_path / "profile_clickbench.py"
    script.touch()

    toolchain = RecordingToolchain()
    argv, env, artifact = ProfileTarget(Repo(tmp_path), script).prepare(
        toolchain, tmp_path
    )

    kind, source, _, options = toolchain.calls[0]
    assert kind == "shared_lib"
    assert source.endswith("python/bindings/lib.mojo")
    # The library is the thing under test, so it needs the same debug info.
    assert options == BuildOptions.for_profiling()
    assert argv == [sys.executable, str(script)]
    assert env["PYTHONPATH"].endswith("python")
    assert artifact is None


def test_a_build_failure_is_reported_not_profiled(tmp_path):
    script = tmp_path / "broken.mojo"
    script.touch()

    from devkit.mojo import CommandResult

    class FailingToolchain:
        def build(self, source, out, options, label):
            return CommandResult(
                argv=(), returncode=1, stdout="", stderr="boom", elapsed=0.0
            )

    with pytest.raises(RuntimeError, match="boom"):
        ProfileTarget(Repo(tmp_path), script).prepare(FailingToolchain(), tmp_path)


def test_each_recorder_names_its_own_output(tmp_path):
    """The two answer different questions and must not overwrite each other."""
    assert TraceRecorder.SUFFIX == ".trace"
    assert SampleRecorder.SUFFIX == ".sample.txt"
    # Instruments resolves symbols when the trace is opened; `sample` writes
    # them into its own output.
    assert TraceRecorder.KEEPS_BINARY and not SampleRecorder.KEEPS_BINARY


def test_the_profiler_puts_the_result_in_the_repository_root(tmp_path):
    script = tmp_path / "profile_sort.mojo"
    script.touch()
    repo = Repo(tmp_path)

    class StubRecorder:
        SUFFIX = ".trace"
        KEEPS_BINARY = True

        def record(self, argv, out, env):
            out.write_text("trace")
            return out

    destination, kept = Profiler(repo, RecordingToolchain(), StubRecorder()).run(
        ProfileTarget(repo, script)
    )
    assert destination == repo.root / "profile_sort.trace"
    assert destination.exists()
    # The binary is copied out of the tempdir, which is gone by now.
    assert kept is not None and kept.exists()
