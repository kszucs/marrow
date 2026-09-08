"""The compiler flags and the subprocess layer."""

import os
import subprocess
import sys
import time

import pytest

from devkit.mojo import (
    AsanRuntime,
    BuildOptions,
    ProcessRunner,
    Repo,
    SilentProgress,
)


# ---------------------------------------------------------------------------
# Repo
# ---------------------------------------------------------------------------


def test_repo_locates_the_marker(tmp_path):
    (tmp_path / Repo.MARKER).touch()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert Repo.locate(nested / "c.py").root == tmp_path.resolve()


def test_repo_without_a_marker_raises(tmp_path):
    with pytest.raises(RuntimeError, match=Repo.MARKER):
        Repo.locate(tmp_path / "nothing")


def test_repo_keeps_precompile_and_drivers_apart(tmp_path):
    """An artifact in `.test_runners/` shadows the whole `marrow/` source tree.

    Mojo adds a source file's own directory to the import search path, and the
    generated driver lives there -- so the precompile artifact must not.
    """
    repo = Repo(tmp_path)
    assert repo.precompile_dir != repo.runner_dir


# ---------------------------------------------------------------------------
# BuildOptions
# ---------------------------------------------------------------------------


def test_flags_test_vs_bench():
    test_flags = BuildOptions.for_tests().flags()
    bench_flags = BuildOptions.for_benches().flags()

    # A test must keep its assertions; a benchmark must measure optimized code.
    assert "-O1" in test_flags and "-O3" not in test_flags
    assert "ASSERT=all" in test_flags
    assert "-O3" in bench_flags and "-O1" not in bench_flags
    assert "ASSERT=all" not in bench_flags
    # Both compile against the source tree, never a precompiled package.
    assert test_flags[test_flags.index("-I") + 1] == "."
    assert bench_flags[bench_flags.index("-I") + 1] == "."


def test_flags_gpu_is_opt_in():
    """GPU codegen is off by default, so --gpu must ask for it explicitly."""
    assert "MARROW_GPU=true" not in BuildOptions.for_tests().flags()
    gpu_flags = BuildOptions.for_tests(gpu=True).flags()
    assert gpu_flags[gpu_flags.index("MARROW_GPU=true") - 1] == "-D"
    # Benchmarks honour it too -- a GPU benchmark must measure the device path.
    assert "MARROW_GPU=true" in BuildOptions.for_benches(gpu=True).flags()


def test_flags_asan_only_when_requested():
    assert "--sanitize" not in BuildOptions.for_tests().flags()
    runtime = AsanRuntime("/tmp/libclang_rt.asan.dylib")
    flags = BuildOptions.for_tests(asan=True).flags(runtime)
    assert flags[flags.index("--sanitize") + 1] == "address"
    assert str(runtime.path) in flags


def test_asan_without_a_runtime_is_an_error():
    with pytest.raises(RuntimeError, match="libcompiler-rt"):
        BuildOptions.for_tests(asan=True).flags(None)


def test_shared_lib_follows_the_session_not_the_kind():
    """The bindings are one library, so a benchmark session must build it at -O3."""
    assert "-O1" in BuildOptions.for_shared_lib().flags()
    assert "-O3" in BuildOptions.for_shared_lib(bench=True).flags()
    # Debug info costs size in a library nobody symbolicates.
    assert "-g0" in BuildOptions.for_shared_lib().flags()


def test_docs_listings_are_compiled_but_never_run():
    """Nothing executes a documentation snippet, so it carries no debug info."""
    flags = BuildOptions.for_docs().flags()
    assert "-O1" in flags and "-g0" in flags
    assert "ASSERT=all" not in flags


def test_size_gate_and_profiling_flags():
    size = BuildOptions.for_size_gate().flags()
    assert "-O3" in size and "-g0" in size
    assert "ASSERT=all" not in size

    profile = BuildOptions.for_profiling().flags()
    # -O1 keeps frame pointers, which is the whole point of profiling at all.
    assert "-O1" in profile
    assert profile[profile.index("--debug-info-language") + 1] == "C"


def test_libm_is_linked_for_runners_only():
    """`mojo` does not auto-link libm on Linux; the shared library does not need it."""
    test_flags = BuildOptions.for_tests().flags()
    if sys.platform == "darwin":
        assert "-lm" not in test_flags
    else:
        assert test_flags[-2:] == ["-Xlinker", "-lm"]
    assert "-lm" not in BuildOptions.for_shared_lib().flags()
    assert "-lm" not in BuildOptions.for_size_gate().flags()


def test_asan_runtime_locate_prefers_the_conda_prefix(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    names = (
        AsanRuntime.MACOS_LIBS if sys.platform == "darwin" else AsanRuntime.LINUX_LIBS
    )
    planted = lib / names[0]
    planted.touch()
    found = AsanRuntime.locate({"CONDA_PREFIX": str(tmp_path)})
    assert found is not None and found.path == planted


# ---------------------------------------------------------------------------
# ProcessRunner
# ---------------------------------------------------------------------------


def test_process_runner_reports_output_and_success(tmp_path):
    runner = ProcessRunner(tmp_path, SilentProgress())
    result = runner.run([sys.executable, "-c", "print('hi')"], "saying hi")
    assert result.ok
    assert result.stdout.strip() == "hi"
    assert result.returncode == 0
    assert not result.timed_out


def test_process_runner_reports_failure(tmp_path):
    runner = ProcessRunner(tmp_path, SilentProgress())
    result = runner.run(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        "failing",
    )
    assert not result.ok
    assert result.returncode == 3
    assert "boom" in result.output


#: Long enough that the child is certainly running, short enough that the
#: selftest suite stays a one-second command.  The deadline is a float all the
#: way down to `communicate(timeout=...)`; only `--mojo-timeout` is an int.
DEADLINE = 0.3


def test_process_runner_turns_a_hang_into_an_ordinary_failure(tmp_path):
    """A killed process reports a signal returncode; callers check `!= 0`."""
    runner = ProcessRunner(tmp_path, SilentProgress(), timeout=DEADLINE)
    result = runner.run(
        [sys.executable, "-c", "import time; time.sleep(30)"], "hanging"
    )
    assert result.timed_out
    assert result.returncode == 124
    assert "TIMEOUT" in result.stderr
    assert "hung, not slow" in result.stderr


def test_process_runner_uses_the_injected_suspender(tmp_path):
    """The harness lifts pytest's capture around a compile; nothing else does."""
    import contextlib

    entered = []

    @contextlib.contextmanager
    def suspend():
        entered.append("in")
        yield
        entered.append("out")

    runner = ProcessRunner(tmp_path, SilentProgress(), suspend=suspend)
    runner.run([sys.executable, "-c", "pass"], "quiet")
    assert entered == ["in", "out"]


def test_process_runner_runs_in_the_given_directory(tmp_path):
    (tmp_path / "marker.txt").touch()
    runner = ProcessRunner(tmp_path, SilentProgress())
    result = runner.run(
        [sys.executable, "-c", "import os; print(os.listdir('.'))"], "listing"
    )
    assert "marker.txt" in result.stdout


def test_command_result_output_joins_both_streams(tmp_path):
    runner = ProcessRunner(tmp_path, SilentProgress())
    result = runner.run(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); sys.stderr.write('err\\n'); sys.exit(1)",
        ],
        "both",
    )
    assert "out" in result.output and "err" in result.output


def test_a_failure_message_carries_both_streams(tmp_path):
    """Mojo splits its diagnostics; reading `stderr` alone can report nothing.

    Both the pytest boundary and the CLI report a failed bindings build through
    this, so a build failing on stdout must not produce an empty message.
    """
    runner = ProcessRunner(tmp_path, SilentProgress())
    result = runner.run(
        [sys.executable, "-c", "import sys; print('on stdout'); sys.exit(1)"],
        "failing on stdout",
    )
    message = result.failure("Failed to build libmarrow.so")
    assert "Failed to build libmarrow.so" in message
    assert "on stdout" in message


def test_toolchain_builds_the_expected_command_line(tmp_path):
    """Every `mojo` invocation in the tree goes through here, so the shape matters."""
    from devkit.mojo import MojoToolchain

    recorded = []

    class RecordingRunner:
        def run(self, argv, label):
            recorded.append(([str(a) for a in argv], label))
            return None

    toolchain = MojoToolchain(RecordingRunner(), executable="mojo")

    toolchain.run("driver.mojo", BuildOptions.for_tests(), ("--json",), "running")
    argv, label = recorded[-1]
    assert argv[:2] == ["mojo", "run"]
    assert argv[-2:] == ["driver.mojo", "--json"]
    assert label == "running"

    toolchain.build("driver.mojo", "out", BuildOptions.for_tests(), "building")
    assert recorded[-1][0][:2] == ["mojo", "build"]
    assert recorded[-1][0][-3:] == ["driver.mojo", "-o", "out"]

    toolchain.build_shared_lib(
        "lib.mojo", "lib.so", BuildOptions.for_shared_lib(), "linking"
    )
    assert recorded[-1][0][-5:] == [
        "lib.mojo",
        "--emit",
        "shared-lib",
        "-o",
        "lib.so",
    ]

    toolchain.precompile("marrow", tmp_path / "out" / "marrow.mojoc")
    assert recorded[-1][0][:3] == ["mojo", "precompile", "marrow"]
    # `mojo precompile` rejects -D, so it never carries BuildOptions flags.
    assert "-D" not in recorded[-1][0]


def test_toolchain_resolves_the_compiler_from_path(tmp_path):
    """Naming .pixi/envs/default/bin/mojo compiles with the wrong environment."""
    from devkit.mojo import MojoToolchain

    recorded = []

    class RecordingRunner:
        def run(self, argv, label):
            recorded.append([str(a) for a in argv])

    MojoToolchain(RecordingRunner()).precompile("marrow", tmp_path / "x")
    assert recorded[-1][0] == "mojo"
    assert "/" not in recorded[-1][0]


def test_asan_runtime_flags_link_the_library(tmp_path):
    runtime = AsanRuntime(tmp_path / "lib" / "libclang_rt.asan.dylib")
    flags = runtime.flags()
    assert flags[:2] == ["--sanitize", "address"]
    assert flags[-2:] == ["-Xlinker", str(runtime.path)]
    if sys.platform == "darwin":
        # dyld must resolve the pixi env's copy, not the Xcode toolchain's.
        assert "-rpath" in flags
        assert str(runtime.path.parent) in flags


def test_process_runner_stringifies_path_arguments(tmp_path):
    """Callers hand it Paths; subprocess wants str."""
    runner = ProcessRunner(tmp_path, SilentProgress())
    script = tmp_path / "s.py"
    script.write_text("print('ran')\n")
    result = runner.run([sys.executable, script], "path arg")
    assert result.stdout.strip() == "ran"
    assert all(isinstance(part, str) for part in result.argv)


def test_a_failure_to_launch_still_tears_the_display_down(tmp_path):
    """rich's live region must not outlive the command.

    A region that is never stopped keeps redirecting stdout, so the rest of the
    session's output disappears -- on top of a display stuck on screen.
    """
    from devkit.progress import ConsoleProgress

    progress = ConsoleProgress()
    runner = ProcessRunner(tmp_path, progress)
    with pytest.raises(FileNotFoundError):
        runner.run(["definitely-not-a-real-program-xyz"], "doomed")
    assert progress._progress is None


def test_the_timeout_takes_the_whole_process_tree(tmp_path):
    """`mojo` spawns children, and they inherit the pipes.

    Killing only the parent leaves them holding the write end, so the second
    `communicate()` blocks until *they* exit -- the very hang the deadline
    exists to bound.
    """
    script = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "time.sleep(30)"
    )
    runner = ProcessRunner(tmp_path, SilentProgress(), timeout=DEADLINE)
    started = time.monotonic()
    result = runner.run([sys.executable, "-c", script], "hanging with a child")
    assert result.timed_out and result.returncode == 124
    # Without the process group this waits for the grandchild's full 30 s.
    assert time.monotonic() - started < 15


def test_process_runner_propagates_a_missing_program(tmp_path):
    runner = ProcessRunner(tmp_path, SilentProgress())
    with pytest.raises((FileNotFoundError, subprocess.SubprocessError)):
        runner.run(["definitely-not-a-real-program-xyz"], "missing")


def test_the_module_imports_without_rich_or_psutil():
    """`devkit.mojo` must need nothing but the standard library.

    `python/build.py` imports it to build `libmarrow.so`, and cibuildwheel gives
    that hook an environment holding `hatchling` and the Mojo compiler and
    nothing else.  A third-party import added here therefore breaks the wheel
    and nothing else -- a failure no local run reaches, since every environment
    on this machine has both packages installed.  `sys.modules[name] = None`
    makes `import name` raise, which is what that environment looks like.
    """
    root = Repo.locate().root
    script = (
        "import sys\n"
        "sys.modules['rich'] = None\n"
        "sys.modules['psutil'] = None\n"
        "import devkit.mojo\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_precompile_is_judged_by_its_output_not_its_status():
    """A parse failure ends with `failed to parse` and still exits 0.

    Reading the status alone reports a broken tree as a clean one, which is how
    a `use of unknown declaration` can survive a green build-only check.
    """
    from devkit.mojo import CommandResult, MojoToolchain

    def result(returncode, stdout="", stderr=""):
        return CommandResult(
            argv=("mojo", "precompile"),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=0.0,
        )

    clean = result(0, stdout="0 errors\n")
    assert not MojoToolchain.reports_errors(clean)

    lying = result(
        0,
        stderr=(
            "marrow/kernels/filter.mojo:12:5: error: use of unknown declaration 'x'\n"
            "mojo: error: failed to parse the provided Mojo source module\n"
        ),
    )
    assert MojoToolchain.reports_errors(lying)

    # A non-zero status is still a failure even with nothing to say about it.
    assert MojoToolchain.reports_errors(result(1))


def test_libm_is_linked_on_linux_for_both_runner_kinds(monkeypatch):
    """`mojo` does not auto-link libm there, and log10f lives in it.

    The platform guard makes the real behaviour invisible on macOS, so the
    platform is faked rather than left to whoever runs the suite.
    """
    import devkit.mojo as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    for options in (BuildOptions.for_tests(), BuildOptions.for_benches()):
        assert options.flags()[-2:] == ["-Xlinker", "-lm"]
    # Still only the runners: the shared library and the size gate go without.
    assert "-lm" not in BuildOptions.for_shared_lib().flags()
    assert "-lm" not in BuildOptions.for_size_gate().flags()

    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert "-lm" not in BuildOptions.for_tests().flags()
