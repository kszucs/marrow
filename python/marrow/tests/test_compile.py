"""Unit tests for `marrow.compile` — the pure functions only.

A single `mojo build` takes 1-2 minutes, so these tests never shell out to
`mojo`: `check_mojo_version`'s subprocess/PATH lookups are monkeypatched, and
the end-to-end "does it actually compile" check is a manual smoke test (see
the task report), not a unit test.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from marrow.compile import (
    _build_arg_parser,
    build_command,
    bundle,
    check_mojo_version,
    dylib_closure,
    main,
    resolve_marrow_path,
)


def test_build_command_uses_o3_and_include_path(tmp_path):
    cmd = build_command(tmp_path / "q.mojo", tmp_path / "q", tmp_path / "src")
    assert cmd[:2] == ["mojo", "build"]
    assert "-O3" in cmd and "-g0" in cmd
    assert cmd[cmd.index("-I") + 1] == str(tmp_path / "src")
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "q")


# --- `marrow compile` subcommand ---------------------------------------------
#
# The UX contract is `marrow compile <file> [out]`. The bare form
# (`marrow <file>`) is deliberately no longer accepted: nothing outside this
# repo has ever depended on it, and requiring the subcommand keeps a file
# literally named `compile` from becoming ambiguous with the subcommand
# itself. `_run_compile`'s build-invoking behaviour is already covered by
# the `build_command` tests above; these exercise the argparse wiring that
# gets a parsed `Namespace` to it.


def test_bare_form_is_no_longer_accepted():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["q.mojo"])


def test_compile_subcommand_parses_file_and_output_flag():
    parser = _build_arg_parser()
    args = parser.parse_args(["compile", "q.mojo", "-o", "out"])
    assert args.command == "compile"
    assert args.file == Path("q.mojo")
    assert args.output == Path("out")


def test_compile_subcommand_parses_positional_out():
    parser = _build_arg_parser()
    args = parser.parse_args(["compile", "q.mojo", "out"])
    assert args.file == Path("q.mojo")
    assert args.out == Path("out")
    assert args.output is None


def test_compile_subcommand_help_lists_every_flag(capsys):
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["compile", "--help"])
    out = capsys.readouterr().out
    for flag in (
        "-o",
        "--output",
        "--marrow-path",
        "--bundle",
        "--no-writers",
        "--fast",
        "--no-strip",
        "-v",
    ):
        assert flag in out


def test_top_level_help_lists_compile_subcommand(capsys):
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "compile" in out


def test_main_compile_reaches_build_command_with_flag(tmp_path, monkeypatch):
    src = tmp_path / "q.mojo"
    src.write_text("")
    marrow_dir = tmp_path / "src"
    (marrow_dir / "marrow").mkdir(parents=True)

    monkeypatch.setattr("marrow.compile.check_mojo_version", lambda: "1.1.0")
    monkeypatch.setattr("marrow.compile.shutil.which", lambda name: None)

    captured = {}

    def fake_build_command(src, out, marrow_path, opt="-O3", writers=True):
        captured["opt"] = opt
        captured["marrow_path"] = marrow_path
        return ["mojo", "build", opt]

    monkeypatch.setattr("marrow.compile.build_command", fake_build_command)

    def fake_run(cmd, check):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("marrow.compile.subprocess.run", fake_run)

    rc = main(["compile", str(src), "--marrow-path", str(marrow_dir), "--fast"])

    assert rc == 0
    assert captured["opt"] == "-O1"
    assert captured["marrow_path"] == marrow_dir


def test_resolve_marrow_path_prefers_explicit(tmp_path):
    (tmp_path / "marrow").mkdir()
    assert resolve_marrow_path(str(tmp_path)) == tmp_path


def test_resolve_marrow_path_reads_env(tmp_path, monkeypatch):
    (tmp_path / "marrow").mkdir()
    monkeypatch.setenv("MARROW_MOJO_PATH", str(tmp_path))
    assert resolve_marrow_path() == tmp_path


def test_resolve_marrow_path_reports_every_location(tmp_path, monkeypatch):
    monkeypatch.delenv("MARROW_MOJO_PATH", raising=False)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_marrow_path(str(tmp_path / "nope"))
    assert "MARROW_MOJO_PATH" in str(exc.value)


# --- CLI_WRITERS define -----------------------------------------------------
#
# Task 7 gated the Parquet/IPC output writers behind
# `-D MARROW_CLI_WRITERS=true` in `marrow/expr/relations.mojo` (off by
# default at the Mojo level, since linking them costs 572,288 bytes of
# `__text`). `marrow compile` has to pass that define by default so the
# documented `-o result.parquet` contract works out of the box; `--no-writers`
# (exercised through `build_command`'s `writers=False`) is the opt-out for the
# minimum-size build. Note the *compiler* define is `MARROW_CLI_WRITERS`, not
# the internal Mojo comptime name `CLI_WRITERS_ENABLED`.


def test_build_command_default_includes_writers_define(tmp_path):
    cmd = build_command(tmp_path / "q.mojo", tmp_path / "q", tmp_path / "src")
    assert "-D" in cmd
    assert cmd[cmd.index("-D") + 1] == "MARROW_CLI_WRITERS=true"


def test_build_command_no_writers_omits_define(tmp_path):
    cmd = build_command(
        tmp_path / "q.mojo", tmp_path / "q", tmp_path / "src", writers=False
    )
    assert "-D" not in cmd
    assert "MARROW_CLI_WRITERS=true" not in cmd


# --- check_mojo_version ------------------------------------------------------
#
# `mojo` is monkeypatched out entirely -- these test the argument-handling
# and error-formatting logic, never the real compiler.


def test_check_mojo_version_missing_raises_with_range(monkeypatch):
    monkeypatch.setattr("marrow.compile.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as exc:
        check_mojo_version()
    message = str(exc.value)
    assert ">=1.1.0,<2" in message
    assert "nightly" in message


def test_check_mojo_version_too_old_raises_with_found_version(monkeypatch):
    monkeypatch.setattr("marrow.compile.shutil.which", lambda name: "/usr/bin/mojo")

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Mojo 1.0.0 (deadbeef)\n", stderr=""
        )

    monkeypatch.setattr("marrow.compile.subprocess.run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        check_mojo_version()
    message = str(exc.value)
    assert "1.0.0" in message
    assert ">=1.1.0,<2" in message


def test_check_mojo_version_in_range_returns_version_string(monkeypatch):
    monkeypatch.setattr("marrow.compile.shutil.which", lambda name: "/usr/bin/mojo")

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Mojo 1.1.0.dev2026081705 (18b45e5c)\n", stderr=""
        )

    monkeypatch.setattr("marrow.compile.subprocess.run", fake_run)
    assert check_mojo_version() == "1.1.0"


# --- dylib_closure / bundle --------------------------------------------------
#
# These exercise `otool`/`install_name_tool` against an *already-built* gate
# binary (`benchmarks/binary_size/query_scan_typed`, built by
# `pixi run binary_size`) rather than invoking `mojo build` — a build takes
# 1-2 minutes and these tests only need a Mach-O to introspect. Both skip if
# that binary is not present locally. macOS-only: `dylib_closure`/`bundle`
# also have a Linux path (`ldd`/`patchelf`), but that is unverified here —
# see the task report.

_GATE_BINARY = Path("benchmarks/binary_size/query_scan_typed")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only: otool")
def test_dylib_closure_is_transitive_and_excludes_system():
    if not _GATE_BINARY.exists():
        pytest.skip("gate binary not built")
    names = {p.name for p in dylib_closure(_GATE_BINARY)}
    assert "libAsyncRTMojoBindings.dylib" in names
    assert "libKGENCompilerRTShared.dylib" in names
    assert "libAsyncRTRuntimeGlobals.dylib" in names  # transitive, not direct
    assert "libMSupportGlobals.dylib" in names  # transitive, not direct
    assert not any(n.startswith("libSystem") for n in names)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only: install_name_tool")
def test_bundle_copies_closure_and_rewrites_rpath_to_loader_path(tmp_path):
    if not _GATE_BINARY.exists():
        pytest.skip("gate binary not built")
    dest = tmp_path / "bundled"
    deps = dylib_closure(_GATE_BINARY)

    out = bundle(_GATE_BINARY, dest)

    assert out == dest / _GATE_BINARY.name
    assert out.exists()
    for dep in deps:
        assert (dest / dep.name).exists()

    result = subprocess.run(
        ["otool", "-l", str(out)], capture_output=True, text=True, check=True
    )
    assert "@loader_path" in result.stdout
    assert ".pixi" not in result.stdout
