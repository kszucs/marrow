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
    _CODEC_LIB_CANDIDATES,
    _copy_deduped,
    _find_codec_lib,
    build_command,
    bundle,
    check_mojo_version,
    codec_lib_dir,
    dylib_closure,
    main,
    resolve_marrow_path,
    stage_codec_libs,
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


# --- output writers ---------------------------------------------------------
#
# The Parquet/IPC writers were once gated by a `-D MARROW_CLI_WRITERS=true`
# build define, which `marrow compile` passed by default and `--no-writers`
# opted out of. That flag is gone: `QueryCli.run[parquet=True, ipc=True]` puts
# the choice in the query source instead, so the file states which formats its
# binary supports and no build invocation can disagree with it. There is
# nothing left for `build_command` to pass, and the three tests that asserted
# it did are removed rather than rewritten -- they pinned the mechanism, and
# the mechanism is what changed.


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


# --- codec library staging ---------------------------------------------------
#
# marrow's Parquet compression codecs (zstd, snappy, lz4, zlib, brotli) are
# `dlopen`-ed, not linked, so `dylib_closure()` above cannot see them — a
# bundle built from that alone drops every codec, and snappy is pyarrow's
# default, so real Parquet files failed to read from a bundled binary. These
# tests cover the pure directory/filename logic (`codec_lib_dir`,
# `_find_codec_lib`, `_copy_deduped`) with fakes, plus a couple that check
# the real pixi `dev` environment actually has the libraries `compression.mojo`
# tries to `dlopen` — no `mojo build` involved either way.


def test_codec_lib_dir_prefers_conda_prefix(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert codec_lib_dir() == tmp_path / "lib"


def test_codec_lib_dir_falls_back_to_mojo_location(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    (tmp_path / "lib").mkdir()
    (tmp_path / "bin").mkdir()
    fake_mojo = tmp_path / "bin" / "mojo"
    fake_mojo.write_text("")
    monkeypatch.setattr(
        "marrow.compile.shutil.which",
        lambda name: str(fake_mojo) if name == "mojo" else None,
    )
    assert codec_lib_dir() == tmp_path / "lib"


def test_codec_lib_dir_returns_none_when_unresolved(monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr("marrow.compile.shutil.which", lambda name: None)
    assert codec_lib_dir() is None


def test_find_codec_lib_returns_first_existing_candidate(tmp_path):
    (tmp_path / "libfoo.so.1").write_bytes(b"stub")
    found = _find_codec_lib(tmp_path, ["libfoo.dylib", "libfoo.so", "libfoo.so.1"])
    assert found == tmp_path / "libfoo.so.1"


def test_find_codec_lib_returns_none_when_missing(tmp_path):
    assert _find_codec_lib(tmp_path, ["libfoo.dylib", "libfoo.so"]) is None


def test_stage_codec_libs_none_dir_warns_and_returns_empty(capsys):
    assert stage_codec_libs(None) == []
    assert "codec library directory" in capsys.readouterr().err


def test_stage_codec_libs_skips_missing_codec_with_warning(tmp_path, capsys):
    # An empty directory: every codec candidate misses.
    assert stage_codec_libs(tmp_path) == []
    err = capsys.readouterr().err
    for codec in _CODEC_LIB_CANDIDATES:
        assert codec in err


def test_stage_codec_libs_finds_a_present_codec_and_skips_the_rest(tmp_path, capsys):
    (tmp_path / "libzstd.dylib").write_bytes(b"stub")
    found = stage_codec_libs(tmp_path)
    assert [p.name for p in found] == ["libzstd.dylib"]
    err = capsys.readouterr().err
    assert "zstd library not found" not in err  # zstd was found, not warned about
    assert "snappy library not found" in err  # every other codec still warned


def test_copy_deduped_writes_one_real_file_and_symlinks_aliases(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    real = src_dir / "libfoo.1.2.3.dylib"
    real.write_bytes(b"payload")
    dest = tmp_path / "dest"
    dest.mkdir()

    staged = {"libfoo.dylib": real, "libfoo.1.dylib": real}
    _copy_deduped(staged, dest)

    names = sorted(p.name for p in dest.iterdir())
    assert names == ["libfoo.1.dylib", "libfoo.dylib"]
    real_files = [p for p in dest.iterdir() if not p.is_symlink()]
    symlinks = [p for p in dest.iterdir() if p.is_symlink()]
    assert len(real_files) == 1
    assert len(symlinks) == 1
    assert real_files[0].read_bytes() == b"payload"
    assert symlinks[0].resolve().read_bytes() == b"payload"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only: otool")
def test_pixi_dev_env_has_every_codec_compression_mojo_dlopens():
    """Not a fake: this is the actual environment `pixi run -e dev` builds
    in. If a codec listed in `_CODEC_LIB_CANDIDATES` (kept in sync with
    `compression.mojo`) goes missing from the `zstd`/`snappy`/`lz4-c`/
    `brotli`/`zlib` conda dependencies, this is what would catch it before
    a `--bundle` run does."""
    lib_dir = codec_lib_dir()
    if lib_dir is None:
        pytest.skip("no active pixi/conda environment")
    missing = [
        codec
        for codec, names in _CODEC_LIB_CANDIDATES.items()
        if _find_codec_lib(lib_dir, names) is None
    ]
    assert missing == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only: otool")
def test_stage_codec_libs_includes_brotli_transitive_dependency():
    lib_dir = codec_lib_dir()
    if (
        lib_dir is None
        or _find_codec_lib(lib_dir, _CODEC_LIB_CANDIDATES["brotlienc"]) is None
    ):
        pytest.skip("brotli not present in this environment")
    names = {p.name for p in stage_codec_libs(lib_dir)}
    assert "libbrotlienc.dylib" in names
    assert "libbrotlicommon.1.dylib" in names  # transitive, not a direct candidate


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only: otool")
def test_bundle_includes_codec_libraries(tmp_path):
    if not _GATE_BINARY.exists():
        pytest.skip("gate binary not built")
    if codec_lib_dir() is None:
        pytest.skip("no active pixi/conda environment")
    dest = tmp_path / "bundled"

    bundle(_GATE_BINARY, dest)

    names = {p.name for p in dest.iterdir()}
    for codec in ("zstd", "snappy", "lz4"):
        assert any(n in names for n in _CODEC_LIB_CANDIDATES[codec]), (
            f"{codec} missing from bundle: {names}"
        )
