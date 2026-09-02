"""``marrow compile`` — compile a ``.mojo`` query file into a standalone binary.

A user writes a ``.mojo`` file that builds a plan against a known schema,
declares its late-bound values with ``QueryCli.param``/``.argument``, and ends
in ``cli.run(plan)``. This module builds that file with the same recipe
``benchmarks/binary_size/compare.py:build_and_strip`` uses to measure gate
binaries: ``mojo build -O3 -g0 -I <marrow> <src> -o <out>``, then ``strip``.

See ``benchmarks/binary_size/query_cli.mojo`` for a complete example, and
``docs/guide/compile.qmd`` for the guide.

**The output writers are chosen in the source, not here.** They used to be
gated by a ``-D MARROW_CLI_WRITERS=true`` define this module passed by default,
which is why it once carried a ``--no-writers`` flag. ``QueryCli.run`` takes
them as comptime parameters instead — ``cli.run[parquet=True](plan)`` links the
Parquet writer, ``cli.run(plan)`` does not — so the query file says which
formats its binary supports and there is no build flag that can disagree with
it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_VERSION = "1.1.0"
MAX_VERSION = "2"
REQUIRED_RANGE = f">={MIN_VERSION},<{MAX_VERSION}"

# marrow pins this nightly build exactly (see `pixi.toml`); kept as a
# constant purely so it shows up in one place if the pin moves.
PINNED_NIGHTLY = "1.1.0.dev2026081705"

_NIGHTLY_HELP = (
    "marrow pins a Mojo *nightly* build (currently "
    f"{PINNED_NIGHTLY}), while PyPI's stable `mojo` package tops out at "
    "1.0.0. A wheel cannot force `--extra-index-url`, so `pip install "
    "marrow[compile]` cannot resolve marrow's exact compiler today. "
    "Install the nightly explicitly with:\n"
    "  pip install --pre mojo-compiler --extra-index-url "
    "https://whl.modular.com/nightly/simple/\n"
    "or build from a checkout of the marrow repo with pixi, which pins the "
    "nightly automatically (`pixi run -e dev mojo build ...`)."
)

_VERSION_RE = re.compile(r"Mojo (\d+)\.(\d+)\.(\d+)")


def build_command(
    src: Path,
    out: Path,
    marrow_path: Path,
    opt: str = "-O3",
) -> list[str]:
    """The `mojo build` invocation for compiling `src` into `out`.

    Mirrors `benchmarks/binary_size/compare.py:build_and_strip`'s recipe:
    `mojo build -O3 -g0 -I <marrow_path> <src> -o <out>`, so a compiled
    query's size is directly comparable to the gate numbers.
    """
    return [
        "mojo",
        "build",
        opt,
        "-g0",
        "-I",
        str(marrow_path),
        str(src),
        "-o",
        str(out),
    ]


def _is_system_dep(path: str) -> bool:
    """True for a dependency marrow does not need to ship.

    `/usr/lib` and `/System` (macOS) and `/lib`, `/lib64` and `/usr/lib`
    (Linux) are present on any target machine by construction — the OS
    would not boot without them.
    """
    return (
        path.startswith("/usr/lib")
        or path.startswith("/System")
        or path.startswith("/lib")
    )


def _otool_lines(flag: str, path: Path) -> list[str]:
    result = subprocess.run(
        ["otool", flag, str(path)], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def _otool_deps(path: Path) -> list[str]:
    """The dependency list from `otool -L`, skipping the self-id/path line."""
    lines = _otool_lines("-L", path)[1:]
    return [line.strip().split(" (")[0] for line in lines if line.strip()]


_RPATH_RE = re.compile(r"path (.+) \(offset \d+\)")


def _otool_rpaths(path: Path) -> list[str]:
    """The `LC_RPATH` entries baked into `path`, in load-command order."""
    lines = _otool_lines("-l", path)
    rpaths = []
    for i, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            match = _RPATH_RE.search(lines[i + 2])
            if match is not None:
                rpaths.append(match.group(1))
    return rpaths


def _resolve_macos_dep(dep: str, loader: Path) -> Path | None:
    """Resolve one `otool -L` dependency string to a file on disk.

    `@rpath/libX.dylib` is resolved against `loader`'s own `LC_RPATH`
    entries (each of which may itself be `@loader_path`-relative) —
    dependency resolution happens per-Mach-O-file, not against the
    original binary's rpath. `@loader_path/...` and `@executable_path/...`
    are resolved directly against `loader`'s directory. An absolute path
    is returned as-is.

    Deliberately **not** `.resolve()`d: a conda-forge library typically
    installs as a chain of version symlinks (`libbrotlicommon.dylib` ->
    `libbrotlicommon.1.dylib` -> `libbrotlicommon.1.2.0.dylib`), and the
    dependency string names the *symlink* (`@rpath/libbrotlicommon.1.dylib`)
    — collapsing that to the real file's name broke the copy `bundle()`
    made: a Mach-O looks up its dependency by the exact name it recorded, so
    shipping the file under the resolved `.1.2.0` name left the referenced
    `.1.dylib` name missing at runtime. Returning the un-resolved candidate
    keeps `.name` equal to what the dependent actually asks for; the caller
    still dereferences the symlink when it copies the bytes
    (`shutil.copy2`'s default `follow_symlinks=True`).
    """
    if dep.startswith("@rpath/"):
        name = dep.removeprefix("@rpath/")
        for rpath in _otool_rpaths(loader):
            base = rpath.replace("@loader_path", str(loader.parent)).replace(
                "@executable_path", str(loader.parent)
            )
            candidate = Path(base) / name
            if candidate.exists():
                return candidate
        return None
    if dep.startswith("@loader_path/"):
        candidate = loader.parent / dep.removeprefix("@loader_path/")
        return candidate if candidate.exists() else None
    if dep.startswith("@executable_path/"):
        candidate = loader.parent / dep.removeprefix("@executable_path/")
        return candidate if candidate.exists() else None
    if dep.startswith("/"):
        return Path(dep)
    return None


def _dylib_closure_macos(binary: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    frontier = [binary]
    while frontier:
        current = frontier.pop()
        for dep in _otool_deps(current):
            if _is_system_dep(dep):
                continue
            resolved = _resolve_macos_dep(dep, current)
            if resolved is None or not resolved.exists() or resolved == binary:
                continue
            if resolved not in seen:
                seen[resolved] = None
                frontier.append(resolved)
    return list(seen.keys())


def _ldd_deps(path: Path) -> list[str]:
    result = subprocess.run(
        ["ldd", str(path)], capture_output=True, text=True, check=True
    )
    deps = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=>" in line:
            target = line.split("=>", 1)[1].strip().split(" (")[0].strip()
            if target and target != "not found":
                deps.append(target)
        elif line.startswith("/"):
            deps.append(line.split(" (")[0].strip())
    return deps


def _dylib_closure_linux(binary: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    frontier = [binary]
    while frontier:
        current = frontier.pop()
        for dep in _ldd_deps(current):
            if _is_system_dep(dep):
                continue
            resolved = Path(dep).resolve()
            if not resolved.exists() or resolved == binary:
                continue
            if resolved not in seen:
                seen[resolved] = None
                frontier.append(resolved)
    return list(seen.keys())


def dylib_closure(binary: Path) -> list[Path]:
    """The transitive closure of `binary`'s non-system shared-library deps.

    Direct deps come from a single `otool -L` (macOS) / `ldd` (Linux) call;
    each newly discovered dependency is then walked the same way, so a lib
    pulled in only transitively (`libAsyncRTRuntimeGlobals.dylib`, pulled in
    by `libKGENCompilerRTShared.dylib` rather than linked directly into the
    query binary) is still found. `/usr/lib`, `/System` and `/lib*` entries
    are excluded throughout — they are present on any target machine.

    This must recurse rather than hardcode the dylib list: marrow's own
    query binaries currently link 2 dylibs directly and depend on 4
    transitively, and a `-D MARROW_GPU=true` build pulls in a 5th
    (`libMGPRT.dylib`). A fixed list silently ships a broken bundle the
    moment that closure changes.
    """
    binary = binary.resolve()
    if sys.platform == "darwin":
        return _dylib_closure_macos(binary)
    return _dylib_closure_linux(binary)


# --- Parquet compression codecs -------------------------------------------
#
# marrow's block codecs (zstd, snappy, lz4, zlib, brotli) are `dlopen`-ed at
# runtime, not linked (see `marrow/utils/compression.mojo`), so `otool -L` /
# `ldd` — and therefore `dylib_closure()` above — cannot see them: a bundle
# built from those alone silently drops every compression codec, and snappy
# is pyarrow's default, so that broke reading most real Parquet files.
# `marrow/utils/compression.mojo` also tries an executable-relative
# candidate before the bare soname now (see its `_exe_dir`/`_with_exe_dir`),
# which is what makes a copy staged here actually get found at runtime —
# copying alone is necessary but not sufficient.
#
# Keep this candidate table in sync with `compression.mojo`'s
# `_ZSTD_PATHS`/`_SNAPPY_PATHS`/`_LZ4_PATHS`/`_ZLIB_PATHS`/
# `_BROTLI_ENC_PATHS`/`_BROTLI_DEC_PATHS` — same names, same order.
_CODEC_LIB_CANDIDATES: dict[str, list[str]] = {
    "zstd": ["libzstd.dylib", "libzstd.1.dylib", "libzstd.so", "libzstd.so.1"],
    "snappy": ["libsnappy.dylib", "libsnappy.so", "libsnappy.so.1"],
    "lz4": ["liblz4.dylib", "liblz4.so", "liblz4.so.1"],
    "zlib": ["libz.dylib", "libz.1.dylib", "libz.so", "libz.so.1"],
    "brotlienc": ["libbrotlienc.dylib", "libbrotlienc.so", "libbrotlienc.so.1"],
    "brotlidec": ["libbrotlidec.dylib", "libbrotlidec.so", "libbrotlidec.so.1"],
}


def codec_lib_dir() -> Path | None:
    """The directory holding marrow's `dlopen`-ed codec libraries, resolved
    from the active environment rather than a hardcoded pixi path.

    Tries `$CONDA_PREFIX/lib` first — set by `pixi run`/`pixi shell` for
    whichever environment is active — then falls back to two directories up
    from `mojo` on `PATH` (`.../bin/mojo` -> `.../lib`), the same layout any
    conda/pixi environment uses. Returns `None` if neither resolves, so the
    caller can skip codec staging with a warning instead of guessing a path
    that may not exist on this machine.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "lib"
        if candidate.is_dir():
            return candidate
    mojo = shutil.which("mojo")
    if mojo:
        candidate = Path(mojo).resolve().parent.parent / "lib"
        if candidate.is_dir():
            return candidate
    return None


def _find_codec_lib(lib_dir: Path, names: list[str]) -> Path | None:
    """The first of `names` that exists in `lib_dir`, unresolved (a
    version-symlink chain is dereferenced only when its bytes are copied,
    not when picking the destination filename — see `_resolve_macos_dep`).
    """
    for name in names:
        candidate = lib_dir / name
        if candidate.exists():
            return candidate
    return None


def stage_codec_libs(lib_dir: Path | None) -> list[Path]:
    """The compression-codec libraries marrow's Parquet reader can `dlopen`,
    plus their own transitive dependency closure (`libbrotlienc.dylib` pulls
    in `libbrotlicommon.dylib`, for instance).

    A codec whose library is not installed in `lib_dir` — or `lib_dir`
    itself unresolved — is skipped with a warning rather than raising: a
    bundle missing one codec still reads every file compressed with the
    others, and still reads uncompressed Parquet.
    """
    if lib_dir is None:
        print(
            "marrow: warning: could not resolve the codec library directory "
            "(checked $CONDA_PREFIX/lib and the mojo binary's ../lib); "
            "--bundle will ship with no zstd/snappy/lz4/zlib/brotli support",
            file=sys.stderr,
        )
        return []
    staged: dict[str, Path] = {}
    for codec, names in _CODEC_LIB_CANDIDATES.items():
        lib = _find_codec_lib(lib_dir, names)
        if lib is None:
            print(
                f"marrow: warning: {codec} library not found in {lib_dir}, "
                "skipping (--bundle will not support that codec)",
                file=sys.stderr,
            )
            continue
        staged.setdefault(lib.name, lib)
        for dep in dylib_closure(lib):
            staged.setdefault(dep.name, dep)
    return list(staged.values())


def _copy_deduped(staged: dict[str, Path], dest: Path) -> None:
    """Copy `staged` (destination filename -> source path) into `dest`,
    writing each distinct file's bytes exactly once.

    A conda-forge codec library's un-resolved candidate names
    (`_resolve_macos_dep`'s self-id-as-dependency case: `libzstd.dylib` and
    `libzstd.1.dylib` both naming the same real file) would otherwise be
    copied twice under `shutil.copy2` — harmless for correctness but doubled
    the codec footprint. Every name past the first real copy of a given file
    becomes a symlink to it instead.
    """
    by_real: dict[Path, list[str]] = {}
    for name, src in staged.items():
        by_real.setdefault(src.resolve(), []).append(name)
    for real, names in by_real.items():
        primary, *aliases = names
        shutil.copy2(real, dest / primary)
        for alias in aliases:
            (dest / alias).symlink_to(primary)


def bundle(binary: Path, dest: Path) -> Path:
    """Copy `binary`, its dylib closure, and marrow's compression codec
    libraries into `dest`, relocatably.

    The binary's `LC_RPATH` (macOS) / `RUNPATH` (Linux) is rewritten to
    `@loader_path` / `$ORIGIN` so it resolves its link-time dylibs next to
    itself instead of inside the local pixi environment — `dest` can then be
    zipped, copied to another machine, or shipped as a Lambda deployment
    unit and run without the build machine's pixi environment present. The
    codec libraries (zstd, snappy, lz4, zlib, brotli — see `stage_codec_libs`)
    are `dlopen`-ed rather than linked, so they need no rpath entry; they are
    found via the executable-relative candidate `compression.mojo` now tries
    before the bare soname. Returns the path to the copied binary inside
    `dest`.
    """
    binary = binary.resolve()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    staged: dict[str, Path] = {}
    for dep in dylib_closure(binary):
        staged.setdefault(dep.name, dep)
    for lib in stage_codec_libs(codec_lib_dir()):
        staged.setdefault(lib.name, lib)

    out_binary = dest / binary.name
    shutil.copy2(binary, out_binary)
    _copy_deduped(staged, dest)

    if sys.platform == "darwin":
        for rpath in _otool_rpaths(out_binary):
            subprocess.run(
                ["install_name_tool", "-delete_rpath", rpath, str(out_binary)],
                check=True,
            )
        subprocess.run(
            ["install_name_tool", "-add_rpath", "@loader_path", str(out_binary)],
            check=True,
        )
    else:
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", str(out_binary)], check=True
        )

    return out_binary


def _bundled_path() -> Path:
    return Path(__file__).resolve().parent / "_mojo"


def _autodetect_repo_root() -> Path | None:
    """Walk up from the current directory looking for a marrow checkout.

    A directory qualifies if it has both a `marrow/` subdirectory and a
    `pixi.toml` — the second check keeps an unrelated directory that
    happens to be named `marrow` from matching.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "marrow").is_dir() and (candidate / "pixi.toml").is_file():
            return candidate
    return None


def _unresolved_message(explicit: str | None) -> str:
    bundled = _bundled_path()
    lines = [
        "could not locate the marrow Mojo package. Resolution order:",
        f"  1. --marrow-path" + (f" = {explicit}" if explicit else " (not given)"),
        "  2. $MARROW_MOJO_PATH"
        + (
            f" = {os.environ['MARROW_MOJO_PATH']}"
            if os.environ.get("MARROW_MOJO_PATH")
            else " (not set)"
        ),
        f"  3. bundled marrow/_mojo/ = {bundled}",
        "  4. repo-root autodetect (walking up from the current directory"
        " for a `marrow/` + `pixi.toml` pair)",
    ]
    lines.append(
        "Pass --marrow-path, or set the MARROW_MOJO_PATH environment "
        "variable, to the directory containing the `marrow/` package."
    )
    return "\n".join(lines)


def resolve_marrow_path(explicit: str | None = None) -> Path:
    """Resolve the directory to pass to `mojo build -I`.

    Resolution order: `--marrow-path` (`explicit`) -> `$MARROW_MOJO_PATH` ->
    the bundled `marrow/_mojo/` -> repo-root autodetect. An explicitly given
    location (the flag or the environment variable) is authoritative: if it
    does not contain a `marrow/` package, resolution fails immediately
    rather than silently falling back to a location the caller did not ask
    for. Only when neither is given do the bundled copy and autodetection
    run. Raises `FileNotFoundError` listing all four locations when nothing
    resolves.
    """
    if explicit is not None:
        path = Path(explicit)
        if (path / "marrow").is_dir():
            return path
        raise FileNotFoundError(_unresolved_message(explicit))

    env = os.environ.get("MARROW_MOJO_PATH")
    if env:
        path = Path(env)
        if (path / "marrow").is_dir():
            return path
        raise FileNotFoundError(_unresolved_message(None))

    bundled = _bundled_path()
    if (bundled / "marrow").is_dir():
        return bundled

    autodetected = _autodetect_repo_root()
    if autodetected is not None:
        return autodetected

    raise FileNotFoundError(_unresolved_message(None))


def check_mojo_version() -> str:
    """Verify `mojo` is on PATH and within marrow's required range.

    Raises `RuntimeError` naming the required range (`>=1.1.0,<2`), plus —
    since the most likely reason mojo is missing or too old is that only
    PyPI's stable wheel was installed — the nightly-vs-stable limitation and
    where to get the nightly instead of letting an opaque compiler error
    surface later.
    """
    if shutil.which("mojo") is None:
        raise RuntimeError(
            f"mojo compiler not found on PATH (requires {REQUIRED_RANGE}).\n"
            f"{_NIGHTLY_HELP}"
        )

    try:
        result = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`mojo --version` failed: {e}") from e

    match = _VERSION_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(
            f"could not parse `mojo --version` output: {result.stdout!r} "
            f"(requires {REQUIRED_RANGE})"
        )

    version = match.group(0).removeprefix("Mojo ")
    major, minor, _patch = (int(g) for g in match.groups())
    if not (major == 1 and minor >= 1):
        raise RuntimeError(
            f"marrow requires mojo {REQUIRED_RANGE}, found {version}.\n{_NIGHTLY_HELP}"
        )
    return version


def _add_compile_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "compile",
        help="compile a marrow query (.mojo) into a standalone binary",
        description="Compile a marrow query (.mojo, ending in "
        "cli.run(plan)) into a standalone binary.",
    )
    parser.add_argument("file", type=Path, help="the .mojo source file to compile")
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=None,
        help="output binary path (default: the source file's name, "
        "without .mojo, in the current directory; same as -o/--output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        default=None,
        help="output binary path, same as the positional [out] (takes "
        "precedence if both are given)",
    )
    parser.add_argument(
        "--marrow-path",
        default=None,
        help="directory containing the marrow/ package "
        "(default: $MARROW_MOJO_PATH, the bundled copy, or repo-root "
        "autodetection)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="build with -O1 instead of -O3 (faster compile, slower binary)",
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="skip stripping debug symbols from the output binary",
    )
    parser.add_argument(
        "--bundle",
        metavar="DIR",
        default=None,
        help="copy the built binary and its dylib closure into DIR, with "
        "the rpath rewritten to @loader_path/$ORIGIN, so DIR is a "
        "self-contained, relocatable directory that runs without the "
        "local pixi environment (default: emit a bare binary)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print the build command"
    )
    return parser


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marrow",
        description="marrow's command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_compile_subparser(subparsers)
    return parser


def _run_compile(args: argparse.Namespace) -> int:
    try:
        check_mojo_version()
    except RuntimeError as e:
        print(f"marrow: {e}", file=sys.stderr)
        return 1

    try:
        marrow_path = resolve_marrow_path(args.marrow_path)
    except FileNotFoundError as e:
        print(f"marrow: {e}", file=sys.stderr)
        return 1

    src = args.file
    if not src.is_file():
        print(f"marrow: no such file: {src}", file=sys.stderr)
        return 1

    if args.output is not None:
        out = args.output
    elif args.out is not None:
        out = args.out
    else:
        out = Path.cwd() / src.stem
    opt = "-O1" if args.fast else "-O3"
    cmd = build_command(src, out, marrow_path, opt=opt)

    if args.verbose:
        print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"marrow: build failed ({e})", file=sys.stderr)
        return 1

    if not args.no_strip:
        strip = shutil.which("strip")
        if strip is None:
            print(
                "marrow: warning: `strip` not found on PATH, skipping", file=sys.stderr
            )
        else:
            try:
                subprocess.run([strip, str(out)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"marrow: warning: strip failed ({e})", file=sys.stderr)

    if args.bundle is not None:
        try:
            bundled = bundle(out, Path(args.bundle))
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"marrow: bundle failed ({e})", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"bundled: {bundled}")

    return 0


_SUBCOMMANDS = {"compile": _run_compile}


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return _SUBCOMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
