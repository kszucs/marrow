"""``marrow compile`` — compile a ``.mojo`` query file into a standalone binary.

A user writes a ``.mojo`` file that declares its parameters with ``param(...)``
and ends in ``plan.execute_cli()``. This module builds that file with the same
recipe ``benchmarks/binary_size/compare.py:build_and_strip`` uses to measure
gate binaries: ``mojo build -O3 -g0 -I <marrow> <src> -o <out>``, then
``strip``.

**Output writers are opt-in at the Mojo level, opt-out here.** Task 7 measured
that linking the Parquet + Arrow IPC output writers into ``execute_cli`` costs
572,288 bytes of ``__text`` — enough that they sit behind
``-D MARROW_CLI_WRITERS=true`` (``marrow/expr/relations.mojo``), off by
default for anyone building with plain ``mojo build``. But the CLI's
documented contract is that ``-o result.parquet`` / ``-o result.arrow`` work
out of the box, so ``marrow compile`` passes that define **by default** —
``--no-writers`` opts back out for the smaller binary.
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

_WRITERS_DEFINE = "MARROW_CLI_WRITERS=true"


def build_command(
    src: Path,
    out: Path,
    marrow_path: Path,
    opt: str = "-O3",
    writers: bool = True,
) -> list[str]:
    """The `mojo build` invocation for compiling `src` into `out`.

    Mirrors `benchmarks/binary_size/compare.py:build_and_strip`'s recipe:
    `mojo build -O3 -g0 -I <marrow_path> <src> -o <out>`. `writers=True`
    (the default) adds `-D MARROW_CLI_WRITERS=true`, which is what makes
    `execute_cli`'s documented `-o result.parquet` / `-o result.arrow`
    contract work in the compiled binary — see the module docstring.
    """
    cmd = ["mojo", "build", opt, "-g0"]
    if writers:
        cmd += ["-D", _WRITERS_DEFINE]
    cmd += ["-I", str(marrow_path), str(src), "-o", str(out)]
    return cmd


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
            f"marrow requires mojo {REQUIRED_RANGE}, found {version}.\n"
            f"{_NIGHTLY_HELP}"
        )
    return version


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marrow",
        description="Compile a marrow query (.mojo, ending in "
        "plan.execute_cli()) into a standalone binary.",
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
        "--no-writers",
        action="store_true",
        help="build without -D MARROW_CLI_WRITERS=true: saves roughly "
        "572 KB of __text but disables `-o result.parquet` / "
        "`-o result.arrow` in the compiled binary (it raises instead)",
    )
    parser.add_argument(
        "--bundle",
        metavar="DIR",
        default=None,
        help="(not yet implemented) bundle the marrow Mojo sources into DIR "
        "for a self-contained build",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print the build command"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.bundle is not None:
        print("marrow: --bundle is not implemented yet", file=sys.stderr)
        return 2

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
    cmd = build_command(src, out, marrow_path, opt=opt, writers=not args.no_writers)

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
            print("marrow: warning: `strip` not found on PATH, skipping", file=sys.stderr)
        else:
            try:
                subprocess.run([strip, str(out)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"marrow: warning: strip failed ({e})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
