"""Profile a Mojo or Python workload using Instruments Time Profiler.

Builds the target with debug line tables (-g1), records an xctrace trace,
and opens it in Instruments.

Usage (via pixi task):
    pixi run profile marrow/kernels/tests/profile_filter.mojo
    pixi run profile --open python/tests/profile_filter.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
MOJO = ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo"


def build_mojo(script: Path, out: Path):
    """Compile a .mojo file into an executable with line-table debug info."""
    cmd = [
        str(MOJO), "build", "-I", ".",
        str(script), "-g", "--debug-info-language", "C",
        "-O2", "-o", str(out),
    ]
    print(f"Building {script.name} (-g, C debug info)...")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit("mojo build failed")


def record_trace(launch_cmd: list[str], trace_dir: Path, env: dict | None = None) -> Path:
    """Run a command under Instruments Time Profiler, return trace path."""
    trace_path = trace_dir / "profile.trace"
    cmd = [
        "xcrun", "xctrace", "record",
        "--template", "Time Profiler",
        "--output", str(trace_path),
        "--launch", "--",
        *launch_cmd,
    ]
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    if not trace_path.exists():
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"xctrace failed (exit {result.returncode})")
    return trace_path


def main():
    parser = argparse.ArgumentParser(description="Profile a marrow workload with Instruments")
    parser.add_argument("script", help="Mojo (.mojo) or Python (.py) script to profile")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open the trace in Instruments GUI (default)")
    parser.add_argument("--no-open", action="store_true", help="Don't open Instruments")
    args = parser.parse_args()

    script = Path(args.script).resolve()
    if not script.exists():
        sys.exit(f"Script not found: {script}")

    tmp = tempfile.mkdtemp(prefix="marrow_profile_")
    try:
        if script.suffix == ".mojo":
            exe = Path(tmp) / script.stem
            build_mojo(script, exe)
            launch_cmd = [str(exe)]
            env = None
        elif script.suffix == ".py":
            # Build the shared lib with debug info for Python workloads
            so = ROOT / "python" / "marrow.so"
            build_cmd = [
                str(MOJO), "build", "-I", ".",
                "python/lib.mojo", "--emit", "shared-lib",
                "-g1", "-o", str(so),
            ]
            print("Building marrow.so (-g1)...")
            result = subprocess.run(build_cmd, cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
                sys.exit("mojo build failed")
            launch_cmd = [str(PYTHON), str(script)]
            env = {"PYTHONPATH": str(ROOT / "python")}
        else:
            sys.exit(f"Unsupported file type: {script.suffix}")

        print(f"Recording trace for {script.name}...")
        trace_path = record_trace(launch_cmd, Path(tmp), env)

        dest = ROOT / f"{script.stem}.trace"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(trace_path), str(dest))
        print(f"Trace saved to: {dest}")

        if args.open and not args.no_open:
            subprocess.run(["open", str(dest)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
