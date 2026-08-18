"""Profile a Mojo or Python workload using Instruments Time Profiler.

Builds the target with debug line tables (-g1), records an xctrace trace,
and opens it in Instruments.

Usage (via pixi task):
    pixi run profile marrow/kernels/tests/profile_filter.mojo
    pixi run profile --open python/marrow/tests/profile_filter.py
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


def build_mojo(script: Path, out: Path, opt: str = "-O1"):
    """Compile a .mojo file into an executable with line-table debug info."""
    cmd = [
        str(MOJO), "build", "-I", ".",
        str(script), "-g", "--debug-info-language", "C",
        opt,  # -O1: keeps frame pointers; -O2 omits them; --no-optimization breaks masked.gather
        "-o", str(out),
    ]
    print(f"Building {script.name} ({opt}, -g, C debug info)...")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit("mojo build failed")


def record_trace(launch_cmd: list[str], trace_dir: Path, env: dict | None = None) -> Path:
    """Run a command under Instruments CPU Profiler, return trace path."""
    trace_path = trace_dir / "profile.trace"
    cmd = [
        "xcrun", "xctrace", "record",
        "--template", "CPU Profiler",
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


def record_sample(launch_cmd: list[str], out: Path, env: dict | None = None) -> Path:
    """Run the binary and collect a call-tree profile with macOS `sample`.

    Launches the target, waits for it to start, attaches `sample` for the
    duration of the run, then waits for the target to exit.  The resulting
    file can be opened in any text editor (call tree with self/inclusive
    sample counts) or converted to a flamegraph with flamegraph.pl.
    """
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(launch_cmd, env=full_env)
    # Give the process a moment to load its dylibs before attaching.
    import time
    time.sleep(0.3)
    sample_cmd = [
        "sample", str(proc.pid),
        "600",   # sample for up to 600 s (exits when target exits)
        "1",     # 1 ms interval
        "-f", str(out),
    ]
    sample_proc = subprocess.Popen(sample_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    sample_proc.wait()
    if not out.exists():
        sys.exit("sample failed to write output")
    return out


def main():
    parser = argparse.ArgumentParser(description="Profile a marrow workload with Instruments or sample")
    parser.add_argument(
        "script",
        help=(
            "Mojo (.mojo) or Python (.py) script, or a ClickBench query named "
            "`clickbench-q1` / `clickbench-q34` (see python/marrow/tests/clickbench.py)"
        ),
    )
    parser.add_argument("--sample", action="store_true",
                        help="Use macOS `sample` instead of xctrace (resolves Mojo frames)")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open the result after recording (default)")
    parser.add_argument("--no-open", action="store_true", help="Don't open the result")
    args = parser.parse_args()

    # `clickbench-q1` is a shorthand for the shared query registry rather than a
    # path, so the thing you profile is the same definition the correctness suite
    # checks and the benchmark times -- there is no separate profiling copy to
    # drift out of sync.
    profile_query = None
    if args.script.lower().startswith("clickbench-"):
        profile_query = args.script.split("-", 1)[1]
        script = ROOT / "python" / "marrow" / "tests" / "profile_clickbench.py"
    else:
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
            so = ROOT / "python" / "marrow" / "libmarrow.so"
            build_cmd = [
                str(MOJO), "build", "-I", ".",
                "python/bindings/lib.mojo", "--emit", "shared-lib",
                "-g", "--debug-info-language", "C",
                "-O1", "-o", str(so),
            ]
            print("Building libmarrow.so (-g, C debug info)...")
            result = subprocess.run(build_cmd, cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
                sys.exit("mojo build failed")
            launch_cmd = [str(PYTHON), str(script)]
            env = {"PYTHONPATH": str(ROOT / "python")}
            if profile_query is not None:
                env["MARROW_PROFILE_QUERY"] = profile_query
                # The query is looped so the profiler samples the query itself
                # rather than interpreter start-up; override for a long one.
                env.setdefault(
                    "MARROW_PROFILE_REPEATS",
                    os.environ.get("MARROW_PROFILE_REPEATS", "20"),
                )
        else:
            sys.exit(f"Unsupported file type: {script.suffix}")

        if args.sample:
            stem = f"{script.stem}-{profile_query}" if profile_query else script.stem
            dest = ROOT / f"{stem}.sample.txt"
            print(f"Recording sample for {script.name}...")
            record_sample(launch_cmd, dest, env)
            print(f"Sample saved to: {dest}")
            if args.open and not args.no_open:
                subprocess.run(["open", str(dest)])
        else:
            print(f"Recording trace for {script.name}...")
            trace_path = record_trace(launch_cmd, Path(tmp), env)

            stem = f"{script.stem}-{profile_query}" if profile_query else script.stem
            dest = ROOT / f"{stem}.trace"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(trace_path), str(dest))
            print(f"Trace saved to: {dest}")

            # Keep the binary next to the trace so Instruments can symbolicate it.
            if script.suffix == ".mojo":
                bin_dest = ROOT / script.stem
                shutil.copy2(str(exe), str(bin_dest))
                print(f"Binary kept at: {bin_dest} (needed for symbolication)")

            if args.open and not args.no_open:
                subprocess.run(["open", str(dest)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
