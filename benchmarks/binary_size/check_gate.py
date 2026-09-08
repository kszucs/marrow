#!/usr/bin/env python3
"""CI gate for `benchmarks/binary_size`.

Builds the gates recorded in `baseline.json`, measures each one's `__text`
section via `compare.py`'s `text_section_size` (never file size -- see that
function's docstring for why: `__text` is code only, while file size and the
`__TEXT` segment are both padded to a 16 KB page boundary on Apple Silicon and
cannot see a change smaller than that), and fails if any of them grew past
`threshold_pct` relative to a baseline measurement.

**The baseline has to be measured on the machine doing the checking.** The same
source builds 0.5-1.6% larger on a macOS runner than on a developer's Mac, so a
baseline recorded by hand and committed reads as a REGRESSION on every gate the
moment CI runs it -- which is what the binary-size job did on every run, against
a baseline re-recorded the same day. CI therefore measures the commit under test
*and* the commit it descends from, on one runner, and compares those:

    # the base commit, built from a second checkout with this script
    python3 benchmarks/binary_size/check_gate.py --repo ../base --out base.json
    # the commit under test, checked against it
    python3 benchmarks/binary_size/check_gate.py --baseline base.json

`baseline.json` still names the gates and carries `threshold_pct`, and its
numbers remain the reference for a local run:

    pixi run -e dev python3 benchmarks/binary_size/check_gate.py
    pixi run -e dev python3 benchmarks/binary_size/check_gate.py --update

Those numbers are a developer-machine record, useful for judging a change by
hand and meaningless to compare against a runner's.

`threshold_pct` is 0.5, tighter than the 1% often used as a rule of thumb,
because the regression that motivated this gate
added 8,260 bytes to `query_streaming` -- 0.63% of its ~1.3M baseline -- and
was caught only by a human re-running `pixi run binary_size` by hand. A 1%
threshold would not have caught it. 0.5% still leaves well over an order of
magnitude of headroom above the few-bytes-to-low-hundreds drift ordinary,
unrelated commits cause (e.g. +128 bytes / 0.003% on `query_streaming_agg_fused`
recorded the same day), so it should not false-positive on incidental noise.
"""

import argparse
import json
import sys
from pathlib import Path

from compare import HERE, REPO_ROOT, build_and_strip, gates_dir, text_section_size

BASELINE_PATH = HERE / "baseline.json"


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text())


def measure(names: list[str], root: Path) -> dict[str, int]:
    """Build and measure each gate from the checkout at `root`."""
    here = gates_dir(root)
    measured = {}
    for name in names:
        if not (here / f"{name}.mojo").exists():
            # A gate the change adds does not exist at the commit it is
            # measured against; there is nothing to compare it to yet.
            print(f"{name} is absent from {root} -- skipping", file=sys.stderr)
            continue
        print(f"building {name} ...", file=sys.stderr)
        build_and_strip(name, root=root)
        measured[name] = text_section_size(here / f"{name}_stripped")
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="checkout to build the gates from (default: this one)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="measurements to compare against (default: baseline.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="also write this run's measurements here",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="measure and write --out, without comparing (used for the base commit)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-record baseline.json from this run",
    )
    args = parser.parse_args()

    gates = load_baseline()["gates"]
    # Absolute: `build_and_strip` passes the source path to a `mojo build` whose
    # cwd is the repo root, so a relative one would resolve twice.
    measured = measure(list(gates), args.repo.resolve())

    if args.out:
        args.out.write_text(json.dumps(measured, indent=2) + "\n")
        print(f"wrote {len(measured)} measurements to {args.out}")

    if args.measure_only:
        return

    if args.update:
        baseline = load_baseline()
        baseline["gates"] = measured
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"wrote new baseline to {BASELINE_PATH}")
        return

    baseline = load_baseline(args.baseline)
    threshold_pct = load_baseline()["threshold_pct"]
    # `--baseline base.json` holds bare measurements; baseline.json wraps them.
    against = baseline.get("gates", baseline)

    print(f"{'gate':<28} {'baseline':>12} {'measured':>12} {'delta':>10} {'pct':>8}")
    failed = []
    for name, text in measured.items():
        base = against.get(name)
        if base is None:
            print(f"{name:<28} {'-':>12} {text:>12,}   new gate, no baseline")
            continue
        delta = text - base
        pct = 100.0 * delta / base
        flag = "  REGRESSION" if pct > threshold_pct else ""
        if flag:
            failed.append(name)
        print(f"{name:<28} {base:>12,} {text:>12,} {delta:>+10,} {pct:>+7.3f}%{flag}")

    if failed:
        print()
        print(
            f"FAIL: {', '.join(failed)} grew more than {threshold_pct}% in "
            f"`__text` versus {args.baseline}."
        )
        print(
            "If the growth is intentional, re-record the local reference with "
            "--update and commit benchmarks/binary_size/baseline.json, and say "
            "in the commit message why the growth is worth it."
        )
        sys.exit(1)

    print()
    print(f"OK: no gate grew more than {threshold_pct}%.")


if __name__ == "__main__":
    main()
