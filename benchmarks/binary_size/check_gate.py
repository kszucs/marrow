#!/usr/bin/env python3
"""CI gate for `benchmarks/binary_size`.

Builds the gates recorded in `baseline.json`, measures each one's `__text`
section via `compare.py`'s `text_section_size` (never file size -- see that
function's docstring for why: `__text` is code only, while file size and the
`__TEXT` segment are both padded to a 16 KB page boundary on Apple Silicon and
cannot see a change smaller than that), and fails if any of them grew past
`threshold_pct` relative to its recorded baseline.

Run via:
    pixi run -e dev python3 benchmarks/binary_size/check_gate.py

After an intentional size change, re-record the baseline:
    pixi run -e dev python3 benchmarks/binary_size/check_gate.py --update

The baseline was re-recorded 2026-08-29 after the `exprold` deletion and the
aggregate rearchitecture, because five of the seven gates changed program and the
previous floor could not distinguish a regression from the port. A full run on
2026-08-30 confirms it: four gates match exactly, three are within 0.05% and
below. **Do not re-record without a reason** — this docstring previously carried
a blanket "every number is stale" warning that outlived the re-record by a day
and was then cited as evidence that the gate was meaningless.

`threshold_pct` is 0.5, tighter than the 1% often used as a rule of thumb,
because the regression that motivated this gate
added 8,260 bytes to `query_streaming` -- 0.63% of its ~1.3M baseline -- and
was caught only by a human re-running `pixi run binary_size` by hand. A 1%
threshold would not have caught it. 0.5% still leaves well over an order of
magnitude of headroom above the few-bytes-to-low-hundreds drift ordinary,
unrelated commits cause (e.g. +128 bytes / 0.003% on `query_streaming_agg_fused`
recorded the same day), so it should not false-positive on incidental noise.
"""

import json
import sys

from compare import HERE, build_and_strip, text_section_size

BASELINE_PATH = HERE / "baseline.json"


def load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def main() -> None:
    update = "--update" in sys.argv
    baseline = load_baseline()
    threshold_pct = baseline["threshold_pct"]
    gates = baseline["gates"]

    measured = {}
    for name in gates:
        print(f"building {name} ...", file=sys.stderr)
        build_and_strip(name)
        measured[name] = text_section_size(HERE / f"{name}_stripped")

    if update:
        baseline["gates"] = measured
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"wrote new baseline to {BASELINE_PATH}")
        return

    print(f"{'gate':<28} {'baseline':>12} {'measured':>12} {'delta':>10} {'pct':>8}")
    failed = []
    for name, base in gates.items():
        text = measured[name]
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
            "`__text` versus the recorded baseline."
        )
        print(
            "If the growth is intentional, re-run with --update and commit "
            "the updated benchmarks/binary_size/baseline.json."
        )
        sys.exit(1)

    print()
    print(f"OK: no gate grew more than {threshold_pct}%.")


if __name__ == "__main__":
    main()
