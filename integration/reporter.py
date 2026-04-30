"""Aggregate archery integration test output into a summary table.

`run_all_tests` from `archery.integration.runner` prints results to stdout but
returns no structured data.  This module captures that stream via `Tee`, parses
it with `parse()`, and renders a per-phase + per-case summary via `report()`.
"""

from __future__ import annotations

import re
from collections import defaultdict


_PHASE_RE = re.compile(
    r"##########################################################\n"
    r"([^\n]+)\n"
    r"##########################################################"
)
_FILE_CASE_RE = re.compile(r"Testing file .*?(?:generated_)?([^/]+?)\.json\s*$")
_C_DATA_CASE_RE = re.compile(r"Testing C Arrow(?:Schema|Array) from file '([^']+)'")


class Tee:
    """Write to multiple streams (e.g. terminal + in-memory buffer)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def parse(log: str) -> dict[str, dict[str, set[str]]]:
    """Group archery's stdout by phase → {pass: {cases}, skip: {cases}}."""
    phases = [(m.start(), m.group(1).strip()) for m in _PHASE_RE.finditer(log)]
    phases.append((len(log), None))

    results: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"pass": set(), "skip": set()}
    )
    for i in range(len(phases) - 1):
        start, phase = phases[i]
        body = log[start : phases[i + 1][0]]
        cur = None
        for line in body.splitlines():
            m = _FILE_CASE_RE.match(line) or _C_DATA_CASE_RE.match(line)
            if m:
                cur = m.group(1)
                continue
            if cur is None:
                continue
            if line.startswith("-- Skipping"):
                results[phase]["skip"].add(cur)
                cur = None
            elif line.startswith("-- Validating") or line.startswith(
                "... with record batch"
            ):
                results[phase]["pass"].add(cur)
    return dict(results)


def report(log: str) -> None:
    """Print a per-phase + per-case summary table for an archery run."""
    results = parse(log)
    if not results:
        return

    phases = sorted(results)
    print("\n=== Per-phase summary ===")
    print(f"{'Phase':<55} | {'Pass':>5} | {'Skip':>5}")
    print("-" * 73)
    total_pass = total_skip = 0
    for p in phases:
        np = len(results[p]["pass"])
        ns = len(results[p]["skip"])
        total_pass += np
        total_skip += ns
        print(f"{p:<55} | {np:>5} | {ns:>5}")
    print("-" * 73)
    print(f"{'TOTAL':<55} | {total_pass:>5} | {total_skip:>5}")

    all_cases: set[str] = set()
    for p in phases:
        all_cases |= results[p]["pass"] | results[p]["skip"]

    n_phases = len(phases)
    pass_count = {
        c: sum(1 for p in phases if c in results[p]["pass"]) for c in all_cases
    }
    print("\n=== Per-case coverage ===")
    print(f"{'Case':<30} | {'Phases passing':>15}")
    print("-" * 52)
    for case, n in sorted(pass_count.items(), key=lambda x: (-x[1], x[0])):
        print(f"{case:<30} | {n:>5} / {n_phases:<7}")
