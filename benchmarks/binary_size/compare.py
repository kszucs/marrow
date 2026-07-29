#!/usr/bin/env python3
"""Builds query_streaming.mojo, query_dynvalue.mojo, and query_runtime.mojo,
strips each, and reports a size/symbol-count comparison table plus a
per-module symbol breakdown so regressions (or improvements) can be traced
back to a specific package.

Run via `pixi run binary_size` from the repo root, or directly:
    python3 benchmarks/binary_size/compare.py

Pass gate names to measure only those (query_streaming is always included, as
the ratio baseline) -- a full sweep is five -O3 builds and about ten minutes:
    python3 benchmarks/binary_size/compare.py query_dynvalue

**Sizes are reported as the `__text` section, not file size.** See
`text_section_size` for why that distinction is the whole point of this script.
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

NAMES = [
    "query_streaming",
    "query_arith",
    "query_exprs",
    "query_sort",
    "query_join",
    "query_scan",
    "query_streaming_agg_fused",
    "query_streaming_agg",
    "query_dynvalue",
    "query_runtime",
]

# Symbol-name substrings to group by, in report order. A symbol may match
# more than one (mangled names embed nested generic type params), so these
# are proportional buckets, not a strict partition.
MODULE_BUCKETS = [
    "marrow::kernels::execution",
    "marrow::dtypes",
    "marrow::views",
    "marrow::arrays",
    "marrow::kernels::arithmetic",
    "marrow::builders",
    "marrow::kernels::hashing",
    "marrow::kernels::compare",
    "marrow::kernels::filter",
    "marrow::kernels::join",
    "marrow::kernels::groupby",
    "marrow::kernels::boolean",
    "marrow::scalars",
    "marrow::buffers",
    "marrow::expr::values",
    "marrow::expr::dynamic",
    "marrow::expr::relations",
    "marrow::expr::execution",
    "marrow::tabular",
    "marrow::c_data",
    "marrow::schema",
]


def run(cmd: list[str], **kwargs) -> str:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True, **kwargs
    ).stdout


def build_and_strip(name: str) -> None:
    binary = HERE / name
    stripped = HERE / f"{name}_stripped"
    subprocess.run(
        [
            "mojo",
            "build",
            "-O3",
            "-g0",
            "-I",
            ".",
            str(HERE / f"{name}.mojo"),
            "-o",
            str(binary),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    stripped.write_bytes(binary.read_bytes())
    subprocess.run(["strip", str(stripped)], check=True)


def nm_lines(binary: Path) -> list[str]:
    out = run(["nm", str(binary)])
    return [l for l in out.splitlines() if l.strip()]


def symbol_names(binary: Path) -> list[str]:
    """Extract symbol names for module-bucketing. Each nm line is either
    `<address> <type> <name...>` (defined) or `<blank> U <name...>`
    (undefined) -- demangled Mojo names routinely contain embedded spaces
    (nested generic signatures), so the name is everything after the first
    one or two fields, not just the last whitespace-separated token.
    """
    names = []
    for line in nm_lines(binary):
        parts = line.split(None, 2)
        if len(parts) == 3:
            names.append(parts[2])
        elif len(parts) == 2:
            names.append(parts[1])
    return names


def count_symbols(binary: Path) -> int:
    return len(nm_lines(binary))


def text_section_size(binary: Path) -> int | None:
    """Bytes of machine code: the `__text` *section*.

    Not the `__TEXT` segment and not the file size, both of which are padded up
    to a page boundary — 16 KB on Apple Silicon. Measured 2026-07-29: a change
    that added 1,728 bytes of code moved the stripped file size by 16,504 and
    the segment by exactly 16,384, while the symbol count went *down* by one.
    Any gate reading either of those cannot see a change smaller than a page,
    and reports a phantom page jump for a small one that crosses a boundary.
    """
    out = run(["size", "-m", str(binary)])
    for line in out.splitlines():
        parts = line.split()
        # "\tSection __text: 5266164"
        if len(parts) >= 3 and parts[0] == "Section" and parts[1] == "__text:":
            return int(parts[2])
    return None


def bucket_counts(names: list[str]) -> Counter:
    counts = Counter()
    for bucket in MODULE_BUCKETS:
        counts[bucket] = sum(1 for n in names if bucket in n)
    return counts


def main() -> None:
    global NAMES
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        unknown = wanted - set(NAMES)
        if unknown:
            sys.exit(f"unknown gate(s): {', '.join(sorted(unknown))}")
        # query_streaming is the ratio baseline, so it is always measured.
        NAMES = [n for n in NAMES if n in wanted or n == "query_streaming"]

    for name in NAMES:
        print(f"building {name} ...", file=sys.stderr)
        build_and_strip(name)

    rows = []
    for name in NAMES:
        binary = HERE / name
        stripped = HERE / f"{name}_stripped"
        rows.append(
            {
                "name": name,
                "unstripped": binary.stat().st_size,
                "stripped": stripped.stat().st_size,
                "syms": count_symbols(binary),
                "syms_stripped": count_symbols(stripped),
                "text": text_section_size(stripped),
            }
        )

    print()
    print(
        f"{'binary':<16} {'unstripped':>12} {'stripped':>12} "
        f"{'syms':>8} {'syms(strip)':>12} {'__text':>12}"
    )
    for r in rows:
        print(
            f"{r['name']:<16} {r['unstripped']:>12,} {r['stripped']:>12,} "
            f"{r['syms']:>8,} {r['syms_stripped']:>12,} {r['text']:>12,}"
        )

    base = next(r for r in rows if r["name"] == "query_streaming")
    print()
    print("ratio vs. query_streaming (__text -- code only, not page-padded):")
    for r in rows:
        ratio = r["text"] / base["text"]
        print(f"  {r['name']:<16} {ratio:>6.1f}x")
    print()
    print("Compare runs on __text. The stripped column is page-granular (16 KB on")
    print("Apple Silicon) and moves in steps -- do not quote deltas from it.")

    print()
    print("=== per-module symbol counts (unstripped) ===")
    header = f"{'module':<32}" + "".join(f"{n:>16}" for n in NAMES)
    print(header)
    per_name_names = {name: symbol_names(HERE / name) for name in NAMES}
    per_name_counts = {name: bucket_counts(per_name_names[name]) for name in NAMES}
    for bucket in MODULE_BUCKETS:
        row = f"{bucket:<32}"
        for name in NAMES:
            row += f"{per_name_counts[name][bucket]:>16,}"
        print(row)


if __name__ == "__main__":
    main()
