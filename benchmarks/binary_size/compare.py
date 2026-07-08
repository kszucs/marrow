#!/usr/bin/env python3
"""Builds query_comptime.mojo, query_hybrid.mojo, and query_runtime.mojo,
strips each, and reports a size/symbol-count comparison table plus a
per-module symbol breakdown so regressions (or improvements) can be traced
back to a specific package.

Run via `pixi run binary_size` from the repo root, or directly:
    python3 benchmarks/binary_size/compare.py
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
    "query_dynvalue",
    "query_hybrid",
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
    "marrow::expr::runtime",
    "marrow::expr::relations",
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


def text_segment_size(binary: Path) -> int | None:
    out = run(["size", str(binary)])
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split()
    values = lines[1].split()
    try:
        idx = header.index("__TEXT")
    except ValueError:
        return None
    return int(values[idx])


def bucket_counts(names: list[str]) -> Counter:
    counts = Counter()
    for bucket in MODULE_BUCKETS:
        counts[bucket] = sum(1 for n in names if bucket in n)
    return counts


def main() -> None:
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
                "text": text_segment_size(stripped),
            }
        )

    print()
    print(
        f"{'binary':<16} {'unstripped':>12} {'stripped':>12} "
        f"{'syms':>8} {'syms(strip)':>12} {'__TEXT':>12}"
    )
    for r in rows:
        print(
            f"{r['name']:<16} {r['unstripped']:>12,} {r['stripped']:>12,} "
            f"{r['syms']:>8,} {r['syms_stripped']:>12,} {r['text']:>12,}"
        )

    base = next(r for r in rows if r["name"] == "query_streaming")
    print()
    print("ratio vs. query_streaming (stripped size):")
    for r in rows:
        ratio = r["stripped"] / base["stripped"]
        print(f"  {r['name']:<16} {ratio:>6.1f}x")

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
