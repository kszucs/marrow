"""How much of a Parquet file each index kind could prune, per query predicate.

Produces the table in `docs/superpowers/specs/2026-08-27-index-and-pruning-plan.md`
§0, and exists because that table corrects an earlier one that did not
distinguish the two things this script prints side by side:

- **real** — the file's own row groups. This is what any reader can prune today.
- **simulated** — fixed-size row windows. These granules do not exist in the
  file; the figures are reachable only after rewriting it with a page index or
  smaller row groups, so they are statements about `marrow/parquet/writer.mojo`
  and not about index architecture.

Reporting a simulated figure as if it were achievable is the mistake this file
is here to prevent. `hits_0.parquet` reports `has_column_index = False`.

    pixi run -e dev python benchmarks/pruning/measure_prunability.py [PATH]
"""

import sys

import pyarrow.parquet as pq

DEFAULT_PATH = "~/Workspace/ClickBench/data/hits_0.parquet"

# (label, queries it appears in, column, kind, argument)
PREDICATES = [
    ("CounterID = 62", "Q36-Q42", "CounterID", "eq", 62),
    ("UserID = 435090932899640449", "Q19", "UserID", "eq", 435090932899640449),
    (
        "RefererHash = 3594120000172545465",
        "Q40",
        "RefererHash",
        "eq",
        3594120000172545465,
    ),
    ("URLHash = 2868770270353813622", "Q41", "URLHash", "eq", 2868770270353813622),
    ("AdvEngineID <> 0", "Q01,Q07", "AdvEngineID", "ne", 0),
    ("SearchPhrase <> ''", "Q12-Q14,Q24-Q26,Q30,Q31", "SearchPhrase", "ne_empty", None),
    ("URL <> ''", "Q27", "URL", "ne_empty", None),
    ("Referer <> ''", "Q28", "Referer", "ne_empty", None),
    ("MobilePhoneModel <> ''", "Q10,Q11", "MobilePhoneModel", "ne_empty", None),
    ("URL LIKE '%google%'", "Q20-Q23", "URL", "contains", b"google"),
    ("Title LIKE '%Google%'", "Q22", "Title", "contains", b"google"),
]


def _as_bytes(v):
    return v if isinstance(v, bytes) else v.encode()


def prunable(values, kind, arg):
    """Can this granule be skipped, per index kind? Returns (minmax, membership).

    `minmax` is what an order summary proves; `membership` is what an exact
    value set proves, which upper-bounds any bloom or range filter. A LIKE
    predicate is unprunable by min/max by construction, and the membership
    column there is ngram semantics -- a splitting tokenizer yields no tokens
    at all for '%x%'.
    """
    if kind == "eq":
        return (not (min(values) <= arg <= max(values)), arg not in set(values))
    if kind == "ne":
        allsame = min(values) == max(values) == arg
        return (allsame, allsame)
    if kind == "ne_empty":
        allsame = all(len(v) == 0 for v in values)
        return (allsame, allsame)
    if kind == "contains":
        return (False, not any(arg in _as_bytes(v).lower() for v in values))
    raise ValueError(kind)


def granules(meta, num_rows, window):
    if window is None:  # the file's own row groups
        out, start = [], 0
        for i in range(meta.num_row_groups):
            n = meta.row_group(i).num_rows
            out.append((start, start + n))
            start += n
        return out
    return [(i, min(i + window, num_rows)) for i in range(0, num_rows, window)]


def main(path):
    f = pq.ParquetFile(path)
    meta = f.metadata
    cc = meta.row_group(0).column(0)
    print(f"{path}\n  created_by      {meta.created_by}")
    print(f"  rows            {meta.num_rows:,}")
    print(
        f"  row groups      {[meta.row_group(i).num_rows for i in range(meta.num_row_groups)]}"
    )
    print(f"  column_index    {cc.has_column_index}")
    print(f"  offset_index    {cc.has_offset_index}")

    cols = sorted({p[2] for p in PREDICATES})
    table = pq.read_table(path, columns=cols)
    data = {c: table.column(c).combine_chunks() for c in cols}

    layouts = [("real", None), ("sim 65536", 65536), ("sim 8192", 8192)]
    grids = {name: granules(meta, meta.num_rows, w) for name, w in layouts}

    head = "".join(f"{n:>20s}" for n, _ in layouts)
    print(f"\n{'predicate':38s}{'queries':26s}{head}")
    print(
        f"{'':38s}{'':26s}" + "".join(f"{'minmax / membership':>20s}" for _ in layouts)
    )

    for label, queries, col, kind, arg in PREDICATES:
        numeric = kind in ("eq", "ne")
        series = data[col].to_numpy() if numeric else data[col].to_pylist()
        cells = []
        for name, _ in layouts:
            g = grids[name]
            mm = mem = 0
            for lo, hi in g:
                a, b = prunable(series[lo:hi], kind, arg)
                mm += a
                mem += b
            cells.append(f"{mm / len(g) * 100:6.1f}% /{mem / len(g) * 100:6.1f}%")
        print(f"{label:38s}{queries:26s}" + "".join(f"{c:>20s}" for c in cells))

    print(
        "\nThe 'real' column is the only one a reader can act on today.\n"
        "Simulated columns require rewriting the file; see writer.mojo:63,396."
    )


if __name__ == "__main__":
    import os

    main(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH))
