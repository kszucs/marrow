"""The datasets every golden case runs against.

Fixtures are **files**, not construction code: `regenerate.py` writes each one
to `golden/fixtures/<name>.arrow`, and all three consumers — the runtime lane,
the AOT lane and the DuckDB twin that produces the expectations — read the same
bytes. Building the table in each lane instead would let the lanes drift apart
on exactly the thing under test.

Nulls are present in every column that can hold one. A fixture without them
tests the happy path of kernels whose null handling is the interesting part.
"""

from pathlib import Path

import pyarrow as pa

DIR = Path(__file__).parent / "fixtures"


# `k` repeats so grouping has something to do, and carries a null key — the
# case Arrow and SQL engines disagree about most often. `v` and `w` carry nulls
# at different rows so a binary op sees each side null independently.
TABLES = {
    "basic": pa.table(
        {
            "k": pa.array(["a", "b", "a", "c", "b", "a", None], pa.string()),
            "v": pa.array([1, 2, 3, 4, None, 6, 7], pa.int64()),
            "w": pa.array([10, None, 30, 40, 50, 60, 70], pa.int64()),
        }
    ),
    # Null-semantics fixture. `a` is entirely null, `b` has none, so an
    # aggregate over `a` exercises "no valid input" and one over `b` gives an
    # exact mean (20 / 4 = 5.0) — a non-exact one would compare two different
    # double roundings rather than two implementations.
    "nulls": pa.table(
        {
            "a": pa.array([None, None, None, None], pa.int64()),
            "b": pa.array([2, 4, 6, 8], pa.int64()),
            "g": pa.array(["x", None, "x", "y"], pa.string()),
        }
    ),
}


def path(name):
    return DIR / f"{name}.arrow"


def write_all():
    """Materialise every fixture as an Arrow IPC file."""
    DIR.mkdir(parents=True, exist_ok=True)
    for name, table in TABLES.items():
        with pa.ipc.new_file(path(name), table.schema) as writer:
            writer.write_table(table)
    return sorted(TABLES)


def read(name):
    with pa.ipc.open_file(path(name)) as reader:
        return reader.read_all()
