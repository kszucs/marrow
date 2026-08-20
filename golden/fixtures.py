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
    # Three-valued logic. The AOT lane has no boolean column leaf — `col` has
    # no `BoolType` overload — so a fused expression cannot read a `bool`
    # column at all. These are int columns whose *derived* predicates
    # (`x > 0`, `y > 0`) cover the full 3x3 Kleene table, which keeps the
    # cases expressible in both lanes.
    #
    #   x > 0 : T T T F F F N N N
    #   y > 0 : T F N T F N T F N
    "kleene": pa.table(
        {
            "x": pa.array([1, 1, 1, -1, -1, -1, None, None, None], pa.int64()),
            "y": pa.array([1, -1, None, 1, -1, None, 1, -1, None], pa.int64()),
        }
    ),
    # Join fixtures. `emp.dept` covers every interesting case against
    # `dept.did`: a unique match (10), a duplicated match (20, twice), a key
    # with no match (99), and a NULL key — which must match nothing, not even
    # another NULL. `dept.did` 30 is unmatched from the right, so outer joins
    # have something to widen in both directions.
    "emp": pa.table(
        {
            "eid": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "dept": pa.array([10, 20, 20, 99, None], pa.int64()),
        }
    ),
    "dept": pa.table(
        {
            "did": pa.array([10, 20, 30], pa.int64()),
            "dname": pa.array(["eng", "sales", "ops"], pa.string()),
        }
    ),
    # Strings worth asking questions about: mixed case, surrounding
    # whitespace, the empty string (which is not a null), a multi-byte
    # character so `length` has to say whether it counts bytes or codepoints,
    # and a null.
    "words": pa.table(
        {
            "s": pa.array(
                ["Hello", "wORLD", "  pad  ", "", "héllo", None], pa.string()
            )
        }
    ),
    # Actual boolean columns, covering the 3x3 Kleene table directly rather
    # than through derived predicates. Unreadable by the AOT lane until
    # `BoolColumn` existed.
    "flags": pa.table(
        {
            "p": pa.array(
                [True, True, True, False, False, False, None, None, None],
                pa.bool_(),
            ),
            "q": pa.array(
                [True, False, None, True, False, None, True, False, None],
                pa.bool_(),
            ),
        }
    ),
    # Cast material: a negative and an out-of-int32-range integer, fractions
    # that round and truncate differently, a string that will not parse, and
    # a null in every column so each conversion has to carry validity through.
    "nums": pa.table(
        {
            "i": pa.array([1, -2, 300, None], pa.int64()),
            "f": pa.array([1.7, -2.7, 0.5, None], pa.float64()),
            "s": pa.array(["1", "-2", "abc", None], pa.string()),
            "b": pa.array([True, False, True, None], pa.bool_()),
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
