"""The corpus's input tables, and the files they live in.

Fixtures are **files**, not construction code: all three consumers -- the AOT
lane, the runtime lane, and the DuckDB twin producing the expectations -- read
the same bytes.  Building the table per lane would let them drift on exactly
the thing under test.

Nulls appear in every column that can hold one.  A fixture without them tests
the happy path of kernels whose null handling is the interesting part.

Separate from `devkit.golden` because it is data: 250 lines of table literals
in the middle of the case format and the three lanes made that module read as
though it had more responsibilities than it has.
"""

from datetime import date, datetime
from pathlib import Path

import pyarrow as pa

TABLES = {
    # `k` repeats so grouping has something to do, and carries a null key — the
    # case Arrow and SQL engines disagree about most often. `v` and `w` carry
    # nulls at different rows so a binary op sees each side null independently.
    "basic": pa.table(
        {
            "k": pa.array(["a", "b", "a", "c", "b", "a", None], pa.string()),
            "v": pa.array([1, 2, 3, 4, None, 6, 7], pa.int64()),
            "w": pa.array([10, None, 30, 40, 50, 60, 70], pa.int64()),
        }
    ),
    # Null semantics. `a` is entirely null, `b` has none, so an aggregate over
    # `a` exercises "no valid input" and one over `b` gives an exact mean
    # (20 / 4 = 5.0) — a non-exact one would compare two double roundings
    # rather than two implementations.
    "nulls": pa.table(
        {
            "a": pa.array([None, None, None, None], pa.int64()),
            "b": pa.array([2, 4, 6, 8], pa.int64()),
            "g": pa.array(["x", None, "x", "y"], pa.string()),
        }
    ),
    # Three-valued logic through *derived* predicates (`x > 0`, `y > 0`),
    # covering the full 3x3 Kleene table.
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
        {"s": pa.array(["Hello", "wORLD", "  pad  ", "", "héllo", None], pa.string())}
    ),
    # Actual boolean columns, covering the 3x3 Kleene table directly rather
    # than through derived predicates.
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
    # The type-widening matrix's home: one row set carrying int32, float64,
    # bool and string, so `join`, `aggregate`, `sort` and `filter` can each be
    # asked the same question of every type. Every column has a null.
    #
    # `price` holds only exact binary fractions (1.5, 2.25, 0.5, 4.0, -1.25).
    # A sum of those is exact whatever order the engine adds them in, so a
    # float aggregate compares two implementations rather than two roundings.
    #
    # `ref` covers the join cases against itself: 2 repeats, 99 matches
    # nothing, and one NULL — which must match nothing, not even another NULL.
    "sales": pa.table(
        {
            "region": pa.array(
                ["north", "south", "north", None, "east", "south"], pa.string()
            ),
            "qty": pa.array([10, 20, None, 40, 50, 5], pa.int32()),
            "price": pa.array([1.5, 2.25, 0.5, None, 4.0, -1.25], pa.float64()),
            "active": pa.array([True, False, True, None, True, False], pa.bool_()),
            "ref": pa.array([1, 2, 2, 3, None, 99], pa.int64()),
        }
    ),
    # The string-key join partner. `west` is unmatched from the right, and
    # `east` and the NULL region are unmatched from the left, so an outer join
    # has something to widen in both directions on a *string* key.
    "regions": pa.table(
        {
            "region": pa.array(["north", "south", "west"], pa.string()),
            "country": pa.array(["ca", "mx", None], pa.string()),
        }
    ),
    # Floating-point edge values, for the scalar kernels that only get
    # interesting here: NaN (which is not null, and is not equal to itself),
    # both infinities, and a negative zero that compares equal to +0.0 while
    # having a different sign bit. `y` has no zero, so a division case asks
    # about arithmetic rather than about what marrow and DuckDB each do with
    # division by zero — a separate question, and one they answer differently.
    "floats": pa.table(
        {
            "x": pa.array(
                [
                    1.5,
                    -2.0,
                    0.0,
                    -0.0,
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    None,
                ],
                pa.float64(),
            ),
            "y": pa.array([2.0, 4.0, 8.0, 1.0, 1.0, 2.0, 2.0, None], pa.float64()),
            "n": pa.array([4, -9, 0, 1, 2, 3, -1, None], pa.int64()),
        }
    ),
    # Temporal inputs. Naive (zone-free) timestamps, so nothing here depends on
    # a DST rule or a tz database. Covers a leap day, the last microsecond of a
    # year, a repeated instant so grouping has something to do, and a null.
    "events": pa.table(
        {
            "ts": pa.array(
                [
                    datetime(2021, 1, 1, 0, 0, 0),
                    datetime(2021, 6, 15, 12, 30, 45),
                    datetime(2021, 6, 15, 12, 30, 45),
                    None,
                    datetime(2020, 2, 29, 23, 59, 59),
                    datetime(2021, 12, 31, 23, 59, 59, 999999),
                ],
                pa.timestamp("us"),
            ),
            "d": pa.array(
                [
                    date(2021, 1, 1),
                    date(2021, 6, 15),
                    None,
                    date(2020, 2, 29),
                    date(2021, 12, 31),
                    date(2021, 6, 15),
                ],
                pa.date32(),
            ),
            "label": pa.array(["a", "b", "a", None, "c", "b"], pa.string()),
        }
    ),
    # Cast inputs. `f` holds values where truncation and rounding disagree
    # (1.7, -2.7, 0.5); `s` holds one string that does not parse; `i` holds a
    # value too wide for int32 is deliberately absent — 300 fits, so the
    # int32 case tests conversion rather than overflow.
    "nums": pa.table(
        {
            "i": pa.array([1, -2, 300, None], pa.int64()),
            "f": pa.array([1.7, -2.7, 0.5, None], pa.float64()),
            "s": pa.array(["1", "-2", "abc", None], pa.string()),
            "b": pa.array([True, False, True, None], pa.bool_()),
        }
    ),
    # String-function inputs. `words` answers case, padding, emptiness and
    # multi-byte questions about the *unary* kernels; this one exists for the
    # functions that need structure inside the string:
    #
    #   'a,b,c'       a separator, so split / split_part / position have
    #                 something to find, at three different offsets
    #   'xyz'         no separator at all — the not-found answer, which is 0
    #                 in some engines, NULL in others and an error in a third
    #   ''            the empty string, which is not a null and which every
    #                 pad / substr / repeat function answers differently about
    #   NULL          the null
    #   '  Ab  '      surrounding whitespace *and* mixed case, so trim-with-a
    #                 -character-set and case-insensitive compare separate
    #   'héllo wörld' multi-byte plus a space, so substr / left / lpad have to
    #                 say whether they count bytes or characters
    #
    # `n` drives the count-taking functions from a column rather than a
    # literal, and carries 0 and -1 — the two counts engines disagree about
    # for `repeat`, `lpad` and `substr`.
    "text": pa.table(
        {
            "t": pa.array(
                ["a,b,c", "xyz", "", None, "  Ab  ", "héllo wörld"], pa.string()
            ),
            "n": pa.array([1, 2, 3, 0, -1, None], pa.int64()),
        }
    ),
    # List inputs. Five rows, each asking a different question:
    #
    #   [1, 2, 3]  the ordinary case
    #   []         the empty list — length 0, and *not* a null
    #   NULL       the null list — length NULL, and not length 0
    #   [NULL, 5]  a null *element*, which neither nulls the list nor is
    #              skipped by its length
    #   [7]        one element, so `l[1]` and `l[2]` differ in whether an
    #              out-of-bounds index is NULL or an error
    #
    # `id` is a stable sort key: nothing here can order rows by `l` itself.
    "lists": pa.table(
        {
            "l": pa.array([[1, 2, 3], [], None, [None, 5], [7]], pa.list_(pa.int64())),
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    ),
    # Struct and map inputs, for the field-access and key-lookup families.
    # `st` separates a null *field* from a null *struct* — the distinction an
    # engine loses when it flattens — and `m` covers a present key, an absent
    # key, an empty map and a null map.
    "nested": pa.table(
        {
            "st": pa.array(
                [{"a": 1, "b": "x"}, {"a": None, "b": "y"}, None],
                pa.struct([("a", pa.int64()), ("b", pa.string())]),
            ),
            "m": pa.array(
                [[("k", 1), ("j", 2)], [], None],
                pa.map_(pa.string(), pa.int64()),
            ),
            "id": pa.array([1, 2, 3], pa.int64()),
        }
    ),
    # Integer extremes. The other fixtures hold values a kernel cannot get
    # wrong by a sentinel or a wrap; these are the ones it can.
    #
    #   `i`  int64 max and int64 min — what a `min`/`max` seeded with 0, or an
    #        `abs` that negates in place, gets wrong; plus a null.
    #   `j`  a zero divisor, a negative divisor and a positive one, so
    #        division and modulo have every sign combination against `i`.
    "edges": pa.table(
        {
            "i": pa.array(
                [9223372036854775807, -9223372036854775808, 0, 1, None],
                pa.int64(),
            ),
            "j": pa.array([1, -1, 0, 3, 2], pa.int64()),
        }
    ),
}


class FixtureSet:
    """The corpus's input tables, on disk as Arrow IPC files."""

    def __init__(self, directory):
        self.directory = Path(directory)

    @property
    def names(self):
        return sorted(TABLES)

    def path(self, name):
        return self.directory / f"{name}.arrow"

    def write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, table in TABLES.items():
            with pa.ipc.new_file(self.path(name), table.schema) as writer:
                writer.write_table(table)
        return self.names

    def read(self, name):
        with pa.ipc.open_file(self.path(name)) as reader:
            return reader.read_all()
