"""The vocabulary a golden case body may use, on the Python side.

`helpers.mojo` is the same list for the Mojo lane, and `runner.MOJO_HEADER`
imports it. Between them they are the convergence contract: a name a case can
write is a name both lanes answer to.

`SHIMS` is the convergence metric — one entry per spelling the two lanes still
disagree about, and the goal is an empty set. It was forty-four names: the
fused lane's node vocabulary (`Upper(x)` for `x.upper()`,
`NumericCast[Float64Type](x)` for `x.cast(float64)`), the join-kind constants,
and a `_Relation` adapter carrying four verb shapes. All of those went the way
the file's own advice said they would — *check whether the nicer spelling
already exists before designing one*: most of the string verbs were already
methods on `StringValue`, and the rest converged when the comptime lane grew
`is_null`, `cast`, `coalesce`, `fill_null` and `nullif`.

What is left is one genuine language difference, described at `SHIMS`.
"""

import pyarrow as pa

import marrow
from marrow import col, count_star, if_else, lit

import runner

# Names that are *not* real marrow API — the outstanding convergence debt.
# Keep this in sync with what is defined below; `test_cases.py` reports it.
SHIMS = {
    # dtype spellings, and the only disagreement left. Mojo has `int64` as a
    # dtype *value*; Python has `marrow.int64()`, a constructor, because that
    # is PyArrow's shape and a parameterised type (`timestamp("us")`,
    # `list_(int64)`) has to be a call in any case. A case therefore writes
    # `int64` and the two lanes bind it differently.
    "int64",
    "int32",
    "float64",
    "string",
    "bool_",
    # `timestamp(microsecond)` vs `marrow.timestamp("us")`: the unit half of
    # the same difference. Mojo names it with a `TimeUnit` constant, Python
    # with the string the Arrow spec uses.
    "microsecond",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
# Mojo has `int64` as a dtype *value*; Python has `marrow.int64()`. The
# `Int64Type` spellings are gone with the cast nodes that needed them: a cast
# is `x.cast(float64)` in both lanes now, and takes the dtype value.

int64 = marrow.int64()
int32 = marrow.int32()
float64 = marrow.float64()
string = marrow.string()
bool_ = marrow.bool_()


# The temporal types are the other way round: `date32` and `timestamp` are
# *constructors* in both lanes (`date32()`, `timestamp(microsecond)`), because
# Mojo has no comptime singleton for a type carrying a runtime unit. So they
# pass straight through, and only the unit needs a spelling.
date32 = marrow.date32
list_ = marrow.list_
microsecond = "us"


timestamp = marrow.timestamp


# ---------------------------------------------------------------------------
# The relation surface
# ---------------------------------------------------------------------------
#
# There is no adapter here any more. `LazyTable` accepts `DynRelation`'s own
# argument shapes -- `sort_by(keys, ascending)`, `aggregate(aggs, keys)`,
# `join(other, left_keys, right_keys, kind)` and `rename(names, new_names)` --
# alongside its friendlier ones, so a case body is one text in both lanes
# rather than one text and a translation.


def table(name):
    """A fixture as an in-memory source — never a file scan.

    What is under test is the engine, so the source is a memtable in every
    lane; Parquet and IPC keep their own suites.
    """
    batch = marrow.read_ipc_file(str(runner.fixture_path(name)))[0]
    return marrow.memtable(batch)


def check(name, plan):
    """Run the plan and hold it to the shared expectation."""
    expected = _expectations()[name]
    actual = pa.table(plan.to_pyarrow(num_threads=runner.NUM_THREADS))
    if actual.equals(expected):
        return
    raise AssertionError(
        f"{name} does not match its expectation\n\n"
        f"--- expected (duckdb) ---\n{expected}\n"
        f"--- actual (marrow) ---\n{actual}\n"
    )


_EXPECTED = None


def _expectations():
    global _EXPECTED
    if _EXPECTED is None:
        _EXPECTED = {case.name: case.expected for case in runner.load_cases()}
    return _EXPECTED


# The namespace a case body executes in. Built explicitly rather than from
# `globals()` so that adding a private helper here does not silently widen the
# vocabulary a case may use.
NAMESPACE = {
    "table": table,
    "col": col,
    "lit": lit,
    "count_star": count_star,
    "if_else": if_else,
    "row_number": marrow.row_number,
    "rank": marrow.rank,
    "dense_rank": marrow.dense_rank,
    "array_length": marrow.array_length,
    "int64": int64,
    "int32": int32,
    "float64": float64,
    "string": string,
    "bool_": bool_,
    "date32": date32,
    "list_": list_,
    "timestamp": timestamp,
    "microsecond": microsecond,
    "JOIN_INNER": marrow.JOIN_INNER,
    "JOIN_LEFT": marrow.JOIN_LEFT,
    "JOIN_RIGHT": marrow.JOIN_RIGHT,
    "JOIN_FULL": marrow.JOIN_FULL,
    "JOIN_SEMI": marrow.JOIN_SEMI,
    "JOIN_ANTI": marrow.JOIN_ANTI,
    "JOIN_ALL": marrow.JOIN_ALL,
}
