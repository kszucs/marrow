"""Multi-table joins: marrow vs PyArrow vs Polars vs DuckDB.

``bench_join.py`` asks how fast one hash join is. This file asks a different
question — **what the engine does with three and four of them** — on the shape
where the written order is the wrong one:

    A   n rows, key `ak` over 25 values
    B   1,000 rows, `bk` over the same 25 values, `bc` unique
    C   1,000 rows, `cc` — 25 of which are B's, and 975 that match nothing
    D   1,000 rows, `dv` covering every `cv` in C

``A ⋈ B`` is many-to-many: 40 B rows per key, so it really produces ``40n``.
``B ⋈ C`` produces 25, and every association answers the same ``n`` rows. So
the query as written materialises an intermediate forty times the size of its
own answer, and an engine that reorders never builds it. DuckDB and Polars
both have planners that may; PyArrow has none and runs what it is given.

Three marrow columns, because marrow's answer depends on which of them you
ask for and conflating them would flatter it:

``marrow``
    the plan as written, ``collect()`` — which applies **no** rules at all.
``marrow_opt``
    ``.optimize()``, the whole of ``AllRules``. On these inputs that is
    `SelectBuildSide` and nothing else: `JoinReassociation` needs to see that
    ``A ⋈ B`` explodes, and an `InMemoryTable` records no distinct count, so
    `ColumnEstimate.max_distinct` falls back to the row count, the containment
    denominator is ``max(|L|, |R|)`` and the 40n-row intermediate is estimated
    at 1,000 rows. The rule declines, correctly, on what it can see.
``marrow_reassoc``
    the right-deep plan written by hand and then optimized — the plan a cost
    model with a distinct count in hand would have chosen, and the ceiling the
    rule is reaching for.

``join2`` is the control: one join, no association to choose, so
``marrow_reassoc`` *is* ``marrow_opt`` there — the same plan through the same
rules — and the spread between those two columns is this run's noise floor,
read off the table rather than remembered. The ``marrow`` column beside them
is the same join with no rules at all, which makes that row the price of
`SelectBuildSide` alone on a single join.

**Thread posture.** Everything is pinned to one thread:
``POLARS_MAX_THREADS=1`` and ``OMP_NUM_THREADS=1`` are set *before* Polars is
imported — it reads its pool size at import, so setting them afterwards leaves
it multi-threaded — ``pa.set_cpu_count(1)``, DuckDB ``config={"threads": 1}``,
and marrow ``collect(num_threads=1)``.

**And the posture is asserted, not assumed.** Setting the variable here only
works when this module is the one that imports Polars, which is true when the
file is run on its own and false in a whole-session run — ``bench_cast.py``
and ``bench_compute.py`` import Polars with no variable set and sort ahead of
this file. ``test_every_library_runs_single_threaded`` is what turns that from
a silently flattering Polars column into a failure.

**DuckDB is drained.** ``con.execute(q).arrow()`` returns a
``RecordBatchReader`` on duckdb 1.5, not a table: it comes back in 2.5 ms
while the join it describes takes 15, so a benchmark that stops there times
the reader and not the query. ``.read_all()`` is what makes the column
comparable.

Run with:
    pixi run -e bench pytest python/marrow/tests/bench_join_multi.py \\
        --benchmark --competition
"""

import os

# Polars and OpenMP read their pool size at import, so pin before importing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import pytest

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

import polars as pl
import pyarrow as pa

import marrow as ma

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

SIZES = [50_000, 200_000]

KEYS = 25
"""Distinct `ak` values, shared by A and B."""
BRIDGE = 1_000
"""Rows in B, C and D. ``BRIDGE // KEYS`` = 40 B rows per key, which is what
makes ``A ⋈ B`` many-to-many."""
LIVE = 25
"""C rows whose `cc` is one of B's `bc`. The other 975 match nothing, so
``B ⋈ C`` keeps 25 rows carrying all 25 distinct `ak` between them — every A
row still finds exactly one partner, whichever way the joins associate."""

_skip_no_duckdb = pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed")

OUTPUT = {
    2: ["ak", "av", "bc"],
    3: ["ak", "av", "bc", "cv"],
    4: ["ak", "av", "bc", "cv", "dc"],
}
"""The columns every engine must answer with.

Spelled out rather than left to ``SELECT *``, because the four disagree about
what a join emits: PyArrow and Polars **drop the right key column** — their
join treats `ak` and `bk` as one logical key — where marrow and DuckDB keep
both. Naming the projection is what makes the *answers* identical.

It does not make the work identical, and the residual is worth stating:
marrow and DuckDB still carry four columns through the 40n-row intermediate
where PyArrow and Polars carry three, so those two move about a third fewer
bytes through the join this file is about. Against the 40x the association is
worth that is second order — but it is real, and it runs in PyArrow's and
Polars' favour."""

_SQL = {
    2: "SELECT a.ak, a.av, b.bc FROM a JOIN b ON a.ak = b.bk",
    3: (
        "SELECT a.ak, a.av, b.bc, c.cv "
        "FROM a JOIN b ON a.ak = b.bk JOIN c ON b.bc = c.cc"
    ),
    4: (
        "SELECT a.ak, a.av, b.bc, c.cv, d.dc "
        "FROM a JOIN b ON a.ak = b.bk JOIN c ON b.bc = c.cc "
        "JOIN d ON c.cv = d.dv"
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _columns(n):
    """The four tables as plain int64 column lists — no randomness, so two
    runs on two machines join the same rows."""
    return {
        "a": {"ak": [i % KEYS for i in range(n)], "av": list(range(n))},
        "b": {"bk": [i % KEYS for i in range(BRIDGE)], "bc": list(range(BRIDGE))},
        "c": {
            "cc": [i if i < LIVE else BRIDGE + i for i in range(BRIDGE)],
            "cv": list(range(BRIDGE)),
        },
        "d": {"dv": list(range(BRIDGE)), "dc": list(range(BRIDGE))},
    }


@pytest.fixture(params=SIZES, ids=[f"n={n}" for n in SIZES], scope="session")
def n(request):
    return request.param


@pytest.fixture(scope="session")
def tables(n):
    raw = _columns(n)
    arrow = {
        name: pa.table({c: pa.array(v, pa.int64()) for c, v in cols.items()})
        for name, cols in raw.items()
    }
    return {
        "pa": arrow,
        "pl": {name: pl.DataFrame(cols) for name, cols in raw.items()},
        "ma": {
            name: ma.memtable(
                ma.record_batch(
                    {c: ma.array(v, type=ma.int64()) for c, v in cols.items()}
                )
            )
            for name, cols in raw.items()
        },
    }


@pytest.fixture(scope="session")
def duck_con(tables):
    if not _HAS_DUCKDB:
        return None
    con = duckdb.connect(config={"threads": 1})
    for name, table in tables["pa"].items():
        con.register(name, table)
    return con


# ---------------------------------------------------------------------------
# The plans, one per library
# ---------------------------------------------------------------------------


def _marrow_written(t, depth):
    """``A ⋈ B ⋈ C ⋈ D``, left-deep, exactly as a reader would write it."""
    plan = t["a"].join(t["b"], left_on="ak", right_on="bk")
    if depth >= 3:
        plan = plan.join(t["c"], left_on="bc", right_on="cc")
    if depth >= 4:
        plan = plan.join(t["d"], left_on="cv", right_on="dv")
    return plan.select(*OUTPUT[depth])


def _marrow_right_deep(t, depth):
    """The same joins, associated the other way — ``A ⋈ (B ⋈ C ⋈ D)``.

    Same output schema and same rows: `Join._output_schema` is left fields
    then right fields, so ``A|B|C`` and ``A|(B|C)`` are the same field list in
    the same order, which is what makes the association a free choice.
    """
    if depth == 2:
        return _marrow_written(t, 2)
    right = t["b"].join(t["c"], left_on="bc", right_on="cc")
    if depth >= 4:
        right = right.join(t["d"], left_on="cv", right_on="dv")
    return t["a"].join(right, left_on="ak", right_on="bk").select(*OUTPUT[depth])


def _pyarrow(t, depth):
    out = t["a"].join(t["b"], keys="ak", right_keys="bk", join_type="inner")
    if depth >= 3:
        out = out.join(t["c"], keys="bc", right_keys="cc", join_type="inner")
    if depth >= 4:
        out = out.join(t["d"], keys="cv", right_keys="dv", join_type="inner")
    return out.select(OUTPUT[depth])


def _polars(t, depth):
    """Through `LazyFrame`, so Polars' own optimizer sees the whole chain."""
    out = t["a"].lazy().join(t["b"].lazy(), left_on="ak", right_on="bk", how="inner")
    if depth >= 3:
        out = out.join(t["c"].lazy(), left_on="bc", right_on="cc", how="inner")
    if depth >= 4:
        out = out.join(t["d"].lazy(), left_on="cv", right_on="dv", how="inner")
    return out.select(OUTPUT[depth]).collect()


def _duckdb(con, depth):
    return con.execute(_SQL[depth]).arrow().read_all()


# ---------------------------------------------------------------------------
# Fairness guard
# ---------------------------------------------------------------------------


def test_every_library_runs_single_threaded(duck_con):
    """One thread each, or the table is not a comparison.

    Polars is the one that cannot be fixed after the fact: it sizes its pool
    at import, so ``POLARS_MAX_THREADS`` set at the top of this module is
    ignored when some other module imported Polars first. This is the check
    that says so out loud.
    """
    assert pl.thread_pool_size() == 1, (
        f"polars has {pl.thread_pool_size()} threads: something imported it "
        "before POLARS_MAX_THREADS was set. Run this file on its own."
    )
    assert pa.cpu_count() == 1
    if _HAS_DUCKDB:
        assert duck_con.execute("SELECT current_setting('threads')").fetchone()[0] == 1


def test_every_library_answers_the_same_table(tables, duck_con, n):
    """Six plans, one row count and one column list — otherwise the table
    compares four engines doing different amounts of work."""
    expected = {2: n * (BRIDGE // KEYS), 3: n, 4: n}
    for depth, rows in expected.items():
        answers = {
            "marrow": _marrow_written(tables["ma"], depth).collect(1),
            "marrow_opt": _marrow_written(tables["ma"], depth).optimize().collect(1),
            "marrow_reassoc": _marrow_right_deep(tables["ma"], depth)
            .optimize()
            .collect(1),
            "pyarrow": _pyarrow(tables["pa"], depth),
            "polars": _polars(tables["pl"], depth),
        }
        if _HAS_DUCKDB:
            answers["duckdb"] = _duckdb(duck_con, depth)
        for lib, answer in answers.items():
            got = answer.height if lib == "polars" else answer.num_rows
            names = list(answer.columns) if lib == "polars" else answer.column_names
            assert got == rows, f"{lib} join{depth}: {got} rows, expected {rows}"
            assert names == OUTPUT[depth], f"{lib} join{depth}: {names}"


# ---------------------------------------------------------------------------
# join2 -- the control: one join, no association to choose
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="join_multi")
def test_marrow_join2(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    plan = _marrow_written(tables["ma"], 2)
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_opt_join2(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_opt", n=n)
    plan = _marrow_written(tables["ma"], 2).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_reassoc_join2(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_reassoc", n=n)
    plan = _marrow_right_deep(tables["ma"], 2).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_pyarrow_join2(benchmark, tables, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(_pyarrow, tables["pa"], 2)


@pytest.mark.benchmark(group="join_multi")
def test_polars_join2(benchmark, tables, n):
    benchmark.extra_info.update(lib="polars", n=n)
    benchmark(_polars, tables["pl"], 2)


@_skip_no_duckdb
@pytest.mark.benchmark(group="join_multi")
def test_duckdb_join2(benchmark, duck_con, n):
    benchmark.extra_info.update(lib="duckdb", n=n)
    benchmark(_duckdb, duck_con, 2)


# ---------------------------------------------------------------------------
# join3 -- three tables, written left-deep, optimal right-deep
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="join_multi")
def test_marrow_join3(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    plan = _marrow_written(tables["ma"], 3)
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_opt_join3(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_opt", n=n)
    plan = _marrow_written(tables["ma"], 3).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_reassoc_join3(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_reassoc", n=n)
    plan = _marrow_right_deep(tables["ma"], 3).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_pyarrow_join3(benchmark, tables, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(_pyarrow, tables["pa"], 3)


@pytest.mark.benchmark(group="join_multi")
def test_polars_join3(benchmark, tables, n):
    benchmark.extra_info.update(lib="polars", n=n)
    benchmark(_polars, tables["pl"], 3)


@_skip_no_duckdb
@pytest.mark.benchmark(group="join_multi")
def test_duckdb_join3(benchmark, duck_con, n):
    benchmark.extra_info.update(lib="duckdb", n=n)
    benchmark(_duckdb, duck_con, 3)


# ---------------------------------------------------------------------------
# join4 -- four tables, two associations to choose between
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="join_multi")
def test_marrow_join4(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow", n=n)
    plan = _marrow_written(tables["ma"], 4)
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_opt_join4(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_opt", n=n)
    plan = _marrow_written(tables["ma"], 4).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_marrow_reassoc_join4(benchmark, tables, n):
    benchmark.extra_info.update(lib="marrow_reassoc", n=n)
    plan = _marrow_right_deep(tables["ma"], 4).optimize()
    benchmark(plan.collect, 1)


@pytest.mark.benchmark(group="join_multi")
def test_pyarrow_join4(benchmark, tables, n):
    benchmark.extra_info.update(lib="pyarrow", n=n)
    benchmark(_pyarrow, tables["pa"], 4)


@pytest.mark.benchmark(group="join_multi")
def test_polars_join4(benchmark, tables, n):
    benchmark.extra_info.update(lib="polars", n=n)
    benchmark(_polars, tables["pl"], 4)


@_skip_no_duckdb
@pytest.mark.benchmark(group="join_multi")
def test_duckdb_join4(benchmark, duck_con, n):
    benchmark.extra_info.update(lib="duckdb", n=n)
    benchmark(_duckdb, duck_con, 4)
