"""Group-by aggregation benchmarks: marrow vs pyarrow, polars, duckdb.

Every library is driven through its Python API so the comparison is
apples-to-apples — each pays its own dispatch + result-materialization cost.
Cases span row count and cardinality (few large groups → many small groups),
since the winning strategy depends on both.

Run periodically with::

    pixi run -e bench pytest python/marrow/tests/bench_groupby.py \\
        --benchmark --competition

marrow's group-by is exposed via ``RecordBatch.group_by(keys).aggregate(...)``
(PyArrow-compatible).
"""

import numpy as np
import pytest
import pyarrow as pa
import polars as pl
import marrow as ma

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False


# (rows, num_groups): low / mid / high cardinality across sizes.
CASES = [
    (10_000, 10),
    (100_000, 10),
    (1_000_000, 10),
    (1_000_000, 1_000),
    (1_000_000, 100_000),
]


def _label(n, g):
    size = f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}m"
    grp = f"{g // 1000}k" if g >= 1000 else str(g)
    return f"{size}_g{grp}"


_IDS = [_label(n, g) for n, g in CASES]


def _data(n, g):
    keys = (np.arange(n) % g).astype(np.int32)
    vals = np.arange(n).astype(np.float64)
    return keys, vals


@pytest.fixture(params=CASES, ids=_IDS, scope="session")
def case(request):
    return request.param


@pytest.fixture(scope="session")
def ma_batch(case):
    keys, vals = _data(*case)
    # Ingest through the Arrow C protocol (zero-copy), then benchmark the group-by.
    return ma.record_batch(pa.record_batch({"k": keys, "v": vals}))


@pytest.fixture(scope="session")
def pa_table(case):
    keys, vals = _data(*case)
    return pa.table({"k": keys, "v": vals})


@pytest.fixture(scope="session")
def pl_df(case):
    keys, vals = _data(*case)
    return pl.DataFrame({"k": keys, "v": vals})


@pytest.fixture(scope="session")
def duck_con(case):
    if not _HAS_DUCKDB:
        pytest.skip("duckdb not installed")
    keys, vals = _data(*case)
    con = duckdb.connect()
    con.register("t", pa.table({"k": keys, "v": vals}))
    return con


def _info(benchmark, lib, case):
    benchmark.extra_info.update(lib=lib, n=case[0], groups=case[1])


# ── group-by sum ────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="groupby_sum")
def test_marrow_groupby_sum(benchmark, ma_batch, case):
    _info(benchmark, "marrow", case)
    benchmark(lambda: ma_batch.group_by("k").aggregate([("v", "sum")]))


@pytest.mark.benchmark(group="groupby_sum")
def test_pyarrow_groupby_sum(benchmark, pa_table, case):
    _info(benchmark, "pyarrow", case)
    benchmark(lambda: pa_table.group_by("k").aggregate([("v", "sum")]))


@pytest.mark.benchmark(group="groupby_sum")
def test_polars_groupby_sum(benchmark, pl_df, case):
    _info(benchmark, "polars", case)
    benchmark(lambda: pl_df.group_by("k").agg(pl.col("v").sum()))


@pytest.mark.benchmark(group="groupby_sum")
def test_duckdb_groupby_sum(benchmark, duck_con, case):
    _info(benchmark, "duckdb", case)
    benchmark(lambda: duck_con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall())


# ── group-by mean ───────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="groupby_mean")
def test_marrow_groupby_mean(benchmark, ma_batch, case):
    _info(benchmark, "marrow", case)
    benchmark(lambda: ma_batch.group_by("k").aggregate([("v", "mean")]))


@pytest.mark.benchmark(group="groupby_mean")
def test_pyarrow_groupby_mean(benchmark, pa_table, case):
    _info(benchmark, "pyarrow", case)
    benchmark(lambda: pa_table.group_by("k").aggregate([("v", "mean")]))


@pytest.mark.benchmark(group="groupby_mean")
def test_polars_groupby_mean(benchmark, pl_df, case):
    _info(benchmark, "polars", case)
    benchmark(lambda: pl_df.group_by("k").agg(pl.col("v").mean()))


@pytest.mark.benchmark(group="groupby_mean")
def test_duckdb_groupby_mean(benchmark, duck_con, case):
    _info(benchmark, "duckdb", case)
    benchmark(lambda: duck_con.execute("SELECT k, avg(v) FROM t GROUP BY k").fetchall())


# ── group-by, five aggregates ───────────────────────────────────────────────
#
# The shape that AOT aggregate fusion targets. A single-aggregate benchmark
# cannot show fusion's win at all: fusion amortises the group lookup and the
# scan of the value column across *every* aggregate, so its entire benefit
# lives in the marginal cost of the 2nd..Nth aggregate.
#
# Read this against the 1-aggregate groups above, on identical data — the
# interesting quantity is the marginal cost per added aggregate,
#
#     (groupby_multi - groupby_sum) / 4
#
# which an engine that revisits the column once per aggregate pays in full,
# and a fused single-pass engine should pay a small fraction of.

_MULTI_SQL = "SELECT k, sum(v), min(v), max(v), count(v), avg(v) FROM t GROUP BY k"

_MULTI = [("v", "sum"), ("v", "min"), ("v", "max"), ("v", "count"), ("v", "mean")]


@pytest.mark.benchmark(group="groupby_multi")
def test_marrow_groupby_multi(benchmark, ma_batch, case):
    _info(benchmark, "marrow", case)
    benchmark(lambda: ma_batch.group_by("k").aggregate(_MULTI))


@pytest.mark.benchmark(group="groupby_multi")
def test_pyarrow_groupby_multi(benchmark, pa_table, case):
    _info(benchmark, "pyarrow", case)
    benchmark(lambda: pa_table.group_by("k").aggregate(_MULTI))


@pytest.mark.benchmark(group="groupby_multi")
def test_polars_groupby_multi(benchmark, pl_df, case):
    _info(benchmark, "polars", case)
    benchmark(
        lambda: pl_df.group_by("k").agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").min().alias("mn"),
            pl.col("v").max().alias("mx"),
            pl.col("v").count().alias("c"),
            pl.col("v").mean().alias("a"),
        )
    )


@pytest.mark.benchmark(group="groupby_multi")
def test_duckdb_groupby_multi(benchmark, duck_con, case):
    _info(benchmark, "duckdb", case)
    benchmark(lambda: duck_con.execute(_MULTI_SQL).fetchall())
