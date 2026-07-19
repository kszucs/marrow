"""Shared ClickBench-inspired query suite — definitions + per-engine runners.

Used by both ``bench_clickbench.py`` (timing) and ``test_clickbench.py``
(correctness). Every engine is driven through its Python API; results are
normalized to a ``{key_tuple: (agg values)}`` map so comparisons are
order-independent and float-tolerant.

The queries are the GROUP BY core of a spread of ClickBench queries
(https://github.com/ClickHouse/ClickBench), restricted to the numeric-key
count/sum/avg/min/max/count_distinct marrow's ``group_by`` exposes today
(ClickBench proper also uses string LIKE and date functions marrow lacks; most
ORDER BY ... LIMIT is dropped so the comparison is the aggregation itself).

The dataset isn't vendored — set ``MARROW_CLICKBENCH_HITS`` to a ``hits`` parquet
partition, or drop one at ``~/Workspace/ClickBench/data/hits_0.parquet``.
"""

import math
import os

import pyarrow as pa
import pyarrow.parquet as pq
import marrow as ma

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from datafusion import SessionContext

    HAS_DATAFUSION = True
except ImportError:
    HAS_DATAFUSION = False


_DEFAULT = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
HITS = os.environ.get("MARROW_CLICKBENCH_HITS", _DEFAULT)
HAVE_DATA = os.path.exists(HITS)

_COLS = [
    "RegionID",
    "UserID",
    "SearchEngineID",
    "AdvEngineID",
    "ResolutionWidth",
    "IsRefresh",
]

# marrow's count(<non-null key>) == COUNT(*). Each query carries: key column(s),
# the (col, func) aggregate list (marrow + pyarrow), a polars lambda, and SQL
# (duckdb + datafusion).
QUERIES = {
    # Q7 shape — very low cardinality (~6 groups), single aggregate.
    "advengine": dict(
        key="AdvEngineID",
        aggs=[("AdvEngineID", "count")],
        pl=lambda df: df.group_by("AdvEngineID").agg(pl.len()),
        sql="SELECT AdvEngineID, count(*) FROM hits GROUP BY AdvEngineID",
    ),
    # low cardinality (~39 groups), multi-aggregate.
    "searchengine": dict(
        key="SearchEngineID",
        aggs=[
            ("SearchEngineID", "count"),
            ("IsRefresh", "sum"),
            ("ResolutionWidth", "mean"),
        ],
        pl=lambda df: df.group_by("SearchEngineID").agg(
            [pl.len(), pl.col("IsRefresh").sum(), pl.col("ResolutionWidth").mean()]
        ),
        sql="SELECT SearchEngineID, count(*), sum(IsRefresh), avg(ResolutionWidth)"
        " FROM hits GROUP BY SearchEngineID",
    ),
    # Q9 shape — mid cardinality (~1242 groups), multi-aggregate.
    "region": dict(
        key="RegionID",
        aggs=[
            ("RegionID", "count"),
            ("AdvEngineID", "sum"),
            ("ResolutionWidth", "mean"),
        ],
        pl=lambda df: df.group_by("RegionID").agg(
            [pl.len(), pl.col("AdvEngineID").sum(), pl.col("ResolutionWidth").mean()]
        ),
        sql="SELECT RegionID, count(*), sum(AdvEngineID), avg(ResolutionWidth)"
        " FROM hits GROUP BY RegionID",
    ),
    # mid cardinality, min/max aggregates.
    "region_minmax": dict(
        key="RegionID",
        aggs=[("ResolutionWidth", "min"), ("ResolutionWidth", "max")],
        pl=lambda df: df.group_by("RegionID").agg(
            [
                pl.col("ResolutionWidth").min().alias("mn"),
                pl.col("ResolutionWidth").max().alias("mx"),
            ]
        ),
        sql="SELECT RegionID, min(ResolutionWidth), max(ResolutionWidth)"
        " FROM hits GROUP BY RegionID",
    ),
    # Q14 shape — two key columns (~3045 groups).
    "multikey": dict(
        key=["RegionID", "SearchEngineID"],
        aggs=[("RegionID", "count")],
        pl=lambda df: df.group_by(["RegionID", "SearchEngineID"]).agg(pl.len()),
        sql="SELECT RegionID, SearchEngineID, count(*)"
        " FROM hits GROUP BY RegionID, SearchEngineID",
    ),
    # Q15 shape — high cardinality (~80k groups), single aggregate.
    "user": dict(
        key="UserID",
        aggs=[("UserID", "count")],
        pl=lambda df: df.group_by("UserID").agg(pl.len()),
        sql="SELECT UserID, count(*) FROM hits GROUP BY UserID",
    ),
    # Q13 shape — GROUP BY with COUNT(DISTINCT): distinct UserIDs per region
    # (mid-cardinality key, high distinct count per group).
    "region_distinct": dict(
        key="RegionID",
        aggs=[("UserID", "count_distinct")],
        pl=lambda df: df.group_by("RegionID").agg(pl.col("UserID").n_unique()),
        sql="SELECT RegionID, count(distinct UserID) FROM hits"
        " GROUP BY RegionID",
    ),
    # Q16 shape — GROUP BY + ORDER BY count DESC, key ASC LIMIT 10. The key
    # tie-break makes the top-N deterministic (so it's comparable across
    # engines). ``top`` = (aggregate output column, n).
    "user_top": dict(
        key="UserID",
        aggs=[("UserID", "count")],
        top=("UserID_count", 10),
        pl=lambda df: df.group_by("UserID")
        .agg(pl.len().alias("UserID_count"))
        .sort(["UserID_count", "UserID"], descending=[True, False])
        .head(10),
        sql="SELECT UserID, count(*) AS UserID_count FROM hits GROUP BY UserID"
        " ORDER BY UserID_count DESC, UserID ASC LIMIT 10",
    ),
    # ----- whole-table aggregates (no GROUP BY), key=[] -----
    # Q1 — COUNT(*).
    "count_star": dict(
        key=[],
        aggs=[("AdvEngineID", "count")],
        pl=lambda df: df.select(pl.len()),
        sql="SELECT count(*) FROM hits",
    ),
    # Q5 — COUNT(DISTINCT UserID).
    "distinct_users": dict(
        key=[],
        aggs=[("UserID", "count_distinct")],
        pl=lambda df: df.select(pl.col("UserID").n_unique()),
        sql="SELECT count(distinct UserID) FROM hits",
    ),
    # Q3 — SUM, COUNT(*), AVG.
    "totals": dict(
        key=[],
        aggs=[
            ("AdvEngineID", "sum"),
            ("AdvEngineID", "count"),
            ("ResolutionWidth", "mean"),
        ],
        pl=lambda df: df.select(
            pl.col("AdvEngineID").sum(),
            pl.len(),
            pl.col("ResolutionWidth").mean(),
        ),
        sql="SELECT sum(AdvEngineID), count(*), avg(ResolutionWidth) FROM hits",
    ),
}

ENGINES = ["marrow", "pyarrow", "polars", "duckdb", "datafusion"]
_AVAILABLE = {
    "marrow": True,
    "pyarrow": True,
    "polars": HAS_POLARS,
    "duckdb": HAS_DUCKDB,
    "datafusion": HAS_DATAFUSION,
}


def available(engine):
    return _AVAILABLE[engine]


def keys(q):
    k = q["key"]
    return [k] if isinstance(k, str) else list(k)


def load():
    """Load the hits slice once and build a handle per available engine."""
    tbl = pq.read_table(HITS, columns=_COLS).combine_chunks()
    h = {"tbl": tbl, "rb": ma.record_batch(tbl.to_batches()[0])}
    if HAS_POLARS:
        h["df"] = pl.from_arrow(tbl)
    if HAS_DUCKDB:
        con = duckdb.connect()
        con.register("hits", tbl)
        h["con"] = con
    if HAS_DATAFUSION:
        ctx = SessionContext()
        ctx.register_record_batches("hits", [tbl.to_batches()])
        h["ctx"] = ctx
    return h


def _apply_top(res, q):
    """Apply ORDER BY <agg> DESC, <keys> ASC LIMIT n if the query has ``top``.

    marrow and pyarrow share the ``sort_by([(name, dir), ...]).slice(0, n)`` API.
    The key tie-break makes the top-N deterministic across engines.
    """
    if "top" not in q:
        return res
    by, n = q["top"]
    order = [(by, "descending")] + [(k, "ascending") for k in keys(q)]
    return res.sort_by(order).slice(0, n)


def run_native(engine, h, q):
    """Run a query and materialize the result in the engine's native form
    (this is what the benchmark times)."""
    grouped = len(keys(q)) > 0
    if engine == "marrow":
        if grouped:
            res = h["rb"].group_by(q["key"]).aggregate(q["aggs"])
        else:
            res = h["rb"].aggregate(q["aggs"])
        return _apply_top(res, q)
    if engine == "pyarrow":
        res = h["tbl"].group_by(keys(q)).aggregate(q["aggs"])
        return _apply_top(res, q)
    if engine == "polars":
        return q["pl"](h["df"])
    if engine == "duckdb":
        return h["con"].execute(q["sql"]).fetch_arrow_table()
    if engine == "datafusion":
        return h["ctx"].sql(q["sql"]).to_arrow_table()
    raise ValueError(f"unknown engine {engine!r}")


def to_arrow(engine, result):
    """Convert a native result to a pyarrow Table for comparison."""
    if engine == "marrow":
        return pa.table(result)  # via __arrow_c_array__
    if engine == "polars":
        return result.to_arrow()
    return result  # pyarrow / duckdb / datafusion already return Arrow


def result_map(table, q):
    """``{key_tuple: (agg values in query order)}`` for a result Arrow table.

    Key columns are matched by name (identical across engines); the remaining
    columns are the aggregates, which every engine emits in the query's ``aggs``
    order — so the comparison is robust to differing column *names* and to
    aggregates-first vs keys-first layouts.
    """
    key_names = keys(q)
    key_set = set(key_names)
    agg_names = [n for n in table.column_names if n not in key_set]
    key_data = [table[k].to_pylist() for k in key_names]
    agg_data = [table[a].to_pylist() for a in agg_names]
    out = {}
    for i in range(table.num_rows):
        kt = tuple(key_data[c][i] for c in range(len(key_names)))
        out[kt] = tuple(agg_data[c][i] for c in range(len(agg_names)))
    return out


def assert_maps_equal(a, b, *, rel_tol=1e-9, abs_tol=1e-6):
    """Compare two ``result_map`` outputs: same group keys, and per-group
    aggregate values equal (ints exact, floats within tolerance)."""
    if a.keys() != b.keys():
        only_a = list(set(a) - set(b))[:3]
        only_b = list(set(b) - set(a))[:3]
        raise AssertionError(
            f"group key sets differ: {len(a)} vs {len(b)} groups; "
            f"only-a sample {only_a}, only-b sample {only_b}"
        )
    for k in a:
        av, bv = a[k], b[k]
        assert len(av) == len(bv), f"group {k}: arity {len(av)} vs {len(bv)}"
        for x, y in zip(av, bv):
            if isinstance(x, float) or isinstance(y, float):
                if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                    raise AssertionError(f"group {k}: {av} vs {bv}")
            elif x != y:
                raise AssertionError(f"group {k}: {av} vs {bv}")
