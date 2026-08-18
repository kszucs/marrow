"""ClickBench correctness — all 43 queries against the DuckDB reference.

Query definitions live in ``clickbench.py`` and are shared with
``bench_clickbench.py``, so a query is written once and both checked and timed.

    pixi run -e bench pytest python/marrow/tests/test_clickbench.py -v

``bench`` is the environment: it is ``["dev", "bench"]``, so it has the marrow
extension *and* duckdb. Under ``dev`` the whole module skips for want of a
reference engine; without the dataset it skips too.

Each query runs in its **own subprocess** (``clickbench.run_isolated``). marrow
still has open memory-safety defects on this data, and a query that aborts the
interpreter has to cost one test rather than the whole run.

The four `PROBES` in ``clickbench.py`` — spellings that are silently wrong or
that abort — are deliberately not tests: they assert *wrongness*, which pytest
should not enshrine. ``python python/marrow/tests/clickbench.py`` reports them.
"""

import pytest

import clickbench as cb

pytestmark = pytest.mark.skipif(
    not cb.HAVE_DATA, reason=f"ClickBench hits parquet not found at {cb.HITS}"
)


@pytest.fixture(scope="session")
def reference():
    """DuckDB answers, computed on demand and memoised for the session."""
    if not cb.HAS_DUCKDB:
        pytest.skip("duckdb is not installed — run under `pixi run -e bench`")
    return cb.Reference()


@pytest.mark.parametrize("name", list(cb.QUERIES))
def test_clickbench_query(reference, name):
    q = cb.QUERIES[name]
    if q.unsupported:
        pytest.skip(f"UNSUPPORTED: {q.unsupported}")
    out = cb.run_isolated(name)
    assert out["status"] == "RAN", f"{out['status']}: {out.get('reason')}"
    cb.verify(q, out, reference)
    assert out["status"] == "PASS", out.get("reason")


@pytest.mark.skipif(not cb.HAS_POLARS, reason="polars is not installed")
@pytest.mark.parametrize("name", list(cb.QUERIES))
def test_polars_thunk_matches_reference(reference, name):
    """The polars spelling of each query computes the same answer.

    ``bench_clickbench.py`` times marrow against these, so they have to be the
    same query — a polars thunk that quietly dropped a predicate would make
    marrow look slow for free.
    """
    import polars as pl

    q = cb.QUERIES[name]
    if q.polars is None:
        pytest.skip("no polars spelling")
    df = q.polars_checked(pl.scan_parquet(cb.HITS)).collect()
    if q.probe is not None:  # Q24 is `SELECT *`; the reference is narrowed
        df = df.select(q.probe)
    got = cb.polars_rows(df)
    want, _ = reference.get(q.duckdb_verify_sql)
    reason = cb.compare(got, want)
    if reason is not None and q.tie_key is not None and got:
        assert cb.compare_tie(got, want, q.tie_key), reason
    elif q.compare == "shape":
        assert len(got) == len(want)
    else:
        assert reason is None, reason
