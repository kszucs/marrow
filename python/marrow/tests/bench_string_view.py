# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""ClickBench-shaped queries over Parquet, strings read as ``string`` against
``string_view``.

One file, one engine, one thread; the only difference between the two
contenders is the layout the scan's schema asks the string columns for:

    pixi run -e dev pytest python/marrow/tests/bench_string_view.py \\
        --benchmark --competition

The ``hits`` table imitates the string columns ClickBench's string queries
touch: long, high-cardinality ``URL`` and ``Title``; ``SearchPhrase``, empty on
most rows; ``MobilePhoneModel``, short and almost always empty. Row groups of
128k rows, so the scan streams as it does over the real file.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow as ma
from marrow import col, count_star

ROWS = 2_000_000
ROW_GROUP = 128 * 1024
THREADS = 1

_DOMAINS = [
    "google.com",
    "yandex.ru",
    "example.org",
    "news.site.ru",
    "shop.example.com",
]
_MODELS = ["iPad", "iPhone 12", "Galaxy S9", "Pixel 7", "Redmi Note 8", "Mi 9"]


def _hits():
    """Deterministic, so every run measures the same bytes."""
    phrases = [f"search phrase number {k}" for k in range(20_000)]
    url, title, phrase, model, user, region = [], [], [], [], [], []
    for i in range(ROWS):
        domain = _DOMAINS[0] if i % 50 == 0 else _DOMAINS[1 + i % 4]
        url.append(f"https://www.{domain}/catalog/item/{(i * 7919) % 300_000}?ref=feed")
        title.append(f"Product page {(i * 104729) % 100_000} | Catalog | Example Store")
        phrase.append(phrases[(i * 31) % 20_000] if i % 10 < 3 else "")
        model.append(_MODELS[i % 6] if i % 100 == 0 else "")
        user.append((i * 2654435761) % 100_000)
        region.append(i % 300)
    return pa.table(
        {
            "URL": pa.array(url, pa.string()),
            "Title": pa.array(title, pa.string()),
            "SearchPhrase": pa.array(phrase, pa.string()),
            "MobilePhoneModel": pa.array(model, pa.string()),
            "UserID": pa.array(user, pa.int64()),
            "RegionID": pa.array(region, pa.int32()),
        }
    )


@pytest.fixture(scope="session")
def hits_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("clickbench") / "hits.parquet"
    pq.write_table(_hits(), path, row_group_size=ROW_GROUP)
    return path


def _source(path, view):
    """The file as a lazy table, its string columns declared as ``view``
    asks -- the scan decodes them straight into that layout."""
    fields = []
    for f in ma.read_parquet(path).schema:
        if view and f.type == ma.string():
            fields.append(ma.field(f.name, ma.string_view()))
        else:
            fields.append(f)
    return ma.read_parquet(path, ma.schema(fields))


# ---------------------------------------------------------------------------
# The queries, as plans over a source
# ---------------------------------------------------------------------------


def count_urls(t):
    """Decode only: the cost of producing the column at all."""
    return t.aggregate(n=col("URL").count())


def like_google(t):
    """Q20 -- ``COUNT(*) WHERE URL LIKE '%google%'``."""
    return t.filter(col("URL").like("%google%")).aggregate(n=count_star())


def top_phrases(t):
    """Q13 -- group by a short, mostly-empty key."""
    return (
        t.filter(col("SearchPhrase") != "")
        .aggregate(by=["SearchPhrase"], c=count_star())
        .order_by(("c", "descending"))
        .limit(10)
    )


def models_by_users(t):
    """Q11 -- ``COUNT(DISTINCT UserID)`` per short key."""
    return (
        t.filter(col("MobilePhoneModel") != "")
        .aggregate(by=["MobilePhoneModel"], u=col("UserID").count_distinct())
        .order_by(("u", "descending"))
        .limit(10)
    )


def phrase_min_url(t):
    """Q21 -- a LIKE filter, then ``MIN(URL)`` per phrase."""
    return (
        t.filter(col("URL").like("%google%") & (col("SearchPhrase") != ""))
        .aggregate(by=["SearchPhrase"], u=col("URL").min(), c=count_star())
        .order_by(("c", "descending"))
        .limit(10)
    )


def top_urls(t):
    """Q34 -- group by a long, high-cardinality key."""
    return (
        t.aggregate(by=["URL"], c=count_star()).order_by(("c", "descending")).limit(10)
    )


def sorted_phrases(t):
    """Q25 -- sort a string column, keep the first rows."""
    return (
        t.filter(col("SearchPhrase") != "")
        .select("SearchPhrase")
        .order_by("SearchPhrase")
        .limit(10)
    )


def region_pages(t):
    """Filter on a number, materialise two long string columns -- the shape
    a view gathers without copying the bytes."""
    return t.filter(col("RegionID") == 7).select("URL", "Title")


QUERIES = [
    count_urls,
    like_google,
    top_phrases,
    models_by_users,
    phrase_min_url,
    top_urls,
    sorted_phrases,
    region_pages,
]


def _run(benchmark, hits_file, query, view):
    benchmark.extra_info.update(lib="string_view" if view else "string", n=ROWS)
    # Optimized, as an engine runs it: column pruning narrows the scan to the
    # columns the query reads, so the layout's cost is not buried under
    # decoding the whole file.
    plan = query(_source(hits_file, view)).optimize()
    benchmark.pedantic(
        lambda: plan.collect(num_threads=THREADS), rounds=5, warmup_rounds=1
    )


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.__name__)
def test_string_scan(benchmark, hits_file, query):
    benchmark.group = query.__name__
    _run(benchmark, hits_file, query, view=False)


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.__name__)
def test_string_view_scan(benchmark, hits_file, query):
    benchmark.group = query.__name__
    _run(benchmark, hits_file, query, view=True)


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.__name__)
def test_layouts_agree(hits_file, query):
    """Not a benchmark: both layouts must answer every query identically."""
    plain = query(_source(hits_file, False)).optimize().to_pyarrow(THREADS)
    viewed = query(_source(hits_file, True)).optimize().to_pyarrow(THREADS)
    as_string = pa.record_batch(
        [
            c.cast(pa.string()) if pa.types.is_string_view(c.type) else c
            for c in viewed.columns
        ],
        names=viewed.schema.names,
    )
    assert as_string.equals(plain)
