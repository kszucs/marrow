"""ClickBench for marrow — one set of query definitions, three consumers.

All 43 canonical ClickBench queries
(https://github.com/ClickHouse/ClickBench, ``clickhouse/queries.sql``) live
here exactly once. Each carries:

* the **canonical SQL** in its docstring — which is also, after the rewrites
  below, the DuckDB text, so there is no second copy to drift;
* a **marrow** thunk against the lazy Python frontend (``LazyTable``);
* a **polars** thunk against ``pl.LazyFrame``;
* its **status** — ``UNSUPPORTED`` / ``DEVIATED`` with the reason, or nothing,
  which means it is expected to pass.

Three consumers drive that one registry:

* ``test_clickbench.py``  — correctness against the DuckDB reference.
* ``bench_clickbench.py`` — timing, marrow vs polars vs duckdb.
* ``python clickbench.py`` — the standalone PASS/FAIL/ABORT report.

Environment
-----------
``bench`` runs everything: ``bench = ["dev", "bench"]`` in ``pixi.toml``, so it
is a superset of ``dev`` and has the marrow extension, pyarrow, polars *and*
duckdb in one interpreter. (The alpha split this across two files because it
believed duckdb and marrow could not be imported together; they can.)

    pixi run -e bench pytest python/marrow/tests/test_clickbench.py
    pixi run -e bench python python/marrow/tests/bench_clickbench.py
    pixi run -e bench python python/marrow/tests/clickbench.py

Under ``dev`` (no polars, no duckdb) every cross-engine check skips cleanly and
the marrow queries still run.

Dataset
-------
``~/Workspace/ClickBench/data/hits_0.parquet`` — one partition, 1,000,000 rows,
105 columns. Not vendored; override with ``MARROW_CLICKBENCH_HITS``. Everything
skips when it is absent.

**The column types are not what the SQL implies**, and three facts drive most
of the code below:

* ``EventDate`` is **uint16** (days since the epoch), not a date. The literal
  date predicates are therefore integer predicates: ``'2013-07-01'`` is 15887,
  ``'2013-07-31'`` is 15917, ``'2013-07-14'`` is 15900, ``'2013-07-15'`` is
  15901.
* ``EventTime`` is **int64** (unix seconds), not a timestamp. Q19 and Q43 cast
  it to ``timestamp('s')`` before asking for a minute.
* All 28 string-looking columns (``URL``, ``SearchPhrase``, ``Title``,
  ``Referer``, ``MobilePhoneModel``, …) are **binary**. marrow's string kernels
  are bound on ``StringLikeType`` and reject ``binary``, so those columns are
  ``.cast(ma.string())`` at the point of use — the ``s()`` helper. polars and
  DuckDB have the same problem (``pl.Binary`` has no ``.str``; BLOB binds to
  neither ``like`` nor ``length``), so all three sides cast and all three are
  comparing the same thing.

Two reference traps
-------------------
Getting the *reference* right was harder than getting marrow right, and both of
these initially read as marrow failures when marrow was correct. Preserve them:

1. **``CAST(blob AS VARCHAR)`` escapes non-printable bytes as ``\\xNN``.** So
   ``length()`` counts four characters per byte and ``ORDER BY`` sorts escaped
   text. The reference connection instead reads the file with PyArrow and
   ``view()``s the binary columns to ``string`` — a zero-copy reinterpret with
   no validation — so DuckDB sees real VARCHAR holding the original bytes.
   Results come back through Arrow and are ``view()``-ed back to ``binary``, so
   the round trip is byte-exact. ``fetchall()`` would decode VARCHAR to ``str``
   and choke on data that is not all UTF-8.
2. **``length`` means bytes.** DuckDB's ``length(VARCHAR)`` counts *characters*;
   ClickHouse's ``length()`` — which is what ClickBench means — and marrow's
   ``LengthKernel`` both count *bytes*. ``strlen`` is DuckDB's byte-length
   function. Without the rewrite Q28 reads as a marrow mismatch (76.44 vs
   73.97) when marrow is right.

Correcting those two moved the measured score from 25/43 to 38/43.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYROOT = os.path.dirname(os.path.dirname(_HERE))
if _PYROOT not in sys.path:
    sys.path.insert(0, _PYROOT)

import marrow as ma  # noqa: E402
from marrow import col, count_star, if_else, lit  # noqa: E402

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:  # pragma: no cover - environment dependent
    pl = None
    HAS_POLARS = False

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover - environment dependent
    duckdb = None
    HAS_DUCKDB = False


_DEFAULT = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
HITS = os.environ.get("MARROW_CLICKBENCH_HITS", _DEFAULT)
HAVE_DATA = os.path.exists(HITS)

STRING = ma.string()
TIMESTAMP_S = ma.timestamp("s", None)

# `'2013-07-01'` … as uint16 epoch days, because `EventDate` is uint16.
JUL_01 = 15887
JUL_14 = 15900
JUL_15 = 15901
JUL_31 = 15917


def s(name):
    """A binary column read as a string — ``.cast(ma.string())``.

    Needed on 28 of the 105 columns; see the module docstring.
    """
    return col(name).cast(STRING)


def ps(name):
    """The polars counterpart of :func:`s` — ``pl.Binary`` has no ``.str``."""
    return pl.col(name).cast(pl.String)


# ── the registry ───────────────────────────────────────────────────────────

QUERIES = {}


class Query:
    """One ClickBench query and everything the three consumers need of it."""

    def __init__(self, fn, **kw):
        self.name = fn.__name__
        self.fn = fn
        self.sql = " ".join(fn.__doc__.split()).rstrip(";")
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<Query {self.name} {self.status}>"

    @property
    def status(self):
        """The *declared* status. The measured one comes from a run."""
        if self.unsupported:
            return "UNSUPPORTED"
        if self.deviation:
            return "DEVIATED"
        return "PASS"

    @property
    def duckdb_sql(self):
        """The docstring SQL, rewritten for this dataset's column types."""
        return self.reference_sql or _rewrite(self.sql)

    @property
    def duckdb_timing_sql(self):
        """The SQL to *time*: always the real query, never a narrowed probe.

        Q24 compares on three columns but must be timed as the ``SELECT *`` it
        is, or the comparison measures a different query than marrow runs.
        """
        return _rewrite(self.sql)

    @property
    def duckdb_verify_sql(self):
        """The reference text for :attr:`marrow_verify` — see ``tie_key``."""
        return self.verify_sql or self.duckdb_sql

    def marrow(self, table):
        return self.fn(table)

    @property
    def marrow_verify(self):
        """The variant used for *checking*, when it differs from the query.

        Q25 and Q27 sort on a column they do not project, so their answers are
        ambiguous wherever the sort key ties across the LIMIT boundary. The
        verify variant also projects the sort key, which is what lets the tie
        argument be made at all.
        """
        return self.verify or self.fn

    @property
    def polars_checked(self):
        """The polars variant used for *checking* — see :attr:`marrow_verify`."""
        return self.polars_verify or self.polars


def query(
    feature,
    *,
    polars=None,
    polars_verify=None,
    deviation=None,
    unsupported=None,
    compare="rows",
    probe=None,
    tie_key=None,
    verify=None,
    verify_sql=None,
    reference_sql=None,
    metadata_shortcut=False,
):
    """Register a query function with the metadata the three consumers need.

    ``compare`` is ``"rows"`` (full row-set diff against DuckDB) or ``"shape"``
    (row/column counts only, for the one query whose SQL has no ORDER BY and is
    therefore genuinely nondeterministic).

    ``probe`` narrows the marrow result to named columns before comparing, for
    the one ``SELECT *`` query.

    ``tie_key`` is the 0-based index of the ``ORDER BY`` column in the output.
    ``ORDER BY k DESC LIMIT n`` has no unique answer when the value at the
    boundary is tied, and most of these queries order by a count with hundreds
    of tied groups. So on a row-set mismatch the harness falls back to comparing
    the *multiset of ``tie_key`` values*: if those are identical then both
    engines returned a valid top-N and only the tie-break differs. That is a
    sound argument, not a fudge — same row count, same ordering values, both
    sorted by that column.

    ``metadata_shortcut`` marks a query an engine can answer from Parquet
    metadata without scanning (``pl.scan_parquet(...).select(pl.len())`` returns
    ``COUNT(*)`` in ~0.4 ms). Those rows are excluded from the benchmark's
    headline totals — timing a metadata read against a full scan is not a
    comparison.
    """

    def decorate(fn):
        q = Query(
            fn,
            feature=feature,
            polars=polars,
            polars_verify=polars_verify,
            deviation=deviation,
            unsupported=unsupported,
            compare=compare,
            probe=probe,
            tie_key=tie_key,
            verify=verify,
            verify_sql=verify_sql,
            reference_sql=reference_sql,
            metadata_shortcut=metadata_shortcut,
        )
        QUERIES[fn.__name__] = q
        return q

    return decorate


# ── Q1–Q10: whole-table and low-cardinality aggregation ────────────────────


@query(
    "COUNT(*)",
    polars=lambda t: t.select(pl.len()),
    metadata_shortcut=True,
)
def q01(t):
    """SELECT COUNT(*) FROM hits;"""
    return t.aggregate(by=[], c=count_star())


@query(
    "COUNT(*) + WHERE <>",
    polars=lambda t: t.filter(pl.col("AdvEngineID") != 0).select(pl.len()),
)
def q02(t):
    """SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;"""
    return t.filter(col("AdvEngineID") != lit(0)).aggregate(by=[], c=count_star())


@query(
    "SUM + COUNT(*) + AVG, no GROUP BY",
    polars=lambda t: t.select(
        pl.col("AdvEngineID").sum().alias("s"),
        pl.len().alias("c"),
        pl.col("ResolutionWidth").mean().alias("a"),
    ),
)
def q03(t):
    """SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;"""
    return t.aggregate(
        by=[],
        s=("sum", "AdvEngineID"),
        c=count_star(),
        a=("mean", "ResolutionWidth"),
    )


@query("AVG over int64", polars=lambda t: t.select(pl.col("UserID").mean()))
def q04(t):
    """SELECT AVG(UserID) FROM hits;"""
    return t.aggregate(by=[], a=("mean", "UserID"))


@query(
    "COUNT(DISTINCT) over int64",
    polars=lambda t: t.select(pl.col("UserID").n_unique()),
)
def q05(t):
    """SELECT COUNT(DISTINCT UserID) FROM hits;"""
    return t.aggregate(by=[], u=("count_distinct", "UserID"))


@query(
    "COUNT(DISTINCT) over string",
    polars=lambda t: t.select(ps("SearchPhrase").n_unique()),
)
def q06(t):
    """SELECT COUNT(DISTINCT SearchPhrase) FROM hits;"""
    return t.aggregate(by=[], u=s("SearchPhrase").aggregate("count_distinct"))


@query(
    "MIN/MAX",
    # `.min()` and `.max()` on the same column collide on output name in one
    # `select` — polars raises `DuplicateError` without the aliases.
    polars=lambda t: t.select(
        pl.col("EventDate").min().alias("mn"),
        pl.col("EventDate").max().alias("mx"),
    ),
)
def q07(t):
    """SELECT MIN(EventDate), MAX(EventDate) FROM hits;"""
    return t.aggregate(by=[], mn=("min", "EventDate"), mx=("max", "EventDate"))


@query(
    "GROUP BY int + ORDER BY COUNT(*) DESC",
    tie_key=1,
    polars=lambda t: t.filter(pl.col("AdvEngineID") != 0)
    .group_by("AdvEngineID")
    .agg(pl.len().alias("c"))
    .sort("c", descending=True),
)
def q08(t):
    """SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;"""
    return (
        t.filter(col("AdvEngineID") != lit(0))
        .aggregate(by=["AdvEngineID"], c=count_star())
        .order_by(("c", False))
    )


@query(
    "grouped COUNT(DISTINCT) + ORDER BY + LIMIT",
    tie_key=1,
    polars=lambda t: t.group_by("RegionID")
    .agg(pl.col("UserID").n_unique().alias("u"))
    .sort("u", descending=True)
    .head(10),
)
def q09(t):
    """SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;"""
    return (
        t.aggregate(by=["RegionID"], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query(
    "4 heterogeneous aggregates over one grouping",
    tie_key=2,
    polars=lambda t: t.group_by("RegionID")
    .agg(
        pl.col("AdvEngineID").sum().alias("s"),
        pl.len().alias("c"),
        pl.col("ResolutionWidth").mean().alias("a"),
        pl.col("UserID").n_unique().alias("u"),
    )
    .sort("c", descending=True)
    .head(10),
)
def q10(t):
    """SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;"""
    return (
        t.aggregate(
            by=["RegionID"],
            s=("sum", "AdvEngineID"),
            c=count_star(),
            a=("mean", "ResolutionWidth"),
            u=("count_distinct", "UserID"),
        )
        .order_by(("c", False))
        .limit(10)
    )


# ── Q11–Q19: string group keys ─────────────────────────────────────────────


@query(
    "GROUP BY string key + COUNT(DISTINCT)",
    tie_key=1,
    polars=lambda t: t.filter(ps("MobilePhoneModel") != "")
    .group_by(ps("MobilePhoneModel"))
    .agg(pl.col("UserID").n_unique().alias("u"))
    .sort("u", descending=True)
    .head(10),
)
def q11(t):
    """SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;"""
    return (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(by=[s("MobilePhoneModel")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query(
    "GROUP BY (int, string)",
    tie_key=2,
    polars=lambda t: t.filter(ps("MobilePhoneModel") != "")
    .group_by("MobilePhone", ps("MobilePhoneModel"))
    .agg(pl.col("UserID").n_unique().alias("u"))
    .sort("u", descending=True)
    .head(10),
)
def q12(t):
    """SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;"""
    return (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(
            by=["MobilePhone", s("MobilePhoneModel")],
            u=("count_distinct", "UserID"),
        )
        .order_by(("u", False))
        .limit(10)
    )


@query(
    "GROUP BY high-cardinality string",
    tie_key=1,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .group_by(ps("SearchPhrase"))
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q13(t):
    """SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=[s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY string + COUNT(DISTINCT int64)",
    tie_key=1,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .group_by(ps("SearchPhrase"))
    .agg(pl.col("UserID").n_unique().alias("u"))
    .sort("u", descending=True)
    .head(10),
)
def q14(t):
    """SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=[s("SearchPhrase")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query(
    "GROUP BY (int, string)",
    tie_key=2,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .group_by("SearchEngineID", ps("SearchPhrase"))
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q15(t):
    """SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=["SearchEngineID", s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY very-high-cardinality int64",
    tie_key=1,
    polars=lambda t: t.group_by("UserID")
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q16(t):
    """SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;"""
    return t.aggregate(by=["UserID"], c=count_star()).order_by(("c", False)).limit(10)


@query(
    "GROUP BY (int64, string), very high cardinality",
    tie_key=2,
    polars=lambda t: t.group_by("UserID", ps("SearchPhrase"))
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q17(t):
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"""
    return (
        t.aggregate(by=["UserID", s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY + LIMIT with no ORDER BY",
    compare="shape",
    polars=lambda t: t.group_by("UserID", ps("SearchPhrase"))
    .agg(pl.len().alias("c"))
    .head(10),
)
def q18(t):
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;"""
    # No ORDER BY: which ten groups come back is engine-defined, so this is
    # compared on shape (10 rows x 3 columns) rather than on values.
    return t.aggregate(by=["UserID", s("SearchPhrase")], c=count_star()).limit(10)


@query(
    "extract(minute FROM ts) as a group key",
    tie_key=3,
    polars=lambda t: t.group_by(
        "UserID",
        pl.from_epoch(pl.col("EventTime"), time_unit="s").dt.minute().alias("m"),
        ps("SearchPhrase"),
    )
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q19(t):
    """SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"""
    minute = col("EventTime").cast(TIMESTAMP_S).minute()
    return (
        t.aggregate(by=["UserID", minute, s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


# ── Q20–Q29: filters, LIKE, top-N ──────────────────────────────────────────


@query(
    "point lookup on int64",
    polars=lambda t: t.filter(pl.col("UserID") == 435090932899640449).select("UserID"),
)
def q20(t):
    """SELECT UserID FROM hits WHERE UserID = 435090932899640449;"""
    return t.filter(col("UserID") == lit(435090932899640449)).select("UserID")


@query(
    "LIKE '%…%' + COUNT(*)",
    polars=lambda t: t.filter(ps("URL").str.contains("google", literal=True)).select(
        pl.len()
    ),
)
def q21(t):
    """SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';"""
    return t.filter(s("URL").like("%google%")).aggregate(by=[], c=count_star())


@query(
    "LIKE + MIN(string) + GROUP BY string",
    tie_key=2,
    polars=lambda t: t.filter(
        ps("URL").str.contains("google", literal=True) & (ps("SearchPhrase") != "")
    )
    .group_by(ps("SearchPhrase"))
    .agg(ps("URL").min().alias("m"), pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q22(t):
    """SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("URL").like("%google%") & (s("SearchPhrase") != lit("")))
        .aggregate(by=[s("SearchPhrase")], m=s("URL").min(), c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "LIKE + NOT LIKE + MIN(string) x2 + COUNT(DISTINCT)",
    tie_key=3,
    polars=lambda t: t.filter(
        ps("Title").str.contains("Google", literal=True)
        & ~ps("URL").str.contains(".google.", literal=True)
        & (ps("SearchPhrase") != "")
    )
    .group_by(ps("SearchPhrase"))
    .agg(
        ps("URL").min().alias("mu"),
        ps("Title").min().alias("mt"),
        pl.len().alias("c"),
        pl.col("UserID").n_unique().alias("u"),
    )
    .sort("c", descending=True)
    .head(10),
)
def q23(t):
    """SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(
            s("Title").like("%Google%")
            & ~s("URL").like("%.google.%")
            & (s("SearchPhrase") != lit(""))
        )
        .aggregate(
            by=[s("SearchPhrase")],
            mu=s("URL").min(),
            mt=s("Title").min(),
            c=count_star(),
            u=("count_distinct", "UserID"),
        )
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "SELECT * (105 columns) + ORDER BY + LIMIT",
    probe=["WatchID", "EventTime", "URL"],
    # `SELECT *` over 105 columns: keep a stable three-column probe on both
    # sides rather than dumping 105 columns of reference rows.
    reference_sql=(
        "SELECT WatchID, EventTime, URL FROM hits "
        "WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10"
    ),
    deviation=(
        "The query itself is the real SELECT * over all 105 columns; only the "
        "*comparison* is narrowed to WatchID/EventTime/URL, because dumping "
        "105 columns of reference rows is not useful."
    ),
    polars=lambda t: t.filter(ps("URL").str.contains("google", literal=True))
    .sort("EventTime")
    .head(10),
)
def q24(t):
    """SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;"""
    return t.filter(s("URL").like("%google%")).order_by("EventTime").limit(10)


def _q25(t, keep_key):
    q = (
        t.filter(s("SearchPhrase") != lit(""))
        .project(SearchPhrase=s("SearchPhrase"), EventTime=col("EventTime"))
        .order_by("EventTime")
        .limit(10)
    )
    return q if keep_key else q.select("SearchPhrase")


@query(
    "top-N on int64 sort key",
    tie_key=1,
    # `ORDER BY EventTime` ties thousands of rows per second, and the SQL does
    # not project the sort key — so without it there is nothing to prove the two
    # engines' answers differ only in the tie-break. Both sides are checked on a
    # variant that keeps `EventTime`.
    verify=lambda t: _q25(t, keep_key=True),
    verify_sql=(
        "SELECT SearchPhrase, EventTime FROM hits WHERE SearchPhrase <> '' "
        "ORDER BY EventTime LIMIT 10"
    ),
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .select(ps("SearchPhrase"), pl.col("EventTime"))
    .sort("EventTime")
    .head(10)
    .select("SearchPhrase"),
    # `ORDER BY EventTime` ties, so the check needs the sort key projected —
    # the polars mirror of `verify` / `verify_sql` above.
    polars_verify=lambda t: t.filter(ps("SearchPhrase") != "")
    .select(ps("SearchPhrase"), pl.col("EventTime"))
    .sort("EventTime")
    .head(10),
)
def q25(t):
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;"""
    return _q25(t, keep_key=False)


@query(
    "top-N on string sort key",
    tie_key=0,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .select(ps("SearchPhrase"))
    .sort("SearchPhrase")
    .head(10),
)
def q26(t):
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .project(SearchPhrase=s("SearchPhrase"))
        .order_by("SearchPhrase")
        .limit(10)
    )


def _q27(t, keep_key):
    q = (
        t.filter(s("SearchPhrase") != lit(""))
        .project(SearchPhrase=s("SearchPhrase"), EventTime=col("EventTime"))
        .order_by("EventTime", "SearchPhrase")
        .limit(10)
    )
    return q if keep_key else q.select("SearchPhrase")


@query(
    "top-N on a two-column (int, string) sort key",
    tie_key=1,
    verify=lambda t: _q27(t, keep_key=True),  # same reason as Q25
    verify_sql=(
        "SELECT SearchPhrase, EventTime FROM hits WHERE SearchPhrase <> '' "
        "ORDER BY EventTime, SearchPhrase LIMIT 10"
    ),
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .select(ps("SearchPhrase"), pl.col("EventTime"))
    .sort("EventTime", "SearchPhrase")
    .head(10)
    .select("SearchPhrase"),
    polars_verify=lambda t: t.filter(ps("SearchPhrase") != "")
    .select(ps("SearchPhrase"), pl.col("EventTime"))
    .sort("EventTime", "SearchPhrase")
    .head(10),
)
def q27(t):
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;"""
    return _q27(t, keep_key=False)


@query(
    "AVG(length(str)) + HAVING",
    tie_key=1,
    polars=lambda t: t.filter(ps("URL") != "")
    .group_by("CounterID")
    .agg(ps("URL").str.len_bytes().mean().alias("l"), pl.len().alias("c"))
    .filter(pl.col("c") > 100000)
    .sort("l", descending=True)
    .head(25),
)
def q28(t):
    """SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;"""
    return (
        t.filter(s("URL") != lit(""))
        .aggregate(by=["CounterID"], l=s("URL").length().mean(), c=count_star())
        .filter(col("c") > lit(100000))
        .order_by(("l", False))
        .limit(25)
    )


@query(
    "REGEXP_REPLACE",
    unsupported=(
        "no regex kernel exists anywhere in marrow — neither "
        "`marrow/kernels/string.mojo` nor `DynValue` has one, and there is no "
        "regex engine in the Mojo standard library to bind to "
        "(docs/alpha-findings/g2-regex-evaluation.md)"
    ),
    polars=lambda t: t.filter(ps("Referer") != "")
    .group_by(
        ps("Referer").str.replace(r"^https?://(?:www\.)?([^/]+)/.*$", "$1").alias("k")
    )
    .agg(
        ps("Referer").str.len_bytes().mean().alias("l"),
        pl.len().alias("c"),
        ps("Referer").min().alias("m"),
    )
    .filter(pl.col("c") > 100000)
    .sort("l", descending=True)
    .head(25),
)
def q29(t):
    r"""SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;"""
    raise NotImplementedError(QUERIES["q29"].unsupported)


# ── Q30–Q36: wide aggregation and composite keys ───────────────────────────


@query(
    "90 SUM aggregates over computed expressions in one pass",
    polars=lambda t: t.select(
        [pl.col("ResolutionWidth").sum().alias("s0")]
        + [(pl.col("ResolutionWidth") + k).sum().alias(f"s{k}") for k in range(1, 90)]
    ),
)
def q30(t):
    """SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 2), SUM(ResolutionWidth + 3), SUM(ResolutionWidth + 4), SUM(ResolutionWidth + 5), SUM(ResolutionWidth + 6), SUM(ResolutionWidth + 7), SUM(ResolutionWidth + 8), SUM(ResolutionWidth + 9), SUM(ResolutionWidth + 10), SUM(ResolutionWidth + 11), SUM(ResolutionWidth + 12), SUM(ResolutionWidth + 13), SUM(ResolutionWidth + 14), SUM(ResolutionWidth + 15), SUM(ResolutionWidth + 16), SUM(ResolutionWidth + 17), SUM(ResolutionWidth + 18), SUM(ResolutionWidth + 19), SUM(ResolutionWidth + 20), SUM(ResolutionWidth + 21), SUM(ResolutionWidth + 22), SUM(ResolutionWidth + 23), SUM(ResolutionWidth + 24), SUM(ResolutionWidth + 25), SUM(ResolutionWidth + 26), SUM(ResolutionWidth + 27), SUM(ResolutionWidth + 28), SUM(ResolutionWidth + 29), SUM(ResolutionWidth + 30), SUM(ResolutionWidth + 31), SUM(ResolutionWidth + 32), SUM(ResolutionWidth + 33), SUM(ResolutionWidth + 34), SUM(ResolutionWidth + 35), SUM(ResolutionWidth + 36), SUM(ResolutionWidth + 37), SUM(ResolutionWidth + 38), SUM(ResolutionWidth + 39), SUM(ResolutionWidth + 40), SUM(ResolutionWidth + 41), SUM(ResolutionWidth + 42), SUM(ResolutionWidth + 43), SUM(ResolutionWidth + 44), SUM(ResolutionWidth + 45), SUM(ResolutionWidth + 46), SUM(ResolutionWidth + 47), SUM(ResolutionWidth + 48), SUM(ResolutionWidth + 49), SUM(ResolutionWidth + 50), SUM(ResolutionWidth + 51), SUM(ResolutionWidth + 52), SUM(ResolutionWidth + 53), SUM(ResolutionWidth + 54), SUM(ResolutionWidth + 55), SUM(ResolutionWidth + 56), SUM(ResolutionWidth + 57), SUM(ResolutionWidth + 58), SUM(ResolutionWidth + 59), SUM(ResolutionWidth + 60), SUM(ResolutionWidth + 61), SUM(ResolutionWidth + 62), SUM(ResolutionWidth + 63), SUM(ResolutionWidth + 64), SUM(ResolutionWidth + 65), SUM(ResolutionWidth + 66), SUM(ResolutionWidth + 67), SUM(ResolutionWidth + 68), SUM(ResolutionWidth + 69), SUM(ResolutionWidth + 70), SUM(ResolutionWidth + 71), SUM(ResolutionWidth + 72), SUM(ResolutionWidth + 73), SUM(ResolutionWidth + 74), SUM(ResolutionWidth + 75), SUM(ResolutionWidth + 76), SUM(ResolutionWidth + 77), SUM(ResolutionWidth + 78), SUM(ResolutionWidth + 79), SUM(ResolutionWidth + 80), SUM(ResolutionWidth + 81), SUM(ResolutionWidth + 82), SUM(ResolutionWidth + 83), SUM(ResolutionWidth + 84), SUM(ResolutionWidth + 85), SUM(ResolutionWidth + 86), SUM(ResolutionWidth + 87), SUM(ResolutionWidth + 88), SUM(ResolutionWidth + 89) FROM hits;"""
    aggs = {"s0": ("sum", "ResolutionWidth")}
    for k in range(1, 90):
        aggs[f"s{k}"] = (col("ResolutionWidth") + lit(k)).sum()
    return t.aggregate(by=[], **aggs)


@query(
    "GROUP BY (int16, int32) + 3 aggregates",
    tie_key=2,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .group_by("SearchEngineID", "ClientIP")
    .agg(
        pl.len().alias("c"),
        pl.col("IsRefresh").sum().alias("r"),
        pl.col("ResolutionWidth").mean().alias("a"),
    )
    .sort("c", descending=True)
    .head(10),
)
def q31(t):
    """SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(
            by=["SearchEngineID", "ClientIP"],
            c=count_star(),
            r=("sum", "IsRefresh"),
            a=("mean", "ResolutionWidth"),
        )
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY (int64, int32), high cardinality",
    tie_key=2,
    polars=lambda t: t.filter(ps("SearchPhrase") != "")
    .group_by("WatchID", "ClientIP")
    .agg(
        pl.len().alias("c"),
        pl.col("IsRefresh").sum().alias("r"),
        pl.col("ResolutionWidth").mean().alias("a"),
    )
    .sort("c", descending=True)
    .head(10),
)
def q32(t):
    """SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(
            by=["WatchID", "ClientIP"],
            c=count_star(),
            r=("sum", "IsRefresh"),
            a=("mean", "ResolutionWidth"),
        )
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY (int64, int32) over the whole table",
    tie_key=2,
    polars=lambda t: t.group_by("WatchID", "ClientIP")
    .agg(
        pl.len().alias("c"),
        pl.col("IsRefresh").sum().alias("r"),
        pl.col("ResolutionWidth").mean().alias("a"),
    )
    .sort("c", descending=True)
    .head(10),
)
def q33(t):
    """SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;"""
    return (
        t.aggregate(
            by=["WatchID", "ClientIP"],
            c=count_star(),
            r=("sum", "IsRefresh"),
            a=("mean", "ResolutionWidth"),
        )
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY high-cardinality string over the whole table",
    tie_key=1,
    polars=lambda t: t.group_by(ps("URL"))
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q34(t):
    """SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;"""
    return t.aggregate(by=[s("URL")], c=count_star()).order_by(("c", False)).limit(10)


@query(
    "GROUP BY constant + string",
    tie_key=2,
    polars=lambda t: t.group_by(pl.lit(1).alias("one"), ps("URL"))
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q35(t):
    """SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"""
    return (
        t.aggregate(by=[lit(1), s("URL")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query(
    "GROUP BY four computed integer keys",
    tie_key=4,
    polars=lambda t: t.group_by(
        "ClientIP",
        (pl.col("ClientIP") - 1).alias("i1"),
        (pl.col("ClientIP") - 2).alias("i2"),
        (pl.col("ClientIP") - 3).alias("i3"),
    )
    .agg(pl.len().alias("c"))
    .sort("c", descending=True)
    .head(10),
)
def q36(t):
    """SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;"""
    ip = col("ClientIP")
    return (
        t.aggregate(by=[ip, ip - lit(1), ip - lit(2), ip - lit(3)], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


# ── Q37–Q43: the date-range "page views" family ────────────────────────────


def _july(t):
    """``EventDate BETWEEN '2013-07-01' AND '2013-07-31'`` as a uint16 range."""
    return (col("EventDate") >= lit(JUL_01)) & (col("EventDate") <= lit(JUL_31))


def _pl_july():
    return (pl.col("EventDate") >= JUL_01) & (pl.col("EventDate") <= JUL_31)


@query(
    "multi-predicate filter + GROUP BY string",
    tie_key=1,
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & _pl_july()
        & (pl.col("DontCountHits") == 0)
        & (pl.col("IsRefresh") == 0)
        & (ps("URL") != "")
    )
    .group_by(ps("URL"))
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .head(10),
)
def q37(t):
    """SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;"""
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & _july(t)
            & (col("DontCountHits") == lit(0))
            & (col("IsRefresh") == lit(0))
            & (s("URL") != lit(""))
        )
        .aggregate(by=[s("URL")], PageViews=count_star())
        .order_by(("PageViews", False))
        .limit(10)
    )


@query(
    "multi-predicate filter + GROUP BY string",
    tie_key=1,
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & _pl_july()
        & (pl.col("DontCountHits") == 0)
        & (pl.col("IsRefresh") == 0)
        & (ps("Title") != "")
    )
    .group_by(ps("Title"))
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .head(10),
)
def q38(t):
    """SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;"""
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & _july(t)
            & (col("DontCountHits") == lit(0))
            & (col("IsRefresh") == lit(0))
            & (s("Title") != lit(""))
        )
        .aggregate(by=[s("Title")], PageViews=count_star())
        .order_by(("PageViews", False))
        .limit(10)
    )


@query(
    "LIMIT … OFFSET 1000",
    tie_key=1,
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & _pl_july()
        & (pl.col("IsRefresh") == 0)
        & (pl.col("IsLink") != 0)
        & (pl.col("IsDownload") == 0)
    )
    .group_by(ps("URL"))
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .slice(1000, 10),
)
def q39(t):
    """SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"""
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & _july(t)
            & (col("IsRefresh") == lit(0))
            & (col("IsLink") != lit(0))
            & (col("IsDownload") == lit(0))
        )
        .aggregate(by=[s("URL")], PageViews=count_star())
        .order_by(("PageViews", False))
        .limit(10, 1000)
    )


@query(
    "CASE WHEN as a group key, 5 keys",
    tie_key=5,
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62) & _pl_july() & (pl.col("IsRefresh") == 0)
    )
    .group_by(
        "TraficSourceID",
        "SearchEngineID",
        "AdvEngineID",
        pl.when((pl.col("SearchEngineID") == 0) & (pl.col("AdvEngineID") == 0))
        .then(ps("Referer"))
        .otherwise(pl.lit(""))
        .alias("Src"),
        ps("URL").alias("Dst"),
    )
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .slice(1000, 10),
)
def q40(t):
    """SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"""
    src = if_else(
        (col("SearchEngineID") == lit(0)) & (col("AdvEngineID") == lit(0)),
        s("Referer"),
        lit(""),
    )
    return (
        t.filter(
            (col("CounterID") == lit(62)) & _july(t) & (col("IsRefresh") == lit(0))
        )
        .aggregate(
            by=["TraficSourceID", "SearchEngineID", "AdvEngineID", src, s("URL")],
            PageViews=count_star(),
        )
        .order_by(("PageViews", False))
        .limit(10, 1000)
    )


@query(
    "IN (…) + int64 equality + OFFSET",
    tie_key=2,
    deviation=(
        "`TraficSourceID IN (-1, 6)` is spelled "
        "`isin(ma.array([-1, 6], ma.int16()))`, not `isin([-1, 6])`. The list "
        "form builds an int64 value set. It used to match an int16 column "
        "against it and silently return zero rows "
        "(docs/alpha-findings/e1-clickbench.md, bug 1); as of this measurement "
        "it raises `is_in: dtype mismatch: int16 vs int64` instead — see the "
        "`p_isin_untyped` probe. Still a deviation, because the plain list "
        "spelling the SQL implies does not work; semantics are unchanged, only "
        "the spelling is."
    ),
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & _pl_july()
        & (pl.col("IsRefresh") == 0)
        & pl.col("TraficSourceID").is_in([-1, 6])
        & (pl.col("RefererHash") == 3594120000172545465)
    )
    .group_by("URLHash", "EventDate")
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .slice(100, 10),
)
def q41(t):
    """SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;"""
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & _july(t)
            & (col("IsRefresh") == lit(0))
            & col("TraficSourceID").isin(ma.array([-1, 6], ma.int16()))
            & (col("RefererHash") == lit(3594120000172545465))
        )
        .aggregate(by=["URLHash", "EventDate"], PageViews=count_star())
        .order_by(("PageViews", False))
        .limit(10, 100)
    )


@query(
    "OFFSET 10000 past the end of the result",
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & _pl_july()
        & (pl.col("IsRefresh") == 0)
        & (pl.col("DontCountHits") == 0)
        & (pl.col("URLHash") == 2868770270353813622)
    )
    .group_by("WindowClientWidth", "WindowClientHeight")
    .agg(pl.len().alias("PageViews"))
    .sort("PageViews", descending=True)
    .slice(10000, 10),
)
def q42(t):
    """SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;"""
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & _july(t)
            & (col("IsRefresh") == lit(0))
            & (col("DontCountHits") == lit(0))
            & (col("URLHash") == lit(2868770270353813622))
        )
        .aggregate(
            by=["WindowClientWidth", "WindowClientHeight"], PageViews=count_star()
        )
        .order_by(("PageViews", False))
        .limit(10, 10000)
    )


@query(
    "DATE_TRUNC('minute', ts) as a group key + ORDER BY it",
    tie_key=0,
    polars=lambda t: t.filter(
        (pl.col("CounterID") == 62)
        & (pl.col("EventDate") >= JUL_14)
        & (pl.col("EventDate") <= JUL_15)
        & (pl.col("IsRefresh") == 0)
        & (pl.col("DontCountHits") == 0)
    )
    .group_by(
        pl.from_epoch(pl.col("EventTime"), time_unit="s").dt.truncate("1m").alias("M")
    )
    .agg(pl.len().alias("PageViews"))
    .sort("M")
    .slice(1000, 10),
)
def q43(t):
    """SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;"""
    minute = col("EventTime").cast(TIMESTAMP_S).date_trunc("minute")
    return (
        t.filter(
            (col("CounterID") == lit(62))
            & (col("EventDate") >= lit(JUL_14))
            & (col("EventDate") <= lit(JUL_15))
            & (col("IsRefresh") == lit(0))
            & (col("DontCountHits") == lit(0))
        )
        .aggregate(by=[minute], PageViews=count_star())
        # The group key has no name of its own; the aggregate node calls a
        # computed key `key0`. See docs/alpha-findings/e1-clickbench.md, gap 3.
        .order_by("key0")
        .limit(10, 1000)
    )


# ── extra probes: spellings that do not work, kept so changes are visible ──
#
# These were all *silently wrong* when the alpha measured them. Two now raise
# and one no longer aborts, which is why they stay in the report: the probe is
# how that improvement was noticed. Each docstring records the old behaviour
# next to the current one.

PROBES = {}


def probe(name, note):
    def decorate(fn):
        fn.feature = note
        fn.unsupported = None
        fn.deviation = None
        fn.probe = None
        fn.compare = "rows"
        fn.tie_key = None
        PROBES[name] = fn
        return fn

    return decorate


@probe("p_binary_group_key", "GROUP BY a raw `binary` key, no cast")
def p_binary_group_key(t):
    """SELECT SearchPhrase, COUNT(*) FROM hits GROUP BY SearchPhrase LIMIT 3;

    Deliberately omits the `.cast(string)` every real query above applies.
    This used to hard-abort the process above `_PARALLEL_ALWAYS_ROWS` rows
    (`arrays.mojo` `get: wrong variant type`). It now runs.
    """
    return t.aggregate(by=["SearchPhrase"], c=count_star()).limit(3)


@probe("p_isin_untyped", "isin() with a plain Python list against an int16 column")
def p_isin_untyped(t):
    """SELECT COUNT(*) FROM hits WHERE TraficSourceID IN (-1, 6);

    The correct answer is the same as `p_isin_typed`. This spelling used to
    return 0 silently; it now raises `is_in: dtype mismatch: int16 vs int64`.
    """
    return t.filter(col("TraficSourceID").isin([-1, 6])).aggregate(
        by=[], c=count_star()
    )


@probe("p_isin_typed", "isin() with a dtype-matched marrow Array")
def p_isin_typed(t):
    """SELECT COUNT(*) FROM hits WHERE TraficSourceID IN (-1, 6);"""
    return t.filter(
        col("TraficSourceID").isin(ma.array([-1, 6], ma.int16()))
    ).aggregate(by=[], c=count_star())


@probe("p_binary_literal_ne", "`binary_col != lit('')` without a cast")
def p_binary_literal_ne(t):
    """SELECT COUNT(*) FROM hits WHERE SearchPhrase <> '';

    Correct answer 69354 (what the cast spelling returns). This used to return
    0 silently; it now raises `dispatch_primitive: dtype is not primitive`.
    """
    return t.filter(col("SearchPhrase") != lit(b"")).aggregate(by=[], c=count_star())


# ── the DuckDB reference ───────────────────────────────────────────────────

# `EventDate` is uint16 epoch days and `EventTime` is int64 unix seconds, so the
# literal date predicates have to be rewritten for DuckDB too.
# `make_timestamp` (not `to_timestamp`) keeps the result a naive TIMESTAMP,
# matching marrow's `cast(timestamp('s'))`. `strlen` is byte length; see the
# module docstring, trap 2.
REWRITES = [
    ("EventDate >= '2013-07-01'", f"EventDate >= {JUL_01}"),
    ("EventDate <= '2013-07-31'", f"EventDate <= {JUL_31}"),
    ("EventDate >= '2013-07-14'", f"EventDate >= {JUL_14}"),
    ("EventDate <= '2013-07-15'", f"EventDate <= {JUL_15}"),
    (
        "extract(minute FROM EventTime)",
        "extract(minute FROM make_timestamp(EventTime * 1000000))",
    ),
    (
        "DATE_TRUNC('minute', EventTime)",
        "DATE_TRUNC('minute', make_timestamp(EventTime * 1000000))",
    ),
    ("AVG(length(URL))", "AVG(strlen(URL))"),
    ("AVG(length(Referer))", "AVG(strlen(Referer))"),
]


def _rewrite(sql):
    for old, new in REWRITES:
        sql = sql.replace(old, new)
    return sql


def _binary_columns():
    """The names of the ``binary`` columns in the hits file (metadata only)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(HITS)
    return [f.name for f in schema if pa.types.is_binary(f.type)]


def load_hits_as_strings():
    """The file as an Arrow table with every ``binary`` column viewed as UTF-8.

    ``view()`` is a zero-copy reinterpret with no validation, so DuckDB sees
    real VARCHAR holding the original bytes rather than the ``\\xNN``-escaped
    text ``CAST(blob AS VARCHAR)`` would produce. Module docstring, trap 1.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(HITS)
    fields, arrays = [], []
    for name in table.schema.names:
        arr = table[name].combine_chunks()
        if pa.types.is_binary(arr.type):
            arr = arr.view(pa.string())
        fields.append(pa.field(name, arr.type))
        arrays.append(arr)
    return pa.table(arrays, schema=pa.schema(fields))


def duckdb_reference_connection():
    """A DuckDB connection over the byte-exact in-memory ``hits`` table.

    This is the *correctness* connection. It is eager on purpose: the zero-copy
    binary→string view is the only way to give DuckDB the original bytes.
    """
    con = duckdb.connect()
    con.register("hits", load_hits_as_strings())
    return con


def duckdb_lazy_connection():
    """A DuckDB connection that scans the Parquet file lazily.

    This is the *timing* connection — ``ma.read_parquet`` and
    ``pl.scan_parquet`` are lazy, so DuckDB has to be too or the comparison is
    not one. The binary columns are cast to VARCHAR in a view; that cast escapes
    non-printable bytes (module docstring, trap 1), which is wrong for answers
    but fine for timing, and it is why correctness uses the other connection.
    """
    con = duckdb.connect()
    replace = ", ".join(f'"{c}"::VARCHAR AS "{c}"' for c in _binary_columns())
    path = HITS.replace("'", "''")
    con.execute(
        f"CREATE VIEW hits AS SELECT * REPLACE ({replace}) FROM read_parquet('{path}')"
    )
    return con


def duckdb_rows(con, sql):
    """Run ``sql`` and return ``(normalised rows, column names)``.

    Results come back through Arrow and strings are ``view()``-ed back to
    ``binary``, so the round trip is byte-exact. ``fetchall()`` would decode
    VARCHAR to ``str`` and choke on data that is not all UTF-8.
    """
    import pyarrow as pa

    reader = con.sql(sql).to_arrow_reader(1 << 20)
    table = pa.Table.from_batches(list(reader), schema=reader.schema)
    columns = []
    for arr in table.columns:
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            arr = arr.combine_chunks().view(pa.binary())
        columns.append(arr.to_pylist())
    rows = [[_normalise(c[i]) for c in columns] for i in range(table.num_rows)]
    return rows, list(table.schema.names)


def polars_rows(df):
    """A polars ``DataFrame`` as normalised rows, byte-exact.

    Same treatment as :func:`duckdb_rows`: out through Arrow, strings viewed
    back to ``binary``, because the data is not all valid UTF-8.
    """
    import pyarrow as pa

    table = df.to_arrow()
    columns = []
    for arr in table.columns:
        arr = arr.combine_chunks()
        if pa.types.is_large_string(arr.type):
            arr = arr.view(pa.large_binary())
        elif pa.types.is_string(arr.type):
            arr = arr.view(pa.binary())
        columns.append(arr.to_pylist())
    return [[_normalise(c[i]) for c in columns] for i in range(table.num_rows)]


class Reference:
    """Lazily computed, memoised DuckDB answers for the 43 queries."""

    def __init__(self):
        self._con = None
        self._cache = {}

    @property
    def con(self):
        if self._con is None:
            self._con = duckdb_reference_connection()
        return self._con

    def get(self, sql):
        if sql not in self._cache:
            self._cache[sql] = duckdb_rows(self.con, sql)
        return self._cache[sql]


# ── result normalisation and comparison ────────────────────────────────────


def _normalise(value):
    """One cell as a ``(kind, value)`` pair the engines can be compared on.

    ``"b"`` carries bytes, ``"i"`` an integer, ``"f"`` a float. Strings and
    timestamps both become bytes, because every side renders them the same way
    and the data is not all valid UTF-8.
    """
    import datetime
    import decimal

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return ("b", bytes(value))
    if isinstance(value, str):
        return ("b", value.encode("utf-8", "surrogateescape"))
    if isinstance(value, (datetime.datetime, datetime.date)):
        return ("b", str(value).encode())
    if isinstance(value, decimal.Decimal):
        return ("f", float(value))
    if isinstance(value, bool):
        return ("i", int(value))
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, float):
        return ("f", value)
    return ("b", str(value).encode())


def _cells_equal(a, b, rel=1e-9):
    if a is None or b is None:
        return a is b
    if a[0] == "f" or b[0] == "f":
        x, y = float(a[1]), float(b[1])
        if math.isnan(x) and math.isnan(y):
            return True
        return math.isclose(x, y, rel_tol=rel, abs_tol=1e-9)
    return a == b


def _sort_key(row):
    return tuple((c[0], repr(c[1])) if c is not None else ("", "") for c in row)


def _show(cell):
    if cell is None:
        return "None"
    kind, v = cell
    if kind == "b":
        return repr(v[:60] + b"..." if len(v) > 60 else v)
    return repr(v)


def compare(got, want):
    """Diff two normalised row sets; return ``None`` or a reason string."""
    if len(got) != len(want):
        return f"row count {len(got)} != reference {len(want)}"
    if got and len(got[0]) != len(want[0]):
        return f"column count {len(got[0])} != reference {len(want[0])}"
    g = sorted(got, key=_sort_key)
    w = sorted(want, key=_sort_key)
    for i, (gr, wr) in enumerate(zip(g, w)):
        for j, (gc, wc) in enumerate(zip(gr, wr)):
            if not _cells_equal(gc, wc):
                return f"row {i} col {j}: {_show(gc)} != reference {_show(wc)}"
    return None


def compare_tie(got, want, key):
    """Whether two top-N answers differ only in how they broke a tie.

    Both engines ran ``ORDER BY <key> [DESC] LIMIT n``. If they returned the
    same number of rows *and the same multiset of key values*, then every row
    either side returned sits at a key value the other also returned, so both
    are valid answers and the difference is the tie-break. That is the whole
    argument; nothing weaker is accepted.
    """
    if len(got) != len(want):
        return False
    return all(
        _cells_equal(a[key], b[key])
        for a, b in zip(
            sorted(got, key=lambda r: _sort_key([r[key]])),
            sorted(want, key=lambda r: _sort_key([r[key]])),
        )
    )


def verify(q, out, reference):
    """Turn a ``RAN`` result dict into PASS / PASS(tie) / MISMATCH, in place."""
    sql = q.duckdb_verify_sql
    if reference is None:
        out["status"] = "PASS"
        out["note"] = "ran; no reference to check against"
        return out
    want, want_columns = reference.get(sql)
    if q.compare == "shape":
        ok = len(out["rows"]) == len(want) and (
            not out["rows"] or len(out["rows"][0]) == len(want_columns)
        )
        out["status"] = "PASS" if ok else "MISMATCH"
        out["note"] = "shape-only (the SQL has no ORDER BY)"
        if not ok:
            got_cols = len(out["rows"][0]) if out["rows"] else 0
            out["reason"] = (
                f"{len(out['rows'])} rows x {got_cols} cols vs "
                f"{len(want)} x {len(want_columns)}"
            )
        return out
    got = [[_from_json(c) for c in r] for r in out["rows"]]
    reason = compare(got, want)
    if reason is None:
        out["status"] = "PASS"
        return out
    key = q.tie_key
    if key is not None and got and key < len(got[0]) and compare_tie(got, want, key):
        out["status"] = "PASS"
        out["note"] = (
            f"top-N tie: same {len(got)} rows and an identical multiset of "
            f"ORDER BY values (column {key}); which tied rows are returned is "
            "engine-defined"
        )
        return out
    out["status"] = "MISMATCH"
    out["reason"] = reason
    return out


# ── running marrow ─────────────────────────────────────────────────────────

_MARKER = "@@CLICKBENCH_RESULT@@"


def marrow_scan():
    """The lazy handle every marrow query starts from."""
    return ma.read_parquet(HITS)


def _batch_rows(batch, probe_columns=None):
    """A marrow ``RecordBatch`` as normalised rows, via PyArrow.

    String columns are cast to ``binary`` first: the data in this file is not
    all valid UTF-8, so decoding it to ``str`` is not safe, and bytes is the
    form the reference is normalised to anyway.

    A **zero-row** batch never reaches PyArrow: ``pa.record_batch(batch)`` on an
    empty marrow batch raises ``SystemError: returned NULL without setting an
    exception``. That is a marrow C-Data defect (see
    docs/alpha-findings/e1-clickbench.md, bug 2), not a query failure, so the
    empty case is answered from the schema instead. Two of the 43 queries — Q20
    and Q42 — legitimately return no rows.
    """
    import pyarrow as pa

    if str(batch).startswith("RecordBatch(num_rows=0,"):
        return [], list(probe_columns or ["<empty>"])
    table = pa.record_batch(batch)
    if probe_columns is not None:
        table = table.select(probe_columns)
    columns = []
    for arr in table.columns:
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            arr = arr.cast(pa.binary())
        columns.append(arr.to_pylist())
    rows = [[_normalise(c[i]) for c in columns] for i in range(table.num_rows)]
    return rows, list(table.schema.names)


def _jsonable(v):
    return v.decode("latin1") if isinstance(v, bytes) else v


def _from_json(cell):
    if cell is None:
        return None
    kind, v = cell
    return (kind, v.encode("latin1") if kind == "b" else v)


def run_one(name):
    """Execute one query in *this* process and return its report dict."""
    q = QUERIES.get(name) or PROBES[name]
    if getattr(q, "unsupported", None):
        return {"status": "UNSUPPORTED", "reason": q.unsupported}
    build = q.marrow_verify if isinstance(q, Query) else q
    t0 = time.perf_counter()
    try:
        batch = build(marrow_scan()).collect()
        rows, names = _batch_rows(batch, getattr(q, "probe", None))
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "seconds": round(time.perf_counter() - t0, 3),
        }
    return {
        "status": "RAN",
        "seconds": round(time.perf_counter() - t0, 3),
        "columns": names,
        "rows": [
            [None if c is None else [c[0], _jsonable(c[1])] for c in r] for r in rows
        ],
    }


def run_isolated(name):
    """Execute one query in its **own subprocess** and return its report dict.

    A query that aborts the interpreter costs one row of the report rather than
    the whole report — which is the only reason this suite can report a number
    at all while memory-safety defects are open.
    """
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--only", name],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER) :])
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return {
        "status": "ABORT",
        "reason": tail[-1] if tail else f"exit {proc.returncode}",
    }


# ── the standalone report ──────────────────────────────────────────────────


def run_all(names=None, isolate=True):
    """Run every query and print the PASS/FAIL/ABORT report."""
    reference = Reference() if HAS_DUCKDB else None
    if reference is None:
        print("note: duckdb is absent, so answers are unverified.\n")
    targets = names or (list(QUERIES) + list(PROBES))
    results = {}
    for name in targets:
        q = QUERIES.get(name) or PROBES[name]
        out = run_isolated(name) if isolate else run_one(name)
        if out["status"] == "RAN":
            if isinstance(q, Query):
                verify(q, out, reference)
            else:
                out["status"] = "RAN"
        results[name] = out
        _report_line(name, q, out)
    _summary(results)
    return results


def _report_line(name, q, out):
    bits = [f"{name:<20}", f"{out['status']:<12}"]
    bits.append(f"{out['seconds']:>6.2f}s" if "seconds" in out else "       ")
    bits.append(getattr(q, "feature", ""))
    line = "  ".join(bits)
    if out.get("reason"):
        line += f"\n    -> {out['reason']}"
    if out.get("note"):
        line += f"\n    note: {out['note']}"
    if getattr(q, "deviation", None):
        line += f"\n    DEVIATED: {q.deviation}"
    print(line, flush=True)


def _summary(results):
    real = {k: v for k, v in results.items() if k in QUERIES}
    counts = {}
    for v in real.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print("\n" + "=" * 78)
    print(f"{counts.get('PASS', 0)}/{len(QUERIES)} ClickBench queries PASS")
    for k in sorted(counts):
        print(f"  {k:<12} {counts[k]}")
    deviated = [k for k, q in QUERIES.items() if q.deviation]
    if deviated:
        print(f"  DEVIATED     {len(deviated)}  {', '.join(deviated)}")
    print("=" * 78)


def main(argv):
    if "--only" in argv:
        print(_MARKER + json.dumps(run_one(argv[argv.index("--only") + 1])), flush=True)
        return 0
    if not HAVE_DATA:
        print(f"dataset not found: {HITS}")
        return 2
    names = [a for a in argv[1:] if not a.startswith("-")]
    run_all(names or None, isolate="--no-isolate" not in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
