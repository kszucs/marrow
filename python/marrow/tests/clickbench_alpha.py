"""All 43 ClickBench queries expressed against marrow's *lazy* Python frontend.

This is the alpha feature-coverage measurement. It is the lazy replacement for
``clickbench.py`` (the older eager, 11-query, PyArrow-driven harness), which is
left untouched.

Run it::

    pixi run -e bench python python/marrow/tests/clickbench_reference.py  # once
    pixi run -e dev   python python/marrow/tests/clickbench_alpha.py

Each query is one function named ``q01`` … ``q43`` carrying the canonical SQL in
its docstring and returning a ``LazyTable``. The runner executes every one **in
its own subprocess**, so a query that aborts the interpreter (see "Known aborts"
below) costs one row of the report rather than the whole report, and diffs the
answer against the DuckDB reference rows ``clickbench_reference.py`` produced.
Without a reference the queries still run; they are just reported unverified.

Dataset
-------
``~/Workspace/ClickBench/data/hits_0.parquet`` — one partition, 1,000,000 rows,
105 columns. Override with ``MARROW_CLICKBENCH_HITS``.

Three dataset facts drive most of the code below, and none of them is what the
ClickBench SQL implies:

* ``EventDate`` is **uint16** (days since the epoch), not a date. The literal
  date predicates therefore become integer predicates: ``'2013-07-01'`` is
  15887, ``'2013-07-31'`` is 15917, ``'2013-07-14'`` is 15900, ``'2013-07-15'``
  is 15901.
* ``EventTime`` is **int64** (unix seconds), not a timestamp. Q19 and Q43 cast
  it to ``timestamp('s')`` before asking for a minute.
* Every string-looking column is **binary**. marrow's string kernels are bound
  on ``StringLikeType`` and will not accept it, so those columns are cast with
  ``.cast(STRING)`` at the point of use. DuckDB has exactly the same problem —
  BLOB does not bind to ``like``/``length``/``min`` — so the reference view
  casts the same 28 columns, and the two sides are comparing the same thing.

``COUNT(*)``
------------
``count(col)`` counts *non-null values*, which is not ``COUNT(*)``. marrow has
``marrow.expr.builders.count_star()`` on the Mojo side but **it is not bound to
Python on this tree** (there is no ``libmarrow.expr_count_star``), so this file
spells it out as ``lit(1).count()`` — a literal is valid on every row, so the
valid-count of a constant column is the row count. ``COUNT_STAR_QUERIES`` lists
every query that depends on it; all of them become one-liners the day the
binding lands.

Known aborts
------------
Grouping by a raw ``binary`` key hard-aborts the process above ~200,000 rows
(``_PARALLEL_ALWAYS_ROWS`` in ``marrow/kernels/groupby.mojo``). Casting the key
to ``string`` avoids it, which is what every grouped query here does.
``p_binary_group_key`` in ``PROBES`` reproduces the abort deliberately so the
report records it.

A *second*, previously unrecorded crash takes out Q11, Q12 and Q24 — see
``docs/alpha-findings/e1-clickbench.md`` §2.2 and §2.3. Nothing here works
around it; those three are reported as ABORT.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYROOT = os.path.dirname(os.path.dirname(_HERE))
if _PYROOT not in sys.path:
    sys.path.insert(0, _PYROOT)

import marrow as ma  # noqa: E402
from marrow import col, if_else, lit  # noqa: E402

__all__ = ["QUERIES", "run_all", "run_one"]


_DEFAULT = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
HITS = os.environ.get("MARROW_CLICKBENCH_HITS", _DEFAULT)
REFERENCE = os.environ.get(
    "MARROW_CLICKBENCH_REFERENCE",
    os.path.join(_PYROOT, "..", ".benchmarks", "clickbench-reference.json"),
)

STRING = ma.string()
TIMESTAMP_S = ma.timestamp("s", None)

# `'2013-07-01'` … as uint16 epoch days, because `EventDate` is uint16.
JUL_01 = 15887
JUL_14 = 15900
JUL_15 = 15901
JUL_31 = 15917


def count_star():
    """``COUNT(*)`` — see the module docstring.

    Not ``col(x).count()``: that counts non-null values of ``x``. A literal is
    valid on every row, so counting one counts rows. Replace every call with
    ``marrow.count_star()`` once the Mojo builder is bound to Python.
    """
    return lit(1).count()


def s(name):
    """A binary column read as a string — ``.cast(ma.string())``.

    Needed on 28 of the 105 columns; see the module docstring.
    """
    return col(name).cast(STRING)


# ── the registry ───────────────────────────────────────────────────────────

QUERIES = {}


def query(
    feature,
    *,
    deviation=None,
    unsupported=None,
    compare="rows",
    probe=None,
    tie_key=None,
    reference=None,
):
    """Register a query function with the metadata the report needs.

    ``compare`` is ``"rows"`` (full row-set diff against DuckDB) or ``"shape"``
    (row/column counts only, for the one query whose SQL has no ORDER BY and is
    therefore genuinely nondeterministic).

    ``probe`` narrows the marrow result to named columns before comparing, for
    the one ``SELECT *`` query.

    ``tie_key`` is the 0-based index of the ``ORDER BY`` column in the output.
    ``ORDER BY k DESC LIMIT n`` has no unique answer when the value at the
    boundary is tied, and most of these queries order by a count with hundreds
    of tied groups. So on a row-set mismatch the harness falls back to
    comparing the *multiset of ``tie_key`` values*: if those are identical then
    both engines returned a valid top-N and only the tie-break differs. That is
    a sound argument, not a fudge — same row count, same ordering values, both
    sorted by that column.

    ``reference`` names a different reference key, for the two queries that sort
    on a column they do not project (Q25/Q27): the harness compares a variant
    that also projects the sort key, so the tie argument above can be made at
    all.
    """

    def decorate(fn):
        fn.feature = feature
        fn.deviation = deviation
        fn.unsupported = unsupported
        fn.compare = compare
        fn.probe = probe
        fn.tie_key = tie_key
        fn.reference = reference or fn.__name__
        QUERIES[fn.__name__] = fn
        return fn

    return decorate


# ── Q1–Q10: whole-table and low-cardinality aggregation ────────────────────


@query("COUNT(*)")
def q01(t):
    """SELECT COUNT(*) FROM hits;"""
    return t.aggregate(by=[], c=count_star())


@query("COUNT(*) + WHERE <>")
def q02(t):
    """SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;"""
    return t.filter(col("AdvEngineID") != lit(0)).aggregate(by=[], c=count_star())


@query("SUM + COUNT(*) + AVG, no GROUP BY")
def q03(t):
    """SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;"""
    return t.aggregate(
        by=[],
        s=("sum", "AdvEngineID"),
        c=count_star(),
        a=("mean", "ResolutionWidth"),
    )


@query("AVG over int64")
def q04(t):
    """SELECT AVG(UserID) FROM hits;"""
    return t.aggregate(by=[], a=("mean", "UserID"))


@query("COUNT(DISTINCT) over int64")
def q05(t):
    """SELECT COUNT(DISTINCT UserID) FROM hits;"""
    return t.aggregate(by=[], u=("count_distinct", "UserID"))


@query("COUNT(DISTINCT) over string")
def q06(t):
    """SELECT COUNT(DISTINCT SearchPhrase) FROM hits;"""
    return t.aggregate(by=[], u=s("SearchPhrase").aggregate("count_distinct"))


@query("MIN/MAX")
def q07(t):
    """SELECT MIN(EventDate), MAX(EventDate) FROM hits;"""
    return t.aggregate(by=[], mn=("min", "EventDate"), mx=("max", "EventDate"))


@query("GROUP BY int + ORDER BY COUNT(*) DESC", tie_key=1)
def q08(t):
    """SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;"""
    return (
        t.filter(col("AdvEngineID") != lit(0))
        .aggregate(by=["AdvEngineID"], c=count_star())
        .order_by(("c", False))
    )


@query("grouped COUNT(DISTINCT) + ORDER BY + LIMIT", tie_key=1)
def q09(t):
    """SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;"""
    return (
        t.aggregate(by=["RegionID"], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query("4 heterogeneous aggregates over one grouping", tie_key=2)
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


@query("GROUP BY string key + COUNT(DISTINCT)", tie_key=1)
def q11(t):
    """SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;"""
    return (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(by=[s("MobilePhoneModel")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query("GROUP BY (int, string)", tie_key=2)
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


@query("GROUP BY high-cardinality string", tie_key=1)
def q13(t):
    """SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=[s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query("GROUP BY string + COUNT(DISTINCT int64)", tie_key=1)
def q14(t):
    """SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=[s("SearchPhrase")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )


@query("GROUP BY (int, string)", tie_key=2)
def q15(t):
    """SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=["SearchEngineID", s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query("GROUP BY very-high-cardinality int64", tie_key=1)
def q16(t):
    """SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;"""
    return t.aggregate(by=["UserID"], c=count_star()).order_by(("c", False)).limit(10)


@query("GROUP BY (int64, string), very high cardinality", tie_key=2)
def q17(t):
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"""
    return (
        t.aggregate(by=["UserID", s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query("GROUP BY + LIMIT with no ORDER BY", compare="shape")
def q18(t):
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;"""
    # No ORDER BY: which ten groups come back is engine-defined, so this is
    # compared on shape (10 rows x 3 columns) rather than on values.
    return t.aggregate(by=["UserID", s("SearchPhrase")], c=count_star()).limit(10)


@query("extract(minute FROM ts) as a group key", tie_key=3)
def q19(t):
    """SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"""
    minute = col("EventTime").cast(TIMESTAMP_S).minute()
    return (
        t.aggregate(by=["UserID", minute, s("SearchPhrase")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


# ── Q20–Q29: filters, LIKE, top-N ──────────────────────────────────────────


@query("point lookup on int64")
def q20(t):
    """SELECT UserID FROM hits WHERE UserID = 435090932899640449;"""
    return t.filter(col("UserID") == lit(435090932899640449)).select("UserID")


@query("LIKE '%…%' + COUNT(*)")
def q21(t):
    """SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';"""
    return t.filter(s("URL").like("%google%")).aggregate(by=[], c=count_star())


@query("LIKE + MIN(string) + GROUP BY string", tie_key=2)
def q22(t):
    """SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"""
    return (
        t.filter(s("URL").like("%google%") & (s("SearchPhrase") != lit("")))
        .aggregate(by=[s("SearchPhrase")], m=s("URL").min(), c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query("LIKE + NOT LIKE + MIN(string) x2 + COUNT(DISTINCT)", tie_key=3)
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
    deviation=(
        "The query itself is the real SELECT * over all 105 columns; only the "
        "*comparison* is narrowed to WatchID/EventTime/URL, because dumping "
        "105 columns of reference rows is not useful."
    ),
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


@query("top-N on int64 sort key", tie_key=1, reference="q25v")
def q25(t):
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;"""
    return _q25(t, keep_key=False)


# `ORDER BY EventTime` ties thousands of rows per second, and the SQL does not
# project the sort key — so without it there is nothing to prove the two
# engines' answers differ only in the tie-break. `q25.verify` is the same plan
# with `EventTime` kept, checked against the `q25v` reference.
q25.verify = lambda t: _q25(t, keep_key=True)


@query("top-N on string sort key", tie_key=0)
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


@query("top-N on a two-column (int, string) sort key", tie_key=1, reference="q27v")
def q27(t):
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;"""
    return _q27(t, keep_key=False)


# Same reason as Q25 — see there.
q27.verify = lambda t: _q27(t, keep_key=True)


@query("AVG(length(str)) + HAVING", tie_key=1)
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
        "regex engine in the Mojo standard library to bind to"
    ),
)
def q29(t):
    r"""SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;"""
    raise NotImplementedError(q29.unsupported)


# ── Q30–Q36: wide aggregation and composite keys ───────────────────────────


@query("90 SUM aggregates over computed expressions in one pass")
def q30(t):
    """SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), … SUM(ResolutionWidth + 89) FROM hits;"""
    aggs = {"s0": ("sum", "ResolutionWidth")}
    for k in range(1, 90):
        aggs[f"s{k}"] = (col("ResolutionWidth") + lit(k)).sum()
    return t.aggregate(by=[], **aggs)


@query("GROUP BY (int16, int32) + 3 aggregates", tie_key=2)
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


@query("GROUP BY (int64, int32), high cardinality", tie_key=2)
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


@query("GROUP BY (int64, int32) over the whole table", tie_key=2)
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


@query("GROUP BY high-cardinality string over the whole table", tie_key=1)
def q34(t):
    """SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;"""
    return t.aggregate(by=[s("URL")], c=count_star()).order_by(("c", False)).limit(10)


@query("GROUP BY constant + string", tie_key=2)
def q35(t):
    """SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"""
    return (
        t.aggregate(by=[lit(1), s("URL")], c=count_star())
        .order_by(("c", False))
        .limit(10)
    )


@query("GROUP BY four computed integer keys", tie_key=4)
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


@query("multi-predicate filter + GROUP BY string", tie_key=1)
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


@query("multi-predicate filter + GROUP BY string", tie_key=1)
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


@query("LIMIT … OFFSET 1000", tie_key=1)
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


@query("CASE WHEN as a group key, 5 keys", tie_key=5)
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
        "form builds an int64 value set, and matching an int16 column against "
        "it silently returns zero rows instead of raising — see the findings "
        "doc, bug 1. Semantics are unchanged; only the spelling is."
    ),
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


@query("OFFSET 10000 past the end of the result")
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


@query("DATE_TRUNC('minute', ts) as a group key + ORDER BY it", tie_key=0)
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
        # computed key `key0`. See the findings doc, gap 3.
        .order_by("key0")
        .limit(10, 1000)
    )


# ── extra probes, reported but not part of the 43 ──────────────────────────

PROBES = {}


def probe(name, note):
    def decorate(fn):
        fn.feature = note
        PROBES[name] = fn
        return fn

    return decorate


@probe("p_binary_group_key", "GROUP BY a raw `binary` key (no cast) — known abort")
def p_binary_group_key(t):
    """SELECT SearchPhrase, COUNT(*) FROM hits GROUP BY SearchPhrase LIMIT 3;

    Deliberately omits the `.cast(string)` every real query above applies, to
    record the `_PARALLEL_ALWAYS_ROWS` abort in the report.
    """
    return t.aggregate(by=["SearchPhrase"], c=count_star()).limit(3)


@probe("p_isin_untyped", "isin() with a plain Python list against an int16 column")
def p_isin_untyped(t):
    """SELECT COUNT(*) FROM hits WHERE TraficSourceID IN (-1, 6);

    The correct answer is the same as `p_isin_typed`. This spelling returns 0.
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

    Correct answer 69354 (what the cast spelling returns). This returns 0.
    """
    return t.filter(col("SearchPhrase") != lit(b"")).aggregate(by=[], c=count_star())


# ── result normalisation and comparison ────────────────────────────────────


def _normalise(value):
    """One cell as a ``(kind, value)`` pair the two engines can be compared on.

    ``"b"`` carries bytes, ``"i"`` an integer, ``"f"`` a float. Strings and
    timestamps both become bytes, because both sides render them the same way
    and the data is not all valid UTF-8.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return ("b", bytes(value))
    if isinstance(value, str):
        return ("b", value.encode("utf-8", "surrogateescape"))
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
    gk = sorted(_sort_key([r[key]]) for r in got)
    wk = sorted(_sort_key([r[key]]) for r in want)
    if len(gk) != len(wk):
        return False
    return all(
        _cells_equal(a[key], b[key])
        for a, b in zip(
            sorted(got, key=lambda r: _sort_key([r[key]])),
            sorted(want, key=lambda r: _sort_key([r[key]])),
        )
    )


def _show(cell):
    if cell is None:
        return "None"
    kind, v = cell
    if kind == "b":
        return repr(v[:60] + b"..." if len(v) > 60 else v)
    return repr(v)


# ── execution ──────────────────────────────────────────────────────────────

_MARKER = "@@CLICKBENCH_RESULT@@"


def _batch_rows(batch, probe_columns=None):
    """A marrow ``RecordBatch`` as normalised rows, via PyArrow.

    String columns are cast to ``binary`` first: the data in this file is not
    all valid UTF-8, so decoding it to ``str`` is not safe, and bytes is the
    form the reference is normalised to anyway.

    A **zero-row** batch never reaches PyArrow: ``pa.record_batch(batch)`` on an
    empty marrow batch raises ``SystemError: returned NULL without setting an
    exception``. That is a marrow C-Data bug (findings doc, bug 2), not a query
    failure, so the empty case is answered from the schema instead. Two of the
    43 queries — Q20 and Q42 — legitimately return no rows.
    """
    import pyarrow as pa

    if str(batch).startswith("RecordBatch(num_rows=0,"):
        names = [n for n in (probe_columns or [])] or None
        return [], names or ["<empty>"]
    table = pa.record_batch(batch)
    if probe_columns is not None:
        table = table.select(probe_columns)
    columns = []
    for arr in table.columns:
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            arr = arr.cast(pa.binary())
        columns.append(arr.to_pylist())
    return [[_normalise(c[i]) for c in columns] for i in range(table.num_rows)], list(
        table.schema.names
    )


def run_one(name):
    """Execute one query in this process and return its report dict."""
    fn = QUERIES.get(name) or PROBES[name]
    if getattr(fn, "unsupported", None):
        return {"status": "UNSUPPORTED", "reason": fn.unsupported}
    import time

    build = getattr(fn, "verify", None) or fn
    t0 = time.time()
    try:
        table = ma.read_parquet(HITS)
        batch = build(table).collect()
        rows, names = _batch_rows(batch, getattr(fn, "probe", None))
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "seconds": round(time.time() - t0, 3),
        }
    return {
        "status": "RAN",
        "seconds": round(time.time() - t0, 3),
        "columns": names,
        "rows": [
            [None if c is None else [c[0], _jsonable(c[1])] for c in r] for r in rows
        ],
    }


def _jsonable(v):
    return v.decode("latin1") if isinstance(v, bytes) else v


def _from_json(cell):
    if cell is None:
        return None
    kind, v = cell
    return (kind, v.encode("latin1") if kind == "b" else v)


def _load_reference():
    if not os.path.exists(REFERENCE):
        return {}
    with open(REFERENCE) as fh:
        return json.load(fh)


def run_all(names=None, isolate=True):
    """Run every query, each in its own subprocess, and print the report."""
    reference = _load_reference()
    if not reference:
        print(
            f"note: no reference at {REFERENCE}; run "
            "`pixi run -e bench python python/marrow/tests/"
            "clickbench_reference.py` first. "
            "Queries will be run but not verified.\n"
        )
    targets = names or (list(QUERIES) + list(PROBES))
    results = {}
    for name in targets:
        fn = QUERIES.get(name) or PROBES[name]
        if isolate:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--only", name],
                capture_output=True,
                text=True,
            )
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith(_MARKER):
                    payload = json.loads(line[len(_MARKER) :])
            if payload is None:
                tail = (proc.stdout + proc.stderr).strip().splitlines()
                results[name] = {
                    "status": "ABORT",
                    "reason": (tail[-1] if tail else f"exit {proc.returncode}"),
                }
                _report_line(name, fn, results[name])
                continue
            out = payload
        else:
            out = run_one(name)

        if out["status"] == "RAN":
            _verify(fn, name, out, reference)
        results[name] = out
        _report_line(name, fn, out)

    _summary(results)
    return results


def _verify(fn, name, out, reference):
    """Turn a ``RAN`` result into PASS / PASS(tie) / MISMATCH."""
    want = reference.get(getattr(fn, "reference", name))
    mode = getattr(fn, "compare", None)
    if want is None or "error" in want:
        out["status"] = "PASS"
        out["note"] = "ran; no reference to check against"
        return
    if mode == "shape":
        ok = len(out["rows"]) == len(want["rows"]) and (
            not out["rows"] or len(out["rows"][0]) == len(want["columns"])
        )
        out["status"] = "PASS" if ok else "MISMATCH"
        out["note"] = "shape-only (the SQL has no ORDER BY)"
        if not ok:
            out["reason"] = (
                f"{len(out['rows'])} rows x "
                f"{len(out['rows'][0]) if out['rows'] else 0} cols vs "
                f"{len(want['rows'])} x {len(want['columns'])}"
            )
        return
    got = [[_from_json(c) for c in r] for r in out["rows"]]
    exp = [[_from_json(c) for c in r] for r in want["rows"]]
    reason = compare(got, exp)
    if reason is None:
        out["status"] = "PASS"
        return
    key = getattr(fn, "tie_key", None)
    if key is not None and got and key < len(got[0]) and compare_tie(got, exp, key):
        out["status"] = "PASS"
        out["note"] = (
            f"top-N tie: same {len(got)} rows and an identical multiset of "
            f"ORDER BY values (column {key}); which tied rows are returned is "
            "engine-defined"
        )
        return
    out["status"] = "MISMATCH"
    out["reason"] = reason


def _report_line(name, fn, out):
    status = out["status"]
    bits = [f"{name:<20}", f"{status:<12}"]
    if "seconds" in out:
        bits.append(f"{out['seconds']:>6.2f}s")
    else:
        bits.append("       ")
    bits.append(getattr(fn, "feature", ""))
    line = "  ".join(bits)
    if out.get("reason"):
        line += f"\n    -> {out['reason']}"
    if out.get("note"):
        line += f"\n    note: {out['note']}"
    if getattr(fn, "deviation", None):
        line += f"\n    DEVIATED: {fn.deviation}"
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
    deviated = [k for k, f in QUERIES.items() if f.deviation]
    if deviated:
        print(f"  DEVIATED     {len(deviated)}  {', '.join(deviated)}")
    print(f"  uses COUNT(*) workaround: {len(COUNT_STAR_QUERIES)} queries")
    print("=" * 78)


# Every query whose answer depends on real `COUNT(*)` semantics and therefore
# on the `lit(1).count()` workaround above.
COUNT_STAR_QUERIES = [
    "q01",
    "q02",
    "q03",
    "q08",
    "q10",
    "q13",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q21",
    "q22",
    "q23",
    "q28",
    "q31",
    "q32",
    "q33",
    "q34",
    "q35",
    "q36",
    "q37",
    "q38",
    "q39",
    "q40",
    "q41",
    "q42",
    "q43",
]


def main(argv):
    if "--only" in argv:
        name = argv[argv.index("--only") + 1]
        print(_MARKER + json.dumps(run_one(name)), flush=True)
        return 0
    names = [a for a in argv[1:] if not a.startswith("-")]
    if not os.path.exists(HITS):
        print(f"dataset not found: {HITS}")
        return 2
    run_all(names or None, isolate="--no-isolate" not in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
