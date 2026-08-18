"""DuckDB reference answers for the 43 ClickBench queries.

The companion to ``clickbench_alpha.py``: that module runs the queries through
marrow, this one produces the answers it is checked against. Separate because
they need different pixi environments — DuckDB lives in ``bench``, the marrow
extension in ``dev``.

    pixi run -e bench python python/marrow/tests/clickbench_reference.py
    pixi run -e dev   python python/marrow/tests/clickbench_alpha.py

Writes ``.benchmarks/clickbench-reference.json`` (gitignored; ~1 MB). Override
with ``MARROW_CLICKBENCH_REFERENCE``.

Two things have to be got right or the reference is silently wrong, and both
initially read as marrow failures when they were not:

1. **Binary columns.** Every string-looking column in the file is ``binary``,
   which DuckDB reads as BLOB, and BLOB binds to neither ``like`` nor ``length``
   nor ``min``. ``CAST(blob AS VARCHAR)`` is *not* the fix: DuckDB's blob→varchar
   cast escapes every non-printable byte as ``\\xNN``, so ``length()`` counts four
   characters per byte and ``ORDER BY`` sorts the escaped text. Instead the file
   is read with PyArrow and the 28 columns are ``view()``-ed — zero-copy
   reinterpret, no validation — from ``binary`` to ``string``, so DuckDB sees
   real VARCHAR holding the original bytes. Results come back through Arrow and
   are ``view()``-ed back to ``binary``, so the round trip is byte-exact.
   ``fetchall()`` would decode VARCHAR to ``str`` and choke on this data.
2. **``length`` means bytes.** DuckDB's ``length(VARCHAR)`` counts *characters*;
   ClickHouse's ``length()`` — which is what ClickBench means — and marrow's
   ``LengthKernel`` both count *bytes*. ``strlen`` is DuckDB's byte-length
   function. Without the rewrite, Q28 reads as a marrow mismatch (76.44 vs
   73.97) when marrow is right.
"""

import datetime
import decimal
import json
import os
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_HITS = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
_DEFAULT_SQL = os.path.expanduser("~/Workspace/ClickBench/clickhouse/queries.sql")
_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

HITS = os.environ.get("MARROW_CLICKBENCH_HITS", _DEFAULT_HITS)
QUERIES_SQL = os.environ.get("MARROW_CLICKBENCH_SQL", _DEFAULT_SQL)
OUT = os.environ.get(
    "MARROW_CLICKBENCH_REFERENCE",
    os.path.join(_REPO, ".benchmarks", "clickbench-reference.json"),
)

# `EventDate` is uint16 epoch days and `EventTime` is int64 unix seconds, so the
# literal date predicates have to be rewritten for DuckDB too. 2013-07-01 ==
# 15887, 2013-07-31 == 15917, 2013-07-14 == 15900, 2013-07-15 == 15901.
# `make_timestamp` (not `to_timestamp`) keeps the result a naive TIMESTAMP,
# matching marrow's `cast(timestamp('s'))`.
REWRITES = [
    ("EventDate >= '2013-07-01'", "EventDate >= 15887"),
    ("EventDate <= '2013-07-31'", "EventDate <= 15917"),
    ("EventDate >= '2013-07-14'", "EventDate >= 15900"),
    ("EventDate <= '2013-07-15'", "EventDate <= 15901"),
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

# Extra rows the harness needs that the canonical SQL does not project: Q25 and
# Q27 sort on EventTime without selecting it, so their answers are ambiguous
# wherever EventTime ties across the LIMIT boundary. These variants add the sort
# key, which is what lets the harness prove the tie is the only difference.
EXTRA = {
    "q25v": (
        "SELECT SearchPhrase, EventTime FROM hits WHERE SearchPhrase <> '' "
        "ORDER BY EventTime LIMIT 10"
    ),
    "q27v": (
        "SELECT SearchPhrase, EventTime FROM hits WHERE SearchPhrase <> '' "
        "ORDER BY EventTime, SearchPhrase LIMIT 10"
    ),
}


def convert(value):
    """One cell as the ``[kind, value]`` pair ``clickbench_alpha`` compares on.

    ``latin1`` is a byte-preserving round trip through JSON, not a character
    encoding claim — the data is not all valid UTF-8.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return ["b", bytes(value).decode("latin1")]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return ["b", str(value)]
    if isinstance(value, decimal.Decimal):
        return ["f", float(value)]
    if isinstance(value, bool):
        return ["i", int(value)]
    if isinstance(value, int):
        return ["i", value]
    if isinstance(value, float):
        return ["f", value]
    return ["b", str(value)]


def _columns_as_bytes(table):
    """A result table's columns as Python lists, strings viewed back as bytes."""
    out = []
    for arr in table.columns:
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            arr = arr.combine_chunks().view(pa.binary())
        out.append(arr.to_pylist())
    return out


def load_hits(path):
    """The file as an Arrow table with every ``binary`` column viewed as UTF-8."""
    table = pq.read_table(path)
    fields, arrays = [], []
    for name in table.schema.names:
        arr = table[name].combine_chunks()
        if pa.types.is_binary(arr.type):
            arr = arr.view(pa.string())
        fields.append(pa.field(name, arr.type))
        arrays.append(arr)
    return pa.table(arrays, schema=pa.schema(fields))


def main():
    if not os.path.exists(HITS):
        print(f"dataset not found: {HITS}", file=sys.stderr)
        return 2
    queries = [line.strip() for line in open(QUERIES_SQL) if line.strip()]
    print(f"{len(queries)} queries from {QUERIES_SQL}", file=sys.stderr)

    con = duckdb.connect()
    con.register("hits", load_hits(HITS))

    todo = [(f"q{i:02d}", sql) for i, sql in enumerate(queries, 1)]
    todo += sorted(EXTRA.items())
    out = {}
    for key, sql in todo:
        for old, new in REWRITES:
            sql = sql.replace(old, new)
        if key == "q24":
            # `SELECT *` over 105 columns: keep a stable three-column probe.
            sql = sql.replace("SELECT *", "SELECT WatchID, EventTime, URL")
        try:
            reader = con.sql(sql).to_arrow_reader(1 << 20)
            arrow = pa.Table.from_batches(list(reader), schema=reader.schema)
            columns = _columns_as_bytes(arrow)
            out[key] = {
                "sql": sql,
                "columns": list(arrow.schema.names),
                "rows": [
                    [convert(c[i]) for c in columns] for i in range(arrow.num_rows)
                ],
            }
            print(f"{key} ok {arrow.num_rows}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            out[key] = {"sql": sql, "error": str(e)}
            print(f"{key} ERR {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
