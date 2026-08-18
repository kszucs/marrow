"""Flexible bisection driver for the Q11/Q12 SIGSEGV.

    python repros/bisect2.py <cols-csv|ALL> <predicate> <morsel_size> [source]

`predicate` is one of the keys in PREDS below, or `none`.
`source` is `parquet` (default) or `mem`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "python"))
import marrow as ma
from marrow import col, lit

HITS = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
STRING = ma.string()


def s(n):
    return col(n).cast(STRING)


PREDS = {
    "none": None,
    "mpm_ne_empty": lambda: s("MobilePhoneModel") != lit(""),
    "sp_ne_empty": lambda: s("SearchPhrase") != lit(""),
    "url_ne_empty": lambda: s("URL") != lit(""),
    "mp_ne_0": lambda: col("MobilePhone") != lit(0),
    "age_eq_31": lambda: col("Age") == lit(31),
    "isrefresh_ne_0": lambda: col("IsRefresh") != lit(0),
    "true": lambda: col("MobilePhone") >= lit(-32768),
    "false": lambda: col("MobilePhone") == lit(31337),
    "watchid_eq": lambda: col("WatchID") == lit(1),
}

cols = sys.argv[1]
pred = sys.argv[2]
morsel = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
source = sys.argv[4] if len(sys.argv) > 4 else "parquet"

if source == "mem":
    import pyarrow.parquet as pq

    names = None if cols == "ALL" else cols.split(",")
    tbl = pq.read_table(HITS, columns=names)
    batch = tbl.combine_chunks().to_batches()[0]
    t = ma.memtable(ma.record_batch(batch), morsel_size=morsel)
else:
    if cols == "ALL":
        t = ma.read_parquet(HITS, morsel_size=morsel)
    else:
        import pyarrow.parquet as pq

        names = cols.split(",")
        pa_schema = pq.ParquetFile(HITS).schema_arrow
        sub = pa_schema.empty_table().select(names).schema
        t = ma.read_parquet(HITS, schema=ma.schema(sub), morsel_size=morsel)

if pred != "none":
    t = t.filter(PREDS[pred]())
r = t.aggregate(by=[], c=lit(1).count())
b = r.collect()
print("OK", cols, pred, morsel, source, b, b.to_pydict())
