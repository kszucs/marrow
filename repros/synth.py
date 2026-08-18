"""Synthetic reproduction attempt for the Q11/Q12 SIGSEGV.

Builds a small Parquet file with a page index and many data pages, then runs a
lazy marrow scan whose predicate prunes *pages* (not just row groups).

    python repros/synth.py <nrows> <page_size> <morsel> <mode>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "python"))

import pyarrow as pa
import pyarrow.parquet as pq

import marrow as ma
from marrow import col, lit

nrows = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
morsel = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
mode = sys.argv[4] if len(sys.argv) > 4 else "pruning"

PATH = os.path.join(_HERE, "synth.parquet")

# `model` is empty on all but a sparse scattered minority, so whole data pages
# have min == max == "" and a `model <> ''` predicate prunes them outright.
model = [("m%d" % (i % 7)) if i % 997 == 0 else "" for i in range(nrows)]
user = [i * 2654435761 % 1_000_003 for i in range(nrows)]
tbl = pa.table(
    {
        "model": pa.array(model, pa.string()),
        "user": pa.array(user, pa.int64()),
    }
)
pq.write_table(
    tbl,
    PATH,
    data_page_size=page_size,
    write_page_index=True,
    compression="snappy",
)
pf = pq.ParquetFile(PATH)
print(
    "file: rows=%d row_groups=%d" % (pf.metadata.num_rows, pf.num_row_groups),
    flush=True,
)

t = ma.read_parquet(PATH, morsel_size=morsel)
if mode == "pruning":
    t = t.filter(col("model") != lit(""))
elif mode == "nonpruning":
    t = t.filter(col("user") >= lit(0))
elif mode == "nofilter":
    pass
r = t.aggregate(by=[], c=lit(1).count())
print("OK", mode, r.collect().to_pydict(), flush=True)
