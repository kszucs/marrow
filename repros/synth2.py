"""Synthetic reproduction attempt #2: clustered filter selectivity.

    python repros/synth2.py <nrows> <cluster> <morsel> <source>

`cluster` is how the surviving rows are distributed:
  head   — only the first 1000 rows match (every later morsel filters to 0 rows)
  sparse — every 997th row matches (no morsel is ever empty)
  blocks — one 200-row run every 50_000 rows (most morsels empty)
  all    — every row matches
`source` is `parquet` or `mem`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "python"))

import pyarrow as pa
import pyarrow.parquet as pq

import marrow as ma
from marrow import col, lit

nrows = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
cluster = sys.argv[2] if len(sys.argv) > 2 else "blocks"
morsel = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
source = sys.argv[4] if len(sys.argv) > 4 else "parquet"

PATH = os.path.join(_HERE, "synth2.parquet")


def matches(i):
    if cluster == "head":
        return i < 1000
    if cluster == "sparse":
        return i % 997 == 0
    if cluster == "blocks":
        return (i % 50_000) < 200
    return True


model = [("m%d" % (i % 7)) if matches(i) else "" for i in range(nrows)]
user = [i * 2654435761 % 1_000_003 for i in range(nrows)]
tbl = pa.table(
    {
        "model": pa.array(model, pa.binary()),
        "user": pa.array(user, pa.int64()),
    }
)
expected = sum(1 for i in range(nrows) if matches(i))

if source == "mem":
    t = ma.memtable(
        ma.record_batch(tbl.combine_chunks().to_batches()[0]), morsel_size=morsel
    )
else:
    pq.write_table(tbl, PATH, compression="snappy")
    t = ma.read_parquet(PATH, morsel_size=morsel)

r = t.filter(col("model").cast(ma.string()) != lit("")).aggregate(
    by=[], c=lit(1).count()
)
got = r.collect().to_pydict()
print("OK", cluster, source, "expected", expected, "got", got, flush=True)
