"""Cross-check ClickBench Q11/Q12 against PyArrow on the real dataset."""

import os

import pyarrow.compute as pc
import pyarrow.parquet as pq

HITS = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
t = pq.read_table(HITS, columns=["MobilePhoneModel", "MobilePhone", "UserID"])
t = t.filter(pc.not_equal(t["MobilePhoneModel"], b""))

q11 = (
    t.group_by(["MobilePhoneModel"])
    .aggregate([("UserID", "count_distinct")])
    .sort_by([("UserID_count_distinct", "descending")])
    .slice(0, 10)
)
print("Q11:")
for r in q11.to_pylist():
    print("  ", r)

q12 = (
    t.group_by(["MobilePhone", "MobilePhoneModel"])
    .aggregate([("UserID", "count_distinct")])
    .sort_by([("UserID_count_distinct", "descending")])
    .slice(0, 10)
)
print("Q12:")
for r in q12.to_pylist():
    print("  ", r)
