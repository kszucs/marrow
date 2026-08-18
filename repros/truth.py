import os

import pyarrow.compute as pc
import pyarrow.parquet as pq

HITS = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
t = pq.read_table(HITS, columns=["MobilePhoneModel", "MobilePhone", "Age", "IsRefresh"])
print("rows", t.num_rows)
print("MobilePhoneModel != ''", pc.sum(pc.not_equal(t["MobilePhoneModel"], b"")))
print("MobilePhone != 0", pc.sum(pc.not_equal(t["MobilePhone"], 0)))
print("Age == 31", pc.sum(pc.equal(t["Age"], 31)))
print("IsRefresh != 0", pc.sum(pc.not_equal(t["IsRefresh"], 0)))
