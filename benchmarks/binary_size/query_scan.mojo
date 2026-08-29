"""Binary-size gate: a fused query over a **Parquet scan**.

`SELECT a FROM file.parquet WHERE a > b` — the same shape as
`query_streaming.mojo`, but the leaf is a `ParquetScan` instead of an
`InMemoryTable`. The delta between the two is the cost of linking the Parquet
reader: the footer/Thrift decode, the page and column-chunk readers, every
`LeafBuilder`, and the compression shims.

**This gate exists because that cost was invisible.** T2.4 rewrote the whole
Parquet read path and the scan operator and moved *zero bytes* on every gate
then in the suite, because none of them constructed a `ParquetScan`. The DCE
property is real — that is why it was zero — but it meant the entire AOT
Parquet surface had never been measured.

The file is never opened: `compare.py` builds and strips these programs, it does
not run them.

    pixi run binary_size query_scan

**Ported from the old expression package on 2026-08-29; the recorded baseline
predates the port and is stale.** One thing it no longer links:
**statistics-based pruning.** The old scan took the predicate pushed into it
and skipped row groups and pages whose statistics proved no row could match.
`marrow.expr` has no pruning module, no `PruneStats` and no pushdown — the
`Filter` above the scan applies the predicate to every decoded row — so the
pruning path is not in this binary and is not measured by anything. That is a
smaller scan, not a wrong one; restore the measurement when pruning lands.
"""

from marrow.dtypes import field, int64, string
from marrow.expr import col, scan
from marrow.expr import Gt
from marrow.expr import DynValue
from marrow.schema import schema


def main() raises:
    var sch = schema(
        [field("a", int64), field("b", int64), field("name", string)]
    )
    var values: List[DynValue] = [col("a", int64)]
    print(
        scan(String("orders.parquet"), sch^)
        .filter(Gt(col("a", int64), col("b", int64)))
        .project(["a"], values^)
        .execute()
    )
