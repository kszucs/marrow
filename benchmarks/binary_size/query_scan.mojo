"""Binary-size gate: a fused query over a **Parquet scan**.

`SELECT a FROM file.parquet WHERE a > b` — the same shape as
`query_streaming.mojo`, but the leaf is a `ParquetScan` instead of an
`InMemoryTable`. The delta between the two is the cost of linking the Parquet
reader: the footer/Thrift decode, the page and column-chunk readers, every
`LeafBuilder`, the compression shims, and the pruning path.

**This gate exists because that cost was invisible.** T2.4 rewrote the whole
Parquet read path and the scan processor and moved *zero bytes* on every gate
then in the suite, because none of them constructed a `ParquetScan`. The DCE
property is real — that is why it was zero — but it meant the entire AOT Parquet
surface had never been measured.

The file is never opened: `compare.py` builds and strips these programs, it does
not run them.

    pixi run binary_size query_scan
"""

from marrow.dtypes import int64, string, field
from marrow.schema import schema
from marrow.expr.values import col, AnyValue
from marrow.expr.relations import ParquetScan, Project, DynRelation


def main() raises:
    var sch = schema(
        [field("a", int64), field("b", int64), field("name", string)]
    )
    var filtered = DynRelation(
        ParquetScan(path=String("orders.parquet"), schema=sch)
    ).filter(AnyValue(col("a", int64) > col("b", int64)))

    var values = List[AnyValue]()
    values.append(AnyValue(col("a", int64)))
    var proj = Project(
        input=filtered,
        names=["a"],
        values=values^,
        schema=schema([field("a", int64)]),
    )
    print(DynRelation(proj^).execute())
