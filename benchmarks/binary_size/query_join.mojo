"""Binary-size gate: an equi-**join**, which nothing else measures.

`SELECT ... FROM l JOIN r ON l.k = r.k` over two small in-memory batches. The
delta against `query_streaming.mojo` is the cost of the join machinery: the
`SwissHashTable` build and probe, `rapidhash` over the key columns, the radix
partitioner, and the gather that assembles the output.

That is the largest single block of kernel code an AOT query can pull in, and
before this gate nothing linked it — a plan that never joins should not pay for
any of it, and this is what proves it still doesn't.

    pixi run binary_size query_join
"""

from marrow.builders import array
from marrow.dtypes import int64, field
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.relations import InMemoryTable, Join, DynRelation
from marrow.kernels.join import JOIN_INNER


def main() raises:
    var lk = array([1, 2, 3], int64)
    var lv = array([10, 20, 30], int64)
    var left = record_batch([lk.copy(), lv.copy()], names=["k", "lv"])

    var rk = array([2, 3, 4], int64)
    var rv = array([200, 300, 400], int64)
    var right = record_batch([rk.copy(), rv.copy()], names=["k", "rv"])

    var joined = Join(
        left=DynRelation(InMemoryTable(batch=left)),
        right=DynRelation(InMemoryTable(batch=right)),
        left_key_indices=[0],
        right_key_indices=[0],
        join_kind=JOIN_INNER,
        strictness=0,
        schema=schema(
            [
                field("k", int64),
                field("lv", int64),
                field("rv", int64),
            ]
        ),
    )
    print(DynRelation(joined^).execute())
