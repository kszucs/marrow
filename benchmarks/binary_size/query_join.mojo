"""Binary-size gate: an equi-**join**, which nothing else measures.

`SELECT ... FROM l JOIN r ON l.k = r.k` over two small in-memory batches. The
delta against `query_streaming.mojo` is the cost of the join machinery: the
`SwissHashTable` build and probe, `rapidhash` over the key columns, the radix
partitioner, and the gather that assembles the output.

That is the largest single block of kernel code an AOT query can pull in, and
before this gate nothing linked it — a plan that never joins should not pay for
any of it, and this is what proves it still doesn't.

    pixi run binary_size query_join

**Ported from the old expression package on 2026-08-29.** The join carries
over whole: same kernel, same key indices, same `JOIN_INNER` from
`marrow.kernels.join`.
Two mechanical differences, neither of which changes what is measured — the
output schema is derived by `Join._output_schema` instead of being passed in,
and the plan is spelled with the `join` verb rather than by naming the node.
The recorded baseline predates the port and is stale.
"""

from marrow.builders import array
from marrow.expr.builders import table
from marrow.dtypes import int64
from marrow.kernels.join import JOIN_INNER
from marrow.tabular import record_batch


def main() raises:
    var lk = array([1, 2, 3], int64)
    var lv = array([10, 20, 30], int64)
    var left = record_batch([lk.copy(), lv.copy()], names=["k", "lv"])

    var rk = array([2, 3, 4], int64)
    var rv = array([200, 300, 400], int64)
    var right = record_batch([rk.copy(), rv.copy()], names=["k", "rv"])

    print(table(left^).join(table(right^), [0], [0], JOIN_INNER).execute())
