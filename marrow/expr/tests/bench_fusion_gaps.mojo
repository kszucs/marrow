"""What the string breakers actually cost (Q7.1).

`StringLength` and `StringPredicate` are `Breaker`s: they materialise a full
column in `prepare` and read it back per lane in `vectorwise`. So `s.len() + 1`
is two passes over the data, not the one the fusion design claims, and
`s1 == s2 and a > b` likewise.

Q7.1 proposes fusing them. That means giving `StringValue` a way to expose its
offsets vectorwise, which is a change to a trait the binary-size gate watches --
so it should be paid for with a measurement, not with a plausible story. Q6.1 is
the cautionary case: AOT-resolved aggregates *sound* faster than runtime-named
ones and measure as identical, because the cost they remove is per-plan rather
than per-row.

The measurement here isolates the extra pass. Each pair differs by exactly one
fused arithmetic step on an already-materialised column:

    s.len()            materialise the length column, nothing else
    s.len() + 1        the same, plus one pass over that column

The delta between them is what fusing `StringLength` could recover -- an upper
bound on it, in fact, since a fused version still has to read the offsets. If the
delta is small against the total, Q7.1's payoff is small and the trait change is
not worth its size.
"""

from std.benchmark import BenchMetric, keep

from ...testing import Benchmark
from ...builders import StringBuilder
from ...dtypes import string, int32
from ...tabular import record_batch, RecordBatch
from ...expr.values import col, lit, into_array


def _strings(n: Int) raises -> RecordBatch:
    """`n` short strings of varying length, so the length column is not
    constant and the offsets are genuinely read."""
    var b = StringBuilder(n)
    for i in range(n):
        b.append(String("row", i))
    return record_batch([b.finish().to_dyn()], names=["s"])


def _bench_len_only(mut bm: Benchmark, n: Int) raises:
    var batch = _strings(n)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(into_array(col("s", string).length().execute(batch), n))

    bm.iter[call]()
    keep(batch)


def _bench_len_plus_one(mut bm: Benchmark, n: Int) raises:
    var batch = _strings(n)
    bm.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(
            into_array(
                (col("s", string).length() + lit(1, int32)).execute(batch), n
            )
        )

    bm.iter[call]()
    keep(batch)


def bench_fusion_len_only_1m(mut b: Benchmark) raises:
    _bench_len_only(b, 1_000_000)


def bench_fusion_len_plus_one_1m(mut b: Benchmark) raises:
    _bench_len_plus_one(b, 1_000_000)
