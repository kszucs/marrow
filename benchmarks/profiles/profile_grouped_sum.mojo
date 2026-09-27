# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Single-shot profiling driver for `sum(qty * price) GROUP BY g`, per lane.

The query `marrow/expr/tests/bench_comptime.mojo` times as
`bench_comptime_aggregate_sum_grouped_100` /
`bench_runtime_aggregate_sum_grouped_100`, run in a loop with no benchmark
harness so a sampling profiler sees only the query.

    MARROW_PROFILE_LANE=fused   pixi run profile benchmarks/profiles/profile_grouped_sum.mojo --sample --no-open
    MARROW_PROFILE_LANE=runtime pixi run profile benchmarks/profiles/profile_grouped_sum.mojo --sample --no-open

`MARROW_PROFILE_EXPR=column` folds `sum(qty)` instead of `sum(qty * price)`.
Overrides: `MARROW_PROFILE_N` (default 100_000), `MARROW_PROFILE_GROUPS`
(default 100), `MARROW_PROFILE_ITERS` (default 5_000).
"""

from std.benchmark import keep
from std.os.env import getenv
from std.time import perf_counter_ns

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr import DynValue, col, table
from marrow.tabular import RecordBatch, record_batch


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def _batch(n: Int, groups: Int) raises -> RecordBatch:
    var g = List[Optional[Int]](capacity=n)
    var qty = List[Optional[Int]](capacity=n)
    var price = List[Optional[Int]](capacity=n)
    for i in range(n):
        g.append(i % groups)
        qty.append(i % 17)
        price.append(100 + (i % 23))
    return record_batch(
        [
            array(g^, int64).to_dyn(),
            array(qty^, int64).to_dyn(),
            array(price^, int64).to_dyn(),
        ],
        names=["g", "qty", "price"],
    )


def main() raises:
    var n = _parse_int("MARROW_PROFILE_N", 100_000)
    var groups = _parse_int("MARROW_PROFILE_GROUPS", 100)
    var iters = _parse_int("MARROW_PROFILE_ITERS", 5_000)
    var fused = getenv("MARROW_PROFILE_LANE", "fused") == "fused"
    var product = getenv("MARROW_PROFILE_EXPR", "product") == "product"
    var batch = _batch(n, groups)

    var aggs: List[DynValue]
    var keys: List[DynValue]
    if fused:
        keys = [col("g", int64)]
        if product:
            aggs = [(col("qty", int64) * col("price", int64)).sum().alias("r")]
        else:
            aggs = [col("qty", int64).sum().alias("r")]
    else:
        keys = [col("g")]
        if product:
            aggs = [(col("qty") * col("price")).sum().alias("r")]
        else:
            aggs = [col("qty").sum().alias("r")]

    var start = perf_counter_ns()
    for _ in range(iters):
        var plan = table(batch.copy()).aggregate(aggs.copy(), keys.copy())
        keep(plan.execute().num_rows())
    var elapsed = perf_counter_ns() - start
    print(
        "profile_grouped_sum:",
        "fused" if fused else "runtime",
        "product" if product else "column",
        " us/iter =",
        Float64(elapsed) / Float64(iters) / 1000.0,
    )
    keep(batch)
