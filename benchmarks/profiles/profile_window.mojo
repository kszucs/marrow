# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Single-shot profiling driver for the shapes in `bench_window.mojo`.

    pixi run profile benchmarks/profiles/profile_window.mojo --sample --no-open

`sum` runs over its frames in one pass (`Windowable.over`): a running fold
while the frame's start holds still, a segment-tree query otherwise.
`MARROW_PROFILE_SHAPE` picks what the trace shows:

- `sliding` (default) — `ROWS BETWEEN 9 PRECEDING AND CURRENT ROW`: every
  frame's start moves, so every row is a tree query.
- `cumulative` — the default `RANGE` frame over distinct keys: the running fold.
- `ties` — the default frame over 16 ordering values: 16 peer groups.
- `row_number` — no frame: the sort, boundary scan and scatter every shape pays.

Overrides: `MARROW_PROFILE_N` (default 20_000), `MARROW_PROFILE_ITERS`
(default 10).
"""

from std.benchmark import keep
from std.os.env import getenv

from marrow.builders import Int64Builder
from marrow.dtypes import int64
from marrow.expr import DynRelation, col, row_number, table
from marrow.tabular import record_batch


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def _plan(shape: String, rows: DynRelation) raises -> DynRelation:
    if shape == "sliding":
        return rows.with_columns(
            ["s"],
            [
                col("v", int64)
                .sum()
                .over(order_by=[col("o", int64)], rows=(-9, 0))
            ],
        )
    if shape == "cumulative" or shape == "ties":
        return rows.with_columns(
            ["s"], [col("v", int64).sum().over(order_by=[col("o", int64)])]
        )
    if shape == "row_number":
        return rows.with_columns(
            ["rn"], [row_number().over(order_by=[col("o", int64)])]
        )
    raise Error("MARROW_PROFILE_SHAPE: unknown shape ", shape)


def main() raises:
    var n = _parse_int("MARROW_PROFILE_N", 20_000)
    var iters = _parse_int("MARROW_PROFILE_ITERS", 10)
    var shape = getenv("MARROW_PROFILE_SHAPE", "sliding")

    var o = Int64Builder(capacity=n)
    var v = Int64Builder(capacity=n)
    for i in range(n):
        o.append(Int64(i % 16 if shape == "ties" else i))
        v.append(Int64((i * 7919) % n))
    var rows = table(record_batch([o.finish(), v.finish()], names=["o", "v"]))

    var plan = _plan(shape, rows)
    for _ in range(iters):
        keep(plan.execute().num_rows())
