"""The AOT lane's product surface, end to end — and the gate that keeps it cheap.

This is what a user of `marrow compile` actually writes: a fused plan over a
known schema, a late-bound threshold, a late-bound input path, and `cli.run()`.
Compiled it is a standalone `SELECT` over a Parquet file with `--help`,
`--describe`, `-o` and `--format`:

    marrow compile benchmarks/binary_size/query_cli.mojo -o orders
    ./orders orders.parquet --min-amount 250
    ./orders orders.parquet --format csv | head

Every expression here is comptime — `col("amount", int64)` is a
`Column[Int64Type]`, the predicate fuses into one SIMD loop — so the gate's job
is to show that wrapping a plan in a command-line surface does **not** drag the
runtime interpreter in behind it. `marrow::expr::runtime` must link 0 symbols:

    pixi run binary_size query_cli query_scan

`run()` is called without `[parquet=True]` / `[ipc=True]`, so the Parquet and
Arrow IPC *writers* are gated out and this binary prints or writes text. That
is the default on purpose: the writers are the largest thing the CLI layer can
pull in, and a query that pipes its result should not pay for them.
"""

from marrow.dtypes import field, int64, string
from marrow.expr import QueryCli, col, scan
from marrow.schema import schema


def main() raises:
    var cli = QueryCli(
        String("orders"),
        description=String("Orders at or above a minimum amount."),
    )
    var min_amount = cli.param(
        String("min-amount"),
        int64,
        default=Int64(0),
        help=String("keep orders whose amount is at least this"),
    )
    cli.argument(String("src"), help=String("input Parquet file"))

    if cli.parse():
        var sch = schema(
            [
                field("id", int64),
                field("amount", int64),
                field("name", string),
            ]
        )
        cli.run(
            scan(cli.get(String("src")), sch^).filter(
                col("amount", int64) >= min_amount
            )
        )
