"""Binary-size gate: what a late-bound query parameter costs.

Identical query and plan to `query_scan_typed.mojo` — same fused predicate,
same projection, same `.filter()` pushdown, same comptime-typed scan. The only
differences are the two parameters:

    ParquetScan(path=param("src", string), ...)
    col("a", int64) > param("min-a", int64)

and the tail: `plan.execute_cli()` instead of `print(...execute())`, so
argv-binding and the output-writer dispatch are linked too.

A parameter is structurally a literal plus a pointer dereference resolved once
per batch (`NumericParam.state()` resolves the cell; `.lane()` splats a plain
`Scalar`, byte-identical to a literal's), so the `__text` delta against
`query_scan_typed` was predicted to be near-zero. **It is not** — the
parameters are free, the tail is not:

    query_scan_typed                              2,025,432
    query_param, writers gated out                2,222,132   +196,700
    query_param, writers linked (default build)   2,794,420   +768,988

The +196,700 is `execute_cli` itself — argv splitting, `parse_params`, the
`--help`/`--describe` renderers and the registry — and the further +572,288 is
the Parquet and Arrow IPC writers plus the codec layer behind them, which is
why they are gated behind `-D MARROW_CLI_WRITERS=true`
(`relations.CLI_WRITERS_ENABLED`). Build this gate both ways to reproduce the
two rows.

    pixi run binary_size query_param query_scan_typed
"""

from marrow.expr.values import BoxedValue
from marrow.dtypes import Int64Type, StringType, int64, string, field
from marrow.parquet import leaf_of
from marrow.schema import schema
from marrow.expr.builders import col, param
from marrow.expr.dynamic import DynValue
from marrow.expr.relations import ParquetScan, Project, DynRelation


def main() raises:
    var sch = schema(
        [field("a", int64), field("b", int64), field("name", string)]
    )

    # The scan path and the predicate's right operand are both late-bound.
    var filtered = DynRelation(
        ParquetScan[leaf_of[Int64Type]() | leaf_of[StringType]()](
            path=param("src", string), schema=sch
        )
    ).filter(BoxedValue(col("a", int64) > param("min-a", int64)))

    var values = List[BoxedValue]()
    values.append(BoxedValue(col("a", int64)))
    var proj = Project(
        input=filtered,
        names=["a"],
        values=values^,
        schema=schema([field("a", int64)]),
    )
    var plan = DynRelation(proj^)
    plan.execute_cli()
