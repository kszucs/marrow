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
`query_scan_typed` should be near-zero. See Task 7 / spec open question 2 for
what "near-zero" means and what to do if it isn't.

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
