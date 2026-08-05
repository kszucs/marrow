"""Binary-size gate: an **AOT expression over a Parquet scan**, comptime-typed.

Identical query and plan to `query_scan.mojo` — same fused predicate, same
projection, same `.filter()` pushdown. The only difference is the one parameter
on the scan:

    ParquetScan[leaf_of[Int64Type]() | leaf_of[StringType]()](...)

which says, in the same types the expression is already written in, which leaf
kinds this read can encounter. `ColumnReader._dispatch` resolves a leaf's dtype
at *runtime*, so by default every arm is reachable and every `LeafBuilder` is
linked: a two-type schema still pays for bool, binary, all eleven numeric
widths, the temporal types and all four decimals.

Note the symmetry with the expression above it. `col("a", int64)` is a
`NumericColumn[Int64Type]` — the dtype is *already* a comptime parameter in the
fused lane, and the kernels already use exactly this to fold away branches they
cannot reach. `leaf_of[T]()` is the same move applied to the decode ladder.

The delta against `query_scan` is therefore what the unused half of that ladder
costs, with everything else held equal. See Q4.6.

    pixi run binary_size query_scan_typed query_scan
"""

from marrow.expr.relations import BoxedValue
from marrow.dtypes import Int64Type, StringType, int64, string, field
from marrow.parquet import leaf_of
from marrow.schema import schema
from marrow.expr.values import col, DynValue
from marrow.expr.relations import ParquetScan, Project, DynRelation


def main() raises:
    var sch = schema(
        [field("a", int64), field("b", int64), field("name", string)]
    )

    # The scan is compiled for exactly the leaf kinds this plan can meet.
    var filtered = DynRelation(
        ParquetScan[leaf_of[Int64Type]() | leaf_of[StringType]()](
            path=String("orders.parquet"), schema=sch
        )
    ).filter(BoxedValue(col("a", int64) > col("b", int64)))

    var values = List[BoxedValue]()
    values.append(BoxedValue(col("a", int64)))
    var proj = Project(
        input=filtered,
        names=["a"],
        values=values^,
        schema=schema([field("a", int64)]),
    )
    print(DynRelation(proj^).execute())
