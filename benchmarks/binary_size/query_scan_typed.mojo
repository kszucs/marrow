"""Binary-size gate: a Parquet scan with its **leaf set pinned at comptime**.

⚠️ **This gate is currently degenerate: it is byte-for-byte the same program as
`query_scan.mojo`, and the delta it exists to report is 0 by construction.**
Read the whole docstring before quoting a number from it.

What it measured, and should measure again. The old package's scan took the
leaf kinds it could encounter as a comptime parameter:

    ParquetScan[leaf_of[Int64Type]() | leaf_of[StringType]()](...)

`ColumnReader._dispatch` resolves a leaf's dtype at *runtime*, so without that
parameter every arm is reachable and every `LeafBuilder` is linked: a two-type
schema still pays for bool, binary, all eleven numeric widths, the temporal
types and all four decimals. Pinning the set is the same move the fused lane
already makes for expressions — `col("a", int64)` is a `Column[Int64Type]`, so
the dtype is a comptime parameter and the kernels fold away branches they
cannot reach; `leaf_of[T]()` applies it to the decode ladder. The delta against
`query_scan` was what the unused half of that ladder costs, with everything
else held equal. See Q4.6.

**Why it is degenerate now.** `marrow.expr`'s `ParquetScan`
(`marrow/expr/logical.mojo`) takes no parameters at all, and
`ParquetScanOperator` (`marrow/expr/physical.mojo`) hardcodes
`ParquetFile[MappedFile, LeafSet.all()]`. There is no way to narrow the leaf
set through the plan layer, so this program can only be `query_scan` again —
it links the *whole* ladder, which is precisely the thing it was written to
show the cost of avoiding.

Measured 2026-08-29: `query_scan` and `query_scan_typed` came out at the
identical `__text` of 2,431,724 bytes, with identical symbol counts and
identical `marrow::parquet::*` buckets. That is the degeneracy, confirmed
rather than assumed.

It is kept rather than deleted because the suite's value is cross-program
comparability and the set is quoted as a whole; the honest reading until
`ParquetScan` regains a leaf-set parameter is that it duplicates `query_scan`,
costs one `-O3` build in every sweep, and reports nothing `query_scan` does
not. The recorded baseline predates the port and is stale.
"""

from marrow.dtypes import field, int64, string
from marrow.expr import col, scan
from marrow.expr import Gt
from marrow.expr import DynValue
from marrow.schema import schema


def main() raises:
    var sch = schema(
        [field("a", int64), field("b", int64), field("name", string)]
    )
    # No `[leaf_of[Int64Type]() | leaf_of[StringType]()]` here: the plan layer
    # has nowhere to put it. See the docstring.
    var values: List[DynValue] = [col("a", int64)]
    print(
        scan(String("orders.parquet"), sch^)
        .filter(Gt(col("a", int64), col("b", int64)))
        .project(["a"], values^)
        .execute()
    )
