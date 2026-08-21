"""Statistics-based predicate pruning for the expression layer.

A *pruning* evaluation answers, for a whole row group or data page: *could this
predicate be true for some row, given only each column's `[min, max]` bounds?*
When the answer is a definite no, the row group / page can be skipped without
decoding it. Correctness never depends on the result: a scan still applies the
exact predicate to whatever it decodes, so a wrong "maybe" costs time and only a
wrong "no" would cost correctness.

This module owns the *plan-level* half: `PruneStats`, the per-column bounds view
fed by a row group's ColumnStatistics or a page's ColumnIndex entry. The
*operator* half — what `<` or `and` mean over `[lo, hi]` — is a kernel family
like any other and lives in `marrow.kernels.interval`, which this imports. Both
`DynValue` (the runtime lane) and the fused comptime `Value` nodes implement
`prune(stats)` by driving those kernels, so either kind of expression can be
evaluated against the index.
"""

from ..scalars import DynScalar
from ..schema import Schema
from ..kernels.interval import Interval


struct PruneStats(Copyable, Movable):
    """Per-column `[min, max]` bounds for one row group or page. A `None` bound
    means the column has no usable statistic (treated as unknown)."""

    var schema: Schema
    var mins: List[Optional[DynScalar]]
    var maxs: List[Optional[DynScalar]]

    def __init__(
        out self,
        var schema: Schema,
        var mins: List[Optional[DynScalar]],
        var maxs: List[Optional[DynScalar]],
    ):
        self.schema = schema^
        self.mins = mins^
        self.maxs = maxs^

    def by_index(
        self, i: Int
    ) -> Tuple[Optional[DynScalar], Optional[DynScalar]]:
        if i < 0 or i >= len(self.mins):
            return (None, None)
        return (self.mins[i].copy(), self.maxs[i].copy())

    def by_name(
        self, name: String
    ) raises -> Tuple[Optional[DynScalar], Optional[DynScalar]]:
        return self.by_index(self.schema.get_field_index(name))
