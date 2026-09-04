"""A whole query as one sentence — filter, group, having, order, limit.

Compiled by `pixi run -e dev docs_check`; included by docs/guide/expressions.qmd.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr import col, lit, table
from marrow.tabular import record_batch


def main() raises:
    var orders = record_batch(
        [
            array([1, 2, 1, 2], int64).copy(),
            array([10, 20, None, 40], int64).copy(),
        ],
        names=["customer", "amount"],
    )

    var out = (
        table(orders^)
        .filter(col("amount", int64) > lit(5, int64))
        .aggregate(
            [col("amount", int64).sum().alias("total")],
            [col("customer", int64)],
        )
        .filter(col("total", int64) > lit(5, int64))
        .sort_by([col("total", int64)], [False])
        .limit(1)
        .execute()
    )

    print(out)
