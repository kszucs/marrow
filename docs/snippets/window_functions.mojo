"""Window functions — ranking and a partitioned running total.

Compiled by `pixi run docs_check`; included by docs/guide/expressions.qmd.
"""

from marrow.builders import array
from marrow.dtypes import int64, string
from marrow.expr import col, row_number, table
from marrow.tabular import record_batch


def main() raises:
    var sales = record_batch(
        [
            array(["east", "west", "east", "west"]).to_dyn(),
            array([10, 20, 30, 40], int64).to_dyn(),
        ],
        names=["region", "amount"],
    )

    var out = (
        table(sales^)
        .with_columns(
            ["rn", "running"],
            [
                row_number().over(order_by=[col("amount", int64)]),
                col("amount", int64)
                .sum()
                .over(
                    partition_by=[col("region", string)],
                    order_by=[col("amount", int64)],
                ),
            ],
        )
        .execute()
    )

    print(out)
