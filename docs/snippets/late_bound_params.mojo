"""A late-bound parameter, supplied per execution through `Bindings`.

Compiled by `pixi run docs_check`; included by docs/guide/expressions.qmd.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr import col, param, table
from marrow.scalars import Int64Scalar
from marrow.tabular import record_batch


def main() raises:
    var batch = record_batch(
        [array([1, 5, 9], int64).to_dyn()], names=["a"]
    )

    var min_a = param("min-a", int64)
    var plan = table(batch^).filter(col("a", int64) > min_a.copy())

    # One plan, two executions -- the value travels through, not into, the plan.
    print(plan.execute(bindings={"min-a": Int64Scalar(4).to_dyn()}))
    print(plan.execute(bindings={"min-a": Int64Scalar(8).to_dyn()}))
